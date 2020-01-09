"""
Helper functions for Trainers

Taken from Allennlp, access via https://github.com/allenai/allennlp/blob/master/allennlp/training/util.py

"""
from typing import Any, Union, Dict, Iterable, List, Optional
import datetime
import logging
import os
import shutil
import statistics
import pandas as pd
import csv

import numpy as np

import torch
from torch.nn.parallel import replicate, parallel_apply
from torch.nn.parallel.scatter_gather import gather

from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common.params import Params
from allennlp.common.tqdm import Tqdm
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator
from allennlp.data.iterators.data_iterator import TensorDict
from allennlp.models.model import Model
from allennlp.models.archival import CONFIG_NAME
from allennlp.nn import util as nn_util

logger = logging.getLogger(__name__)

# We want to warn people that tqdm ignores metrics that start with underscores
# exactly once. This variable keeps track of whether we have.
class HasBeenWarned:
    tqdm_ignores_underscores = False


def sparse_clip_norm(parameters, max_norm, norm_type=2) -> float:
    """Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Supports sparse gradients.
    Parameters
    ----------
    parameters : ``(Iterable[torch.Tensor])``
        An iterable of Tensors that will have gradients normalized.
    max_norm : ``float``
        The max norm of the gradients.
    norm_type : ``float``
        The type of the used p-norm. Can be ``'inf'`` for infinity norm.
    Returns
    -------
    Total norm of the parameters (viewed as a single vector).
    """
    # pylint: disable=invalid-name,protected-access
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            if p.grad.is_sparse:
                # need to coalesce the repeated indices before finding norm
                grad = p.grad.data.coalesce()
                param_norm = grad._values().norm(norm_type)
            else:
                param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad.is_sparse:
                p.grad.data._values().mul_(clip_coef)
            else:
                p.grad.data.mul_(clip_coef)
    return total_norm


def move_optimizer_to_cuda(optimizer):
    """
    Move the optimizer state to GPU, if necessary.
    After calling, any parameter specific state in the optimizer
    will be located on the same device as the parameter.
    """
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.is_cuda:
                param_state = optimizer.state[param]
                for k in param_state.keys():
                    if isinstance(param_state[k], torch.Tensor):
                        param_state[k] = param_state[k].cuda(device=param.get_device())


def get_batch_size(batch: Union[Dict, torch.Tensor]) -> int:
    """
    Returns the size of the batch dimension. Assumes a well-formed batch,
    returns 0 otherwise.
    """
    if isinstance(batch, torch.Tensor):
        return batch.size(0) # type: ignore
    elif isinstance(batch, Dict):
        return get_batch_size(next(iter(batch.values())))
    else:
        return 0


def time_to_str(timestamp: int) -> str:
    """
    Convert seconds past Epoch to human readable string.
    """
    datetimestamp = datetime.datetime.fromtimestamp(timestamp)
    return '{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}'.format(
        datetimestamp.year, datetimestamp.month, datetimestamp.day,
        datetimestamp.hour, datetimestamp.minute, datetimestamp.second
    )


def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split('-')]
    return datetime.datetime(*pieces)


def datasets_from_params(params: Params) -> Dict[str, Iterable[Instance]]:
    """
    Load all the datasets specified by the config.
    """
    dataset_reader = DatasetReader.from_params(params.pop('dataset_reader'))
    validation_dataset_reader_params = params.pop("validation_dataset_reader", None)

    validation_and_test_dataset_reader: DatasetReader = dataset_reader
    if validation_dataset_reader_params is not None:
        logger.info("Using a separate dataset reader to load validation and test data.")
        validation_and_test_dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)

    train_data_path = params.pop('train_data_path')
    logger.info("Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path)

    datasets: Dict[str, Iterable[Instance]] = {"train": train_data}

    validation_data_path = params.pop('validation_data_path', None)
    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = validation_and_test_dataset_reader.read(validation_data_path)
        datasets["validation"] = validation_data

    test_data_path = params.pop("test_data_path", None)
    if test_data_path is not None:
        logger.info("Reading test data from %s", test_data_path)
        test_data = validation_and_test_dataset_reader.read(test_data_path)
        datasets["test"] = test_data

    return datasets


def create_serialization_dir(
        params: Params,
        serialization_dir: str,
        recover: bool,
        force: bool) -> None:
    """
    This function creates the serialization directory if it doesn't exist.  If it already exists
    and is non-empty, then it verifies that we're recovering from a training with an identical configuration.
    Parameters
    ----------
    params: ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: ``str``
        The directory in which to save results and logs.
    recover: ``bool``
        If ``True``, we will try to recover from an existing serialization directory, and crash if
        the directory doesn't exist, or doesn't match the configuration we're given.
    force: ``bool``
        If ``True``, we will overwrite the serialization directory if it already exists.
    """
    if recover and force:
        raise ConfigurationError("Illegal arguments: both force and recover are true.")

    if os.path.exists(serialization_dir) and force:
        shutil.rmtree(serialization_dir)

    if os.path.exists(serialization_dir) and os.listdir(serialization_dir):
        if not recover:
            raise ConfigurationError(f"Serialization directory ({serialization_dir}) already exists and is "
                                     f"not empty. Specify --recover to recover training from existing output.")

        logger.info(f"Recovering from prior training at {serialization_dir}.")

        recovered_config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(recovered_config_file):
            raise ConfigurationError("The serialization directory already exists but doesn't "
                                     "contain a config.json. You probably gave the wrong directory.")
        else:
            loaded_params = Params.from_file(recovered_config_file)

            # Check whether any of the training configuration differs from the configuration we are
            # resuming.  If so, warn the user that training may fail.
            fail = False
            flat_params = params.as_flat_dict()
            flat_loaded = loaded_params.as_flat_dict()
            for key in flat_params.keys() - flat_loaded.keys():
                logger.error(f"Key '{key}' found in training configuration but not in the serialization "
                             f"directory we're recovering from.")
                fail = True
            for key in flat_loaded.keys() - flat_params.keys():
                logger.error(f"Key '{key}' found in the serialization directory we're recovering from "
                             f"but not in the training config.")
                fail = True
            for key in flat_params.keys():
                if flat_params.get(key, None) != flat_loaded.get(key, None):
                    logger.error(f"Value for '{key}' in training configuration does not match that the value in "
                                 f"the serialization directory we're recovering from: "
                                 f"{flat_params[key]} != {flat_loaded[key]}")
                    fail = True
            if fail:
                raise ConfigurationError("Training configuration does not match the configuration we're "
                                         "recovering from.")
    else:
        if recover:
            raise ConfigurationError(f"--recover specified but serialization_dir ({serialization_dir}) "
                                     "does not exist.  There is nothing to recover from.")
        os.makedirs(serialization_dir, exist_ok=True)


def data_parallel(batch_group: List[TensorDict],
                  model: Model,
                  cuda_devices: List) -> Dict[str, torch.Tensor]:
    """
    Performs a forward pass using multiple GPUs.  This is a simplification
    of torch.nn.parallel.data_parallel to support the allennlp model
    interface.
    """
    assert len(batch_group) <= len(cuda_devices)

    moved = [nn_util.move_to_device(batch, device)
             for batch, device in zip(batch_group, cuda_devices)]

    used_device_ids = cuda_devices[:len(moved)]
    replicas = replicate(model, used_device_ids)
    # We pass all our arguments as kwargs. Create a list of empty tuples of the
    # correct shape to serve as (non-existent) positional arguments.
    inputs = [()] * len(batch_group)
    outputs = parallel_apply(replicas, inputs, moved, used_device_ids)

    # Only the 'loss' is needed.
    # a (num_gpu, ) tensor with loss on each GPU
    losses = gather([output['loss'].unsqueeze(0) for output in outputs], used_device_ids[0], 0)
    return {'loss': losses.mean()}


def enable_gradient_clipping(model: Model, grad_clipping: Optional[float]) -> None:
    if grad_clipping is not None:
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.register_hook(lambda grad: nn_util.clamp_tensor(grad,
                                                                          minimum=-grad_clipping,
                                                                          maximum=grad_clipping))


def rescale_gradients(model: Model, grad_norm: Optional[float] = None) -> Optional[float]:
    """
    Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.
    """
    if grad_norm:
        parameters_to_clip = [p for p in model.parameters()
                              if p.grad is not None]
        return sparse_clip_norm(parameters_to_clip, grad_norm)
    return None


def get_metrics(model: Model, total_loss: float, num_batches: int, reset: bool = False) -> Dict[str, float]:
    """
    Gets the metrics but sets ``"loss"`` to
    the total loss divided by the ``num_batches`` so that
    the ``"loss"`` metric is "average loss per batch".
    """
    metrics = model.get_metrics(reset=reset)
    metrics["loss"] = float(total_loss / num_batches) if num_batches > 0 else 0.0
    return metrics


def evaluate(model: Model,
             instances: Iterable[Instance],
             data_iterator: DataIterator,
             cuda_device: int,
             batch_weight_key: str) -> Dict[str, Any]:
    check_for_gpu(cuda_device)
    with torch.no_grad():
        model.eval()

        iterator = data_iterator(instances,
                                 num_epochs=1,
                                 shuffle=False)
        logger.info("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))

        # Number of batches in instances.
        batch_count = 0
        # Number of batches where the model produces a loss.
        loss_count = 0
        # Cumulative weighted loss
        total_loss = 0.0
        # Cumulative weight across all batches.
        total_weight = 0.0

        for batch in generator_tqdm:
            batch_count += 1
            batch = nn_util.move_to_device(batch, cuda_device)
            output_dict = model(**batch)
            loss = output_dict.get("loss")

            metrics = model.get_metrics()

            if loss is not None:
                loss_count += 1
                if batch_weight_key:
                    weight = output_dict[batch_weight_key].item()
                else:
                    weight = 1.0

                total_weight += weight
                total_loss += loss.item() * weight
                # Report the average loss so far.
                metrics["loss"] = total_loss / total_weight

            if (not HasBeenWarned.tqdm_ignores_underscores and
                    any(metric_name.startswith("_") for metric_name in metrics)):
                logger.warning("Metrics with names beginning with \"_\" will "
                               "not be logged to the tqdm progress bar.")
                HasBeenWarned.tqdm_ignores_underscores = True
            description = ', '.join(["%s: %.4f" % (name, value) for name, value
                                     in metrics.items() if not name.startswith("_")]) + " ||"
            generator_tqdm.set_description(description, refresh=False)

        final_metrics = model.get_metrics(reset=True)
        if loss_count > 0:
            # Sanity check
            if loss_count != batch_count:
                raise RuntimeError("The model you are trying to evaluate only sometimes " +
                                   "produced a loss!")
            final_metrics["loss"] = total_loss / total_weight

        return final_metrics


def description_from_metrics(metrics: Dict[str, float]) -> str:
    if (not HasBeenWarned.tqdm_ignores_underscores and
            any(metric_name.startswith("_") for metric_name in metrics)):
        logger.warning("Metrics with names beginning with \"_\" will "
                       "not be logged to the tqdm progress bar.")
        HasBeenWarned.tqdm_ignores_underscores = True
    return ', '.join(["%s: %.4f" % (name, value)
                      for name, value in
                      metrics.items() if not name.startswith("_")]) + " ||"


def is_retweet(source_tweet_json : dict):
    """
    a temporary solution to load actual source tweet data from "retweeted_status" in our augmented dataset
    :param source_tweet_json:
    :return:
    """
    return "retweeted_status" in source_tweet_json


def get_original_source_tweet (source_tweet_json : dict):
    if "retweeted_status" in source_tweet_json:
        return source_tweet_json["retweeted_status"]


def get_tweet_content(tweet_json: dict):
    if "text" in tweet_json:
        return tweet_json["text"]
    elif "full_text" in tweet_json:
        return tweet_json["full_text"]
    else:
        raise Exception("text is not exist in tweet [%s]! " % tweet_json["id_str"])


def social_context_dataset_statistics():
    """
    Utility method to perform social context (PHEME dataset structure based) corpus statistics by a given path

    :return:
    """
    # social_context_data_dir = "C:\\Data\\NLP-corpus\\aug_rnr\\twitter1516"
    from data_loader import load_abs_path
    social_context_data_dir = os.path.join(os.path.dirname(__file__),  '..', "data", "social_context","aug-rnr-annotated-threads-retweets")
    social_context_data_dir = load_abs_path(social_context_data_dir)

    print("check social context corpus [%s] ... " % social_context_data_dir)
    events_dataset_dirs = []
    for root, dirs, files in os.walk(social_context_data_dir):
        # print("root: ", root)
        # print("dirs: ", dirs)
        # print("files size: ", len(files))
        events_dataset_dirs = dirs
        break

    print("total [%s] events dataset" % len(events_dataset_dirs))
    print(events_dataset_dirs)

    # check every individual event corpus
    for event_dataset_dir in events_dataset_dirs:
        labelled_event_dataset_statistics(social_context_data_dir, event_dataset_dir)
        print(" ========================================== ")

    print("complete.")


def labelled_event_dataset_statistics(social_context_data_dir, event_dataset_dir):
    """
    check every individual event corpus

    :param social_context_data_dir:
    :param event_dataset_dir:
    :return: {"non-rumours": {}, "rumours": {}}
    """
    print(" --- [%s] corpus statistics ---- " % event_dataset_dir)
    event_types_dirs = []
    for root, dirs, files in os.walk(os.path.join(social_context_data_dir, event_dataset_dir)):
        event_types_dirs = dirs
        break

    # print("labelled event types : ", event_types_dirs) -> ['non-rumours', 'rumours']
    # check labelled event type (i.e, rumour or non-rumour) corpus
    results = dict()
    for event_types_dir in event_types_dirs:
        event_type_results = labelled_event_type_statistics(social_context_data_dir, event_dataset_dir, event_types_dir)
        results[event_types_dir] = event_type_results

    return results


def labelled_event_type_statistics(social_context_data_dir, event_dataset_dir, event_type_i):
    """
    check every individual event (rumour or non-rumour) corpus and count the number of retweets and reactions

    :param social_context_data_dir:
    :param event_dataset_dir:
    :param event_type_i:
    :return:
    """
    labelled_instances = []
    for root, dirs, files in os.walk(os.path.join(social_context_data_dir, event_dataset_dir, event_type_i)):
        labelled_instances = dirs
        break
    print("total labelled [%s] events : " % event_type_i, len(labelled_instances))

    # every individual labelled instance level statistics
    all_reactions = []
    all_retweets = []
    num_of_non_reactions = 0
    for labelled_instance in labelled_instances:
        instance_statistics = labelled_instance_metadata_statistics(social_context_data_dir, event_dataset_dir, event_type_i, labelled_instance)
        # print(instance_statistics)
        inst_reactions = instance_statistics["reactions"]
        inst_retweets = instance_statistics["retweets"]
        all_reactions.append(inst_reactions)
        all_retweets.append(inst_retweets)

    total_reactions = sum(all_reactions)
    min_reactions = 0 if len(all_reactions) == 0 else min(all_reactions)
    max_reactions = 0 if len(all_reactions) == 0 else max(all_reactions)
    avg_reactions = 0 if len(all_reactions) == 0 else round(sum(all_reactions)/len(all_reactions), 1)
    std_reactions = 0 if len(all_reactions) == 0 else statistics.stdev(all_reactions)
    # Median has a very big advantage over Mean, which is the median value is not skewed so much by extremely large or small values.
    # see also https://www.geeksforgeeks.org/python-statistics-median/
    median_reactions = 0 if len(all_reactions) == 0 else statistics.median(all_reactions)

    total_retweets = sum(all_retweets)
    min_retweets = 0 if len(all_retweets) == 0 else min(all_retweets)
    max_retweets = 0 if len(all_retweets) == 0 else max(all_retweets)
    avg_retweets = 0 if len(all_retweets) == 0 else round(total_retweets / len(all_retweets), 1)
    std_retweets = 0 if len(all_retweets) == 0 else statistics.stdev(all_retweets)
    # Median has a very big advantage over Mean, which is the median value is not skewed so much by extremely large or small values,
    # see also https://www.geeksforgeeks.org/python-statistics-median/
    median_retweets = 0 if len(all_retweets) == 0 else statistics.median(all_retweets)

    print("[%s] total reactions: [%s], min reaction: [%s], max reaction: [%s], avg reaction: [%s], std reactions: [%s], median reactions: [%s]" %
          (event_dataset_dir, total_reactions, min_reactions, max_reactions, avg_reactions, std_reactions, median_reactions))
    print("[%s] total retweets: [%s], min retweets: [%s], max retweets: [%s], avg retweets: [%s], std retweets: [%s], median retweets: [%s]" %
          (event_dataset_dir, total_retweets, min_retweets, max_retweets, avg_retweets, std_retweets, median_retweets))
    print("[%s] total tweets without reaction: [%s]"% (event_dataset_dir, all_reactions.count(0)))
    print("[%s] total tweets without retweets: [%s]"% (event_dataset_dir, all_retweets.count(0)))

    results = dict()
    results["total_reactions"] = total_reactions
    results["min_reactions"] = min_reactions
    results["max_reactions"] = max_reactions
    results["avg_reactions"] = avg_reactions
    results["std_reactions"] = std_reactions
    results["median_reactions"] = median_reactions

    results["total_retweets"] = total_retweets
    results["min_retweets"] = min_retweets
    results["max_retweets"] = max_retweets
    results["avg_retweets"] = avg_retweets
    results["std_retweets"] = std_retweets
    results["median_retweets"] = median_retweets

    return results


def labelled_instance_metadata_statistics(social_context_data_dir, event_dataset_dir, event_type_i, labelled_instance_i) -> dict:
    """
    every individual labelled instance (i.e., rumour instance or non-rumour instance) level statistics

    # e.g., <path>/gurlitt/rumours/536833104280055808

    :param social_context_data_dir:
    :param event_dataset_dir:
    :param event_type_i:
    :param labelled_instance_i:
    :param labelled_instance_metadata_dirs:
    :return: dict, e.g., {'reactions': 175, 'retweets': 79, 'source-tweets': 1}
    """
    results = dict()

    # labelled_instance_i = labelled_instances[0]
    labelled_instance_metadata_dirs = []
    for root, dirs, files in os.walk(os.path.join(social_context_data_dir, event_dataset_dir, event_type_i, labelled_instance_i)):
        labelled_instance_metadata_dirs = dirs
        break

    # print("labelled_instance_metadata_dirs: ", labelled_instance_metadata_dirs) -> ['reactions', 'retweets', 'source-tweets']
    for labelled_instance_metadata_dirs_i in labelled_instance_metadata_dirs:
        # labelled_instance_metadata_dirs_i = labelled_instance_metadata_dirs[0]
        labelled_instance_metadata_dirs_i_size = 0

        for root, dirs, files in os.walk(os.path.join(social_context_data_dir, event_dataset_dir, event_type_i, labelled_instance_i, labelled_instance_metadata_dirs_i)):
            # print(root)
            json_files = [file for file in files if not file.startswith(".")]
            labelled_instance_metadata_dirs_i_size = len(json_files)
            results [labelled_instance_metadata_dirs_i] = labelled_instance_metadata_dirs_i_size

    # set default
    if "reactions" not in results:
        results["reactions"] = 0

    if "retweets" not in results:
        results["retweets"] = 0

    return results


def load_matrix_from_csv(fname, start_col_index, end_col_index, delimiter=',', encoding='utf-8',
                          header=None, result_type = -1):
    """
    load gs terms (one term per line) from "csv" txt file
    :param fname:
    :param start_col_index:
    :param end_col_index:
    :param encoding:
    :param header default as None, header=0 denotes the first line of data
    :return:
    """
    print("reading data set from csv file at: ", fname)
    df = pd.read_csv(fname, header=header, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL,
                     usecols=range(start_col_index, end_col_index), lineterminator='\n',
                     encoding=encoding)

    if result_type > 0:
        return df

    return df.as_matrix()


def statistics_rumour_dnn_dataset(file_name):
    """
    perform statistics of social context for a given training data set file

    :param file_name:
    :return:
    """
    print("statistics of [%s]" % file_name)

    df_file = load_matrix_from_csv(file_name, 0, 1, header=0)

    #for dataset_row in df_file[:]:
    #    print("tweet id: [%s]" % dataset_row[0])

    all_tweet_ids = [dataset_row[0] for dataset_row in df_file[:]]
    print("all_tweet_ids size: ", len(all_tweet_ids))

    from data_loader import load_tweets_context_dataset_dir
    from data_loader import load_abs_path

    social_context_data_dir = os.path.join(os.path.dirname(__file__),  '..', "data", "social_context","aug-rnr-annotated-threads-retweets")
    social_context_data_dir = load_abs_path(social_context_data_dir)
    context_tweets_dataset_dir_dict = load_tweets_context_dataset_dir(social_context_data_dir)

    all_replies_list = []
    all_retweets_list = []
    for tweet_id in all_tweet_ids:
        total_replies, total_retweets = count_social_context(str(tweet_id), context_tweets_dataset_dir_dict)
        all_replies_list.append(total_replies)
        all_retweets_list.append(total_retweets)

    print("total_replies_list: ", all_replies_list)
    print("total_retweets_list: ", all_retweets_list)

    total_reactions = sum(all_replies_list)
    min_reactions = 0 if len(all_replies_list) == 0 else min(all_replies_list)
    max_reactions = 0 if len(all_replies_list) == 0 else max(all_replies_list)
    avg_reactions = 0 if len(all_replies_list) == 0 else round(sum(all_replies_list)/len(all_replies_list), 1)
    std_reactions = 0 if len(all_replies_list) == 0 else statistics.stdev(all_replies_list)
    # Median has a very big advantage over Mean, which is the median value is not skewed so much by extremely large or small values.
    # see also https://www.geeksforgeeks.org/python-statistics-median/
    median_reactions = 0 if len(all_replies_list) == 0 else statistics.median(all_replies_list)

    total_retweets = sum(all_retweets_list)
    min_retweets = 0 if len(all_retweets_list) == 0 else min(all_retweets_list)
    max_retweets = 0 if len(all_retweets_list) == 0 else max(all_retweets_list)
    avg_retweets = 0 if len(all_retweets_list) == 0 else round(total_retweets / len(all_retweets_list), 1)
    std_retweets = 0 if len(all_retweets_list) == 0 else statistics.stdev(all_retweets_list)
    # Median has a very big advantage over Mean, which is the median value is not skewed so much by extremely large or small values,
    # see also https://www.geeksforgeeks.org/python-statistics-median/
    median_retweets = 0 if len(all_retweets_list) == 0 else statistics.median(all_retweets_list)

    print("total reactions: [%s], min reaction: [%s], max reaction: [%s], avg reaction: [%s], std reactions: [%s], median reactions: [%s]" %
          (total_reactions, min_reactions, max_reactions, avg_reactions, std_reactions, median_reactions))
    print("total retweets: [%s], min retweets: [%s], max retweets: [%s], avg retweets: [%s], std retweets: [%s], median retweets: [%s]" %
          (total_retweets, min_retweets, max_retweets, avg_retweets, std_retweets, median_retweets))

    print("total tweets without reaction: [%s]"% (all_replies_list.count(0)))
    print("total tweets without retweets: [%s]"% (all_retweets_list.count(0)))

    results = dict()
    results["total_reactions"] = total_reactions
    results["min_reactions"] = min_reactions
    results["max_reactions"] = max_reactions
    results["avg_reactions"] = avg_reactions
    results["std_reactions"] = std_reactions
    results["median_reactions"] = median_reactions

    results["total_retweets"] = total_retweets
    results["min_retweets"] = min_retweets
    results["max_retweets"] = max_retweets
    results["avg_retweets"] = avg_retweets
    results["std_retweets"] = std_retweets
    results["median_retweets"] = median_retweets

    print("statistics: ")
    print(results)


def count_social_context(source_tweet_id, context_tweets_dataset_dir_dict):
    context_tweets_dataset_dir = context_tweets_dataset_dir_dict[source_tweet_id]

    context_types = ["reactions", 'retweets']

    total_replies = 0
    total_retweets = 0

    for c_type in context_types:
        all_context_data_dir = os.path.join(context_tweets_dataset_dir, "{}".format(c_type))
        reaction_dir = os.path.join(all_context_data_dir)
        if not os.path.isdir(reaction_dir):
            # reaction ('replies' or 'retweets') not exist
            continue

        source_tweet_reaction_json_dataset = os.listdir(reaction_dir)
        for source_tweet_reaction_json_file_name in source_tweet_reaction_json_dataset:
            if source_tweet_reaction_json_file_name.startswith("."):
                continue

            if "reactions" == c_type:
                total_replies += 1

            if "retweets" == c_type:
                total_retweets += 1

    return total_replies, total_retweets


def global_norm(input_x: np.ndarray, global_means: np.ndarray, global_stds: np.ndarray, eps: torch.float32 = 1e-6):
    return (input_x - global_means) / (global_stds + eps)
    

def shuffle_dataset(csv_file_path: str):
    """
    shuffling our training set and test set
    :param file_path:
    :return:
    """
    # df = pd.read_csv(csv_file_path, encoding="utf-8", quoting=csv.QUOTE_MINIMAL, lineterminator='\n', header=0)
    df = load_matrix_from_csv(csv_file_path, 0, 6, header=0, result_type=1)
    ds = df.sample(frac=1)

    ds.to_csv(csv_file_path.replace(".csv", "_shuffled.csv"), encoding="utf-8", quoting=csv.QUOTE_MINIMAL, index=False)


def test_dataset(csv_file_path: str):
    df = load_matrix_from_csv(csv_file_path, start_col_index=0, end_col_index=4, header=0)
    rumor_num = 0
    non_rumor_num = 0
    for tweet_row in df[:]:
        tweet_id = tweet_row[0]
        created_time = tweet_row[1]
        tweet_text = tweet_row[2]
        tag = tweet_row[3]
        # print("tag: ", tag)
        tag = int(tag)
        if tag == 1:
            rumor_num+=1
        else:
            non_rumor_num+=1

    print("num of rumor instance: ", rumor_num)
    print("num of non-rumor instance: ", non_rumor_num)


def generate_shuffled_training_set():
    shuffle_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","bostonbombings", "aug_rnr_train_set_combined.csv"))
    shuffle_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","bostonbombings", "aug_rnr_heldout_set_combined.csv"))

    shuffle_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","charliehebdo", "aug_rnr_train_set_combined.csv"))
    shuffle_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","charliehebdo", "aug_rnr_heldout_set_combined.csv"))

    shuffle_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","ferguson", "aug_rnr_train_set_combined.csv"))
    shuffle_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","ferguson", "aug_rnr_heldout_set_combined.csv"))

    shuffle_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","germanwings", "aug_rnr_train_set_combined.csv"))
    shuffle_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","germanwings", "aug_rnr_heldout_set_combined.csv"))

    shuffle_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","ottawashooting", "aug_rnr_train_set_combined.csv"))
    shuffle_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","ottawashooting", "aug_rnr_heldout_set_combined.csv"))

    shuffle_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","sydneysiege", "aug_rnr_train_set_combined.csv"))
    shuffle_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","sydneysiege", "aug_rnr_heldout_set_combined.csv"))

    test_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","bostonbombings", "aug_rnr_train_set_combined_shuffled.csv"))
    test_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","bostonbombings", "aug_rnr_heldout_set_combined_shuffled.csv"))

    test_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","charliehebdo", "aug_rnr_train_set_combined_shuffled.csv"))
    test_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","charliehebdo", "aug_rnr_heldout_set_combined_shuffled.csv"))

    test_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","ferguson", "aug_rnr_train_set_combined_shuffled.csv"))
    test_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","ferguson", "aug_rnr_heldout_set_combined_shuffled.csv"))

    test_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","germanwings", "aug_rnr_train_set_combined_shuffled.csv"))
    test_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","germanwings", "aug_rnr_heldout_set_combined_shuffled.csv"))

    test_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","ottawashooting", "aug_rnr_train_set_combined_shuffled.csv"))
    test_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","ottawashooting", "aug_rnr_heldout_set_combined_shuffled.csv"))

    test_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","sydneysiege", "aug_rnr_train_set_combined_shuffled.csv"))
    test_dataset(os.path.join(os.path.dirname(__file__),  '..', "data", "train","sydneysiege", "aug_rnr_heldout_set_combined_shuffled.csv"))

if __name__ == '__main__':
    # social_context_dataset_statistics()
    #twitter16_test_set = os.path.join(os.path.dirname(__file__),  '..', "data", "test","twitter16_test_set.csv")
    #statistics_rumour_dnn_dataset(twitter16_test_set)
    generate_shuffled_training_set()