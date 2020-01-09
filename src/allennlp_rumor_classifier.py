from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder, PytorchSeq2SeqWrapper

from training_util import global_norm

try:
    import sys  # Just in case

    # to fix a weird crash due to "ValueError: failed to parse CPython sys.version '3.6.6 |Anaconda, Inc.| (default, Jun 28 2018, 11:27:44) [MSC v.1900 64 bit (AMD64)]'"
    # possible due to a bug on anaconda
    # see https://stackoverflow.com/questions/34145861/valueerror-failed-to-parse-cpython-sys-version-after-using-conda-command
    start = sys.version.index('|')  # Do we have a modified sys.version?
    end = sys.version.index('|', start + 1)
    version_bak = sys.version  # Backup modified sys.version
    sys.version = sys.version.replace(sys.version[start:end + 1], '')  # Make it legible for platform module
    import platform

    platform.python_implementation()  # Ignore result, we just need cache populated
    platform._sys_version_cache[version_bak] = platform._sys_version_cache[sys.version]  # Duplicate cache
    sys.version = version_bak  # Restore modified version string
except ValueError:  # Catch .index() method not finding a pipe
    pass

import csv
import logging
import operator
import os
import sys
from datetime import datetime
from datetime import timedelta
from typing import Iterator, List, Dict, Union, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from allennlp.commands.elmo import ElmoEmbedder
from allennlp.common import JsonDict
from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data import Instance, Field, DataIterator
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField, MetadataField
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder
from allennlp.modules.layer_norm import LayerNorm
from my_layer_norm import MyLayerNorm
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.nn import Activation, InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.predictors import Predictor
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.training.trainer import Trainer
from overrides import overrides

from attention import HierarchicalAttentionNet, StructuredSelfAttention

from context_features_extractor import context_feature_extraction_from_context_status, NUMERICAL_FEATURE_DIM, \
    FEATURE_SETTING_OPTION_METADATA_ONLY, \
    FEATURE_SETTING_OPTION_CONTEXT_CONTENT_ONLY, FEATURE_SETTING_OPTION_CONTEXT_ONLY, FEATURE_SETTING_SC_WITHOUT_CM, \
    FEATURE_SETTING_SC_WITHOUT_CC, DISABLE_SETTING_AUTO_ENCODER
from data_loader import load_source_tweet_json, load_source_tweet_context, readlink_on_windows, load_abs_path
from data_loader import DISABLE_CXT_TYPE_RETWEET

# from context_features_extractor import context_feature_extraction
# https://github.com/mhagiwara/realworldnlp/blob/master/examples/sentiment/sst_classifier.py

# global variables
# ========================================================================
# credbank fine-tuned model path
elmo_credbank_model_path = load_abs_path(
    os.path.join(os.path.dirname(__file__), '..', "resource", "embedding", "elmo_model",
                 "elmo_credbank_2x4096_512_2048cnn_2xhighway_weights_10052019.hdf5"))

FEATURE_SETTING_OPTION_FULL = -1
FEATURE_SETTING_OPTION_SOURCE_TWEET_CONTENT_ONLY = 1
SOCIAL_CONTEXT_ENCODER_OPTION_LSTM = 1
SOCIAL_CONTEXT_ENCODER_OPTION_TRANSFORMER = 2

ATTENTION_OPTION_NONE = 0  # no attention
ATTENTION_OPTION_ATTENTION_WITH_CXT = 1  # AttentionWithContext
ATTENTION_SELF_ATTENTATIVE_NET = 2  # self_attention_net

MAXIMUM_CONTEXT_SEQ_SIZE = 200
# =========================================================================

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def timestamped_print(msg):
    logger.info(msg)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(ts + " :: " + msg)


@DatasetReader.register("rumor_tweets_train_set_reader")
class RumorTweetsDataReader(DatasetReader):
    def __init__(self, tokenizer=None, token_indexers: Dict[str, TokenIndexer] = None, lazy: bool = False) -> None:
        super().__init__(lazy)
        timestamped_print("RumorTweetsDataReader ...")
        # can be replaced with a proper tweets specific tokeniser, e.g., https://github.com/myleott/ark-twokenize-py
        self._tokenizer = tokenizer or WordTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def text_to_instance(self, tweet_id: str, sentence: str, tag: str = None) -> Instance:
        # can normalise emoji via a database, e.g., https://github.com/carpedm20/emoji/blob/e7bff32/emoji/unicode_codes.py

        fields: Dict[str, Field] = {}

        tokenized_sentence = self._tokenizer.tokenize(sentence)
        sentence_field = TextField(tokenized_sentence, self.token_indexers)
        fields["sentence"] = sentence_field

        fields['tweet_id'] = MetadataField(str(tweet_id))

        if tag is not None:
            fields['label'] = LabelField(int(tag), skip_indexing=True)

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        file_abs_path = file_path
        try:
            # if the path is not exist, we assume it is symlink and attempt to read the path from that
            if not os.path.exists(file_path) or os.path.islink(file_path):
                if sys.platform == "win32":
                    file_abs_path = readlink_on_windows(file_path)
                else:
                    # read file path from symlink in linux/Mac OS
                    file_abs_path = os.readlink(file_path)
        except:
            print("Failure to read the file from [%s]. "
                  "Please check if the file or the symlink of the file exists ?" % file_path)

        df = self._load_matrix_from_csv(file_abs_path, header=0, start_col_index=0, end_col_index=4)
        for tweet_row in df[:]:
            tweet_id = tweet_row[0]
            created_time = tweet_row[1]
            tweet_text = tweet_row[2]
            tag = tweet_row[3]

            yield self.text_to_instance(tweet_id, tweet_text, tag)

    def _load_matrix_from_csv(self, fname, start_col_index, end_col_index, delimiter=',', encoding='utf-8',
                              header=None):
        """
        load gs terms (one term per line) from "csv" txt file
        :param fname:
        :param start_col_index:
        :param end_col_index:
        :param encoding:
        :param header default as None, header=0 denotes the first line of data
        :return:
        """
        timestamped_print("reading instances from csv file at: %s" % fname)

        df = pd.read_csv(fname, header=header, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL,
                         usecols=range(start_col_index, end_col_index), lineterminator='\n',
                         encoding=encoding).as_matrix()
        return df


@Model.register("rumor_tweets_classifier")
class RumorTweetsClassifer(Model):
    def __init__(self, tweet_text_embedder: TextFieldEmbedder, lang_model_encoder: Seq2VecEncoder,
                 social_context_encoder: Seq2VecEncoder,
                 context_content_encoder_rnn_wrapper: Seq2VecEncoder,
                 context_metadata_encoder_rnn_wrapper: Seq2VecEncoder,
                 vocab: Vocabulary,
                 classifier_feedforward: FeedForward,
                 cxt_content_encoder: Seq2SeqEncoder,
                 cxt_metadata_encoder: Seq2SeqEncoder,
                 social_context_self_attention_encoder: Seq2SeqEncoder = None,
                 cuda_device: int = -1,
                 max_cxt_size: int = MAXIMUM_CONTEXT_SEQ_SIZE,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)

        self.cuda_device = cuda_device
        # We need the embeddings to convert word IDs to their vector representations
        # word_embeddings
        self.tweet_text_embedder = tweet_text_embedder
        # Seq2VecEncoder is a neural network abstraction that takes a sequence of something
        # (usually a sequence of embedded word vectors), processes it, and returns it as a single
        # vector. Oftentimes, this is an RNN-based architecture (e.g., LSTM or GRU), but
        # AllenNLP also supports CNNs and other simple architectures (for example,
        # just averaging over the input vectors).
        self.lang_model_encoder = lang_model_encoder
        # 'social_context_encoder' is deprecated
        self.social_context_encoder = social_context_encoder

        self.cxt_content_encoder = cxt_content_encoder
        self.cxt_metadata_encoder = cxt_metadata_encoder
        self.cxt_content_encoder_wrapper = context_content_encoder_rnn_wrapper
        self.cxt_metadata_encoder_wrapper = context_metadata_encoder_rnn_wrapper

        self.event_varying_timedelta: timedelta = None
        self.disable_cxt_type = DISABLE_CXT_TYPE_RETWEET

        self.feature_setting = FEATURE_SETTING_OPTION_FULL
        self.disable_setting = -1
        # SET default social encoder
        self.set_social_encoder_option(SOCIAL_CONTEXT_ENCODER_OPTION_LSTM)
        # Need to be consistent with maximum size of context extraction
        self.max_cxt_size = max_cxt_size

        # init attentions
        self.attention_option = None
        self.cxt_content_attention = None
        self.cxt_metadata_attention = None
        self.cxt_multimodal_attention = None
        # set default attention mechanism
        self.set_attention_mechanism()

        self.layer_norm_content_attention = None
        self.layer_norm_metadata_attention = None
        self.layer_norm_multimodal_cxt = None
        self.set_layer_norm_layers()

        self.num_tags = 0

        if self.vocab is not None:
            self.num_tags = self.vocab.get_vocab_size('label')

        LABEL_TYPE_RUMOUR: int = 1
        self.accuracy = CategoricalAccuracy()

        timestamped_print("LOOCV Performance is  evaluated  by  computing  the  precision,  recall  and  F1  scores  "
                          "for  the  target category [%s] only, i.e., rumors." % str(LABEL_TYPE_RUMOUR))

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "f1": F1Measure(positive_label=LABEL_TYPE_RUMOUR)
        }

        # We use the cross-entropy loss because this is a classification task.
        # Note that PyTorch's CrossEntropyLoss combines softmax and log likelihood loss,
        # which makes it unnecessary to add a separate softmax layer.

        self.loss_function = torch.nn.CrossEntropyLoss().cuda(
            device=cuda_device) if torch.cuda.is_available() else torch.nn.CrossEntropyLoss()

        self.classifier_feedforward = classifier_feedforward
        self.social_context_self_attention_encoder = social_context_self_attention_encoder

        # can change the following to conform to allennlp framework
        self.elmo_model = ElmoEmbedder(
            options_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
            weight_file=elmo_credbank_model_path,
            cuda_device=cuda_device)

        # batch normalisation for input data of context features With or Without Learnable Parameters
        # In normalization, we ideally want to use the global mean and variance to standardize our data.
        # batch normalisation performance suffer from small batch size
        # https://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/
        self.global_means: np.ndarray = None
        self.global_stds: np.ndarray = None

        if self.global_means is not None and self.global_stds is not None:
            print("use global means and std to standardize numeric features.")

        if self.disable_setting == DISABLE_SETTING_AUTO_ENCODER:
            print("model is set to disable auto encoder.")

        timestamped_print("social context encoder is set to use [%s]" %
                          (
                              "LSTM" if self.social_encoder_option == SOCIAL_CONTEXT_ENCODER_OPTION_LSTM else "Transformer"))

        timestamped_print("enable batch normalisation after global mean and std feature scaling (in batches) .")

        if self.feature_setting == FEATURE_SETTING_OPTION_METADATA_ONLY or self.feature_setting == FEATURE_SETTING_SC_WITHOUT_CC:
            self.bn_input = None
        elif self.feature_setting == FEATURE_SETTING_OPTION_CONTEXT_CONTENT_ONLY or self.feature_setting == FEATURE_SETTING_SC_WITHOUT_CM:
            self.bn_input = None
        elif self.feature_setting == FEATURE_SETTING_OPTION_SOURCE_TWEET_CONTENT_ONLY:
            self.bn_input = None
        else:
            self.bn_input = None

        initializer(self)

    def set_social_encoder_option(self, social_encoder_option):
        self.social_encoder_option = social_encoder_option
        if self.social_encoder_option == SOCIAL_CONTEXT_ENCODER_OPTION_LSTM:
            # cxt_metadata_encoder or cxt_content_encoder can be NULL depending on the feature setting (metadata only or context content only)
            self.social_context_output_layer_norm = LayerNorm(
                self.cxt_metadata_encoder.get_output_dim() if self.cxt_metadata_encoder else 0 +
                                                                                             self.cxt_content_encoder.get_output_dim() if self.cxt_content_encoder else 0)
        else:
            raise Exception("social context encoder option [%s] is not supported!" % str(social_encoder_option))

    def set_disable_cxt_type_option(self, set_disable_cxt_type_option: int = DISABLE_CXT_TYPE_RETWEET):
        self.disable_cxt_type = set_disable_cxt_type_option
        if self.disable_cxt_type:
            timestamped_print("disable_context type  is set to [%s].0: accept all types of context; "
                              "1: disable reply; 2: disable retweet (default)" % self.disable_cxt_type)

    def set_attention_mechanism(self, attention_option: int = ATTENTION_OPTION_ATTENTION_WITH_CXT):
        """
        attention mechanism depends on the max_context_size setting

        see set_feature_setting()

        :param attention_option:
        :return:
        """
        timestamped_print("set attention mechanism (option: [%s])..." % attention_option)

        if attention_option == int(ATTENTION_OPTION_ATTENTION_WITH_CXT):
            if self.feature_setting == FEATURE_SETTING_OPTION_CONTEXT_CONTENT_ONLY or self.feature_setting == FEATURE_SETTING_SC_WITHOUT_CM:
                # otherwise, it means that the context metadata only option is set
                timestamped_print("init cxt_content_attention with dim [%s] and max_cxt_size: [%s]" % (
                    str(self.cxt_content_encoder.get_output_dim()),
                    str(self.max_cxt_size)))
                self.cxt_content_attention = HierarchicalAttentionNet(self.cxt_content_encoder.get_output_dim(),
                                                                      self.max_cxt_size)
            elif self.feature_setting == FEATURE_SETTING_OPTION_METADATA_ONLY or self.feature_setting == FEATURE_SETTING_SC_WITHOUT_CC:
                # otherwise, it means that the context context only is set
                timestamped_print("init cxt_metadata_attention with dim [%s] and max_cxt_size: [%s]" % (
                    str(self.cxt_metadata_encoder.get_output_dim()),
                    str(self.max_cxt_size)))
                self.cxt_metadata_attention = HierarchicalAttentionNet(self.cxt_metadata_encoder.get_output_dim(),
                                                                       self.max_cxt_size)

            elif self.feature_setting == FEATURE_SETTING_OPTION_CONTEXT_ONLY or self.feature_setting == FEATURE_SETTING_OPTION_FULL:
                #  self.cxt_multimodal_attention is None:
                # apply additional AttentionLayer (+Layernorm) on top of full context (CC+CM) embedding
                timestamped_print("self.feature_setting: [%s]" % self.feature_setting)
                if self.cxt_content_encoder:
                    timestamped_print("init cxt_content_attention with dim [%s] and max_cxt_size: [%s]" % (
                        str(self.cxt_content_encoder.get_output_dim()),
                        str(self.max_cxt_size)))
                    self.cxt_content_attention = HierarchicalAttentionNet(self.cxt_content_encoder.get_output_dim(),
                                                                          self.max_cxt_size)

                if self.cxt_metadata_encoder:
                    timestamped_print("init cxt_metadata_attention with dim [%s] and max_cxt_size: [%s]" % (
                        str(self.cxt_metadata_encoder.get_output_dim()),
                        str(self.max_cxt_size)))
                    self.cxt_metadata_attention = HierarchicalAttentionNet(self.cxt_metadata_encoder.get_output_dim(),
                                                                           self.max_cxt_size)

                if self.cxt_content_encoder is not None and self.cxt_metadata_encoder is not None:
                    cxt_multimodal_feature_dim = self.cxt_content_encoder.get_output_dim() + self.cxt_metadata_encoder.get_output_dim()
                    timestamped_print("initialise cxt multi-modal attention layer. dim [%s] and max_cxt_size: [%s]" % (
                        str(cxt_multimodal_feature_dim),
                        str(self.max_cxt_size)))
                    self.cxt_multimodal_attention = HierarchicalAttentionNet(cxt_multimodal_feature_dim,
                                                                             self.max_cxt_size)

            self.attention_option = attention_option
        elif attention_option == int(ATTENTION_SELF_ATTENTATIVE_NET):
            if self.cxt_content_encoder:
                self.cxt_content_attention = StructuredSelfAttention(self.cxt_content_encoder.get_output_dim())

            if self.cxt_metadata_encoder:
                self.cxt_metadata_attention = StructuredSelfAttention(self.cxt_metadata_encoder.get_output_dim())

            self.attention_option = attention_option
        elif attention_option == ATTENTION_OPTION_NONE:
            self.attention_option = attention_option
            timestamped_print("set to not apply attention.")
        else:
            self.attention_option = None
            raise ValueError(
                "'attention_option' [%s] is not supported! Available options are [%s,%s]" % (attention_option,
                                                                                             ATTENTION_OPTION_ATTENTION_WITH_CXT,
                                                                                             ATTENTION_SELF_ATTENTATIVE_NET))

        timestamped_print("done.")

    def set_layer_norm_layers(self):
        """
         layer-norm applied to the output of a self-attention layer

        :return:
        """
        if self.feature_setting == FEATURE_SETTING_OPTION_CONTEXT_CONTENT_ONLY or self.feature_setting == FEATURE_SETTING_SC_WITHOUT_CM:
            self.layer_norm_content_attention = MyLayerNorm(self.cxt_content_attention.feature_dim)

        if self.feature_setting == FEATURE_SETTING_OPTION_METADATA_ONLY or self.feature_setting == FEATURE_SETTING_SC_WITHOUT_CC:
            self.layer_norm_metadata_attention = MyLayerNorm(self.cxt_metadata_attention.feature_dim)

        if self.feature_setting == FEATURE_SETTING_OPTION_CONTEXT_ONLY or self.feature_setting == FEATURE_SETTING_OPTION_FULL:
            # setting: 1) feature_setting == FEATURE_SETTING_OPTION_CONTEXT_ONLY (4); 2) social_encoder == LSTM ;
            #           3) --attention == ATTENTION_OPTION_ATTENTION_WITH_CXT (1)
            # reset for stacked attention layer with layer norm for concatenated multimodal context
            timestamped_print(
                "apply stacked attention: reset CC layer norm with dim [%s] and CM layer norm with dim [%s]" %
                (str(0 if self.cxt_content_encoder is None else self.cxt_content_encoder.get_output_dim()),
                 str(0 if self.cxt_metadata_encoder is None else self.cxt_metadata_encoder.get_output_dim())))
            if self.cxt_content_encoder:
                self.layer_norm_content_attention = MyLayerNorm(self.cxt_content_encoder.get_output_dim())

            if self.cxt_metadata_encoder:
                self.layer_norm_metadata_attention = MyLayerNorm(self.cxt_metadata_encoder.get_output_dim())

            if self.cxt_content_encoder is not None and self.cxt_metadata_encoder is not None:
                concatenated_multimodal_cxt_size = self.cxt_content_encoder.get_output_dim() + self.cxt_metadata_encoder.get_output_dim()
                timestamped_print("initialise concatenated multimodal context Laynorm. dim [%s]" % str(
                    concatenated_multimodal_cxt_size))
                self.layer_norm_multimodal_cxt = MyLayerNorm(concatenated_multimodal_cxt_size)

    def set_max_cxt_size(self, max_cxt_size: int = MAXIMUM_CONTEXT_SEQ_SIZE):
        """
        this setting will affect attention setting
        :param max_cxt_size:
        :return:
        """
        if max_cxt_size:
            self.max_cxt_size = max_cxt_size
            timestamped_print("maximum context sequence size is [%s]" % self.max_cxt_size)
            timestamped_print("re-set attention1 with the new context size")
            self.set_attention_mechanism(self.attention_option)

    def set_feature_setting(self, feature_setting_option):
        self.feature_setting = feature_setting_option
        if self.feature_setting == FEATURE_SETTING_OPTION_METADATA_ONLY or self.feature_setting == FEATURE_SETTING_SC_WITHOUT_CC:
            print("context feature will be set to use numerical feature only.")

        if self.feature_setting == FEATURE_SETTING_OPTION_CONTEXT_CONTENT_ONLY or self.feature_setting == FEATURE_SETTING_SC_WITHOUT_CM:
            print("context feature will be set to use content embedding only.")

        if self.feature_setting == FEATURE_SETTING_OPTION_SOURCE_TWEET_CONTENT_ONLY:
            print("model is set to use source content only (social context is disabled).")

        if self.feature_setting == FEATURE_SETTING_OPTION_CONTEXT_ONLY:
            print("model is set to use context only (CC + CM).")

    def forward(self, sentence: Dict[str, torch.Tensor], tweet_id: list, label: torch.LongTensor = None) -> Dict[
        str, torch.Tensor]:

        """

        :param sentence:
        :param tweet_id:
        :param label:
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        # In deep NLP, when sequences of tensors in different lengths are batched together,
        # shorter sequences get padded with zeros to make them of equal length.
        # Masking is the process to ignore extra zeros added by padding
        # each 'sentence'tensor is a sequence of token ids

        timestamped_print("--------------------------------------next batch ----------------------------------")
        timestamped_print("rumor classifier forward pass ... ")
        # print out few dataset for sanity check and visual inspect of attention weights
        timestamped_print("First two source tweet ids in current batch: [%s] and [%s]" % (str(tweet_id[0]),
                                                                                          str(tweet_id[1])))
        timestamped_print("First two source tweet label(y): [%s]" % (str(label[:2])))

        mask = get_text_field_mask(sentence)
        # Forward pass
        if self.feature_setting == FEATURE_SETTING_OPTION_METADATA_ONLY:
            # zero-filling approach is simply to set the missing value
            # vast amount of zeros may make the training unstable ???
            lang_model_encoder_out = torch.as_tensor(np.zeros(self.lang_model_encoder.get_output_dim()),
                                                     dtype=torch.float32)
        else:
            embeddings = self.tweet_text_embedder(sentence)
            lang_model_encoder_out = self.lang_model_encoder(embeddings, mask)

        if self.feature_setting != FEATURE_SETTING_OPTION_SOURCE_TWEET_CONTENT_ONLY:
            cxt_content_tensor_in_batch_padded, cxt_metadata_tensor_in_batch_padded, \
            batch_cxt_content_mask, batch_cxt_metadata_mask = self.batch_compute_context_feature_encoding(tweet_id,
                                                                                                          maximum_cxt_sequence_length=self.max_cxt_size)

            if self.social_encoder_option == SOCIAL_CONTEXT_ENCODER_OPTION_TRANSFORMER:
                social_context_encoder_out = self.social_cxt_encoding_with_transformer(
                    cxt_content_tensor_in_batch_padded,
                    cxt_metadata_tensor_in_batch_padded,
                    batch_cxt_content_mask, batch_cxt_metadata_mask)

                timestamped_print("use transformer to encode social context instead")
            else:
                # mask is none when input is a packed variable length sequence
                # see https://pytorch.org/docs/master/nn.html#torch.nn.LSTM and torch.PytorchSeq2VecWrapper
                timestamped_print("encode social context with LSTM")
                social_context_encoder_out = self.social_cxt_encoding_with_lstm(cxt_content_tensor_in_batch_padded,
                                                                                cxt_metadata_tensor_in_batch_padded,
                                                                                batch_cxt_content_mask,
                                                                                batch_cxt_metadata_mask)

            timestamped_print("social context reprsentation shape : %s" % (str(social_context_encoder_out.shape)))

        # concatenate fine-tuned text representation with social context feature representation
        # dim must be 1 for horizontal concatenation, see tech details via https://pytorch.org/docs/stable/torch.html
        if self.feature_setting == FEATURE_SETTING_OPTION_METADATA_ONLY or \
                self.feature_setting == FEATURE_SETTING_OPTION_CONTEXT_CONTENT_ONLY or \
                self.feature_setting == FEATURE_SETTING_OPTION_CONTEXT_ONLY:
            rumor_representation = social_context_encoder_out
        elif self.feature_setting == FEATURE_SETTING_OPTION_SOURCE_TWEET_CONTENT_ONLY:
            rumor_representation = lang_model_encoder_out
            timestamped_print("source tweet text (only) representation shape: %s" % str(rumor_representation.shape))
        else:
            # either FULL model setting or FEATURE_SETTING_SC_WITHOUT_CM or FEATURE_SETTING_SC_WITHOUT_CC (for abalation study)
            rumor_representation = torch.cat((lang_model_encoder_out, social_context_encoder_out), dim=1)
            timestamped_print(
                "rumor tweet representation (textual model + social context model): %s" % str(
                    rumor_representation.shape))

        timestamped_print("rumor_representation before feedforward: %s" % str(rumor_representation.shape))
        logits = self.classifier_feedforward(rumor_representation)

        # In AllenNLP, the output of forward() is a dictionary.
        # Your output dictionary must contain a "loss" key for your model to be trained.
        output_dict = {"logits": logits}

        if label is not None:
            loss = self.loss_function(logits, label.squeeze(-1))
            for metric in self.metrics.values():
                metric(logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=2)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="label")
                  for x in argmax_indices]
        timestamped_print("labels <- decode: %s" % labels)

        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
        # f1 get_metric returns (precision, recall, f1)
        f1_measures = self.metrics["f1"].get_metric(reset=reset)
        return {
            # https://github.com/allenai/allennlp/issues/1863
            # f1 get_metric returns (precision, recall, f1)
            "precision": f1_measures[0],
            "recall": f1_measures[1],
            "f1": f1_measures[2],
            "accuracy": self.metrics["accuracy"].get_metric(reset=reset)
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'RumorTweetsClassifer':  # type: ignore
        # pylint: disable=arguments-differ,bad-super-call

        word_embedder_params = params.pop("tweet_text_embedder")
        model_text_field_embedder = TextFieldEmbedder.from_params(word_embedder_params, vocab=vocab)

        lang_model_encoder = Seq2VecEncoder.from_params(params.pop("lang_model_encoder"))
        # social_context_encoder is deprecated param
        social_context_encoder = Seq2VecEncoder.from_params(params.pop("social_context_encoder"))
        social_context_self_attention_encoder = Seq2SeqEncoder.from_params(
            params.pop("social_context_self_attention_encoder"))

        classifier_feedforward = FeedForward.from_params(params.pop("classifier_feedforward"))

        cxt_content_encoder = Seq2SeqEncoder.from_params(params.pop("cxt_content_encoder"))
        cxt_metadata_encoder = Seq2SeqEncoder.from_params(params.pop("cxt_metadata_encoder"))
        cxt_content_encoder_rnn_wrapper = Seq2VecEncoder.from_params(params.pop("context_content_encoder_rnn_wrapper"))
        cxt_metadata_encoder_rnn_wrapper = Seq2VecEncoder.from_params(
            params.pop("context_metadata_encoder_rnn_wrapper"))

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))

        return cls(tweet_text_embedder=model_text_field_embedder,
                   lang_model_encoder=lang_model_encoder,
                   social_context_encoder=social_context_encoder,
                   context_content_encoder_rnn_wrapper=cxt_content_encoder_rnn_wrapper,
                   context_metadata_encoder_rnn_wrapper=cxt_metadata_encoder_rnn_wrapper,
                   social_context_self_attention_encoder=social_context_self_attention_encoder,
                   vocab=vocab,
                   classifier_feedforward=classifier_feedforward,
                   cxt_content_encoder=cxt_content_encoder,
                   cxt_metadata_encoder=cxt_metadata_encoder,
                   initializer=initializer)

    def social_cxt_encoding_with_transformer(self, cxt_content_tensor_in_batch_padded,
                                             cxt_metadata_tensor_in_batch_padded,
                                             batch_cxt_content_mask: torch.uint8,
                                             batch_cxt_metadata_mask: torch.uint8):
        """

        Referece:
           [Wang 2019] utilize batch size 64, Adam optimizer with learning rate 0.0001 . L2 regularization coefficient is 1e-4.
            As for Contextual Encoding component, the number of layers and heads is 3, 2, respectively.
            [Wang 2019] use the average accuracy of 10 trials as the performance metric

        :param cxt_content_tensor_in_batch_padded:
        :param cxt_metadata_tensor_in_batch_padded:
        :return:
        """
        timestamped_print("encode context with self attention ....")
        timestamped_print("padded <content seq input> shape: [%s]" % str(cxt_content_tensor_in_batch_padded.shape))
        timestamped_print("padded <metadata seq input> shape: [%s]" % str(cxt_metadata_tensor_in_batch_padded.shape))

        # concatenate CC and CM
        packed_input_context_feature_encodings = torch.cat(
            (cxt_content_tensor_in_batch_padded, cxt_metadata_tensor_in_batch_padded), 1)

        timestamped_print("CC + CM <all input> shape: [%s]" % str(packed_input_context_feature_encodings.shape))

        self_attention_encoded_context = self.social_context_self_attention_encoder(
            packed_input_context_feature_encodings, batch_cxt_content_mask)
        timestamped_print(
            "self_attention_encoded_context <output> shape [%s]" % str(self_attention_encoded_context.shape))

        # following the practice in [Cer 2018] (Universal sentence encoder),
        # The context aware reaction representations are converted to a fixed length social context encoding vector by computing the element-wise sum
        # OPTION: it is also possible to use element-wise multiply
        # of the representations at each reaction position
        # self_attention_encoded_context_sum_all = torch.as_tensor([combine_tensors("x+y",tweet_cxt_tensor_i) for tweet_cxt_tensor_i in self_attention_encoded_context])
        self_attention_encoded_context_sum_all = torch.sum(self_attention_encoded_context, dim=1, dtype=torch.float32)
        return self_attention_encoded_context_sum_all

    def social_cxt_encoding_with_lstm(self, cxt_content_tensor_in_batch_padded, cxt_metadata_tensor_in_batch_padded,
                                      batch_cxt_content_mask: torch.uint8, batch_cxt_metadata_mask: torch.uint8):
        timestamped_print("encode context with LSTM / GRU ....")
        timestamped_print("padded <content seq input> shape: [%s]" % str(cxt_content_tensor_in_batch_padded.shape))
        timestamped_print("padded <metadata seq input> shape: [%s]" % str(cxt_metadata_tensor_in_batch_padded.shape))

        # RNN (LSTM) forward pass
        if self.feature_setting == FEATURE_SETTING_OPTION_CONTEXT_CONTENT_ONLY or self.feature_setting == FEATURE_SETTING_SC_WITHOUT_CM:
            cxt_encoder_out = self.context_content_encoding(cxt_content_tensor_in_batch_padded, batch_cxt_content_mask)
        elif self.feature_setting == FEATURE_SETTING_OPTION_METADATA_ONLY or self.feature_setting == FEATURE_SETTING_SC_WITHOUT_CC:
            cxt_encoder_out = self.context_metadata_encoding(cxt_metadata_tensor_in_batch_padded,
                                                             batch_cxt_metadata_mask)
        else:
            cxt_encoder_out = self.full_context_encoding(cxt_content_tensor_in_batch_padded,
                                                         cxt_metadata_tensor_in_batch_padded,
                                                         batch_cxt_content_mask, batch_cxt_metadata_mask)

        return cxt_encoder_out

    def context_metadata_encoding(self, cxt_metadata_tensor_in_batch_padded, batch_cxt_metadata_mask):
        timestamped_print("context metadata only enabled")
        if self.attention_option != ATTENTION_OPTION_NONE:
            # encoding content for attention layer (i.e., use all states)
            cxt_metadata_lstm_out = self.cxt_metadata_encoder.forward(cxt_metadata_tensor_in_batch_padded,
                                                                      batch_cxt_metadata_mask)
        else:
            # encoding content without attention layer (i.e., use final state output for prediction)
            cxt_metadata_lstm_out = self.cxt_metadata_encoder_wrapper.forward(cxt_metadata_tensor_in_batch_padded,
                                                                              batch_cxt_metadata_mask)

        timestamped_print("context metadata RNN(LSTM) embedding shape: %s" % str(cxt_metadata_lstm_out.shape))
        # add attention layer
        if self.attention_option == ATTENTION_OPTION_ATTENTION_WITH_CXT:
            h_cm_lstm_atten, cm_weighted_lstm_output, cm_attention_weights = self.cxt_metadata_attention(
                cxt_metadata_lstm_out, batch_cxt_metadata_mask)
        elif self.attention_option == ATTENTION_SELF_ATTENTATIVE_NET:
            h_cm_lstm_atten, cm_attn_weight_matrix = self.cxt_metadata_attention(cxt_metadata_lstm_out, None)
        elif self.attention_option == ATTENTION_OPTION_NONE:
            timestamped_print("no attention. adopt LSTM final hidden state for prediction")
            h_cm_lstm_atten = cxt_metadata_lstm_out

        cxt_encoder_out = h_cm_lstm_atten
        timestamped_print("context embedding (metadata-only) shape: [%s]" % str(cxt_encoder_out.shape))

        if self.layer_norm_metadata_attention:
            timestamped_print("apply layernorm on top of attention...")
            cxt_encoder_out = self.layer_norm_metadata_attention.forward(cxt_encoder_out, self.cuda_device)
            timestamped_print("context embedding after layernorm shape: [%s]" % str(cxt_encoder_out.shape))

        return cxt_encoder_out

    def context_content_encoding(self, cxt_content_tensor_in_batch_padded, batch_cxt_content_mask):
        timestamped_print("context content only enabled")
        if self.attention_option != ATTENTION_OPTION_NONE:
            # encoding content for attention layer (i.e., use all states)
            cxt_content_lstm_out = self.cxt_content_encoder.forward(cxt_content_tensor_in_batch_padded,
                                                                    batch_cxt_content_mask)
        else:
            # encoding content without attention layer (i.e., use final state output for prediction)
            cxt_content_lstm_out = self.cxt_content_encoder_wrapper.forward(cxt_content_tensor_in_batch_padded,
                                                                            batch_cxt_content_mask)

        timestamped_print("context content RNN(LSTM) embedding shape: %s" % str(cxt_content_lstm_out.shape))

        # add attention layer
        if self.attention_option == ATTENTION_OPTION_ATTENTION_WITH_CXT:
            h_cc_lstm_atten, cc_weighted_lstm_output, cc_attention_weights = self.cxt_content_attention(
                cxt_content_lstm_out, batch_cxt_content_mask)
        elif self.attention_option == ATTENTION_SELF_ATTENTATIVE_NET:
            h_cc_lstm_atten, cc_attn_weight_matrix = self.cxt_content_attention(cxt_content_lstm_out, None)
        elif self.attention_option == ATTENTION_OPTION_NONE:
            timestamped_print("no attention. adopt LSTM final hidden state for prediction")
            h_cc_lstm_atten = cxt_content_lstm_out

        cxt_encoder_out = h_cc_lstm_atten
        timestamped_print("context embedding (reply content-only) shape: [%s]" % str(cxt_encoder_out.shape))

        if self.layer_norm_content_attention:
            timestamped_print("apply layernorm on top of attention...")
            cxt_encoder_out = self.layer_norm_content_attention.forward(cxt_encoder_out, self.cuda_device)
            timestamped_print("context embedding after layernorm shape: [%s]" % str(cxt_encoder_out.shape))

        return cxt_encoder_out

    def full_context_encoding(self, cxt_content_tensor_in_batch_padded, cxt_metadata_tensor_in_batch_padded,
                              batch_cxt_content_mask, batch_cxt_metadata_mask):
        """
        learn social context embedding for both reply content and associated metadata

        LSTM weights are adjusted with stacked attention

        :param cxt_content_tensor_in_batch_padded:
        :param cxt_metadata_tensor_in_batch_padded:
        :param batch_cxt_content_mask:
        :param batch_cxt_metadata_mask:
        :return:
        """
        # ========== RNN LAYER =============
        timestamped_print("LSTM layer ..................")
        if self.attention_option != ATTENTION_OPTION_NONE:
            timestamped_print("stacked LSTM for attention ...... ")
            # instead of providing initialised hidden states, we can enable stateful=True
            cxt_content_lstm_out = self.cxt_content_encoder.forward(cxt_content_tensor_in_batch_padded,
                                                                    batch_cxt_content_mask)
            cxt_metadata_lstm_out = self.cxt_metadata_encoder.forward(cxt_metadata_tensor_in_batch_padded,
                                                                      batch_cxt_metadata_mask)
        else:
            timestamped_print("stacked LSTM for no-attention ...... ")
            # RNN (LSTM or GRU) only
            cxt_content_lstm_out = self.cxt_content_encoder_wrapper.forward(cxt_content_tensor_in_batch_padded,
                                                                            batch_cxt_content_mask)
            cxt_metadata_lstm_out = self.cxt_metadata_encoder_wrapper.forward(cxt_metadata_tensor_in_batch_padded,
                                                                              batch_cxt_metadata_mask)

        # =========== add attention layer ==========
        timestamped_print("attention layer ...... ")
        timestamped_print(" ...... attention_option : %s" % (str(self.attention_option)))
        cm_weighted_lstm_output = None
        cc_weighted_lstm_output = None
        if self.attention_option == ATTENTION_OPTION_ATTENTION_WITH_CXT:
            timestamped_print(" ----- ATTENTION_OPTION_ATTENTION_WITH_CXT ----- ")
            timestamped_print("..... apply attention on content LSTM output .....")
            # print("self.cxt_content_attention weight before feedforward: ", self.cxt_content_attention.weight)
            h_cc_lstm_atten, cc_weighted_lstm_output, cc_attention_weights = self.cxt_content_attention(
                cxt_content_lstm_out, batch_cxt_content_mask)
            timestamped_print("...... apply attention on metadata LSTM output .....")
            # print("self.cxt_metadata_attention weight before feedforward: ", self.cxt_metadata_attention.weight)
            h_cm_lstm_atten, cm_weighted_lstm_output, cm_attention_weights = self.cxt_metadata_attention(
                cxt_metadata_lstm_out, batch_cxt_metadata_mask)
            timestamped_print(" ++++++++++ finish attention ++++++++++ ")
        elif self.attention_option == ATTENTION_SELF_ATTENTATIVE_NET:
            h_cc_lstm_atten, cc_weighted_lstm_output = self.cxt_content_attention(cxt_content_lstm_out, None)
            h_cm_lstm_atten, cm_weighted_lstm_output = self.cxt_metadata_attention(cxt_metadata_lstm_out, None)

            timestamped_print(" concatenate the CC hidden_matrix  (b4 feeding into FC and softmax) shape: %s" % (
                str(h_cc_lstm_atten.shape)))
            timestamped_print(" concatenate the CM hidden_matrix  (b4 feeding into FC and softmax) shape: %s" % (
                str(h_cm_lstm_atten.shape)))
        elif self.attention_option == ATTENTION_OPTION_NONE:
            timestamped_print("no attention. adopt LSTM final hidden state for prediction")
            h_cc_lstm_atten = cxt_content_lstm_out
            h_cm_lstm_atten = cxt_metadata_lstm_out

        timestamped_print("h_cc_lstm_atten shape: %s" % str(h_cc_lstm_atten.shape))
        timestamped_print("h_cm_lstm_atten shape: %s" % str(h_cm_lstm_atten.shape))
        cxt_content_encoder_out = h_cc_lstm_atten
        cxt_metadata_encoder_out = h_cm_lstm_atten

        # ============== Layernorm layer ===================
        if self.attention_option != ATTENTION_OPTION_NONE and self.layer_norm_metadata_attention:
            timestamped_print("fullcxt_metadata: apply layernorm on top of attention...")
            # in stacked attention, only last layer of Layrnorm will take weighted sum of context as input
            cxt_metadata_encoder_out = cm_weighted_lstm_output
            timestamped_print("context embedding after layernorm shape: [%s]" % str(cxt_metadata_encoder_out.shape))

        if self.attention_option != ATTENTION_OPTION_NONE and self.layer_norm_content_attention:
            timestamped_print("fullcxt_content: apply layernorm on top of attention...")
            # in stacked attention, only last layer of Layrnorm will take weighted sum of context as input
            cxt_content_encoder_out = cc_weighted_lstm_output
            timestamped_print("context embedding after layernorm shape: [%s]" % str(cxt_content_encoder_out.shape))

        if self.attention_option == ATTENTION_OPTION_NONE:
            cxt_encoder_out = torch.cat((cxt_content_encoder_out, cxt_metadata_encoder_out), 1)
        else:
            # we now apply stacked attention (only last attention layer will take input of weighted sum of context tensor
            cxt_encoder_out = torch.cat((cxt_content_encoder_out, cxt_metadata_encoder_out), 2)

        timestamped_print("concatenated CC + CM embedding output shape: %s" % str(cxt_encoder_out.shape))

        if self.attention_option != ATTENTION_OPTION_NONE and self.cxt_multimodal_attention is not None:
            timestamped_print("apply attention on CC+CM multimodal output ...")
            timestamped_print(
                "masking tensor (for cxt_multimodal_attention) shape: [%s]" % (str(batch_cxt_content_mask.shape)))
            cxt_lstm_atten_sum, cxt_weighted_lstm_output, cxt_attention_weights = self.cxt_multimodal_attention.forward(
                cxt_encoder_out, batch_cxt_content_mask)
            timestamped_print("stacked attention + layer norm is enabled.")
            cxt_encoder_out = self.layer_norm_multimodal_cxt.forward(cxt_lstm_atten_sum)
            timestamped_print("attended and layrnormed CC+CM output shape: [%s]" % str(cxt_encoder_out.shape))

        return cxt_encoder_out

    def batch_compute_context_feature_encoding(self, source_tweet_ids: List,
                                               maximum_cxt_sequence_length=MAXIMUM_CONTEXT_SEQ_SIZE) -> (
            torch.FloatTensor, torch.FloatTensor,
            torch.uint8, torch.uint8):
        """
        batch processing of context feature encoding

        use batch normalistaion to standardise raw input context features that has different scales

        :param source_tweet_ids: current batch of source tweet id list
        :param maximum_cxt_sequence_length: maximum feasible sequence length of social-temporal context that
                                will feed into LSTM network (i.e., max_input_sequence_length)
        :return: (context content tensor, context metadata tensor, context content mask, context metadata mask)

         # FloatTensor, a packed variable length sequence for RNN, [batch size, context size, embedding/feature dimension]
        """
        timestamped_print("compute context features in batch ...")
        timestamped_print("First two source tweet ids in current batch: [%s] and [%s]" % (str(source_tweet_ids[0]),
                                                                                          str(source_tweet_ids[1])))

        timestamped_print(
            "event social time window: " + (str(self.event_varying_timedelta) if self.event_varying_timedelta else ""))
        # preprocessing: load replies (i.e., propagation history) for current batch of source tweets
        source_tweet_context_set_list = [
            list(load_source_tweet_context(source_tweet_id, self.event_varying_timedelta, self.disable_cxt_type)) for
            source_tweet_id in source_tweet_ids]

        # restrict context temporal sequence to a maximum (feasible) length
        source_tweet_context_set_list = [source_tweet_context_list[:maximum_cxt_sequence_length] for
                                         source_tweet_context_list
                                         in source_tweet_context_set_list]

        # print("source_tweet_context_set_list: ", source_tweet_context_set_list)
        source_tweet_json_list = [load_source_tweet_json(source_tweet_id) for source_tweet_id in source_tweet_ids]

        if len(source_tweet_context_set_list) != len(source_tweet_json_list):
            # check/debug
            timestamped_print("Error: context data size [%s] does not match the size of input source tweets [%s]. "
                              % (len(source_tweet_context_set_list), len(source_tweet_ids)))

        cxt_content_tensor_in_batch = []
        cxt_metadata_tensor_in_batch = []

        for individual_source_tweet_context_set, invidual_source_tweet_json in zip(source_tweet_context_set_list,
                                                                                   source_tweet_json_list):
            cxt_content_tensor, cxt_metadata_tensor = self._context_sequence_encoding(
                individual_source_tweet_context_set, invidual_source_tweet_json)
            cxt_content_tensor_in_batch.append(cxt_content_tensor)
            cxt_metadata_tensor_in_batch.append(cxt_metadata_tensor)

        # normalise data in batch : apply scaling by the mean/variance computed on current batch

        if len(cxt_content_tensor_in_batch) != len(source_tweet_ids):
            # check/debug
            timestamped_print(
                "Error: context propagation encoding size [%s] does not match the size of input source tweets [%s]. "
                % (len(cxt_content_tensor_in_batch), len(source_tweet_ids)))

        timestamped_print(
            "done social context encoding! Padding and normalise sequence tensors for current batch now...")

        cxt_content_tensor_in_batch_padded = torch.Tensor()
        batch_cxt_content_mask = torch.Tensor()
        cxt_metadata_tensor_in_batch_padded = torch.Tensor()
        batch_cxt_metadata_mask = torch.Tensor()

        if self.feature_setting == FEATURE_SETTING_OPTION_METADATA_ONLY or self.feature_setting == FEATURE_SETTING_SC_WITHOUT_CC:
            cxt_metadata_tensor_in_batch_padded, batch_cxt_metadata_mask = self.padding_and_norm_propagation_tensors(
                cxt_metadata_tensor_in_batch,
                maximum_cxt_sequence_length=maximum_cxt_sequence_length)
        elif self.feature_setting == FEATURE_SETTING_OPTION_CONTEXT_CONTENT_ONLY or self.feature_setting == FEATURE_SETTING_SC_WITHOUT_CM:
            cxt_content_tensor_in_batch_padded, batch_cxt_content_mask = self.padding_and_norm_propagation_tensors(
                cxt_content_tensor_in_batch,
                maximum_cxt_sequence_length=maximum_cxt_sequence_length)
        else:
            cxt_content_tensor_in_batch_padded, batch_cxt_content_mask = self.padding_and_norm_propagation_tensors(
                cxt_content_tensor_in_batch,
                maximum_cxt_sequence_length=maximum_cxt_sequence_length)
            cxt_metadata_tensor_in_batch_padded, batch_cxt_metadata_mask = self.padding_and_norm_propagation_tensors(
                cxt_metadata_tensor_in_batch,
                maximum_cxt_sequence_length=maximum_cxt_sequence_length)

        timestamped_print(
            "Done. propagation tensors after batch normalisation: cxt content shape [%s], cxt metadata shape [%s]" %
            (str(cxt_content_tensor_in_batch_padded.size()), str(cxt_metadata_tensor_in_batch_padded.size())))

        timestamped_print("context masking: content [%s], metadata [%s]" % (str(batch_cxt_content_mask.shape),
                                                                            str(batch_cxt_metadata_mask.shape)))

        return cxt_content_tensor_in_batch_padded, cxt_metadata_tensor_in_batch_padded, \
               batch_cxt_content_mask, batch_cxt_metadata_mask

    def padding_and_norm_propagation_tensors(self, list_of_context_seq_tensor: List[torch.FloatTensor],
                                             maximum_cxt_sequence_length=100) -> (torch.FloatTensor, torch.uint8):
        """
        padding and masking

        :param list_of_context_seq_tensor:
        :param maximum_cxt_sequence_length:
        :return: padded batch context tensor, batch context mask (for attention)
        """
        try:
            batch_cxt_mask = self.creating_batch_context_tensor_mask(list_of_context_seq_tensor,
                                                                     maximum_cxt_sequence_length)

            feature_dim = list_of_context_seq_tensor[0].shape[1]
            # pad_sequence func cannot apply padding by a given max length and can only apply max length padding on current batch
            # this workaround is to insert a dummy tensor with maximum length which can be removed after padding
            # this is for the attention which requires a fixed seq size
            # another reason is that although RNN/LSTM can handle variable shape/length, however,
            #               working with a fixed input length can improve performance
            list_of_context_seq_tensor.insert(0, torch.zeros(maximum_cxt_sequence_length, feature_dim))
            # zero padding
            batch_propagation_tensors = torch.nn.utils.rnn.pad_sequence(list_of_context_seq_tensor, batch_first=True)
            # remove the dummy tensor
            batch_propagation_tensors = batch_propagation_tensors[1:]

        except RuntimeError as err:
            print(err)
            timestamped_print(
                "error when pad social-context feature tensors. Check all tensor shapes as following in current batch ...")
            for context_seq_tensor in list_of_context_seq_tensor:
                timestamped_print(str(context_seq_tensor.shape))
            timestamped_print("done")

            raise err

        timestamped_print("tensor size after padding: %s" % str(
            batch_propagation_tensors.size()))  # -> (batch_size, padded size of context sequence, dimension of instance reqpresentation)

        batch_size = batch_propagation_tensors.size()[0]
        if self.bn_input:
            normalised_propagation_tensor = torch.stack(
                [self.bn_input(batch_propagation_tensors[i]) for i in range(batch_size)])
        else:
            normalised_propagation_tensor = torch.stack([batch_propagation_tensors[i] for i in range(batch_size)])

        return normalised_propagation_tensor.float(), batch_cxt_mask

    def creating_batch_context_tensor_mask(self, list_of_context_seq_tensor: List[torch.FloatTensor],
                                           maximum_cxt_sequence_length=300):
        """
        creating mask for attention
        :param context_tensor_seq:
        :return: ByteTensor (a Boolean tensor) (torch.uint8)
        """
        vary_cxt_lengths = torch.stack(
            [torch.as_tensor(tensor_cxt_tensor.shape[0], dtype=torch.float) for tensor_cxt_tensor in
             list_of_context_seq_tensor])

        idxes = torch.arange(0, maximum_cxt_sequence_length,
                             out=torch.FloatTensor(maximum_cxt_sequence_length)).unsqueeze(0)
        idxes = idxes.cuda()
        vary_cxt_lengths = vary_cxt_lengths.cuda()
        mask = idxes < vary_cxt_lengths.unsqueeze(1)

        return mask

    def _context_sequence_encoding(self, source_tweet_context_dataset: List[Dict], source_tweet: Dict) -> (
            torch.FloatTensor, torch.FloatTensor):
        """
        prepare sorted reaction sequence:

        simple sentence embedding only social-context encoder (will replace or combine with context_feature_extractor.py)

        :param source_tweet_context_dataset:context tweets containing replies or retweets or both
        :param source_tweet:
        :return:List[torch.FloatTensor], sequence tensor from temporally sorted encodings of source tweet context/replies according to replies/retweets timeline
        """

        EXPECTED_CC_ENCODER_INPUT_DIM = 1024
        EXPECTED_CM_ENCODER_INPUT_DIM = 28

        if self.cxt_content_encoder:
            EXPECTED_CC_ENCODER_INPUT_DIM = self.cxt_content_encoder.get_input_dim()

        if self.cxt_metadata_encoder:
            EXPECTED_CM_ENCODER_INPUT_DIM = self.cxt_metadata_encoder.get_input_dim()

        if len(source_tweet_context_dataset) == 0:
            # sanity check/debug
            source_tweet_id = source_tweet["id"] if "id" in source_tweet else "source_tweet_id not found"
            timestamped_print("Warning: context data is empty for source tweet [%s]. "
                              "We may remove this type of source tweet in training/test set and avoid it in prediction" % source_tweet_id)

            timestamped_print(
                "generating zero tensor representation for content (with dim[%s]) and metadata (with dim[%s]) now." %
                (EXPECTED_CC_ENCODER_INPUT_DIM, EXPECTED_CM_ENCODER_INPUT_DIM))
            # zero filling for missing data
            cxt_content_seq_tensor = torch.zeros(1, EXPECTED_CC_ENCODER_INPUT_DIM)
            cxt_metadata_seq_tensor = torch.zeros(1, EXPECTED_CM_ENCODER_INPUT_DIM)
        else:
            try:
                source_tweet_user_id = source_tweet['user']['id_str']
                source_tweet_text = source_tweet["text"] if "text" in source_tweet else source_tweet["full_text"]
                source_tweet_timestamp = datetime.strptime(source_tweet['created_at'], '%a %b %d %H:%M:%S %z %Y')
                source_tweet_user_name = '@' + source_tweet['user']['screen_name']

                # We need a maximum number of context (or a strategy to filter non-informative context), say max 500
                # Context size being too long with exhaust GPU memory
                # TODO: we also need a statistics of context size in our data set (train, held and test)
                all_context_metadata_features = []
                all_reply_content_embeddings = []
                for source_tweet_reaction in source_tweet_context_dataset:
                    user_profile_embedding, reply_content_embedding, numerical_features = context_feature_extraction_from_context_status(
                        source_tweet_reaction,
                        source_tweet_user_name,
                        source_tweet_timestamp,
                        source_tweet_user_id,
                        source_tweet_text,
                        cxt_content_encoder=self.cxt_content_encoder,
                        elmo_model=self.elmo_model,
                        feature_setting=self.feature_setting)

                    all_context_metadata_features.append(numerical_features)
                    all_reply_content_embeddings.append(reply_content_embedding)

                # ==================== stack layer: cxt (metadata) numerical features encoding ==========================
                # social numerical feature scaling for current tweet
                all_context_metadata_features = np.array(all_context_metadata_features)
                all_context_metadata_features = global_norm(all_context_metadata_features, self.global_means,
                                                            self.global_stds)

                # auto encoder : linear feature transform
                # ==================== prepare(concatenate) & sort context features in temporal sorted sequence before feeding into LSTM

                cxt_content_seq_tensor, cxt_metadata_seq_tensor = self.sort_cxt_encoding_with_time_sequence(
                    source_tweet_context_dataset,
                    all_reply_content_embeddings,
                    all_context_metadata_features)
            except:
                print("Unexpected error when encoding [%s] tweet context " % source_tweet["id"], sys.exc_info()[0])
                raise

        return cxt_content_seq_tensor, cxt_metadata_seq_tensor

    def sort_cxt_encoding_with_time_sequence(self, source_tweet_context_dataset: List[Dict],
                                             cxt_content_embeddings: List[np.ndarray],
                                             cxt_metadata_encoding: List[np.ndarray]) -> (
            torch.FloatTensor, torch.FloatTensor):
        """
        put two type of rumor tweet context representation in temporal order before feeding into LSTM
        :param source_tweet_context_dataset:
        :param cxt_content_embeddings:
        :param cxt_metadata_encoding:
        :return:
        """
        cxt_size = len(source_tweet_context_dataset)
        cxt_content_embedding_time_seq = []
        cxt_metadata_embedding_time_seq = []
        for i in range(0, cxt_size):
            source_tweet_reaction_i = source_tweet_context_dataset[i]
            cxt_metadata_i = cxt_metadata_encoding[i]
            cxt_content_i = cxt_content_embeddings[i]

            reaction_tweet_timestamp = datetime.strptime(source_tweet_reaction_i["created_at"],
                                                         '%a %b %d %H:%M:%S %z %Y')  # timestamp
            cxt_content_embedding_time_seq.append((reaction_tweet_timestamp, cxt_content_i))
            cxt_metadata_embedding_time_seq.append((reaction_tweet_timestamp, cxt_metadata_i))

        cxt_content_embedding_time_seq_sorted = sorted(cxt_content_embedding_time_seq[:], key=operator.itemgetter(0),
                                                       reverse=False)
        cxt_metadata_embedding_time_seq_sorted = sorted(cxt_metadata_embedding_time_seq[:], key=operator.itemgetter(0),
                                                        reverse=False)

        cxt_content_seq_representation: List[np.ndarray] = [reaction_representation[1] for reaction_representation in
                                                            cxt_content_embedding_time_seq_sorted]
        cxt_metadata_seq_representation: List[np.ndarray] = [reaction_representation[1] for reaction_representation in
                                                             cxt_metadata_embedding_time_seq_sorted]

        # https://pytorch.org/docs/stable/torch.html#torch.as_tensor
        if self.feature_setting == FEATURE_SETTING_OPTION_CONTEXT_CONTENT_ONLY or self.feature_setting == FEATURE_SETTING_SC_WITHOUT_CM:
            return torch.as_tensor(cxt_content_seq_representation, dtype=torch.float32), torch.Tensor()
        elif self.feature_setting == FEATURE_SETTING_OPTION_METADATA_ONLY or self.feature_setting == FEATURE_SETTING_SC_WITHOUT_CC:
            return torch.Tensor(), torch.as_tensor(cxt_metadata_seq_representation, dtype=torch.float32)
        else:
            return torch.as_tensor(cxt_content_seq_representation, dtype=torch.float32), \
                   torch.as_tensor(cxt_metadata_seq_representation, dtype=torch.float32)


@Predictor.register('rumor-tweets-tagger')
class RumorTweetTaggerPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        # self._tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)

    def predict(self, tweet_id: str, tweet_text: str) -> JsonDict:
        """

        :param tweet_id:
        :param tweet_text:
        :return: dict, {'logits': [], 'class_probabilities': [], 'label': str}
        """
        print("type(self): ", type(self._model))
        return self.predict_json({"tweet_id": tweet_id, "sentence": tweet_text})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = json_dict["sentence"]
        tweet_id = json_dict["tweet_id"]
        return self._dataset_reader.text_to_instance(tweet_id, sentence)


def model_training(train_set_path, heldout_set_path, test_set_path, n_gpu: Union[int, List] = -1,
                   train_batch_size: int = 100, model_file_prefix="",
                   feature_setting: int = FEATURE_SETTING_OPTION_FULL,
                   global_means: np.ndarray = None, global_stds: np.ndarray = None, num_epochs: int = 2,
                   social_encoder_option: int = SOCIAL_CONTEXT_ENCODER_OPTION_LSTM,
                   disable_cxt_type_option: int = DISABLE_CXT_TYPE_RETWEET,
                   attention_option: int = ATTENTION_OPTION_ATTENTION_WITH_CXT,
                   max_cxt_size_option: int = MAXIMUM_CONTEXT_SEQ_SIZE):
    timestamped_print(
        "start to train RumourDNN model with training set [%s] and heldout set [%s] with gpu [%s] ... " % (
            train_set_path, heldout_set_path, n_gpu))
    timestamped_print("the model will be evaluated with test set [%s]" % test_set_path)
    # enable GPU here
    n_gpu = config_gpu_use(n_gpu)

    timestamped_print("training batch size: [%s]" % train_batch_size)

    # "elmo_characters"
    elmo_token_indexer = ELMoTokenCharactersIndexer()
    rumor_train_set_reader = RumorTweetsDataReader(token_indexers={'elmo': elmo_token_indexer})

    timestamped_print("loading development dataset and indexing vocabulary  ... ")

    dev_set = rumor_train_set_reader.read(train_set_path)
    heldout_set = rumor_train_set_reader.read(heldout_set_path)

    vocab = Vocabulary.from_instances(dev_set + heldout_set)
    timestamped_print("done. dataset loaded and vocab indexed completely.")

    timestamped_print("initialising RumourDNN model ... ")
    model = instantiate_rumour_model(n_gpu, vocab, feature_setting=feature_setting)

    model.global_means = global_means
    model.global_stds = global_stds
    model.set_feature_setting(feature_setting)
    model.set_max_cxt_size(max_cxt_size_option)
    model.set_social_encoder_option(social_encoder_option)
    model.set_disable_cxt_type_option(disable_cxt_type_option)
    model.set_attention_mechanism(attention_option)
    model.set_layer_norm_layers()

    timestamped_print("model architecture: ")
    print(model)
    timestamped_print("done. RumourDNN model is initialised completely.")

    timestamped_print("initialising optimizer and dataset iteractor ... ")
    # Adam is currently recommeded as the default optim algorithm to use
    # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
    # initial Adam optimisation setting start with beta1 and beta2 values close to 1.0
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # number of batch depends on available memory
    # The BucketIterator batches sequences of similar lengths together to minimize padding.
    # The sorting_keys keyword argument tells the iterator which field to reference when determining the text length of each instance.
    iterator = BucketIterator(batch_size=train_batch_size, biggest_batch_first=True,
                              sorting_keys=[("sentence", "num_tokens")])
    # Iterators are responsible for numericalizing the text fields.
    # We pass the vocabulary we built earlier so that the Iterator knows how to map the words to integers.
    iterator.index_with(vocab)
    timestamped_print("done.")

    timestamped_print("training starting now ... ")
    # set num_epochs > 1 in experiment
    # 30 epochs (which is the necessary time for SGD to get to 94% accuracy with a 1-cycle policy) with Adam and L2 regularization was at 93.96% on average
    # option: set "grad_norm" to 5.0 to avoid NANs in gradients;
    # grad_clipping, https://openreview.net/pdf?id=ZY9xxQDMMu5Pk8ELfEz4
    # grad_norm=5.0,
    # grad_clipping=10.0,
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=dev_set,
                      validation_dataset=heldout_set,
                      shuffle=True,
                      patience=10,
                      num_epochs=num_epochs,
                      cuda_device=n_gpu)
    trainer.train()
    timestamped_print("done.")

    try:
        archive_model_from_memory(model, vocab, model_file_prefix)
    except AttributeError as err:
        timestamped_print("failed to archive model ... ")
        print(err)

    quick_test(model)
    evaluation(test_set_path, model, iterator, n_gpu)


def main():
    # pheme_6392078_train_set_combined.csv, pheme_6392078_heldout_set_combined.csv
    train_set_path = os.path.join(os.path.dirname(__file__), '..', "data", "train", "bostonbombings",
                                  "aug_rnr_train_set_combined.csv")
    heldout_set_path = os.path.join(os.path.dirname(__file__), '..', "data", "train", "bostonbombings",
                                    "aug_rnr_heldout_set_combined.csv")
    evaluation_data_path = os.path.join(os.path.dirname(__file__), '..', "data", "test", "bostonbombings.csv")

    n_gpu = -1
    # Reasonable minibatch sizes are usually: 32, 64, 128, 256, 512, 1024 (powers of 2 are a common convention)
    train_batch_size = 128
    model_file_prefix = "bostonbombings"
    model_training(train_set_path, heldout_set_path, evaluation_data_path, n_gpu, train_batch_size, model_file_prefix)


def instantiate_rumour_model(n_gpu, vocab, feature_setting: int = FEATURE_SETTING_OPTION_FULL,
                             max_cxt_size: int = MAXIMUM_CONTEXT_SEQ_SIZE):
    """
    we use the concatenation operation to combine two representations

    :param n_gpu:
    :param vocab:
    :param feature_setting:
    :param social_encoder_option:
    :return:
    """

    ELMO_EMBEDDING_DIM = 1024
    # dimension of contextual features
    #  option 1: User Profile embedding(1024) + Reply/Retweet embedding (1024) + context numerical features (26)
    #  option 2: reduced dimension of user profile embedding (20) + reduced dimension Reply/Retweet embedding (20) + social numerical features (20)
    # EXPECTED_CONTEXT_INPUT_SIZE = 2076
    # EXPECTED_CONTEXT_INPUT_SIZE = 512 * 3

    global elmo_credbank_model_path
    elmo_embedder = ElmoTokenEmbedder(
        options_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
        weight_file=elmo_credbank_model_path,
        do_layer_norm=False,
        dropout=0.5)
    word_embeddings = BasicTextFieldEmbedder({"elmo": elmo_embedder})
    # LSTM sequence encoder
    # It make no assumptions about the temporal/spatial relationships across the data
    text_embedding_lstm = torch.nn.LSTM(ELMO_EMBEDDING_DIM, ELMO_EMBEDDING_DIM, num_layers=2, batch_first=True)
    # option: torch.nn.GRU
    if n_gpu < 0:
        text_embedding_lstm.cpu()
    else:
        text_embedding_lstm.cuda(n_gpu)

    elmo_lstm_wrapper = PytorchSeq2VecWrapper(text_embedding_lstm)

    # the input dim of context LSTM network is the concatenation of source tweet embedding dim and social context encoding dim
    # we apply propagation context sequence pattern recogntion separately for context content
    #   and context metadata (represented by numerical features extracted from each reaction)
    context_content_encoder = None  # CC
    context_metadata_encoder = None  # CM
    # use Allennlp Seq2Vec wrapper to extract final state of LSTM for prediction
    context_content_encoder_rnn_wrapper = None
    context_metadata_encoder_rnn_wrapper = None
    CM_LSTM_INPUT_DIM = NUMERICAL_FEATURE_DIM
    # Hidden state represent how many different features you want to remember for either short term or long term
    # it's always better to try with 2X-4X of features you identified.
    CM_LSTM_HIDDEN_DIM = CM_LSTM_INPUT_DIM * 2
    CC_LSTM_INPUT_DIM = ELMO_EMBEDDING_DIM
    CC_LSTM_HIDDEN_DIM = CC_LSTM_INPUT_DIM * 2

    # deprecated: 'social_context_encoder2' is a deprecated experimental setting
    # it is kept for supporting to load & evaluate trained models
    PROPAGATION_CONTEXT_LSTM_HIDDEN_DIM = CC_LSTM_INPUT_DIM
    social_context_encoder2 = None

    if feature_setting == FEATURE_SETTING_OPTION_METADATA_ONLY:
        context_features_encoder = torch.nn.LSTM(CM_LSTM_INPUT_DIM,
                                                 CM_LSTM_HIDDEN_DIM, num_layers=2, batch_first=True,
                                                 bidirectional=False)
        context_metadata_encoder = torch.nn.LSTM(CM_LSTM_INPUT_DIM,
                                                 CM_LSTM_HIDDEN_DIM, num_layers=2, batch_first=True,
                                                 bidirectional=False)

        timestamped_print(
            "features are set to use context numerical hand-crafted features only. "
            "Context LSTM input dim is reset to [%s]"
            % CM_LSTM_INPUT_DIM)
        social_context_encoder2 = StackedSelfAttentionEncoder(
            input_dim=CM_LSTM_INPUT_DIM,
            hidden_dim=PROPAGATION_CONTEXT_LSTM_HIDDEN_DIM,
            projection_dim=128,
            feedforward_hidden_dim=128,
            num_layers=2,
            num_attention_heads=4)
    elif feature_setting == FEATURE_SETTING_OPTION_CONTEXT_CONTENT_ONLY or feature_setting == FEATURE_SETTING_SC_WITHOUT_CM:
        context_features_encoder = torch.nn.LSTM(CC_LSTM_INPUT_DIM,
                                                 CC_LSTM_HIDDEN_DIM, num_layers=2, batch_first=True,
                                                 bidirectional=False)
        context_content_encoder = torch.nn.LSTM(CC_LSTM_INPUT_DIM,
                                                CC_LSTM_HIDDEN_DIM, num_layers=2, batch_first=True,
                                                bidirectional=False)

        timestamped_print(
            "features are set to use social context content only. Context LSTM input dim is reset to [%s]"
            % CC_LSTM_INPUT_DIM)

        social_context_encoder2 = StackedSelfAttentionEncoder(
            input_dim=CC_LSTM_INPUT_DIM,
            hidden_dim=CC_LSTM_INPUT_DIM,
            projection_dim=128,
            feedforward_hidden_dim=128,
            num_layers=2,
            num_attention_heads=4)
    else:
        # CC encoding + CM encoding
        PROPAGATION_CONTEXT_LSTM_HIDDEN_DIM = CC_LSTM_INPUT_DIM + CM_LSTM_INPUT_DIM

        context_features_encoder = torch.nn.LSTM(CC_LSTM_INPUT_DIM + CM_LSTM_INPUT_DIM,
                                                 PROPAGATION_CONTEXT_LSTM_HIDDEN_DIM,
                                                 num_layers=2, batch_first=True, bidirectional=False)

        context_metadata_encoder = torch.nn.LSTM(CM_LSTM_INPUT_DIM,
                                                 CM_LSTM_HIDDEN_DIM, num_layers=2, batch_first=True,
                                                 bidirectional=False)
        context_content_encoder = torch.nn.LSTM(CC_LSTM_INPUT_DIM,
                                                CC_LSTM_HIDDEN_DIM, num_layers=2, batch_first=True,
                                                bidirectional=False)

        timestamped_print("context_metadata_encoder and context_content_encoder are set to LSTM.")
        social_context_encoder2 = StackedSelfAttentionEncoder(input_dim=CC_LSTM_INPUT_DIM + CM_LSTM_INPUT_DIM,
                                                              hidden_dim=PROPAGATION_CONTEXT_LSTM_HIDDEN_DIM,
                                                              projection_dim=128,
                                                              feedforward_hidden_dim=128,
                                                              num_layers=2,
                                                              num_attention_heads=4)

    if n_gpu < 0:
        text_embedding_lstm.cpu()
    else:
        context_features_encoder.cuda(n_gpu)

    context_features_lstm_wrapper = PytorchSeq2VecWrapper(context_features_encoder)

    # Seq2Vec is for LSTM only and Seq2Seq is for "LSTM+Attention"
    # Stateful or stateless ?
    # At the moment, we do not consider any dependencies between the social context sequences in our dataset
    # one use case of stateful LSTM is that we have a big sequence and have to split it into smaller sequence.
    #       In this case, if we want LSTM to remember state of last batch, we need to resort to stateful LSTM.
    # So, we do not need LSTM to remember the content of the previous batches.
    if context_content_encoder:
        context_content_encoder_rnn_wrapper = PytorchSeq2VecWrapper(context_content_encoder)
        context_content_encoder = PytorchSeq2SeqWrapper(context_content_encoder, stateful=False)

    if context_metadata_encoder:
        context_metadata_encoder_rnn_wrapper = PytorchSeq2VecWrapper(context_metadata_encoder)
        context_metadata_encoder = PytorchSeq2SeqWrapper(context_metadata_encoder, stateful=False)

    # Finally, we pass each encoded tweet output tensor to the feedforward layer to produce logits corresponding to tags
    # feed-forward layer can be 2 or 3 layers with following options:
    # 2 feed-forward linear layers with dropout[0.2, 0.0]
    # 3 feed-forward linear layers with dropout[0.2, 0.3, 0.0]
    # This consists of two linear transformation with a ReLU activation in between.
    # input dims of feedforward layer is the concatenation of lang_model_encoder output dim and
    #   social_context_encoder output dim (i.e., previous 2 LSTM network)

    if feature_setting == FEATURE_SETTING_OPTION_METADATA_ONLY:
        feedforward_input_dim = context_metadata_encoder.get_output_dim()
        feedforward_hidden_dim_1 = feedforward_input_dim
        feedforward_hidden_dim_2 = feedforward_input_dim
    elif feature_setting == FEATURE_SETTING_OPTION_CONTEXT_CONTENT_ONLY:
        feedforward_input_dim = context_content_encoder.get_output_dim()
        feedforward_hidden_dim_1 = feedforward_input_dim
        feedforward_hidden_dim_2 = int(feedforward_input_dim / 2)
    elif feature_setting == FEATURE_SETTING_OPTION_SOURCE_TWEET_CONTENT_ONLY:
        feedforward_input_dim = elmo_lstm_wrapper.get_output_dim() * 2 if \
            context_features_encoder.bidirectional else elmo_lstm_wrapper.get_output_dim()
        feedforward_hidden_dim_1 = elmo_lstm_wrapper.get_output_dim() * 2 if \
            context_features_encoder.bidirectional else elmo_lstm_wrapper.get_output_dim()
        feedforward_hidden_dim_2 = elmo_lstm_wrapper.get_output_dim() * 2 if \
            context_features_encoder.bidirectional else elmo_lstm_wrapper.get_output_dim()
    elif feature_setting == FEATURE_SETTING_OPTION_CONTEXT_ONLY:
        feedforward_input_dim = CC_LSTM_HIDDEN_DIM + CM_LSTM_HIDDEN_DIM
        feedforward_hidden_dim_1 = feedforward_input_dim
        feedforward_hidden_dim_2 = int(feedforward_hidden_dim_1 / 2)
    elif feature_setting == FEATURE_SETTING_SC_WITHOUT_CM:
        feedforward_input_dim = elmo_lstm_wrapper.get_output_dim() + context_content_encoder.get_output_dim()
        feedforward_hidden_dim_1 = feedforward_input_dim
        feedforward_hidden_dim_2 = int(feedforward_hidden_dim_1 / 2)
    elif feature_setting == FEATURE_SETTING_SC_WITHOUT_CC:
        feedforward_input_dim = elmo_lstm_wrapper.get_output_dim() + context_metadata_encoder.get_output_dim()
        feedforward_hidden_dim_1 = feedforward_input_dim
        feedforward_hidden_dim_2 = int(feedforward_hidden_dim_1 / 2)
    else:
        feedforward_input_dim = elmo_lstm_wrapper.get_output_dim() + context_content_encoder.get_output_dim() + context_metadata_encoder.get_output_dim()
        feedforward_hidden_dim_1 = feedforward_input_dim
        feedforward_hidden_dim_2 = int(feedforward_hidden_dim_1 / 2)
    # set final layer to project encoded tweet text corresponding to output logits (i.e., number of labels)
    class_number = 2
    # feedforward input size = ELMO embedding output dim + social context output size
    classifier_feedforward = FeedForward(input_dim=feedforward_input_dim, num_layers=3,
                                         hidden_dims=[feedforward_hidden_dim_1, feedforward_hidden_dim_2,
                                                      class_number],
                                         activations=[Activation.by_name("leaky_relu")(),
                                                      Activation.by_name("linear")(),
                                                      Activation.by_name("linear")()],
                                         dropout=[0.2, 0.3, 0.3])
    timestamped_print("set fully-connected layer to leaky_relu+linear+linear with dropout (0.2, 0.3, 0.0)")

    if n_gpu < 0:
        classifier_feedforward.cpu()
    else:
        print("enable GPU settings for feedforward layers.")
        classifier_feedforward.cuda(n_gpu)

    model = RumorTweetsClassifer(word_embeddings, elmo_lstm_wrapper,
                                 context_features_lstm_wrapper,
                                 context_content_encoder_rnn_wrapper,
                                 context_metadata_encoder_rnn_wrapper,
                                 vocab,
                                 classifier_feedforward=classifier_feedforward,
                                 cxt_content_encoder=context_content_encoder,
                                 cxt_metadata_encoder=context_metadata_encoder,
                                 social_context_self_attention_encoder=social_context_encoder2,
                                 max_cxt_size=int(max_cxt_size),
                                 cuda_device=n_gpu)
    if n_gpu < 0:
        model.cpu()
    else:
        model.cuda(n_gpu)
    return model


def config_gpu_use(n_gpu: Union[int, List] = -1) -> Union[int, List]:
    """
    set GPU device

    set to 0 for 1 GPU use

    Notes: 1)Dataloaders give normal (non-cuda) tensors by default. They have to be manually cast using the Tensor.to() method.
    2) Many methods are simply not implemented for torch.cuda.*Tensor. Thus, setting the global tensor type to cuda fails.
    3) Conversions to numpy using the numpy() method aren’t’ available for cuda tensors. One has to go x.cpu().numpy().

    :param n_gpu:
    :return:
    """
    if n_gpu != -1:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    return n_gpu


def quick_test(model_in_memory: RumorTweetsClassifer):
    # test the classifier
    timestamped_print("prediction test on trained rumor classifier: ")
    try:
        elmo_token_indexer = ELMoTokenCharactersIndexer()
        rumor_train_set_reader = RumorTweetsDataReader(token_indexers={'elmo': elmo_token_indexer})

        predictor = RumorTweetTaggerPredictor(model_in_memory, dataset_reader=rumor_train_set_reader)
        tweet_id = "580339510883561472"
        test_tweet_sentence = "Single-aisle Airbus A320s are the workhorses of the global fleet &amp; proliferate across European skies #4U9525 http://t.co/pe6OCMiy0q"

        outputs = predictor.predict(tweet_id, test_tweet_sentence)
        # predictor output:  {'logits': [0.10846924781799316, -0.1374501734972],
        #               'class_probabilities': [0.5611718893051147, 0.43882811069488525], 'label': '@@PADDING@@'}
        print("predictor output: ", outputs)

        print("print vocab: ")
        model_in_memory.vocab.print_statistics()
        # model.vocab.get_token_from_index(label_id, 'labels')
        timestamped_print("prediction label on (%s): %s" % (
            test_tweet_sentence, outputs["label"] if "label" in outputs else "label is unknown"))
    except Exception as e:
        timestamped_print("errors in quick model test ")
        print(e)


def evaluation(evaluation_data_path, model_in_memory: Model, data_iterator: DataIterator, cuda_device=-1):
    timestamped_print("evaluating  .... ")
    elmo_token_indexer = ELMoTokenCharactersIndexer()
    rumor_train_set_reader = RumorTweetsDataReader(token_indexers={'elmo': elmo_token_indexer})
    # predictor = RumorTweetTaggerPredictor(model_in_memory, dataset_reader=rumor_train_set_reader)

    # evaluation_data_path = os.path.join(os.path.dirname(__file__),  '..', "data","test","putinmissing-all-rnr-threads.csv")
    test_instances = rumor_train_set_reader.read(evaluation_data_path)
    import ntpath
    output_file_prefix = ntpath.basename(evaluation_data_path)
    output_file_prefix = output_file_prefix.replace(".", "_")

    from training_util import evaluate

    metrics = evaluate(model_in_memory, test_instances, data_iterator, cuda_device, "")
    timestamped_print("Finished evaluating.")
    timestamped_print("Metrics:")
    for key, metric in metrics.items():
        timestamped_print("%s: %s" % (key, metric))

    output_file = os.path.join(os.path.dirname(__file__), '..', "data", "test", output_file_prefix + "_eval.json")
    import json
    if output_file:
        with open(output_file, "w") as file:
            json.dump(metrics, file, indent=4)


def archive_model_from_memory(model_in_memory: Model, vocab: Vocabulary, file_prefix=""):
    """
    https://allennlp.org/tutorials

    :param model_in_memory:
    :param vocab:
    :param file_prefix, optional model file name prefix
    :return:
    """
    timestamped_print("archive model and vocabulary ... ")
    import time

    model_timestamp_version = datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M')
    serialization_dir = os.path.join(os.path.dirname(__file__), '..', "output", file_prefix + model_timestamp_version)
    os.makedirs(serialization_dir, exist_ok=True)
    vocab_dir_name = "vocabulary"
    weights_file_name = "weights_best.th"

    if not os.path.exists(serialization_dir):
        os.mkdir(serialization_dir)

    vocab_dir_path = os.path.join(serialization_dir, vocab_dir_name)
    weights_file_path = os.path.join(serialization_dir, weights_file_name)

    model_state = model_in_memory.state_dict()
    with open(weights_file_path, 'wb') as f:
        torch.save(model_state, f)

    vocab.save_to_files(vocab_dir_path)

    timestamped_print("done. model file and vocab file are archived in [%s]" % (serialization_dir))


def load_classifier_from_archive(vocab_dir_path: str = None,
                                 model_weight_file: str = None,
                                 n_gpu_use: Union[int, List] = -1,
                                 max_cxt_size: int = MAXIMUM_CONTEXT_SEQ_SIZE,
                                 feature_setting: int = FEATURE_SETTING_OPTION_FULL,
                                 global_means: np.ndarray = None, global_stds: np.ndarray = None,
                                 attention_option: int = ATTENTION_OPTION_ATTENTION_WITH_CXT) -> Tuple[
    RumorTweetsClassifer, RumorTweetTaggerPredictor]:
    """
    load model from archive and return rumour predictor

    see also #archive_model_from_memory()

    >>> rumor_dnn_predictor: RumorTweetTaggerPredictor
    >>> tweet_id = "580339510883561472"
    >>> test_tweet_sentence = "Single-aisle Airbus A320s are the workhorses of the global fleet &amp; proliferate across European skies #4U9525 http://t.co/pe6OCMiy0q"
    >>> result = rumor_dnn_predictor.predict(tweet_id, test_tweet_sentence)['logits']
    >>> tag_logits = result['logits']
    >>> prob = result['class_probabilities']


    :param vocab_dir_path:
    :param model_weight_file
    :param n_gpu_use
    :return:
    """
    model_timestamp_version = ""
    serialization_dir = os.path.join(os.path.dirname(__file__), '..', "output", model_timestamp_version)
    vocab_dir_name = "vocabulary"
    weights_file_name = "weights_best.th"

    if model_weight_file is None:
        model_weight_file = os.path.join(serialization_dir, weights_file_name)

    if vocab_dir_path is None:
        vocab_dir_path = os.path.join(serialization_dir, vocab_dir_name)

    n_gpu = config_gpu_use(n_gpu_use)
    vocab = Vocabulary.from_files(vocab_dir_path)
    model = instantiate_rumour_model(n_gpu, vocab, max_cxt_size=int(max_cxt_size), feature_setting=feature_setting)

    model.global_means = global_means
    model.global_stds = global_stds
    model.set_feature_setting(feature_setting)
    model.set_max_cxt_size(max_cxt_size)
    model.set_social_encoder_option(SOCIAL_CONTEXT_ENCODER_OPTION_LSTM)
    model.set_disable_cxt_type_option(DISABLE_CXT_TYPE_RETWEET)
    model.set_attention_mechanism(attention_option)
    model.set_layer_norm_layers()

    with open(model_weight_file, 'rb') as f:
        model.load_state_dict(torch.load(f))

    elmo_token_indexer = ELMoTokenCharactersIndexer()
    rumor_train_set_reader = RumorTweetsDataReader(token_indexers={'elmo': elmo_token_indexer})
    rumor_dnn_predictor = RumorTweetTaggerPredictor(model, dataset_reader=rumor_train_set_reader)

    return model, rumor_dnn_predictor


if __name__ == '__main__':
    main()
