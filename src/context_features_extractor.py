import os
from datetime import datetime
from typing import Dict
import numpy as np
from allennlp.modules import FeedForward, Seq2VecEncoder

from data_loader import load_source_tweet_json, load_source_tweet_context, DISABLE_CXT_TYPE_RETWEET
from preprocessing.CredbankProcessor import preprocessing_tweet_text

np.set_printoptions(suppress=True)


import torch

from data_loader import load_abs_path
from context_features.user_features import user_features_main
from context_features.tweet_features import tweet_features_main

# load ELMos model, check your symlink if it complains "FileNotFoundError"
from allennlp.commands.elmo import ElmoEmbedder

elmo_credbank_model_path = load_abs_path(
    os.path.join(os.path.dirname(__file__), '..', "resource", "embedding", "elmo_model", "elmo_credbank_2x4096_512_2048cnn_2xhighway_weights_10052019.hdf5"))
fine_tuned_elmo = ElmoEmbedder(
    options_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
    weight_file=elmo_credbank_model_path)
ELMO_EMBEDDING_DIM = 1024
NUMERICAL_FEATURE_DIM:int = 28
EXPECTED_CONTEXT_INPUT_SIZE:int = 20 * 3

CONTEXT_TYPE_REPLY = 1
CONTEXT_TYPE_RETWEET = 0

FEATURE_SETTING_OPTION_METADATA_ONLY = 2
FEATURE_SETTING_OPTION_CONTEXT_CONTENT_ONLY = 3
FEATURE_SETTING_OPTION_CONTEXT_ONLY = 4
FEATURE_SETTING_SC_WITHOUT_CM = 5
FEATURE_SETTING_SC_WITHOUT_CC = 6
# to be deprecated
DISABLE_SETTING_AUTO_ENCODER = 1

from embeddings.embedding_layer import sentence_embedding_elmo


def context_feature_extraction_from_context_status(reaction_status_json: Dict, source_tweet_user_screen_name: str,
                                                   source_tweet_timestamp: datetime, source_tweet_user_id: str,
                                                   source_text: str,
                                                   elmo_embedding_dim: int = 1024,
                                                   cxt_content_encoder: Seq2VecEncoder = None,
                                                   elmo_model: ElmoEmbedder = None,
                                                   feature_setting=-1) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    encode reactions (replies and retweets)

    TODO: additional feature:
        1) content level interaction features between rumour correction
            (e.g., contain links to Snopes articles in comments, the spread of rumours on Facebook, [Maddock 2015])

            use "Mention" as an indicator, see also a comprehensive review of featuures for rumour detection via https://arxiv.org/pdf/1711.00726.pdf

        2) diffusion patterns at scale
            (e.g., false rumour features indicating spike say number of reaction in first N hours in early stage [Maddock 2015])
        3) reaction threshold ---> 7 days
            a) top N reactions; b) first N hours/days reactions;
        4) qualitative and quantitative features
            e.g.,  Top-K keywords in replies  ?

        5) additional feature: use of emotions, clarity and credible source attribution [Chua 2018]

        6) threshold to filter noisy context
            we need a practice to have a good combination of posts and timeline to minimise noisies (e.g., minimum replies of replies or minimum likes of replies)
            bing able to get tree structure of propagations is possibly better to get represenative reactions

    :param reaction_status_json: tweet json for either a tweet reply or retweet
    :param source_tweet_user_screen_name: source tweet user name only useful when current reaction is reply
    :param source_tweet_timestamp: source tweet created time
    :param source_text, source tweet text
    :param elmo_embedding_dim, ELMo embedding output dimension
    :param tweet_content_feedforward, auxiliary feedforward layer to reduce dimension for tweet content representation of social context
    :param social_numerical_feature_feedforward, auxiliary feedforward layer to expand dimension for social numerical features
    :param elmo_model, ELMo fine-tuned model
    :return: either np.ndarray or Tuple(np.ndarray, np.ndarray, np.ndarray),
        representation of reactions from the concatenation btw user profile embedding and numerical features from reaction status
        The output dimension is the EXPECTED_CONTEXT_INPUT_SIZE
    """
    numerical_features = extract_social_numerical_features(reaction_status_json, source_tweet_user_screen_name,
                                                           source_tweet_timestamp, source_tweet_user_id, source_text)

    #print("user_profile_embedding shape: ", user_profile_embedding.shape)
    #print("reply_sent_embedding shape: ", reply_sent_embedding.shape)
    #print("numerical_features shape: ", numerical_features.shape)
    if feature_setting == FEATURE_SETTING_OPTION_METADATA_ONLY:
        return torch.Tensor(), torch.Tensor(), numerical_features

    user_profile_description = reaction_status_json['user']['description']
    has_description = 1 if user_profile_description is not None and len(user_profile_description.split()) > 0 else 0  # does this user have a description or not
    if elmo_model is None:
        elmo_model = fine_tuned_elmo
    # textual features
    # vector representation
    user_profile_embedding = np.zeros(elmo_embedding_dim)
    if has_description and feature_setting != FEATURE_SETTING_OPTION_METADATA_ONLY:
        try:
            tokenised_user_profile_description = user_profile_description.split(' ')
            user_profile_embedding = sentence_embedding_elmo(tokenised_user_profile_description, elmo_model, avg_all_layers=True)

        except Exception as e:
            print("failure in tokenising user profile")
            print(e)

    reply_sent_embedding = None
    if cxt_content_encoder and feature_setting != FEATURE_SETTING_OPTION_METADATA_ONLY:
        reply_sent_embedding = encode_reply_content(cxt_content_encoder, elmo_model, feature_setting, reaction_status_json)

    #return np.concatenate((user_profile_embedding, reply_sent_embedding, numerical_features))
    return user_profile_embedding, reply_sent_embedding, numerical_features


def encode_reply_content(cxt_content_encoder, elmo_model, feature_setting, reaction_status_json):
    reply_sent_embedding = np.zeros(cxt_content_encoder.get_input_dim())
    c_type = CONTEXT_TYPE_REPLY if reaction_status_json[
                                       'context_type'] == "reactions" else CONTEXT_TYPE_RETWEET  # whether the context tweet is a retweet or reply
    if c_type == CONTEXT_TYPE_REPLY:
        if 'text' in reaction_status_json:
            source_tweet_reply_text = reaction_status_json["text"]
        elif 'full_text' in reaction_status_json:
            source_tweet_reply_text = reaction_status_json["full_text"]
        else:
            print("check reaction json object")
            raise ValueError

        norm_source_tweet_reply_text_tokens = preprocessing_tweet_text(source_tweet_reply_text)
        if len(norm_source_tweet_reply_text_tokens) > 0 and feature_setting != FEATURE_SETTING_OPTION_METADATA_ONLY:
            reply_sent_embedding = sentence_embedding_elmo(norm_source_tweet_reply_text_tokens, elmo_model,
                                                           avg_all_layers=True)

        else:
            reply_sent_embedding = np.zeros(cxt_content_encoder.get_input_dim())
    return reply_sent_embedding


def extract_social_numerical_features(reaction_status_json: Dict, source_tweet_user_screen_name: str,
                                      source_tweet_timestamp: datetime, source_tweet_user_id: str,
                                      source_text: str) -> np.ndarray:
    numerical_features = []
    context_tweet_timestamp = datetime.strptime(reaction_status_json["created_at"], '%a %b %d %H:%M:%S %z %Y')  # timestamp
    c_type = 1 if reaction_status_json['context_type'] == "reactions" else 0  # whether the context tweet is a retweet or reply
    user_profile_description = reaction_status_json['user']['description']
    has_description = 1 if user_profile_description is not None and len(user_profile_description.split()) > 0 else 0  # does this user have a description or not

    ## Load user_features
    user_features = user_features_main(reaction_status_json, source_tweet_user_id)
    ## Load tweet features
    tweet_features = tweet_features_main(reaction_status_json, source_tweet_user_screen_name, source_text)

    # incorporate time interval explicitly into the model ? or incorporate time difference between observations into our features
    t_diff = (context_tweet_timestamp - source_tweet_timestamp).total_seconds() / 60  # difference between the posting time of the context tweet and that of the source tweet (in minutes)

    numerical_features.extend((user_features+tweet_features))
    numerical_features.extend([t_diff, c_type, has_description])
    numerical_features = np.array(numerical_features)

    return numerical_features


def context_feature_extraction(source_tweet_context_dataset, source_tweet, context_feature_input_dim: int = EXPECTED_CONTEXT_INPUT_SIZE):
    """
    extract features from a collection of tweet context (replies, likes, retweetsetc)

    :param source_tweet_context_dataset: json list
    :param source_tweet: json object of current source tweet
    :param context_feature_input_dim: expected input dimension of contextual features
    :return: torch.Tensor or numpy.ndarray, vector representation of social-temporal contextual feature ready to feed into RNN
    """

    source_user_name = source_tweet['user']['screen_name']
    source_id = source_tweet['id_str']
    if 'full_text' in source_tweet:
        source_text = source_tweet['full_text']
    elif 'text' in source_tweet:
        source_text = source_tweet['text']
    else:
        raise ValueError
    source_tweet_timestamp = source_tweet['created_at']
    source_tweet_user_id = source_tweet['user']['id_str']

    source_tweet_timestamp = datetime.strptime(source_tweet_timestamp, '%a %b %d %H:%M:%S %z %Y')
    # source_tweet_user_id = source_tweet['user']['id_str']
    reply_to_user = '@' + source_user_name

    # print(source_tweet_timestamp)
    replies_in_temporal_order = []
    replies_in_temporal_order = np.array([]).reshape(0, context_feature_input_dim)
    for context_tweet in source_tweet_context_dataset:
        hand_crafted_context_features = context_feature_extraction_from_context_status(context_tweet, reply_to_user,
                                                                                       source_tweet_timestamp,
                                                                                       source_tweet_user_id, source_text)

        if hand_crafted_context_features.size < context_feature_input_dim:
            #for debug
            print("Error: context feature dimension [%s] is lower than expected value [%s]" % (hand_crafted_context_features.size, context_feature_input_dim))

        ## the degree of this user in the user graph
        feature_array = np.array(hand_crafted_context_features)
        # print("feature_array sum ", np.sum(feature_array))
        # print("hand_crafted_context_features shape: ", feature_array.shape)
        replies_in_temporal_order = np.vstack((replies_in_temporal_order, hand_crafted_context_features))
    # features = np.array(sorted(replies_in_temporal_order[:], key=operator.itemgetter(0), reverse=False))
    return replies_in_temporal_order


def test_context_features_shape():
    # tweet_id = "500235112785924096"
    # tweet_id = "499648409767538688"
    # tweet_id = "500295393301647360"
    #test_context_features_shape_by_tweet_id("500327120770301952")
    #test_context_features_shape_by_tweet_id("500235112785924096")
    #test_context_features_shape_by_tweet_id("499648409767538688")
    #test_context_features_shape_by_tweet_id("500295393301647360")
    #test_context_features_shape_by_tweet_id("500375440285237248")
    #test_context_features_shape_by_tweet_id("544316500002484226")

    # test_context_features_shape_by_tweet_id("552846698524184576")
    # test_context_features_shape_by_tweet_id("499360762167848960")
    # test_context_features_shape_by_tweet_id("500364914796802048")
    # test_context_features_shape_by_tweet_id("499606771246567424")
    test_context_features_shape_by_tweet_id("500350777844457473")


def test_context_features_shape_by_tweet_id(source_tweet_id, expected_context_feature_output=EXPECTED_CONTEXT_INPUT_SIZE):
    print("test [%s] context feature" % source_tweet_id)

    source_tweet_json = load_source_tweet_json(source_tweet_id)
    source_tweet_context_dataset = load_source_tweet_context(source_tweet_id)

    cxt_features = context_feature_extraction(source_tweet_context_dataset, source_tweet_json,
                                              context_feature_input_dim=expected_context_feature_output)
    print("context feature shapes: ", cxt_features.shape)
    print("cxt_features.size: ", cxt_features.size)
    assert cxt_features.shape[1] == expected_context_feature_output


def compute_global_values_4_all_dataset_numercial_features():
    from typing import List
    """
    train_set_collection = [os.path.join(os.path.dirname(__file__),  '..', "data", "train","bostonbombings","aug_rnr_train_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "train","bostonbombings","aug_rnr_heldout_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "train","charliehebdo","aug_rnr_train_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "train","charliehebdo","aug_rnr_heldout_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "train","ferguson","aug_rnr_train_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "train","ferguson","aug_rnr_heldout_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "train","germanwings","aug_rnr_train_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "train","germanwings","aug_rnr_heldout_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "train","ottawashooting","aug_rnr_train_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "train","ottawashooting","aug_rnr_heldout_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "train","sydneysiege","aug_rnr_train_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "train","sydneysiege","aug_rnr_heldout_set_combined.csv")]
    """
    train_set_collection = [os.path.join(os.path.dirname(__file__),  '..', "data", "loocv_set_20191002","charliehebdo","all_rnr_train_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "loocv_set_20191002","charliehebdo","all_rnr_heldout_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "loocv_set_20191002","ferguson","all_rnr_train_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "loocv_set_20191002","ferguson","all_rnr_heldout_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "loocv_set_20191002","germanwings","all_rnr_train_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "loocv_set_20191002","germanwings","all_rnr_heldout_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "loocv_set_20191002","ottawashooting","all_rnr_train_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "loocv_set_20191002","ottawashooting","all_rnr_heldout_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "loocv_set_20191002","sydneysiege","all_rnr_train_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "loocv_set_20191002","sydneysiege","all_rnr_heldout_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "loocv_set_20191002","twitter15","all_rnr_train_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "loocv_set_20191002","twitter15","all_rnr_heldout_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "loocv_set_20191002","twitter16","all_rnr_train_set_combined.csv"),
                            os.path.join(os.path.dirname(__file__),  '..', "data", "loocv_set_20191002","twitter16","all_rnr_heldout_set_combined.csv")]

    all_event_source_tweet_ids = []

    for train_set_path in train_set_collection:
        source_tweet_id_list = load_training_set_source_tweets(train_set_path)
        all_event_source_tweet_ids.extend(source_tweet_id_list)

    print("[%s] tweets loaded" % len(all_event_source_tweet_ids))

    all_numerical_features = []
    for source_tweet_id in all_event_source_tweet_ids:
        source_tweet_json = load_source_tweet_json(source_tweet_id)

        source_tweet_user_id = source_tweet_json['user']['id_str']
        source_tweet_text = source_tweet_json["text"] if "text" in source_tweet_json else source_tweet_json["full_text"]
        source_tweet_timestamp = datetime.strptime(source_tweet_json['created_at'], '%a %b %d %H:%M:%S %z %Y')
        source_tweet_user_name = '@' + source_tweet_json['user']['screen_name']

        # from September, experiment showed the evidence that retweets data is very noisy. we now focus on reply content and metadata
        source_tweet_context_dataset: List[Dict] = list(load_source_tweet_context(source_tweet_id, disable_cxt_type=DISABLE_CXT_TYPE_RETWEET))

        for source_tweet_reaction in source_tweet_context_dataset:
            numerical_features = extract_social_numerical_features(source_tweet_reaction, source_tweet_user_name,
                                                                   source_tweet_timestamp, source_tweet_user_id,
                                                                   source_tweet_text)
            # print(numerical_features)
            all_numerical_features.append(numerical_features)

    all_numerical_features = np.array(all_numerical_features)
    print("global mean: ", all_numerical_features.mean(0))
    print("global std: ", all_numerical_features.std(0))
    print("global max: ", all_numerical_features.max(0))
    print("global min: ", all_numerical_features.min(0))


def load_training_set_source_tweets(train_set_path):
    from training_util import load_matrix_from_csv

    dataset_instances = load_matrix_from_csv(train_set_path, 0, 1, header=0)
    source_tweet_id_list = []
    for dataset_instance in dataset_instances:
        source_tweet_id_list.append(str(dataset_instance[0]))
    return source_tweet_id_list


def standard_scaling(numerical_features : np.ndarray, global_means: np.ndarray, global_std: np.ndarray):
    """
    The global mean and standard-deviation are calculated per-dimension over whole training set

    most values will be between -1 and 1

    :param numerical_features:
    :param global_means:
    :param global_std:
    :return:
    """
    return (numerical_features - global_means) / global_std


if __name__ == '__main__':
    compute_global_values_4_all_dataset_numercial_features()

