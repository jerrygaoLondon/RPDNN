### Utility module to load dataset
import os
import json
import pickle

import sys
if sys.platform == "win32":
    import win32com.client
from typing import Generator, Dict, Text

# to fix a weird crash due to "ValueError: failed to parse CPython sys.version '3.6.6 |Anaconda, Inc.| (default, Jun 28 2018, 11:27:44) [MSC v.1900 64 bit (AMD64)]'"
# possible due to a bug on anaconda
# see https://stackoverflow.com/questions/34145861/valueerror-failed-to-parse-cpython-sys-version-after-using-conda-command
try:
    import sys # Just in case
    start = sys.version.index('|') # Do we have a modified sys.version?
    end = sys.version.index('|', start + 1)
    version_bak = sys.version # Backup modified sys.version
    sys.version = sys.version.replace(sys.version[start:end+1], '') # Make it legible for platform module
    import platform
    platform.python_implementation() # Ignore result, we just need cache populated
    platform._sys_version_cache[version_bak] = platform._sys_version_cache[sys.version] # Duplicate cache
    sys.version = version_bak # Restore modified version string
except ValueError: # Catch .index() method not finding a pipe
    pass

import pandas as pd
import csv
from pprint import pprint as pp
from datetime import datetime
from datetime import timedelta

###########Global Varaible ##################
DISABLE_CXT_TYPE_NONE = 0
DISABLE_CXT_TYPE_REPLY = 1
DISABLE_CXT_TYPE_RETWEET = 2
##############################################
# from training_util import is_retweet, get_original_source_tweet

# Global variables
#  the context data will link to PHEME 6392078 dataset directory
# Note: better to keep this symlink directory name general (as "aug-rnr-annotated-threads-retweets") and you can link it to your local/specific name
social_context_data_dir = os.path.join(os.path.dirname(__file__),  '..', "data", "social_context","aug-rnr-annotated-threads-retweets")
# =============================================================


def load_abs_path(data_path: str)-> str:
    """
    read actual data path from either symlink or a absolute path

    :param data_path: either a directory path or a file path
    :return:
    """
    if not os.path.exists(data_path) or os.path.islink(data_path):
        if sys.platform == "win32":
            return readlink_on_windows(data_path)
        else:
            return os.readlink(data_path)

    return data_path


def readlink_on_windows(short_cut_link: str):
    """
    get file actual path via its short cut on windows system
    :param short_cut_link:
    :return: Str, file absolute path
    """

    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(short_cut_link+".lnk")
    return shortcut.Targetpath


def load_tweets_context_dataset_dir(social_context_data_dir):
    """
    load tweet context dataset directory into a dictionary that can be mapped and
        loaded for feature extraction by source tweet id

    This method assumes that the all the context tweets (i.e., replies) are organised under a directory
        named as source tweet id.
    Thus, by the source tweet id, we can load all the replies (from its root directory or subdirectories)
    :return:dict {tweet id: context tweet dataset directory path}
    """
    #global social_context_data_dir
    # social_context_data_dir = os.path.join(os.path.dirname(__file__),  '..', "data", "social_context","pheme-rnr-dataset")
    print("load social context data from dir: ", social_context_data_dir)
    social_context_dataset_dir = social_context_data_dir
    social_context_dataset_abs_path = load_abs_path(social_context_dataset_dir)
    print("social_context_dataset_abs_path: ", social_context_dataset_abs_path)
    all_subdirectories = [x[0] for x in os.walk(social_context_dataset_abs_path)]
    # print(all_subdirectories)
    print("done.")
    return {os.path.basename(subdirectory): subdirectory for subdirectory in all_subdirectories if os.path.basename(subdirectory).isdigit()}


def load_files_from_dataset_dir(dataset_dir) -> dict:
    file_paths = dict()
    all_files = [f for f in os.listdir(dataset_dir)]
    return all_files


context_tweets_dataset_dir_dict = {}


def load_source_tweet_context(source_tweet_id: Text, event_end_timedelta: timedelta = None,
                              disable_cxt_type:int =DISABLE_CXT_TYPE_RETWEET) -> Generator[Dict, None, None]:
    """
    load contextual tweets (replies and retweets) of a source tweet by its tweet id

    For retweets, motivation is that retweets may carry believability characteristic of rumour-bearing tweets
        1) influential twitter accounts are less likely retweet unverified information
        2) Number of retweets can serve as a normative cue that leads a person to presume
        that an unverified rumor is true, increasing the likelihood that they will share it with others,
        according to the study published in Cyberpsychology, Behavior, and Social Networking.

    :param source_tweet_id:
    :param event_end_timedelta: event context end timedelta to filter context
    :param disable_retweets: whether disable retweets in context
    :return: context object (retweet or reply)
    """
    global context_tweets_dataset_dir_dict, social_context_data_dir
    if len(context_tweets_dataset_dir_dict) == 0:
        context_tweets_dataset_dir_dict = load_tweets_context_dataset_dir(social_context_data_dir)

    context_tweets_dataset_dir = context_tweets_dataset_dir_dict[source_tweet_id]
    source_tweet_json = load_source_tweet_json(source_tweet_id)

    event_end_time = None
    event_start_time =datetime.strptime(source_tweet_json["created_at"], '%a %b %d %H:%M:%S %z %Y')
    if event_end_timedelta:
        event_end_time = event_start_time + event_end_timedelta

    if event_end_time:
        print("event start time [%s], end time: [%s]" %(str(event_start_time), str(event_end_time)))

    # the context types here corresponding to the sub-directory names in tweet data set
    context_types = ["reactions", 'retweets']

    for c_type in context_types:
        if disable_cxt_type == DISABLE_CXT_TYPE_REPLY and c_type == "reactions":
            continue

        if disable_cxt_type == DISABLE_CXT_TYPE_RETWEET and c_type == "retweets":
            continue

        all_context_data_dir = os.path.join(context_tweets_dataset_dir, "{}".format(c_type))
        reaction_dir = os.path.join(all_context_data_dir)
        if not os.path.isdir(reaction_dir):
            # reaction ('replies' or 'retweets') not exist
            continue

        source_tweet_reply_json_dataset = os.listdir(reaction_dir)
        for source_tweet_reply_json_file_name in source_tweet_reply_json_dataset:
            if source_tweet_reply_json_file_name.startswith("."):
                continue

            source_tweet_context_json = load_tweet_json(os.path.join(all_context_data_dir, source_tweet_reply_json_file_name))
            source_tweet_context_json['context_type'] = c_type

            reaction_time = datetime.strptime(source_tweet_context_json["created_at"], '%a %b %d %H:%M:%S %z %Y')
            if event_start_time is not None and event_end_time is not None and \
                    (reaction_time < event_start_time or reaction_time > event_end_time):
                continue

            yield source_tweet_context_json


def load_source_tweet_json(source_tweet_id: Text):
    """
        load source tweet json from PHEME dataset by its tweet id
    :param source_tweet_id:
    :return:
    """
    global context_tweets_dataset_dir_dict, social_context_data_dir
    if len(context_tweets_dataset_dir_dict) == 0:
        context_tweets_dataset_dir_dict = load_tweets_context_dataset_dir(social_context_data_dir)

    try:
        source_tweet_dir = context_tweets_dataset_dir_dict[source_tweet_id]
        # source tweet directory is named 'source-tweets' in "6392078" dataset
        source_tweet_data_dir = os.path.join(source_tweet_dir, "source-tweets")
        if not os.path.isdir(source_tweet_data_dir):
            # source tweet directory is named 'source-tweet' in version 1 of PHEME dataset
            source_tweet_data_dir = os.path.join(source_tweet_dir, "source-tweet")

    except KeyError:
        raise Exception("source tweet (id :%s) is not found in current social context dataset!" % source_tweet_id)

    source_tweet_json_dataset = os.listdir(os.path.join(source_tweet_data_dir))

    for source_tweet_json_file_name in source_tweet_json_dataset:
        if source_tweet_json_file_name.startswith(".") or source_tweet_json_file_name.endswith('.csv'):
            continue

        source_tweet_json = load_tweet_json(os.path.join(source_tweet_data_dir, source_tweet_json_file_name))

        ## TODO: it's a temporary fix since our current augmented dataset contains retweets in source tweet data

        # if is_retweet(source_tweet_json):
        #     source_tweet_json = get_original_source_tweet(source_tweet_json)

    return source_tweet_json


def load_tweet_json(tweet_reply_json_path):
    try:
        with open(tweet_reply_json_path, encoding='UTF-8', mode='r') as f:
            data = json.load(f)

    except UnicodeDecodeError as err:
        print("failed to process json file : %s. Error: %s" %(tweet_reply_json_path, err.reason))
        raise err

    return data


def load_tweet_users_posted_rumours():
    """
      load user history (whether a user posted any rumour in the past)

      :return: dict {timestamp at which the user posted a rumour: user_id}
      """

    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tweet_users_posted_rumours'), 'rb') as outfile:
        rumour_users = pickle.load(outfile)
        outfile.close()
    return rumour_users


def load_structure_json(source_tweet_id):
    """
    load structure.json
    This file provides the structure of the conversation
    :return: json
    """
    try:
        structure_json = os.path.join(context_tweets_dataset_dir_dict[source_tweet_id], 'structure.json')
        with open(structure_json, 'r') as f:
            structure = json.load(f)
    except KeyError:
        raise Exception("structure json for the source tweet (id: %s) is not found " % source_tweet_id)
    return structure


def load_user_relations(source_tweet_id):
    """
    load who-follows-whom.dat
    load the relationship between users, within the thread, who are following someone else
    :return:
    """
    try:
        user_relaton_path = os.path.join(context_tweets_dataset_dir_dict[source_tweet_id], 'who-follows-whom.dat')
        user_relations = []
        with open(user_relaton_path, 'r') as f:
            for line in f:
                user_relations.append(tuple(line.strip('\n').split('\t')))
    except KeyError:
        raise Exception("user relationship information for the source tweet (id: %s) is not found " % source_tweet_id)
    return user_relations


def load_matrix_from_csv(fname, start_col_index, end_col_index, delimiter=',', encoding='utf-8',
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
    print("reading instances from csv file at: ", fname)

    df = pd.read_csv(fname, header=header, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL,
                     usecols=range(start_col_index, end_col_index), lineterminator='\n',
                     encoding=encoding).values

    return df


def test_load_tweets_context_dataset_dir():
    x = load_tweets_context_dataset_dir(social_context_data_dir)
    pp(x)

def test_load_source_tweet_context():
    #source_tweet = load_source_tweet_json("552784600502915072")
    #print(list(source_tweet))

    #source_tweet_context_json = load_source_tweet_context("580339510883561472")

    source_tweet_context_json = load_source_tweet_context("524963572083085313")
    print(list(source_tweet_context_json))


if __name__ == '__main__':
    test_load_source_tweet_context()