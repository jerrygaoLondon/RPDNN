import json
import os
import re
from datetime import datetime
import pickle
# make "src" as source root in IntelliJ-Idea if you're encountering compilation err
from data_loader import load_user_relations
from glob import glob
import operator
from collections import OrderedDict
from typing import List

def get_user_history_tweeting_rumours(context_tweets_dataset_dir_dict: dict):
    """
      Create a file that contain user ids and timestamp at which each user posted a rumour
      :param context_tweets_dataset_dir_dict, tweet context dataset dictionary with all the directories that can be mapped and
        loaded for feature extraction by source tweet id
      :return: list {timestamp at which the user posted a rumour: user_id}
    """

    tweet_ids_tweeted_rumours = []  # Create a list to store the ids of users who tweeted rumours in the past

    for source_id, source_dir in context_tweets_dataset_dir_dict.items():  # Iterate each tweet id
        source_tweet_data_dir = os.path.join(source_dir, "source-tweets")
        rumour_type = os.path.splitext(source_dir)[0].split("/")[-2]
        if os.path.exists(source_tweet_data_dir):
            source_tweet_json_dataset = glob(os.path.join(source_tweet_data_dir, '*.json'))
            with open(os.path.join(source_tweet_data_dir, source_tweet_json_dataset[0]), 'r') as f:
                source_tweet_json = json.load(f)
            if rumour_type == "rumours":
                tweet_ids_tweeted_rumours.append((datetime.strptime(source_tweet_json["created_at"],
                                                                    '%a %b %d %H:%M:%S %z %Y'),
                                                  source_tweet_json['user']['id_str']))

    sorted_tweet_ids_tweeted_rumours = sorted(tweet_ids_tweeted_rumours[:], key=operator.itemgetter(0), reverse=False)
    sorted_tweet_ids_tweeted_rumours = OrderedDict(sorted_tweet_ids_tweeted_rumours)
    return sorted_tweet_ids_tweeted_rumours

    ## Save the list in the form of .pkl
    # with open('./tweet_users_posted_rumours', 'wb') as outfile:
    #     pickle.dump(sorted_tweet_ids_tweeted_rumours, outfile)
    #     outfile.close()
    #
    # with open('./tweet_users_posted_rumours', 'rb') as infile:
    #     summary_doc = pickle.load(infile)
    #     infile.close()
    # pp(summary_doc)


def get_account_age(context_tweet_timestamp, user_created_at, unit="day"):
    """
    Get the age of this account
    :param unit: time unit to use (day or hour)
    :return: account age (int)
    """
    if unit == "day":
        account_age = divmod((context_tweet_timestamp - user_created_at).total_seconds(), 86400)  # returns (quotient(days), remainder(seconds)

    elif unit == "hour":
        account_age = divmod((context_tweet_timestamp - user_created_at).total_seconds(), 3600)

    else:
        raise ValueError("Specify a correct time unit for an account age")

    account_age = account_age[0] + 1 if account_age[1] > 0 else account_age[0]
    return account_age


def load_tweet_users_posted_rumours():
    """
      load user history (whether a user posted any rumour in the past)

      :return: dict {timestamp at which the user posted a rumour: user_id}
      """

    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tweet_users_posted_rumours'), 'rb') as outfile:
        rumour_users = pickle.load(outfile)
        outfile.close()
    return rumour_users


def get_user_reliability(context_tweet_timestamp, context_user_id):
    """
    Whether this user posted any rumours in the past
    :param context_tweet_timestamp:
    :param context_user_id:
    :return: binary (0: this user posted rumours in the past, 1: otherwise)
    """
    rumour_users = load_tweet_users_posted_rumours()
    #
    check_user_history = dict(
        (rumour_posting_time, rumour_user) for (rumour_posting_time, rumour_user) in rumour_users.items()
        if (rumour_posting_time <= context_tweet_timestamp) and (rumour_user == context_user_id))
    is_user_reliable = 0 if bool(check_user_history) else 1
    return is_user_reliable


def get_user_relational_features(tweet_id, context_user_id, source_tweet_user_id):
    """
    Features related to the following-follow relationships between users in the conversational thread
    who-follows-whom.dat is required
    :return: cxt_user_to_sc_user, sc_user_to_cxt_user, is_reciprocal
    """
    user_relations = load_user_relations(tweet_id)
    cxt_user_to_sc_user = 0
    sc_user_to_cxt_user = 0
    is_reciprocal = 0

    for follower, following in user_relations:
        ## whether this author follows the author of the source tweet
        cxt_user_to_sc_user = 1 if (follower == context_user_id) and (following == source_tweet_user_id) else 0
        ## whether the source tweet's author follows this author
        sc_user_to_cxt_user = 1 if (follower == source_tweet_user_id) and (following == context_user_id) else 0
        ## whether the source tweet's author and this author follow each other
        is_reciprocal = 1 if (cxt_user_to_sc_user == 1) and (sc_user_to_cxt_user == 1) else 0
    return cxt_user_to_sc_user, sc_user_to_cxt_user, is_reciprocal


def user_features_main(reaction_status_json, source_tweet_user_id) -> List:
    context_tweet_timestamp = datetime.strptime(reaction_status_json["created_at"], '%a %b %d %H:%M:%S %z %Y')  # timestamp

    user_created_at = datetime.strptime(reaction_status_json['user']['created_at'],
                                        '%a %b %d %H:%M:%S %z %Y')  # timestamp at which the account was created
    context_user_id = reaction_status_json['user']['id_str']
    num_posts = reaction_status_json['user']['statuses_count']  # the # of tweets that this user has posted
    listed_count = reaction_status_json['user']['listed_count']  # the # of lists this user belongs to
    followers = reaction_status_json['user']['followers_count']  # the # of followers
    followings = reaction_status_json['user']['friends_count']  # the # of followings
    follow_ratio = followers / (followings + 1)  # the reputation of this user
    follow_ratio_ver2 = followers / (followings + followers + 1)
    user_favourites_count = reaction_status_json['user']['favourites_count']  # the # fo tweets this user has liked in the account's lifetime
    account_age = get_account_age(context_tweet_timestamp, user_created_at, unit="day")
    is_verified = 1 if reaction_status_json['user']['verified'] else 0  # is an account verified or not
    engagement_scr = num_posts / (account_age + 1)
    following_rate = followings / (account_age + 1)
    favourites_scr = user_favourites_count / (account_age + 1)
    is_geo_enabled = 1 if reaction_status_json['user']['geo_enabled'] else 0  # is geo-location enabled or not
    user_profile_description = reaction_status_json['user']['description']
    # has_description = 1 if user_profile_description is not None and len(user_profile_description.split()) > 0 else 0  # does this user have a description or not
    description_len = len(re.findall(r"[\w']+", reaction_status_json['user']['description'])) if \
    reaction_status_json['user']['description'] is not None else 0
    len_profilename = len(reaction_status_json['user']['name'])  # the number of characters in this user's profile name including white spaces)
    is_source_user = 1 if context_user_id == source_tweet_user_id else 0  # 1 if the source tweet user replied or retweeted his/her own source tweet


    # is_user_reliable = get_user_reliability(context_tweet_timestamp, context_user_id)
    # persitence, depth = get_structural_features() # only available with 'structure.json'
    # cxt_user_to_sc_user, sc_user_to_cxt_user, is_reciprocal = get_user_relational_features() # only available with 'who-follows-whom.dat'

    user_features = [num_posts,
                    listed_count,
                    followers,
                    followings,
                    follow_ratio,
                    follow_ratio_ver2,
                    user_favourites_count,
                    account_age,
                    int(is_verified),
                    engagement_scr,
                    following_rate,
                    favourites_scr,
                    int(is_geo_enabled),
                    # int(has_description),
                    description_len,
                    len_profilename,
                    is_source_user]
    # print("Number of user features ", len(user_features))
    return user_features

