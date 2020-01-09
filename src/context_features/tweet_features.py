import json
import os
import re
from datetime import datetime
import numpy as np
import operator
from collections import OrderedDict
from typing import Iterator, List, Dict, Union, Tuple

def tweet_features_main(reaction_status_json, source_tweet_user_screen_name, source_text) -> List:
    num_retweets = reaction_status_json["retweet_count"]
    num_favorites = reaction_status_json["favorite_count"] if reaction_status_json["favorite_count"] is not None else 0  # indicates approximately how many times this Tweet has been liked by Twitter users

    if 'full_text' in reaction_status_json:
        context_text = reaction_status_json['full_text']
    elif 'text' in reaction_status_json:
        context_text = reaction_status_json['text']
    else:
        raise ValueError

    has_question = 1 if re.findall(r"\?", context_text) else 0
    is_duplicate = 1 if context_text.strip() == source_text.strip() else 0
    has_img = 1 if reaction_status_json['user']['profile_use_background_image'] else 0
    has_urls = 1 if reaction_status_json['entities']['urls'] else 0  # if the tweet has urls
    if has_urls ==1:
        num_urls = len(reaction_status_json['entities']['urls'])
    else:
        num_urls = 0
    ## if the tweet contains native media (shared with the Tweet user-interface as opposed via a link to elsewhere)
    ## e.g., https://twitter.com/RT_com/status/500350777844457473/photo/1
    has_native_media = 1 if 'extended_entities' in reaction_status_json else 0
    # native_media_type_map = {'photo':0, 'video':1, 'animated_gif':2}
    # if has_native_media == 1:
    #     media_type = native_media_type_map[reaction_status_json['extended_entities']['media'][0]['type']]
    # else:
        # media_type = 3
    context_len = len(re.findall(r"[\w']+", re.sub(r"(?:{})\s+| http\S+".format(source_tweet_user_screen_name),
                                "", context_text)))  # tweet length after removing the source author's screen name and urls
    tweet_features = [
        num_retweets,
        num_favorites,
        int(has_question),
        int(is_duplicate),
        int(has_img),
        int(has_urls),
        num_urls,
        int(has_native_media),
        context_len
    ]
    # print("Number of tweet features: ", len(tweet_features))
    return tweet_features

