from collections import abc
from src.data_loader import load_structure_json


def looping_nested_dict(structure, depth=0):
    """
    compute the depth of each tweet in a thread
    :param structure: json which contains the structure of a thread
    :param depth: the depth of each reply (source tweet's depth = 0)
    :return: a generator (tweet_id, depth)
    """
    for k, v in structure.items():
        if isinstance(v, abc.Mapping):
            yield k, depth
            depth += 1
            yield from looping_nested_dict(v, depth)
            depth -= 1
        else:
            yield k, depth


def get_structural_features(tweet_id, context_tweet_id):
    """
    Features related to the tree structre of conversation
    structure.json is required
     - persistence: the count of the total number of tweets posted in the thread by the author of the current tweet.
                    High numbers of tweets in a thread indicate that the author participates more (zubiaga2018disclosure)
     - depth: the max depth of this tweet in the thread
    :return: persitence and depth
    """
    structure = load_structure_json(tweet_id)
    thread_structure = list(looping_nested_dict(structure))
    persistence = 0
    depth = 0
    for (id, dep) in thread_structure:
        if id == context_tweet_id:
            persistence += 1
            depth = dep
    return persistence, depth

