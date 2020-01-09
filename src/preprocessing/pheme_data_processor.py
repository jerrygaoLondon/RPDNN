import json
import os
import random
from glob import glob

import numpy as np
import pandas as pd

# from preprocessing.CredbankProcessor import preprocessing_tweet_text
from CredbankProcessor import preprocessing_tweet_text
from src.data_loader import load_files_from_dataset_dir, load_matrix_from_csv
from pprint import pprint as pp

'''
This is a utility script that:

1) Generation of training data for RPDNN model using the PHEME dataset 
Link to the original dataset  
https://figshare.com/articles/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078 

2) generate independent test set from PHEME dataset for CREDBANK fine-tuned ELMo
'''

# Path to the pheme dataset (all-rnr-annotated-threads)
my_path = os.path.join(os.path.dirname(__file__),'..', '..', '..', '..', 'Data/all-rnr-annotated-threads')

train_set = {"charliehebdo-all-rnr-threads", "ebola-essien-all-rnr-threads",
             "ferguson-all-rnr-threads", "germanwings-crash-all-rnr-threads",
             "ottawashooting-all-rnr-threads",
             "prince-toronto-all-rnr-threads", "putinmissing-all-rnr-threads",
             "sydneysiege-all-rnr-threads"}

test_set = {"gurlitt-all-rnr-threads"}


def generate_development_set(my_path, output_dir="aug_rnr_training"):
    for folders in glob(os.path.join(my_path, '*')):
        event_name = os.path.basename(folders) ## Get event name
        print("Event: ", event_name)
        event_types = glob(os.path.join(folders, '*')) # folders named rumours or non-rumours
        # output_df = pd.DataFrame(columns=['tweet_id', 'created_at', 'text', 'retweet_count', 'user_name', 'user_verified', 'follower'])
        # output_df = pd.DataFrame(columns=['tweet_id', 'created_at', 'text', 'label']) # create a dataframe to store output
        output_df = pd.DataFrame(columns=['tweet_id', 'created_at', 'text', 'label', 'user_id', 'user_name']) # create a dataframe to store output

        for event_t in event_types:
            category = os.path.basename(event_t) # rumours or non-rumours
            print("category: ")
            print(event_name, category)
            idfiles = glob(os.path.join(event_t, '*'))
            print(idfiles[:3])
            print("")
            for id_f in idfiles:  # visit each tweet folder
                source_tweet = glob(os.path.join(id_f, 'source-tweets/*'))  # consider source tweets only

                # print(source_tweet)
                # print("id_f: ", id_f)
                tweet_file_name = os.path.basename(id_f)
                # print("tweet_file_name: ", tweet_file_name)
                assert len(source_tweet)==1
                with open(source_tweet[0], 'r') as f:
                    tweet = json.load(f)
                    # if is_retweet(tweet):
                    #     tweet = get_original_source_tweet(tweet)

                    timestamp = pd.DatetimeIndex([tweet['created_at']])
                    tweet_id = tweet["id_str"]
                    # print("tweet id inside: ", tweet_id)
                    if int(tweet_id) != int(tweet_file_name):
                        print("warning: data set problem. check mismatched json name and tweet id inside.")
                        continue

                    if category == 'rumours':
                        #print("saving this rumour source tweet: ")
                        output_df.loc[len(output_df)] = [tweet['id_str'], timestamp[0], tweet['full_text'] if "full_text" in tweet else tweet["text"], 1, tweet['user']['id_str'], tweet['user']['screen_name']]
                    elif category == "non-rumours":
                        #print("saving this non-rumours source tweet: ")
                        output_df.loc[len(output_df)] = [tweet['id_str'], timestamp[0], tweet['full_text'] if "full_text" in tweet else tweet["text"], 0, tweet['user']['id_str'], tweet['user']['screen_name']]
        #print(output_df)
        filename = os.path.join(os.path.dirname(my_path), output_dir+'/{}.csv'.format(event_name))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        output_df.to_csv(filename, index=None, encoding='utf-8')


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


def oversampling_pos(data_X, shuffling_options, X, train_diffs):
    """
    oversampling on positive train examples when dataset is few

    Oversampling should only be applied to train set before the split

    class imbalance like 4:1 above can cause problems

    :param data_X:
    :param shuffling_options:
    :param X:
    :param train_diffs:
    :return:
    """
    print("oversampling %s positive examples ... " % train_diffs)
    random_option = random.randint(0,4)
    data_indices = shuffling_options[random_option][0]

    oversampled_size = 0
    for data_index in data_indices:
            if oversampled_size >= train_diffs:
                break

            dev_data_example = X[data_index]
            # if current instance is positive example
            if dev_data_example[3] == 1:
                data_X = np.append(data_X, [dev_data_example], axis=0)
                oversampled_size +=1

    return data_X


def undersampling_neg(data_X):
    """
    undersampling on negative train examples when dataset is few

    Oversampling should only be applied to train set before the split

    class imbalance like 4:1 above can cause problems

    :param data_X:
    :param shuffling_options:
    :param X:
    :param train_diffs:
    :return:
    """

    pos_samples, neg_samples = separate_pos_neg(data_X)
    train_diffs = len(neg_samples) - len(pos_samples)

    print("undersampling %s negative examples ... " % train_diffs)

    import random
    undersampled_indices = random.sample(range(1, len(neg_samples)), train_diffs)

    print("total %s undersampled examples" % len(undersampled_indices))
    neg_samples = np.delete(neg_samples, undersampled_indices, axis=0)

    pos_samples = np.array(pos_samples)
    print("pos_samples.shape: ", pos_samples.shape)
    print("neg_samples.shape: ", neg_samples.shape)
    return np.concatenate([pos_samples, neg_samples])



def separate_pos_neg(data_X):
    pos_samples = []
    neg_samples = []

    for i in range(0, len(data_X)):
        dev_data_example_i = data_X[i]
        if dev_data_example_i[3] == 1:
            pos_samples.append(dev_data_example_i)
        else:
            neg_samples.append(dev_data_example_i)
    return pos_samples, neg_samples


def export_data(train_X, test_event, file_name):
    """

    :param train_X:
    :param file_name: pheme_6392078_train_set_combined.csv or pheme_6392078_heldout_set_combined.csv
    :return:
    """
    # train_set_file = os.path.join(os.path.dirname(__file__),  '..', '..', "data", "train", test_event, file_name)
    # train_set_file = os.path.join(os.path.dirname(__file__),  '..', '..', "data", "Multitask", "aug-boston", test_event, file_name)
    train_set_file = os.path.join(os.path.dirname(__file__),  '..', '..', "data", "test-balance", test_event, file_name)
    os.makedirs(os.path.dirname(train_set_file), exist_ok=True)

    output_df = pd.DataFrame(train_X)
    # output_df.to_csv(train_set_file, header=['tweet_id', 'created_at', 'text', 'label'], index=None)
    output_df.to_csv(train_set_file, header=['tweet_id', 'created_at', 'text', 'label', 'user_id', 'user_name'], index=None, encoding="utf-8")
    print("export %s to %s" % (output_df.shape, train_set_file))


def generate_combined_dev_set(training_data_dir_path, test_set_dir_path):
    """
    combine every individual development set generated from generate_development_set(),
    shuffle the dataset,
    and split into train and validation set
    and save into a single csv file

    The development set are generated from PHE RNR-annotated threads directory and saved into 4-columns csv files

    We use oversampling of positive examples for original PHEME 6392078 training set
    and use undersampling of negative examples for augmented dataset

    :return:
    """
    # dataset_dir = "C:\\Data\\NLP-corpus\\PHEME-dataset\\pheme_training"
    # dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'Data/pheme_training')
    dataset_dir = training_data_dir_path
    test_dataset_dir = test_set_dir_path
    print(os.path.exists(dataset_dir))

    all_train_dataset_path = load_files_from_dataset_dir(dataset_dir)
    print("all_train_dataset_path: ", all_train_dataset_path)
    dev_set_events = {'bostonbombings','charliehebdo', 'ebola-essien', 'ferguson', 'germanwings', 'gurlitt',
                      'ottawashooting', 'prince-toronto', 'sydneysiege'}

    test_set_events = {'putinmissing'}

    dev_set_path = list(filter(None, [os.path.join(dataset_dir, individual_train_set_path) if individual_train_set_path.replace('.csv','').split('-')[0]
                                                                   in dev_set_events else ''
                                      for individual_train_set_path in all_train_dataset_path]))

    test_set_path = list(filter(None, [os.path.join(test_dataset_dir, individual_train_set_path) if individual_train_set_path.replace('.csv','').split('-')[0]
                                                                    in test_set_events else ''
                                       for individual_train_set_path in all_train_dataset_path]))

    print("dev_set_path: ")
    pp(dev_set_path)
    # assert len(dev_set_path)==5
    print("test_set_path: ")
    pp( test_set_path)

    X = None
    for dev_set_file in dev_set_path:
        df = load_matrix_from_csv(dev_set_file, header=0, start_col_index=0, end_col_index=6)
        print("dataset loaded from %s" % dev_set_file)
        print("current event set size: ", df[:].shape)
        if X is None:
            X = df[:]
        else:
            X = np.append(X, df[:], axis=0)

        pos_inst, neg_inst = check_dataset_balance(df)

    test_X = None
    for test_set_file in test_set_path:
        df = load_matrix_from_csv(test_set_file, header=0, start_col_index=0, end_col_index=6)
        print("dataset loaded from %s" % test_set_file)
        print("current event set size: ", df[:].shape)
        if test_X is None:
            test_X = df[:]
        else:
            test_X = np.append(test_X, df[:], axis=0)

        pos_inst, neg_inst = check_dataset_balance(df)

    print("final total size: ", test_X.shape)
    print("")

    print("undersampling test set ...")
    test_X = undersampling_neg(test_X)


    from sklearn.model_selection import ShuffleSplit
    rs = ShuffleSplit(n_splits=5, random_state=random.randint(1,10), test_size=0.10, train_size=None)

    train_X = np.empty(shape=(0, 6), dtype=object)
    # train_X = np.empty(shape=(0, 4), dtype=object)
    validation_X = np.empty(shape=(0, 6), dtype=object)
    # validation_X = np.empty(shape=(0, 4), dtype=object)

    shuffling_options = list(rs.split(X))

    random_option = random.randint(0,4)
    train_indices = shuffling_options[random_option][0]
    test_indices = shuffling_options[random_option][1]

    print("done. TRAIN:", train_indices, ", size: ", len(train_indices), "TEST:", test_indices, ", size: ", len(test_indices))

    for train_index in train_indices:
        train_X = np.append(train_X, [X[train_index]], axis=0)

    for heldout_index in test_indices:
        validation_X = np.append(validation_X, [X[heldout_index]], axis=0)


    print("train size: ", train_X.shape)
    print("validation set size: ", validation_X.shape)
    print("")
    print("* check balance of train set ... ")
    train_pos_inst, train_neg_inst = check_dataset_balance(train_X)

    print("* check balance of held-out set ... ")
    val_pos_inst, val_neg_inst = check_dataset_balance(validation_X)
    print("")
    # oversampling positive instances
    train_diffs = train_neg_inst - train_pos_inst
    print("difference btw positive examples and negative examples in train set: ", train_diffs)

    val_diffs = val_neg_inst - val_pos_inst
    print("difference btw positive examples and negative examples in validation set: ", val_diffs)

    #print("oversampling (only applied to) train set... ")
    #train_X = oversampling_pos(train_X, shuffling_options, X, train_diffs)
    #print("train side after oversampling positive examples, ", train_X.shape)

    print("undersampling train set ...")
    train_X = undersampling_neg(train_X)
    print("train size after undersampling negative examples, ", train_X.shape)

    print("undersampling heldout set ...")
    validation_X = undersampling_neg(validation_X)

    print("checking dataset balance ...")
    #print("balance after oversampling train set: ")
    print("balance after undersampling train set: ")
    check_dataset_balance(train_X)

    print("balance after undersampling held-out set: ")
    check_dataset_balance(validation_X)

    print("balance after undersampling test set: ")
    check_dataset_balance(test_X)

    #export_data(train_X, "pheme_6392078_train_set_combined.csv")
    #export_data(validation_X, "pheme_6392078_heldout_set_combined.csv")
    # export_data(train_X, "{}".format(list(test_set_events)[0]), "aug_rnr_train_set_combined.csv")
    # export_data(validation_X, "{}".format(list(test_set_events)[0]), "aug_rnr_heldout_set_combined.csv")
    # export_data(test_X, "{}".format(list(test_set_events)[0]), "aug_rnr_test_set_combined.csv")


def check_test_set():
    print("check data balance in test set: ")
    test_set_path = os.path.join(os.path.dirname(__file__),  '..', "..", "data","test","putinmissing-all-rnr-threads.csv")
    df = load_matrix_from_csv(test_set_path, header=0, start_col_index=0, end_col_index=4)
    # print(df)
    check_dataset_balance(df)
    # -> positive instances: 126 , negative instances: 112


def check_dataset_balance(df):
    pos_inst = 0
    neg_inst = 0
    for row in df[:]:
        if row[3] == 1:
            pos_inst += 1
        else:
            neg_inst += 1

    print("positive instances: %s , negative instances: %s" %(pos_inst, neg_inst))
    print("")
    return pos_inst, neg_inst


def generate_corpus_4_elmo_test():
    """
    we fine-tuned pre-trained ELMo model on credbank based on the assumption that
    credbank corpus is a credibility-focus Twitter dataset which can be used as a representative corpus
    for rumour detection domain/task.

    :return:
    """
    print("")

    # source tweet dataset processed from PHEME dataset covering 9 events
    dataset_dir = os.path.join('C:\\Data\\NLP-corpus\\PHEME-dataset', 'pheme_training')

    print(os.path.exists(dataset_dir))

    all_test_dataset_path = load_files_from_dataset_dir(dataset_dir)

    all_events = {'charliehebdo', 'ebola-essien', 'ferguson', 'germanwings', 'gurlitt',
                      'ottawashooting', 'prince-toronto', 'sydneysiege', 'putinmissing'}

    all_set_path = list(filter(None, [os.path.join(dataset_dir, individual_data_set_path) if individual_data_set_path.split('-')[0]
                                                                                              in all_events else ''
                                      for individual_data_set_path in all_test_dataset_path]))

    print("all event data set_path: ", all_set_path)

    X = None
    for event_set_file in all_set_path:
        df = load_matrix_from_csv(event_set_file, header=0, start_col_index=0, end_col_index=4)
        print("dataset loaded from %s" % event_set_file)
        print("current event set size: ", df[:].shape)
        if X is None:
            X = df[:]
        else:
            X = np.append(X, df[:], axis=0)

    # have 6178 source tweets available in PHEME dataset (6392078)
    print("all PHEME source tweets are loaded: ", X.shape)

    print("Export PHEME corpus: ")

    pheme_data_output_dir = os.path.join(os.path.dirname(__file__),  '..', '..',  "output", "elmo")
    try:
        os.mkdir(pheme_data_output_dir)
    except Exception as err:
        print()

    pheme_source_tweet_corpus_path = os.path.join(pheme_data_output_dir, "pheme_source_tweet_corpus.txt")

    with open(pheme_source_tweet_corpus_path, mode='w', encoding='utf-8') as outputfile:
        for row in X[:]:
            normed_text_tokens = preprocessing_tweet_text(row[2])
            if len(normed_text_tokens) > 0:
                outputfile.write("%s\n" % " ".join(normed_text_tokens))
    print("done")


if __name__ == '__main__':
    #generate_corpus_4_elmo_test()

    #my_path = os.path.join('c:\\','Data', 'NLP-corpus', 'aug_rnr', 'aug-rnr-merge-data-14052019-balance')
    # my_path = os.path.join('..','..','data','social_context','aug-rnr-merge-data-23052019-balance')
    # my_path = os.path.join('..','..','data','social_context','aug-rnr-merge-data-13062019')
    # my_path = os.path.join('..','..','data','social_context','pheme-early')
    # C:\Data\NLP-corpus\aug_rnr\test_aug_data
    # my_path = os.path.join('c:\\','Data', 'NLP-corpus', 'aug_rnr', 'test_aug_data')
    # generate_development_set(my_path, output_dir='baseline')



    # development_set_path = os.path.join('c:\\','Data', 'NLP-corpus', 'aug_rnr', 'r')
    development_set_path = os.path.join('..','..','data','social_context','aug_rnr_training')
    # development_set_path = os.path.join('..','..','data','social_context','all_rnr_training')
    test_set_path = os.path.join('..','..','data','social_context','all_rnr_training')
    # test_set_path = os.path.join('..','..','data','social_context','aug_rnr_training')
    generate_combined_dev_set(development_set_path, test_set_path)
    # check_test_set()