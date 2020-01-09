"""
Credbank Processor covers 1) load and dump pre-processed CREDBANK tweets dataset;
    2) shuffling and split dataset into training and hold-out set;
    3) corpus statistics;
    4) sanity check of generated training dataset
"""
import re
from typing import List

from gensim.utils import deaccent
from nltk.tokenize import TweetTokenizer
import os
import pandas as pd
import csv
import unicodedata


def export_credbank_trainset(dataset_dir):
    """
    export credbank as trainset for ELMo fine-tune

    :param dataset_dir: this directory should contain corpus batch files dumped by eu.socialsensor.twcollect.TweetCorpusDownloader

    :return:
    """

    dataset_files = load_all_files_path(dataset_dir)
    export_file_path = os.path.join(dataset_dir, "credbank_dataset_corpus_v2.txt")
    all_tweets_size = 0
    all_dedup_tweets_size = 0
    vocabulary = set()

    # sanity check (with maximum 280 characters per tweet)
    LONG_CHAR_LEN = 280
    min_char_len = 0
    max_char_len = 0
    avg_char_len = 0
    all_char_len = []

    with open(export_file_path, mode='w', encoding='utf-8') as outputfile:
        for credbank_corpus_batch_i in dataset_files:
            print("loading, preprocessing and export corpus in batch from [%s]" % credbank_corpus_batch_i)
            tweet_corpus_batch_i = load_tweets_from_credbank_csv(credbank_corpus_batch_i)
            all_collected_tweets_from_batch_i = list(tweet_corpus_batch_i)
            all_tweets_size_batch_i = len(all_collected_tweets_from_batch_i)
            print("totals tweets collected from current batch: [%s]" % all_tweets_size_batch_i)
            all_tweets_size += all_tweets_size_batch_i
            dedup_tweet_corpus_batch_i = set(all_collected_tweets_from_batch_i)
            all_dedup_tweets_size_batch_i = len(dedup_tweet_corpus_batch_i)
            print("total deduplicated tweets from current batch: [%s]" % all_dedup_tweets_size_batch_i)
            all_dedup_tweets_size += all_dedup_tweets_size_batch_i
            for tweet_text in dedup_tweet_corpus_batch_i:
                tweet_char_len = len(tweet_text)
                if min_char_len == 0 and max_char_len == 0 and avg_char_len == 0:
                    min_char_len = tweet_char_len
                    max_char_len = tweet_char_len
                    avg_char_len = tweet_char_len

                if tweet_char_len < min_char_len:
                    min_char_len = tweet_char_len

                if tweet_char_len > max_char_len:
                    max_char_len = tweet_char_len

                all_char_len.append(tweet_char_len)

                outputfile.write("%s\n" % tweet_text)
            print("done.")

        print("all tweet sentences: ", all_tweets_size) # v1: 77954446; v2: 26695955
        print("all deduplicated tweet sentences: ", all_dedup_tweets_size) # v1: 6157180; v2: 3053922
        print("minimum characer length: ",min_char_len) # 0
        print("maximum character length: ", max_char_len) # 412
        import numpy as np
        print("average character length: ", np.mean(all_char_len)) # 88.967
        print("all complete.")


def generate_train_held_out_set(train_corpus_path):
    """
    generate small held-out set for testing language model perplexity (compare perplexity before and after fine-tune)
    :param train_corpus_path:
    :return:
    """
    from sklearn.model_selection import ShuffleSplit
    with open(train_corpus_path, mode='r', encoding='utf-8') as train_file:
        train_set = train_file.readlines()

    # in v1: with test_size=0.0002, we will have 1232 tweets in held-out set and 6155948 in train set
    # in v2: with test_size=0.0005, we have 3052395 in train set and 1527 in hold-out set

    rs = ShuffleSplit(n_splits=1, random_state=0, test_size=0.0005, train_size=None)
    splitted_sets = list(rs.split(train_set))
    shuffled_train_set = splitted_sets[0][0]
    shuffled_held_set = splitted_sets[0][1]

    print("total number of train set: ", len(shuffled_train_set))
    print("total number of held set: ", len(shuffled_held_set))
    #print("train set: ", shuffled_train_set)
    #print("held set: ", shuffled_held_set)

    train_data_dir = os.path.dirname(train_corpus_path)
    shuffled_train_set_path = os.path.join(train_data_dir, "shuffled_credbank_train_corpus_v2.txt")
    shuffled_held_set_path = os.path.join(train_data_dir, "shuffled_credbank_held_corpus_v2.txt")

    with open(shuffled_train_set_path, mode='w', encoding='utf-8') as outputfile:
        for shuffled_train_indice in shuffled_train_set:
            outputfile.write("%s" % train_set[shuffled_train_indice])

    with open(shuffled_held_set_path, mode='w', encoding='utf-8') as outputfile:
        for shuffled_held_indice in shuffled_held_set:
            outputfile.write("%s" % train_set[shuffled_held_indice])

    print("shuffled train and held-out set are generated and exported.")


def _load_matrix_from_csv(fname,start_col_index, end_col_index, delimiter=',', encoding='utf-8', header=None):
    """
    #WARNING: BUG in reading credbank csv file.

    Cause: credbank CSV file dumped by 'eu.socialsensor.twcollect.TweetCorpusDownloader' has inconsistent columns.
    Most of rows has size 13 column and many/few raws have column size 9.

    load gs terms (one term per line) from "csv" txt file
    :param fname:
    :param start_col_index:
    :param end_col_index:
    :param encoding:
    :param header default as None, header=0 denotes the first line of data
    :return:
    """
    df = pd.read_csv(fname, header=header, delimiter=delimiter, quoting=csv.QUOTE_NONE, usecols=range(start_col_index, end_col_index), lineterminator='\n', encoding=encoding).as_matrix()
    return df


def load_tweets_from_credbank_csv(credbank_dataset_path):
    """
    load tweet text from 9th column

    preprocess the text for fine-tuning ELMo model

    "Each file contains pre-tokenized and white space separated text, one sentence per line. Don't include the <S> or </S> tokens in your training data."
    https://github.com/allenai/bilm-tf

    :param credbank_dataset_path:
    :return:
    """
    print("load tweets from_credbank batch file: ", credbank_dataset_path)
    """
    df = _load_matrix_from_csv(credbank_dataset_path, delimiter="\t", start_col_index=0, end_col_index=13)

    for tweet_row in df[:]:
        tweet_text = tweet_row[8]
        if str(tweet_text) != 'nan':
            print(tweet_row)
            print("tweet row size: ", len(tweet_text))
            norm_tweet = preprocessing_tweet_text(tweet_text)
            if len(norm_tweet) > 100:
                print("abnormal tweet: ", tweet_row)

            yield " ".join(norm_tweet)
    """
    with open(credbank_dataset_path, encoding="utf-8") as f:
        line = f.readline()
        while line:
            record = line.splitlines()[0]
            columns = record.split('\t')
            columns_size = len(columns)
            line = f.readline()

            if columns_size != 13:
                print("error column size: ", columns_size)

                # print(columns)
                # inconsistent_column_size.append(columns_size)
                # inconsistent_record_size += 1
            if columns[0] == 200 or columns[8] != "null":
                # print(columns[8])
                # total_retrived_size += 1
                tweet_text = columns[8]
                norm_tweet = preprocessing_tweet_text(tweet_text)
                if len(norm_tweet) > 140:
                    # few tweets are essentially image, e.g., https://twitter.com/activator_n/status/522308768667693056, 522292180790964225
                    print("abnormally long tweet: ", norm_tweet)

                yield " ".join(norm_tweet)


def load_all_files_path(dataset_dir):
    all_files = []
    for file in os.listdir(dataset_dir):
        if "csv" in file:
            all_files.append(os.path.join(dataset_dir, file))

    return all_files


def _run_strip_accents(self, text):
    """Strips accents from a piece of text.
    Taken from Google bert, https://github.com/google-research/bert/blob/master/tokenization.py
    """
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # Taken from Google bert, https://github.com/google-research/bert/blob/master/tokenization.py
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    # Taken from Google bert, https://github.com/google-research/bert/blob/master/tokenization.py

    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    # Taken from Google bert, https://github.com/google-research/bert/blob/master/tokenization.py
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def preprocessing_tweet_text(tweet_text) -> List[str]:
    """
    Neural Language Model like ELMo does not need much normalisation. Pre-trained ELMo model only need pre-tokenised text.

    see also 'test_preprocessing_tweet_text()' method

    :param tweet_text:
    :return:
    """
    if not isinstance(tweet_text, str):
        raise ValueError("Text parameter must be a Unicode object (str)!")

    # remove URL
    # norm_tweet = re.sub(r'http\S+', '', tweet_text)
    norm_tweet = tweet_text
    norm_tweet = norm_tweet.lower()
    norm_tweet = re.sub(r'http\S+', '', norm_tweet)

    # remove retweets
    # norm_tweet = re.sub('^(rt)( @\w*)?[: ]', '', norm_tweet)

    # http\S+
    # norm_tweet = re.sub(r"https?://(www\.)?", "", norm_tweet)
    # norm_tweet = re.sub(r'^https?:\/\/.*[\r\n]*', '', norm_tweet, flags=re.MULTILINE)
    # norm_tweet = re.sub(r'http\S+', '', norm_tweet)
    # norm_tweet = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", norm_tweet)

    # remove pic URL
    norm_tweet = re.sub(r"pic.twitter.com\S+", "", norm_tweet)
    # remove user mentions
    norm_tweet = re.sub(r"(?:\@|https?\://)\S+", "", norm_tweet)
    # remove punctuations:
    # norm_tweet = re.sub(pattern=r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]', repl='', string=norm_tweet).strip()
    # deaccent
    norm_tweet = deaccent(norm_tweet)

    tknzr = TweetTokenizer()
    tokenised_norm_tweet = tknzr.tokenize(norm_tweet)

    # https://www.shanelynn.ie/word-embeddings-in-python-with-spacy-and-gensim/

    # Set the minimum number of tokens to be considered
    if len(tokenised_norm_tweet) < 4:
        return []

    num_unique_terms = len(set(tokenised_norm_tweet))

    # Set the minimum unique number of tokens to be considered (optional)
    if num_unique_terms < 2:
        return []

    return tokenised_norm_tweet


def corpus_statistics():
    # train_corpus_path = "/userstore/jieg/credbank/corpus/credbank_train_corpus.txt"
    # train_corpus_path = "C:\\Data\\credbank\\tweets_corpus\\credbank_dataset_corpus_v2.txt"
    # train_corpus_path = "C:\\Data\\credbank\\tweets_corpus\\shuffled_credbank_train_corpus_v2.txt"
    # train_corpus_path = "C:\\Data\\credbank\\tweets_corpus\\shuffled_credbank_held_corpus_v2.txt"
    # train_corpus_path = "F:\\Data\\rumour_dataset\\snap-twitter7\\post-processed\\tweets2009-06_tweets.txt"
    train_corpus_path = "F:\\Data\\rumour_dataset\\snap-twitter7\\post-processed\\snap_tweet_hold_out.txt"

    print("Corpus statistics on ", train_corpus_path)

    with open(train_corpus_path, mode='r', encoding='utf-8') as file:
        train_corpus = file.readlines()

    from nltk.tokenize.regexp import WhitespaceTokenizer
    whitespace_tokenize = WhitespaceTokenizer().tokenize
    corpus_size = 0
    for tweet in train_corpus:
        tokens = whitespace_tokenize(tweet)
        corpus_size += len(tokens)

    print("all words (corpus size): ", corpus_size)

    from sklearn.feature_extraction.text import CountVectorizer

    #extract tokens
    text_vectorizer = CountVectorizer(analyzer='word', tokenizer=WhitespaceTokenizer().tokenize, ngram_range=(1, 1), min_df=1)
    X = text_vectorizer.fit_transform(train_corpus)
    # Vocabulary
    vocab = list(text_vectorizer.get_feature_names())
    print("vocabulary size: ", len(vocab)) # 913611
    counts = X.sum(axis=0).A1

    from collections import Counter
    freq_distribution = Counter(dict(zip(vocab, counts)))

    print("top N frequent words: ", freq_distribution.most_common(10))


def test_preprocessing_tweet_text():
    tweet_1 = "RT @TheManilaTimes: Cheers, tears welcome Pope Francis - The Manila Times OnlineThe Manila Times Online http://www.manilatimes.net/cheers-tears-welcome-pope-francis/155612/ …	3:31 am - 15 Jan 2015"
    pre_tweet_1 = preprocessing_tweet_text(tweet_1)
    assert pre_tweet_1 == ['rt', 'cheers', ',', 'tears', 'welcome', 'pope', 'francis', '-', 'the', 'manila', 'times', 'onlinethe', 'manila', 'times', 'online', '…', '3:31', 'am', '-', '15', 'jan', '2015']

    tweet_2 = "Welcome to the Philippines Pope Francis @Pontifex Pray for the Philippines & the entire world.	5:19 pm - 15 Jan 2015"
    pre_tweet_2 = preprocessing_tweet_text(tweet_2)
    assert pre_tweet_2 == ['welcome', 'to', 'the', 'philippines', 'pope', 'francis', 'pray', 'for', 'the', 'philippines', '&', 'the', 'entire', 'world', '.', '5:19', 'pm', '-', '15', 'jan', '2015']

    tweet_3 = "Retweet if you're proud Filipino! \"Welcome to the Philippines Pope Francis\" http://bit.ly/150Zqcq  ۞| http://bit.ly/1INBcie"
    pre_tweet_3 = preprocessing_tweet_text(tweet_3)
    assert pre_tweet_3 == ['retweet', 'if', "you're", 'proud', 'filipino', '!', '"', 'welcome', 'to', 'the', 'philippines', 'pope', 'francis', '"', '۞', '|']

    tweet_4 = "Why Lambert, Lovren and Lallana have struggled at Liverpool http://dlvr.it/8gqKRv  @PLNewsNow"
    pre_tweet_4 = preprocessing_tweet_text(tweet_4)
    assert pre_tweet_4 == ['why', 'lambert', ',', 'lovren', 'and', 'lallana', 'have', 'struggled', 'at', 'liverpool']

    tweet_5 = "UNBELIEVABLE - Senate Democrats Release Scathing #CIA #Torture Report http://www.buzzfeed.com/kyleblaine/senate-democrats-to-release-cia-torture-report-today … via @kyletblaine @buzzfeednews"
    pre_tweet_5 = preprocessing_tweet_text(tweet_5)
    assert pre_tweet_5 == ['unbelievable', '-', 'senate', 'democrats', 'release', 'scathing', '#cia', '#torture', 'report', '…', 'via']


def sanity_check_loaded_credbank_tweets():
    tweet_corpus = load_tweets_from_credbank_csv("C:\\Data\\credbank\\tweets_corpus\\credbank_xad.csv")
    total_size = 0

    tweet_corpus_batch_i = []

    for tweet in tweet_corpus:
        if tweet is not None and tweet.strip() != "":
            tweet_corpus_batch_i.append(tweet)
        # print(tweet)
        total_size += 1
    print("size of original tweets: ", total_size)

    print("total batch size without empty filtering: ", len(tweet_corpus_batch_i))
    tweet_corpus_batch_i = set(tweet_corpus_batch_i)
    print("total batch size  after deduplication: ", len(tweet_corpus_batch_i))
    """
    total_retrived_size = 0
    inconsistent_record_size = 0
    inconsistent_column_size = []
    with open("C:\\Data\\credbank\\tweets_corpus\\credbank_xai.csv", encoding="utf-8") as f:
        line = f.readline()

        while line:
            record = line.splitlines()[0]
            columns = record.split('\t')

            columns_size = len(columns)
            total_size += 1
            # print(columns)
            if columns_size != 13:
                print(columns_size)
                print(columns)
                inconsistent_column_size.append(columns_size)
                inconsistent_record_size += 1

            if columns[0] == 200 or columns[8] != "null":
                total_retrived_size += 1

            line = f.readline()

    print("size of original tweets: ", total_size)
    print("size of retrieved tweets: ", total_retrived_size)
    print("inconsistent row size: ", inconsistent_record_size)
    print("inconsistent column sizes: ", set(inconsistent_column_size))
    """

    # sanity check of sentence sizes
    """
    min_sentence_size = 0
    max_sentence_size = 0
    avg_sentence_size = 0
    all_sentence_size = []

    for credbank_tweet in tweet_corpus:
        tweet_sent_size = len(credbank_tweet)
        # initialisation
        if min_sentence_size == 0 and max_sentence_size == 0 and avg_sentence_size == 0:
            min_sentence_size = tweet_sent_size
            max_sentence_size = tweet_sent_size

        if tweet_sent_size < min_sentence_size:
            min_sentence_size = tweet_sent_size

        if tweet_sent_size > max_sentence_size:
            max_sentence_size = tweet_sent_size

        all_sentence_size.append(tweet_sent_size)

    import numpy as np
    avg_sentence_size = np.mean(all_sentence_size)
    print("minimum tweet size: ", min_sentence_size)
    print("maximum tweet size: ", max_sentence_size)
    print("average tweet size: ", avg_sentence_size)
    """


if __name__ == '__main__':
    # test_preprocessing_tweet_text()

    #weet_corpus = load_tweets_from_credbank_csv("C:\\Data\\credbank\\credbank_xai.csv")

    # sanity_check_loaded_credbank_tweets()

    #all_tweets = set(list(tweet_corpus))
    #for tweet_text in all_tweets:
    #    print(tweet_text)
    #print("all tweets: ", len(all_tweets))

    # export_credbank_trainset("C:\\Data\\credbank\\tweets_corpus")

    corpus_statistics()

    # generate_train_held_out_set("/fastdata/ac1jgx/credbank/train/credbank_train_corpus.txt")
    # generate_train_held_out_set("C:\\Data\\credbank\\tweets_corpus\\credbank_dataset_corpus_v2.txt")