"""
This file contains experiment of various options for sentence embedding
dependencies:
    allennlp
    pip install allennlp/requirements

    CUDA 8+
"""
from typing import List
import string
import numpy as np
import scipy

from nltk.corpus import stopwords

#: Convenience functions
_stop_words = stopwords.words('english')
stop_words_filter = lambda t : filter(lambda a: a not in _stop_words, t)
punctuation_filter = lambda t : filter(lambda a: a not in string.punctuation, t)


from allennlp.commands.elmo import ElmoEmbedder
# use default "original" model

default_elmo = None


def sentence_embedding_elmo(sentence: List[str], elmo_model: ElmoEmbedder,
                            remove_stopwords=False, avg_all_layers=False) -> np.ndarray:
    """
    Parameters
    ----------
    sentence : ``List[str]``, required
    A tokenized sentence to represent context for target word.

    Different from the word embedding model that simply provides a output of mapping/dictionary between a fixed vocabulary and vector representation,
    ELMos needs the input of a given sentence (as context) at test time so as to compute contextual
        representations of words using the biLM based on N nearest neighbor approach.

    Another major difference is ELMo use context insensitive Char CNN filters (i.e., map tokens from input layer into chars) in biLM.
        This character based model avoid the OOV problem that traditional embedding model suffers.

    Taking the average of the word embeddings in a sentence tends to give too much weight to words that are quite irrelevant, semantically speaking.
     Smooth Inverse Frequency seems a way to solve this problem, but it needs a reference corpus.

    a fixed mean-pooled vector representation of the input

    see also, http://nlp.town/blog/sentence-similarity/

    see 'options_file' for the biLM model setting

    see: https://allennlp.org/elmo
    see original paper: https://arxiv.org/pdf/1802.05365.pdf
    see also: https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md
    see also: a paper use ELMo for tweet representation, https://arxiv.org/pdf/1808.08672.pdf
    Returns
    -------
    avg. of sentence of words-in-context embeddings
    """
    # ElmoEmbedder class returns three word-in-context vectors for each word (a set of 2L + 1 representation), each vector corresponding to a layer in the ELMo biLSTM output.
    # 'sentence_vectors' is a tensor containing the ELMo vectors.
    if remove_stopwords:
        sentence = list(stop_words_filter(sentence))
        # print("sentence filtered by stopwords: ", sentence)

    # ELMo will compute representation of words from context given a sentence based on a N nearest neighbor approach
    # "use the biLM to compute representations for a given target word
    #   and take the nearest neighbor sense from the training set,
    #   falling back to the first sense from WordNet for lemmas not observed during training"

    sentence_vectors = elmo_model.embed_sentence(sentence)

    # get the third/top layer's output for the sentence representation (i.e.,contextual representation)
    # In the simplest case, ELMo just selects the top layer
    if not avg_all_layers:
        sentence_word_embeddings = sentence_vectors[2][:]
    else:
        # Weighted sum of the 3 layers to collapses all layers and compute average all token embedding to a single vector
        # There is also a common option to concatentate all layers which convert the final dim to 3072
        # weighted average of all biLM layers
        #  averaging all layers improves development accuracy for SNLI
        # sum_all_layer_sent_embedding = np.sum(sentence_vectors, axis=0, dtype='float32')
        avg_all_layer_sent_embedding = np.mean(sentence_vectors, axis=0, dtype='float32')
        return np.mean(avg_all_layer_sent_embedding, axis=0, dtype='float32')

    # print("dim: ", sentence_word_embeddings.shape)
    return np.mean(sentence_word_embeddings, axis=0).astype('float32')


def word_embedding_elmo(sentence: List[str], elmo_model: ElmoEmbedder, remove_stopwords=False, avg_all_layers=True) -> np.ndarray:
    """
    different from sentence_embedding_elmo, this method returns all word context embedding (with avg of all layers states)

    ELMo will compute representation of words from context given a sentence based on a N nearest neighbor approach
    "use the biLM to compute representations for a given target word and take the nearest neighbor sense from the
    training set, falling back to the first sense from WordNet for lemmas not observed during training"

    :param sentence:
    :param elmo_model:
    :param remove_stopwords:
    :param avg_all_layers:
    :return: (seq_size, feature_dim)
    """
    if remove_stopwords:
        sentence = list(stop_words_filter(sentence))
        # print("sentence filtered by stopwords: ", sentence)

    sentence_vectors = elmo_model.embed_sentence(sentence)

    if not avg_all_layers:
        # get the third/top layer's output for the sentence representation (i.e.,contextual representation)
        # In the simplest case, ELMo just selects the top layer
        sentence_word_embeddings = sentence_vectors[2][:]
    else:
        #  averaging all 3 layers improves development accuracy for SNLI
        avg_all_layer_sent_embedding = np.mean(sentence_vectors, axis=0, dtype='float32')
        return avg_all_layer_sent_embedding

    return sentence_word_embeddings

def export_sentence_embedding():
    import os
    pheme_data_output_path = os.path.join(os.path.dirname(__file__),  '..', '..',  "output", "elmo", "pheme_source_tweet_corpus.txt")

    pheme_data_embedding_output = os.path.join(os.path.dirname(__file__),  '..', '..',  "output", "elmo", "pheme_source_tweet_corpus_elmo_embedding.txt")

    # the method 'embed_file' does not tolerant any empty line and will raise error if encounter any
    with open(pheme_data_output_path, mode='r', encoding='utf-8') as corpus_input:
        default_elmo.embed_file(input_file=corpus_input, output_file_path=pheme_data_embedding_output,output_format="average")


def compare_paraphase_sentence_pair(sent1:str, sent2: str, use_all_layers=False) -> float:
    global default_elmo
    if default_elmo is None:
        default_elmo = ElmoEmbedder()

    sent1_vector = sentence_embedding_elmo(sent1.split(" "), default_elmo, remove_stopwords=False)
    sent2_vector = sentence_embedding_elmo(sent2.split(" "), default_elmo, remove_stopwords=False)
    sim_score_sent = scipy.spatial.distance.cosine(sent1_vector,sent2_vector)
    return sim_score_sent


def test_sentence_embedding_almo():
    global default_elmo
    if default_elmo is None:
        default_elmo = ElmoEmbedder()

    sentence1= ["I", "ate", "an", "apple", "for", "breakfast"]
    sentence1_vector = sentence_embedding_elmo(sentence1, default_elmo)
    sentence2 = ["I", "ate", "a", "carrot", "for", "breakfast"]
    sentence2_vector = sentence_embedding_elmo(sentence2, default_elmo)
    sim_score_sent = scipy.spatial.distance.cosine(sentence1_vector,sentence2_vector)
    print("similarity btw two sentences: ", sim_score_sent) # 0.052996

    sentence1_vector = sentence_embedding_elmo(sentence1, default_elmo, remove_stopwords=True)
    sentence2_vector = sentence_embedding_elmo(sentence2, default_elmo, remove_stopwords=True)
    sim_score_sent = scipy.spatial.distance.cosine(sentence1_vector,sentence2_vector)
    print("similarity btw two sentences filtered by stop words: ", sim_score_sent) # 0.04196

    sentence3= ["9/11", "sandy", "hook", "movie", "shooting", "boston", "bomb", "threats", "from", "n.", "korea", "and", "several", "other", "tragedies", "were", "all", "under", "the", "age", "of", "18"]
    sentence3_vector = sentence_embedding_elmo(sentence3, default_elmo, remove_stopwords=False)
    sentence4 = ["You", "go", "to", "watch", "a", "movie", "You", "get", "shot", "You", "run", "a", "marathon", "You", "get", "bombed", "I", "hate", "this", "world", "#prayforbostonpic.twitter.com/6EBlONvJyC"]
    sentence4_vector = sentence_embedding_elmo(sentence4, default_elmo, remove_stopwords=False)
    sim_score_sent = scipy.spatial.distance.cosine(sentence3_vector,sentence4_vector)
    print("similarity btw two bostonbombing tweets: ", sim_score_sent) # 0.4264

    sentence5=["This", "man","is","good"]
    sentence6=["This", "man","is","bad"]
    sentence7=["This", "man","is","nice"]
    sentence5_vector = sentence_embedding_elmo(sentence5, default_elmo, remove_stopwords=False)
    sentence6_vector = sentence_embedding_elmo(sentence6, default_elmo, remove_stopwords=False)
    sentence7_vector = sentence_embedding_elmo(sentence7, default_elmo, remove_stopwords=False)
    sim_score_sent56 = scipy.spatial.distance.cosine(sentence5_vector,sentence6_vector)
    sim_score_sent57 = scipy.spatial.distance.cosine(sentence5_vector,sentence7_vector)
    print("similarity btw two senti-opposite sentence: ", sim_score_sent56) # 0.02159
    print("similarity btw two senti-similar sentence: ", sim_score_sent57) # 0.02884


    doc1 = 'break report several people injured explosion finish line'
    # doc2 = nlp('break authority investigate repo two explosion finish line')
    doc2 = 'break report several people injured explosion finish line'
    import time
    start_time = time.time()
    print("similarity btw two Sooji's doc tweets: ", compare_paraphase_sentence_pair(doc1, doc2)) # 0.24
    print("ELMo model took", time.time() - start_time, " seconds to run per pair")

    #move example sentence pairs from https://arxiv.org/pdf/1712.02820.pdf

    sentence_8 = " Ricky Clemons’brief, troubled Missouri basketball career is over"
    sentence_9 = "Missouri kicked Ricky Clemons off its team, ending his troubled career there."
    print("similarity btw true paraphrase pair 8-9 ", compare_paraphase_sentence_pair(sentence_8, sentence_9)) # 0.3958

    sentence_10 = " The tech-heavy Nasdaq composite index shot up 5.7 percent for the week."
    sentence_11 = "The Nasdaq composite index advanced 20.59, or 1.3 percent, to 1,616.50, after gaining 5.7 percent last week."
    print("similarity btw non-paraphrase pair 10-11 ", compare_paraphase_sentence_pair(sentence_10, sentence_11))# 0.1803

    sentence_12 = "But 13 people have been killed since 1900 and hundreds injured"
    sentence_13 = "Runners are often injured by bulls and 13 have been killed since 1900."
    print("similarity btw non-paraphrase pair 12-13 ", compare_paraphase_sentence_pair(sentence_12, sentence_13)) #  0.30

    sentence_14 = "I would rather be talking about positive numbers than negative"
    sentence_15 = "But I would rather be talking about high standards rather than low standards."
    print("similarity btw paraphrase pair 14-15 ", compare_paraphase_sentence_pair(sentence_14, sentence_15)) # 0.216

    sentence_16 = "The respected medical journal Lancet has called for a complete ban on tobacco in the United Kingdom."
    sentence_17 = "A leading U.K. medical journal called Friday for a complete ban on tobacco prompting outrage from smokers groups."
    print("similarity btw non-paraphrase pair 16-17 ", compare_paraphase_sentence_pair(sentence_16, sentence_17)) # 0.2027

    sentence_18 = "Mrs. Clinton said she was incredulous that he would endanger their marriage and family."
    sentence_19 = "She hadn’t believed he would jeopardize their marriage and family."
    print("similarity btw paraphrase pair 18-19 ", compare_paraphase_sentence_pair(sentence_18, sentence_19)) # 0.1888

    sentence_20 = "Terrible things happening in Turkey"
    sentence_21 = "Children are dying in Turkey"
    print("similarity btw paraphrase pair 20-21 ", compare_paraphase_sentence_pair(sentence_20, sentence_21)) # 0.411

    sentence_22 = "Anyone trying to see After Earth sometime soon"
    sentence_23 = "Me and my son went to see After Earth last night"
    print("similarity btw non-paraphrase pair 22-23 ", compare_paraphase_sentence_pair(sentence_22, sentence_23)) # 0.2756

    sentence_24 = "hahaha that sounds like me"
    sentence_25 = "That sounds totally reasonable to me"
    print("similarity btw non-paraphrase pair 24-25 ", compare_paraphase_sentence_pair(sentence_24, sentence_25)) # 0.255

    sentence_26 = "I dont understand the hatred for Rafa Benitez"
    sentence_27 = "Top 4 and a trophy and still they dont give any respect for Benitez"
    print("similarity btw paraphrase pair 26-27 ", compare_paraphase_sentence_pair(sentence_26, sentence_27)) # 0.3256

    sentence_28 = "Shonda is a freaking genius"
    sentence_29 = "Dang Shonda knows she can write"
    print("similarity btw paraphrase pair 28-29 ", compare_paraphase_sentence_pair(sentence_28, sentence_29)) # 0.373

    sentence_30 = "Terrible things happening in Turkey"
    sentence_31 = "Be with us to stop the violence in Turkey"
    print("similarity btw paraphrase pair 30-31 ", compare_paraphase_sentence_pair(sentence_30, sentence_31)) # 0.386

    sentence_32 = "I must confess I love Star Wars"
    sentence_33 = "Somebody watch Star Wars with me please"
    print("similarity btw paraphrase pair 32-33 ", compare_paraphase_sentence_pair(sentence_32, sentence_33)) # 0.304

    sentence_34 = "Family guy is really a reality show"
    sentence_35 = "Family guy is such a funny show"
    print("similarity btw non-paraphrase pair 34-35 ", compare_paraphase_sentence_pair(sentence_34, sentence_35))

    sentence_36 = "I see everybody watching family guy tonight"
    sentence_37 = "I havent watched Family Guy in forever"
    print("similarity btw non-paraphrase pair 36-37 ", compare_paraphase_sentence_pair(sentence_36, sentence_37))


    sentence_36 = "I see everybody watching family guy tonight"
    sentence_37 = "I havent watched Family Guy in forever"
    print("similarity btw non-paraphrase pair 36-37 ", compare_paraphase_sentence_pair(sentence_36, sentence_37))


def test_sentence_embedding_almo_all_layers():
    global default_elmo
    if default_elmo is None:
        default_elmo = ElmoEmbedder()
    sentence1= ["I", "ate", "an", "apple", "for", "breakfast"]
    sentence1_vector = sentence_embedding_elmo(sentence1, default_elmo, avg_all_layers=True)
    sentence2 = ["I", "ate", "a", "carrot", "for", "breakfast"]
    sentence2_vector = sentence_embedding_elmo(sentence2, default_elmo, avg_all_layers=True)
    sim_score_sent = scipy.spatial.distance.cosine(sentence1_vector,sentence2_vector)
    print("similarity btw two sentences with all layers: ", sim_score_sent) # 0.05434418663135798

    sentence1_vector = sentence_embedding_elmo(sentence1, default_elmo, avg_all_layers=False)
    sentence2_vector = sentence_embedding_elmo(sentence2, default_elmo, avg_all_layers=False)
    sim_score_sent = scipy.spatial.distance.cosine(sentence1_vector,sentence2_vector)
    print("similarity btw two sentences with top layer: ", sim_score_sent) # 0.045306307620042396

    sentence5=["This", "man","is","good"]
    sentence6=["This", "man","is","bad"]
    sentence7=["This", "man","is","nice"]
    sentence5_vector = sentence_embedding_elmo(sentence5, default_elmo, remove_stopwords=False)
    sentence6_vector = sentence_embedding_elmo(sentence6, default_elmo, remove_stopwords=False)
    sentence7_vector = sentence_embedding_elmo(sentence7, default_elmo, remove_stopwords=False)
    sim_score_sent56 = scipy.spatial.distance.cosine(sentence5_vector,sentence6_vector)
    sim_score_sent57 = scipy.spatial.distance.cosine(sentence5_vector,sentence7_vector)
    print("similarity btw two senti-opposite sentence with top layer: ", sim_score_sent56) #  0.0228955683625649
    print("similarity btw two senti-similar sentence with top layer: ", sim_score_sent57) # 0.0291542572202228

    sentence5_vector = sentence_embedding_elmo(sentence5, default_elmo, remove_stopwords=False, avg_all_layers=False)
    sentence6_vector = sentence_embedding_elmo(sentence6, default_elmo, remove_stopwords=False, avg_all_layers=False)
    sentence7_vector = sentence_embedding_elmo(sentence7, default_elmo, remove_stopwords=False, avg_all_layers=False)
    sim_score_sent56 = scipy.spatial.distance.cosine(sentence5_vector,sentence6_vector)
    sim_score_sent57 = scipy.spatial.distance.cosine(sentence5_vector,sentence7_vector)
    print("similarity btw two senti-opposite sentence with all layers: ", sim_score_sent56) #  0.020847705940927552
    print("similarity btw two senti-similar sentence with all layers: ", sim_score_sent57) # 0.027940926063383942



if __name__ == '__main__':
    # test_sentence_embedding_almo()
    test_sentence_embedding_almo_all_layers()