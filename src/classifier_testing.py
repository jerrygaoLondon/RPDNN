import unittest
import os
import torch
from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer, SingleIdTokenIndexer

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from torch.nn import BatchNorm1d

from allennlp_rumor_classifier import RumorTweetsClassifer, RumorTweetsDataReader, \
    load_classifier_from_archive, timestamped_print
from data_loader import load_abs_path, load_source_tweet_context, load_source_tweet_json


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def tweet_context_encoding_by_tweet_id(self, rumor_classifier, tweet_id, disable_nf: bool = False):
        source_tweet_json = load_source_tweet_json(tweet_id)
        source_tweet_context_dataset = load_source_tweet_context(tweet_id)
        single_source_tweet_tensor = rumor_classifier._context_sequence_encoding(list(source_tweet_context_dataset),
                                                                                 source_tweet_json)
        return single_source_tweet_tensor

    def test_rumor_dataset_reader(self):
        elmo_token_indexer = ELMoTokenCharactersIndexer()
        # token_indexers={'elmo': elmo_token_indexer}
        rumor_train_set_reader = RumorTweetsDataReader(
            token_indexers={"tokens": SingleIdTokenIndexer(lowercase_tokens=True),
                            'elmo': elmo_token_indexer})
        train_dataset = rumor_train_set_reader.read(
            os.path.join(os.path.dirname(__file__), '..', "data", "train", "pheme_6392078_train_set_combined.csv"))
        vocab = Vocabulary.from_instances(train_dataset)
        vocab.print_statistics()

        # vocab_file_name = os.path.join(os.path.dirname(__file__),  '..', "output", "vocabulary")
        # vocab.save_to_files(vocab_file_name)

    def test_context_sequence_encoding(self):
        elmo_credbank_model_path = load_abs_path(
            os.path.join(os.path.dirname(__file__), '..', "resource", "embedding", "elmo_model",
                         "elmo_credbank_2x4096_512_2048cnn_2xhighway_weights_10052019.hdf5"))

        elmo_embedder = ElmoTokenEmbedder(
            options_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
            weight_file=elmo_credbank_model_path,
            do_layer_norm=False,
            dropout=0.5)
        word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})

        EXPECTED_CONTEXT_INPUT_SIZE = 60

        rumor_classifier = RumorTweetsClassifer(word_embeddings, None, None, None,
                                     classifier_feedforward=None,
                                     cxt_content_encoder=None,
                                     cxt_metadata_encoder=None,
                                     social_context_self_attention_encoder = None,
                                     cuda_device=-1)

        tweet_id = "500327120770301952"
        single_source_tweet_tensor_1 = self.tweet_context_encoding_by_tweet_id(rumor_classifier, tweet_id)
        print(type(single_source_tweet_tensor_1))
        print(single_source_tweet_tensor_1.shape)
        assert type(single_source_tweet_tensor_1) == torch.Tensor
        assert single_source_tweet_tensor_1.shape == (
            97, EXPECTED_CONTEXT_INPUT_SIZE), "expected shape is [19, %s]" % EXPECTED_CONTEXT_INPUT_SIZE

        tweet_id = "552806117328568321"  # with three replies
        single_source_tweet_tensor_2 = self.tweet_context_encoding_by_tweet_id(rumor_classifier, tweet_id)
        print(type(single_source_tweet_tensor_2))
        print(single_source_tweet_tensor_2.shape)
        assert type(single_source_tweet_tensor_2) == torch.Tensor
        assert single_source_tweet_tensor_2.shape == (
            94, EXPECTED_CONTEXT_INPUT_SIZE), "expected shape is [3, %s]" % EXPECTED_CONTEXT_INPUT_SIZE

        tweet_id = "552806117328568321"  # with three replies
        print("social context encoding without numerical feature .")
        single_source_tweet_tensor_2 = self.tweet_context_encoding_by_tweet_id(rumor_classifier, tweet_id, disable_nf=True)
        print(type(single_source_tweet_tensor_2))
        print(single_source_tweet_tensor_2.shape)
        assert type(single_source_tweet_tensor_2) == torch.Tensor
        assert single_source_tweet_tensor_2.shape == (
            94, EXPECTED_CONTEXT_INPUT_SIZE), "expected shape is [3, %s]" % EXPECTED_CONTEXT_INPUT_SIZE


    def test_padding_and_norm_propagation_tensors(self):
        dummy_classifier = RumorTweetsClassifer(None, None, None, None, None, None, None, None)
        EXPECTED_CONTEXT_INPUT_SIZE = 60
        expected_context_features_dim = EXPECTED_CONTEXT_INPUT_SIZE

        # invalid_tensor = torch.as_tensor([0], dtype=torch.float32)

        train_data_a = torch.randn(200, expected_context_features_dim)
        train_data_b = torch.randn(14, expected_context_features_dim)
        train_data_c = torch.randn(101, expected_context_features_dim)
        train_data_d = torch.randn(10, expected_context_features_dim)
        train_data_e = torch.randn(14, expected_context_features_dim)

        dummy_classifier.bn_input = BatchNorm1d(expected_context_features_dim, momentum=0.5, affine=False)

        print("type(train_data_a): ", type(train_data_a))
        propagation_representation = [train_data_a, train_data_b, train_data_c, train_data_d, train_data_e]
        normed_train_data = dummy_classifier.padding_and_norm_propagation_tensors(propagation_representation)

        print("shape of padded and normalised propagation dataset: ", normed_train_data.shape)
        assert normed_train_data.shape == (
            5, 200, EXPECTED_CONTEXT_INPUT_SIZE), "expected data shape is (5, 200, %s)" % EXPECTED_CONTEXT_INPUT_SIZE

        train_data_f = torch.randn(4, expected_context_features_dim)
        propagation_representation_f = [train_data_f]
        normed_train_data_f = dummy_classifier.padding_and_norm_propagation_tensors(propagation_representation_f)
        print("normed_train_data_f after padding for only 1 data item: ", normed_train_data_f.shape)

    def test_source_context_loader_4_varying_time_window(self):
        charliehebdo_non_rumour_tweet_id = "552784600502915072"

        from datetime import timedelta

        source_tweet_context_dataset = load_source_tweet_context(charliehebdo_non_rumour_tweet_id)
        assert 119 == len(list(source_tweet_context_dataset))

        charlie_event_end_timedelta = timedelta(minutes=5)
        source_tweet_context_dataset = load_source_tweet_context(charliehebdo_non_rumour_tweet_id,
                                                                 charlie_event_end_timedelta)
        cxt_data_size_after_5mins = len(list(source_tweet_context_dataset))
        print("context size for time window (%s): %s" % (str(charlie_event_end_timedelta), cxt_data_size_after_5mins))
        assert cxt_data_size_after_5mins == 1

        charlie_event_end_timedelta = timedelta(minutes=10)
        source_tweet_context_dataset = load_source_tweet_context(charliehebdo_non_rumour_tweet_id,
                                                                 charlie_event_end_timedelta)
        cxt_data_size_after_10mins = len(list(source_tweet_context_dataset))
        print("context size for time window (%s): %s" % (str(charlie_event_end_timedelta), cxt_data_size_after_10mins))
        assert cxt_data_size_after_10mins == 4

        charlie_event_end_timedelta = timedelta(minutes=15)
        source_tweet_context_dataset = load_source_tweet_context(charliehebdo_non_rumour_tweet_id,
                                                                 charlie_event_end_timedelta)
        cxt_data_size_after_15mins = len(list(source_tweet_context_dataset))
        print("context size for time window (%s): %s" % (str(charlie_event_end_timedelta), cxt_data_size_after_15mins))
        assert cxt_data_size_after_15mins == 6

        charlie_event_end_timedelta = timedelta(minutes=30)
        source_tweet_context_dataset = load_source_tweet_context(charliehebdo_non_rumour_tweet_id,
                                                                 charlie_event_end_timedelta)
        cxt_data_size_after_30mins = len(list(source_tweet_context_dataset))
        print("context size for time window (%s): %s" % (str(charlie_event_end_timedelta), cxt_data_size_after_30mins))
        assert cxt_data_size_after_30mins == 45

        charlie_event_end_timedelta = timedelta(minutes=45)
        source_tweet_context_dataset = load_source_tweet_context(charliehebdo_non_rumour_tweet_id,
                                                                 charlie_event_end_timedelta)
        cxt_data_size_after_45mins = len(list(source_tweet_context_dataset))
        print("context size for time window (%s): %s" % (str(charlie_event_end_timedelta), cxt_data_size_after_45mins))
        assert cxt_data_size_after_45mins == 58

        # charlie_end_date = charlie_start_time +
        charlie_event_end_timedelta = timedelta(hours=1)
        source_tweet_context_dataset = load_source_tweet_context(charliehebdo_non_rumour_tweet_id,
                                                                 charlie_event_end_timedelta)
        cxt_data_size_after_1hr = len(list(source_tweet_context_dataset))
        print("context size for time window (%s): %s" % (str(charlie_event_end_timedelta), cxt_data_size_after_1hr))
        assert cxt_data_size_after_1hr == 70

        charlie_event_end_timedelta = timedelta(minutes=90)
        source_tweet_context_dataset = load_source_tweet_context(charliehebdo_non_rumour_tweet_id,
                                                                 charlie_event_end_timedelta)
        cxt_data_size_after_90mins = len(list(source_tweet_context_dataset))
        print("context size for time window (%s): %s" % (str(charlie_event_end_timedelta), cxt_data_size_after_90mins))
        assert cxt_data_size_after_90mins == 75

        charlie_event_end_timedelta = timedelta(hours=2)
        source_tweet_context_dataset = load_source_tweet_context(charliehebdo_non_rumour_tweet_id,
                                                                 charlie_event_end_timedelta)
        cxt_data_size_after_2hr = len(list(source_tweet_context_dataset))
        print("context size for time window (%s): %s" % (str(charlie_event_end_timedelta), cxt_data_size_after_2hr))
        assert cxt_data_size_after_2hr == 83

        charlie_event_end_timedelta = timedelta(hours=3)
        source_tweet_context_dataset = load_source_tweet_context(charliehebdo_non_rumour_tweet_id,
                                                                 charlie_event_end_timedelta)
        cxt_data_size_after_3hr = len(list(source_tweet_context_dataset))
        print("context size for time window (%s): %s" % (str(charlie_event_end_timedelta), cxt_data_size_after_3hr))
        assert cxt_data_size_after_3hr == 92

        charlie_event_end_timedelta = timedelta(hours=6)
        source_tweet_context_dataset = load_source_tweet_context(charliehebdo_non_rumour_tweet_id,
                                                                 charlie_event_end_timedelta)
        cxt_data_size_after_6hr = len(list(source_tweet_context_dataset))
        print("context size for time window (%s): %s" % (str(charlie_event_end_timedelta), cxt_data_size_after_6hr))
        assert cxt_data_size_after_6hr == 103

        charlie_event_end_timedelta = timedelta(hours=12)
        source_tweet_context_dataset = load_source_tweet_context(charliehebdo_non_rumour_tweet_id,
                                                                 charlie_event_end_timedelta)
        cxt_data_size_after_12hr = len(list(source_tweet_context_dataset))
        print("context size for time window (%s): %s" % (str(charlie_event_end_timedelta), cxt_data_size_after_12hr))
        assert cxt_data_size_after_12hr == 105

        charlie_event_end_timedelta = timedelta(hours=18)
        source_tweet_context_dataset = load_source_tweet_context(charliehebdo_non_rumour_tweet_id,
                                                                 charlie_event_end_timedelta)
        cxt_data_size_after_12hr = len(list(source_tweet_context_dataset))
        print("context size for time window (%s): %s" % (str(charlie_event_end_timedelta), cxt_data_size_after_12hr))
        assert cxt_data_size_after_12hr == 115

        charlie_event_end_timedelta = timedelta(hours=24)
        source_tweet_context_dataset = load_source_tweet_context(charliehebdo_non_rumour_tweet_id,
                                                                 charlie_event_end_timedelta)
        cxt_data_size_after_24hr = len(list(source_tweet_context_dataset))
        print("context size for time window (%s): %s" % (str(charlie_event_end_timedelta), cxt_data_size_after_24hr))
        assert cxt_data_size_after_24hr == 117

        charlie_event_end_timedelta = timedelta(hours=48)
        source_tweet_context_dataset = load_source_tweet_context(charliehebdo_non_rumour_tweet_id,
                                                                 charlie_event_end_timedelta)
        cxt_data_size_after_48hr = len(list(source_tweet_context_dataset))
        print("context size for time window (%s): %s" % (str(charlie_event_end_timedelta), cxt_data_size_after_48hr))
        assert cxt_data_size_after_48hr == 118

        charlie_event_end_no_timedelta = timedelta(minutes=int(-1))
        source_tweet_context_dataset = load_source_tweet_context(charliehebdo_non_rumour_tweet_id,
                                                                 charlie_event_end_no_timedelta)
        cxt_data_size_no_timedelta = len(list(source_tweet_context_dataset))
        print("context size for none time window (%s): %s" % (
            str(charlie_event_end_no_timedelta), cxt_data_size_no_timedelta))
        assert 0 == cxt_data_size_no_timedelta

    def test_context_feature_encoder(self):
        elmo_credbank_model_path = load_abs_path(
            os.path.join(os.path.dirname(__file__), '..', "resource", "embedding", "elmo_model",
                         "elmo_credbank_2x4096_512_2048cnn_2xhighway_weights_10052019.hdf5"))

        # test context feature encoding with small sample data
        #    to make sure that source tweet context are sorted in  temporal order
        elmo_embedder = ElmoTokenEmbedder(
            options_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
            weight_file=elmo_credbank_model_path,
            do_layer_norm=False,
            dropout=0.5)
        word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})
        rumor_classifier = RumorTweetsClassifer(word_embeddings, None, None, None, None)

        propagation_embeddings_tensor = rumor_classifier.batch_compute_context_feature_encoding(
            ['500294803402137600', '500327120770301952'])
        print("propagation_embeddings_tensor: ", propagation_embeddings_tensor)

    def test_on_test_set(self):
        # model_weight_file = os.path.join(os.path.dirname(__file__),  '..', "output", "201905290138", "weights_best.th")
        # vocab_dir_path = os.path.join(os.path.dirname(__file__),  '..', "output", "201905290138", "vocabulary")
        model_weight_file = "C:\\Data\\rumourDNN_models\\output\\bostonbombings-201906241245\\weights_best.th"
        vocab_dir_path = "C:\\Data\\rumourDNN_models\\output\\bostonbombings-201906241245\\vocabulary"

        model, rumor_dnn_predictor = load_classifier_from_archive(vocab_dir_path=vocab_dir_path,
                                                                  model_weight_file=model_weight_file)

        evaluation_data_path = os.path.join(os.path.dirname(__file__), '..', "data", "test", "charliehebdo.csv")

        elmo_token_indexer = ELMoTokenCharactersIndexer()
        rumor_train_set_reader = RumorTweetsDataReader(token_indexers={'elmo': elmo_token_indexer})
        test_instances = rumor_train_set_reader.read(evaluation_data_path)

        from training_util import evaluate
        data_iterator = BucketIterator(batch_size=200, sorting_keys=[("sentence", "num_tokens")])
        data_iterator.index_with(model.vocab)
        metrics = evaluate(model, test_instances, data_iterator, -1, "")

        timestamped_print("Finished evaluating.")
        timestamped_print("Metrics:")
        for key, metric in metrics.items():
            timestamped_print("%s: %s" % (key, metric))

    def test_classifier(self):
        model, rumor_dnn_predictor = load_classifier_from_archive()
        result = rumor_dnn_predictor.predict("553588494661337089",
                                             "The #CharlieHebdo massacre brothers have been killed. Full story here: http://t.co/s7KfpU7vqb http://t.co/jFCrKaZ3DL,1")

        # rumor_dnn_predictor.predict_batch_instance(test_instances)
        # sample result: {'logits': [-0.42800506949424744, 0.5114126205444336], 'class_probabilities': [0.2810179591178894, 0.7189820408821106], 'label': '@@UNKNOWN@@'}
        print(result)

if __name__ == '__main__':
    unittest.main()
