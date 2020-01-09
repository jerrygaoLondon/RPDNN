import os
from datetime import timedelta

from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer

from allennlp_rumor_classifier import load_classifier_from_archive, RumorTweetsDataReader
from allennlp_rumor_classifier import timestamped_print

if __name__ == '__main__':
    import optparse

    parser = optparse.OptionParser()

    parser.add_option('-m', '--model',
                      dest="model",
                      help="model directory to be evaluated", default=None)

    parser.add_option('-t', '--testset',
                      dest="testset",
                      help="test set csv file path", default=None)

    parser.add_option('-g', '--n_gpu',
                      dest="n_gpu",
                      help="gpu device(s) to use (-1: no gpu, 0: 1 gpu)", default=-1)

    parser.add_option('-f', '--feature_setting',
                      dest="feature_setting",
                      help="expriment training setting for 1) source tweet content only (no context); "
                           " 2) context metadata (NF) only (no content); 3) context content only (no source content and no NF) ; "
                           "4) context (CM+CC) only (no source content); 5) Full_Model without CM; 6) Full_Model without CC;",
                      default=-1)

    parser.add_option('-w', '--temporal_window',
                      dest="event_temporal_window_mins",
                      help="varying time window in minutes, accepting str for a list of number in minutes "
                           "(e.g.,'15, 30, 45, 60, 90, 120, 180, 240, 360, 480, 600, 720, 840, "
                           "960,1080,1200,1440,1680,1920,2160,2400,2640,2880') .", default=None)

    # retweet is disabled
    # Our preliminary results shows that retweets metadata are very noisy.
    #     Simply adding retweets into context cause underfitting and poor performance.
    parser.add_option("--disable_context_type", dest="disable_context_type_option",
                      help="disable social context option: 0: accept all types of context; "
                           "1: disable reply; 2: disable retweet (default); ",
                      default=2)

    # We only experimented the first two options 0) and 1).
    parser.add_option('-a', '--attention', dest="attention_option", help="select available attention options: "
                                                                         "0) no attention (use final state of LSTM); "
                                                                         "1) hierarchical attention (default); "
                                                                         "2) self_attention_net", default=1)

    parser.add_option("--max_cxt_size", dest="max_cxt_size_option", help="maximum social context size (default 200)",
                      default=200)

    options, args = parser.parse_args()

    rumour_dnn_model_dir = options.model
    test_set_csv_path = options.testset
    no_gpu = int(options.n_gpu)
    feature_setting_option = int(options.feature_setting)
    event_temporal_filter_hr = options.event_temporal_window_mins
    disable_context_type_option = int(options.disable_context_type_option)
    attention_option = int(options.attention_option)
    max_cxt_size_option = int(options.max_cxt_size_option)

    print("================= model settings for prediction ========================")
    print("rumour_dnn_model_dir: ", rumour_dnn_model_dir)
    print("test_set_csv_path: ", test_set_csv_path)
    print("no_gpu: ", no_gpu)
    print("training (feature_setting option) setting (1: source content only; 2: context metadata only; "
          "3: context content only; 4: context (CC + CM) only); 5: Full_Model without CM; 6: Full_Model without CC; : ",
          feature_setting_option)
    print("event_temporal_filter_hr: ", event_temporal_filter_hr)
    print("disable social context type (0: accept all types of context; 1: disable reply; "
          "2: disable retweet (default)): ", disable_context_type_option)
    print("attention_option ( 0) no attention (use final state of LSTM); 1) hierarchical attention (default); ) ",
          attention_option)
    print("max_cxt_size: ", max_cxt_size_option)
    print("============================================================")

    if feature_setting_option not in [-1, 1, 2, 3, 4, 5, 6]:
        raise ValueError(
            "Supported training (feature_setting option) setting (1: source content only; 2: context metadata only; "
            "3: context content only; 4: context (CC + CM) only); 5: Full_Model without CM; 6: Full_Model without CC. "
            "However, we got [%s]" % feature_setting_option)

    event_varying_time_window_mins = []
    if event_temporal_filter_hr:
        event_varying_time_window_mins = event_temporal_filter_hr.split(",")
        print("event varying time window in hours: ", event_varying_time_window_mins)

    event_varying_timedeltas = []

    if len(event_varying_time_window_mins) > 0:
        for event_varying_time_window_mins_i in event_varying_time_window_mins:
            event_varying_timedeltas.append(timedelta(minutes=int(event_varying_time_window_mins_i)))

        print("Experimental context varying window: ")
        print(event_varying_timedeltas)

    print("evaluating model [%s] on test set [%s]" % (rumour_dnn_model_dir, test_set_csv_path))

    from allennlp_rumor_classifier import config_gpu_use

    config_gpu_use(no_gpu)
    model_weight_file = os.path.join(rumour_dnn_model_dir, "weights_best.th")
    vocab_dir = os.path.join(rumour_dnn_model_dir, "vocabulary")

    import numpy as np

    numerical_feature_global_means = np.array(
        [33482.2189387, 113.50235784, 16110.72212533, 1480.63278539, 1984.97753317,
         0.44584125, 13076.6074171, 1144.7288986, 0.01814169, 67.85327272, 4.6219718,
         40.2049446, 0.46842004, 11.86826366, 11.45688415, 0.02726267, 2.30499867,
         1.51966541, 0.14687646, 0.0030709, 0.88728322, 0.10091743, 0.10361821,
         0.07962697, 12.33783996, 2485.25345758, 1., 0.82859045])

    numerical_feature_global_std = np.array([96641.67591616, 2193.38510189, 577886.49773141, 7192.66662092,
                                             134946.70443628, 0.22736203, 37326.88297683, 741.21114409, 0.13346375,
                                             632.37911747, 47.66767052, 612.61064067, 0.49900171, 9.09843269,
                                             5.25300108,
                                             0.16284784, 127.00542757, 58.64079579, 0.35398272, 0.05533058, 0.31624628,
                                             0.30121936, 0.3149778, 0.27071482, 6.86573998, 45822.31909476, 0.,
                                             0.37686645])

    model, rumor_dnn_predictor = load_classifier_from_archive(vocab_dir_path=vocab_dir,
                                                              model_weight_file=model_weight_file,
                                                              n_gpu_use=no_gpu,
                                                              max_cxt_size=max_cxt_size_option,
                                                              feature_setting=feature_setting_option,
                                                              global_means=numerical_feature_global_means,
                                                              global_stds=numerical_feature_global_std,
                                                              attention_option=attention_option)
    elmo_token_indexer = ELMoTokenCharactersIndexer()
    rumor_train_set_reader = RumorTweetsDataReader(token_indexers={'elmo': elmo_token_indexer})
    test_instances = rumor_train_set_reader.read(test_set_csv_path)

    from training_util import evaluate

    data_iterator = BucketIterator(batch_size=128, sorting_keys=[("sentence", "num_tokens")])
    data_iterator.index_with(model.vocab)

    # =============== apply settings to trained model ===========
    model.feature_setting = feature_setting_option
    model.set_disable_cxt_type_option(disable_context_type_option)

    # cannot reset attention mechanism and cannot reset maximum context size in evaluation
    # model.set_attention_mechanism(attention_option)
    # model.set_max_cxt_size(max_cxt_size_option)
    print("maximum cxt size: ", model.max_cxt_size)

    print("model architecture: ")
    print(model)
    print("==============================")

    all_metrics = []
    if len(event_varying_timedeltas) == 0:
        metrics = evaluate(model, test_instances, data_iterator, no_gpu, "")
        timestamped_print("Evaluation results :")
        for key, metric in metrics.items():
            print("%s: %s" % (key, metric))
    else:
        for event_varying_timedelta in event_varying_timedeltas:
            timestamped_print("evaluate model with time window [%s]" % (str(event_varying_timedelta)))
            model.event_varying_timedelta = event_varying_timedelta

            metrics = evaluate(model, test_instances, data_iterator, no_gpu, "")
            timestamped_print("Evaluation results at [%s] :" % str(event_varying_timedelta))
            for key, metric in metrics.items():
                print("%s: %s" % (key, metric))

            all_metrics.append(metrics)

    print(all_metrics)

    print("completed")
