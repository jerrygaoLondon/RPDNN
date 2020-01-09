import os

from allennlp_rumor_classifier import model_training
from data_loader import load_abs_path
# terminology:
# SC - source content
# cc - context content
# sc - social context

if __name__ == '__main__':

    import optparse

    parser = optparse.OptionParser()

    parser.add_option('-t', '--trainset',
                      dest="trainset",
                      help="train set path", default=None)

    parser.add_option('--heldout',
                      dest="heldout",
                      help="heldout dataset csv file", default=None)

    parser.add_option('-e', '--evaluationset',
                      dest="evalset",
                      help="evaluation/test dataset csv file", default=None)

    parser.add_option('-p', '--model_file_prefix',
                      dest="model_file_prefix",
                      help="model file prefix name for model weight output file", default=None)

    parser.add_option('-g', '--n_gpu',
                      dest="n_gpu",
                      help="gpu device(s) to use (-1: no gpu, 0: 1 gpu). only support int value for device no.",
                      default=-1)

    parser.add_option('-f', '--feature_setting',
                      dest="feature_setting",
                      help="expriment training setting for 1) source tweet content only (no context); "
                           " 2) context metadata (NF) only (no content); 3) context content only (no source content and no NF) ; "
                           "4) context (CM+CC) only (no source content); 5) Full_Model without CM; 6) Full_Model without CC;",
                      default=-1)

    parser.add_option('--epochs', dest="num_epochs", help="set num_epochs for training", default=2)

    parser.add_option("--social_encoder", dest="social_encoder_option", help="social context encoder option: 1) LSTM; "
                                                                             "2) transformer(stacked self attention)",
                      default=1)

    parser.add_option("--disable_context_type", dest="disable_context_type_option",
                      help="disable social context option: 0: accept all types of context; "
                           "1: disable reply; 2: disable retweet (default); ",
                      default=2)

    parser.add_option('-a', '--attention', dest="attention_option", help="select available attention options: "
                                                                         "0) no attention (use final state of LSTM); "
                                                                         "1) hierarchical attention (default); "
                                                                         "2) self_attention_net", default=1)

    parser.add_option("--max_cxt_size", dest="max_cxt_size_option", help="maximum social context size (default 200)",
                      default=200)

    options, args = parser.parse_args()

    train_set_path = options.trainset
    heldout_set_path = options.heldout
    evaluation_data_path = options.evalset
    model_file_prefix = options.model_file_prefix
    no_gpu = int(options.n_gpu)
    feature_setting_option = int(options.feature_setting)

    num_epochs = int(options.num_epochs)
    social_encoder_option = int(options.social_encoder_option)
    disable_context_type_option = int(options.disable_context_type_option)
    attention_option = int(options.attention_option)
    max_cxt_size_option = int(options.max_cxt_size_option)

    print("================= model settings ========================")
    print("trainset file path: ", train_set_path)
    print("heldout file path: ", heldout_set_path)
    print("evaluation set path: ", evaluation_data_path)
    print("model file prefix: ", model_file_prefix)
    print("gpu device: ", no_gpu)
    print("training (feature_setting option) setting (1: source content only; 2: context metadata only; "
          "3: context content only; 4: context (CC + CM) only); 5: Full_Model without CM; 6: Full_Model without CC; : ",
          feature_setting_option)

    print("num_epochs: ", num_epochs)
    print("social_encoder_option (1: LSTM; 2: transformer): ", social_encoder_option)
    print(
        "disable social context type (0: accept all types of context; 1: disable reply; 2: disable retweet (default)): ",
        disable_context_type_option)
    print("attention_option: ", attention_option)
    print("max_cxt_size_option: ", max_cxt_size_option)
    print("============================================================")

    if no_gpu != -1:
        # check GPU device usage and help to set suitable GPU device
        print("================================ check current GPU usage =============================")
        import subprocess

        gpu_usage_result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
        print(gpu_usage_result.stdout.decode('utf-8'))
        print("======================================================================================")

    # see https://pytorch.org/docs/stable/notes/cuda.html
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(no_gpu)

    if not os.path.isfile(train_set_path):
        raise FileNotFoundError("training dataset csv file [%s] not found!", train_set_path)

    if not os.path.isfile(heldout_set_path):
        raise FileNotFoundError("heldout dataset csv file [%s] not found!", heldout_set_path)

    if not os.path.isfile(evaluation_data_path):
        raise FileNotFoundError("test set csv file [%s] not found!", evaluation_data_path)

    if feature_setting_option not in [-1, 1, 2, 3, 4, 5, 6]:
        raise ValueError(
            "Supported training (feature_setting option) setting (1: source content only; 2: context metadata only; "
            "3: context content only; 4: context (CC + CM) only); 5: Full_Model without CM; 6: Full_Model without CC. "
            "However, we got [%s]" % feature_setting_option)

    if attention_option not in [0, 1]:
        raise ValueError("Supported attention setting: 0) no attention (use final state of LSTM); "
                         "1) AttentionWithContext (default). However, we got [%s]" % attention_option)

    print("training RumourDNN model on development dataset [%s] and [%s] with gpu [%s]" %
          (train_set_path, heldout_set_path, no_gpu))

    import allennlp_rumor_classifier
    import data_loader
    from allennlp_rumor_classifier import config_gpu_use

    config_gpu_use(no_gpu)

    allennlp_rumor_classifier.elmo_credbank_model_path = load_abs_path(os.path.join(os.path.dirname(__file__), '..',
                                                                                    "resource", "embedding",
                                                                                    "elmo_model",
                                                                                    "elmo_credbank_2x4096_512_2048cnn_2xhighway_weights_10052019.hdf5"))

    data_loader.social_context_data_dir = os.path.join(os.path.dirname(__file__), '..', "data", "social_context",
                                                       "aug-rnr-annotated-threads-retweets")

    print("Fine-tuned ELMo model is set to [%s]" % allennlp_rumor_classifier.elmo_credbank_model_path)
    print("social context corpus for all events directory is set to [%s]" % data_loader.social_context_data_dir)

    # Reasonable minibatch sizes are usually: 32, 64, 128, 256, 512, 1024 (powers of 2 are a common convention)
    # Usually, you can choose a batch size that is as large as your GPU memory allows
    #   (matrix-multiplication and the size of fully-connected layers are usually the bottleneck)
    # Practical tip: usually, it is a good idea to also make the batch size proportional to the number of classes in the dataset
    train_batch_size = 128
    import numpy as np

    # global mean/std including reply and retweets
    # numerical_feature_global_means = np.array([53687.59651189,102.48283517, 7174.80321466,1679.23193877,163.13357544,
    #                                           0.47222463, 18493.62032805, 1197.23593395,  0.02690504, 82.5726659,
    #                                           3.73283581, 38.35446687,  0.52815819, 11.72352158, 11.6877704, 0.0098383,
    #                                           263.10944539,  0.36006397,  0.05793871,  0.0010532,  0.87620043,
    #                                           0.30828535,  0.31187853,  0.16789344, 15.89250821, 1664.18758394,
    #                                           0.21168127, 0.85294076])
    # numerical_feature_global_std = np.array([127457.99303306, 1228.39836784, 233721.46246049, 8920.48699981,
    #                                         31374.22247502, 0.22822652, 49741.68072626, 686.55474156, 0.16180594,
    #                                         534.09541557, 32.67705202, 407.08631626, 0.49920649, 8.73215805, 5.58364386,
    #                                         0.09869907, 1040.1256501, 36.31889785, 0.23362751, 0.03243597, 0.32935276,
    #                                         0.46178512, 0.47120581, 0.37377163, 5.50114165, 42754.38234763,
    #                                         0.40850008, 0.3541649])

    # global mean/std excluding retweets
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

    print("model training in batches [size: %s]" % train_batch_size)
    model_training(train_set_path, heldout_set_path, evaluation_data_path, no_gpu, train_batch_size,
                   model_file_prefix, global_means=numerical_feature_global_means,
                   global_stds=numerical_feature_global_std, feature_setting=feature_setting_option,
                   num_epochs=num_epochs,
                   social_encoder_option=social_encoder_option, disable_cxt_type_option=disable_context_type_option,
                   attention_option=attention_option, max_cxt_size_option=max_cxt_size_option)
