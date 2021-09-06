import torch
from allennlp.commands.elmo import ElmoEmbedder
from torch import nn

import torch.nn.functional as F
from torch.autograd import Variable

"""
attention mechanism can provides a set of summation weight vectors for the LSTM hidden states.

There are many formulations for attention but they share a common goal: predict a probability distribution called 
attention weights over the sequence elements. 

It is proved to prove better representation than average of all sequence representations/vectors.

Hard or Soft attention ?
    1) hard attention is conceptually simple that requires indexing of dataset features and we have to rely on 
    a score-function estimiate to provide weights. It is not differentiable.
    2) soft attention provides benefit of differentiability and be able to compute weights from gradient decent. However,
    it is added extra computational cost and can be over-parameterised.

"""


class HierarchicalAttentionNet(nn.Module):
    """
    Attention operation, with a context sequence for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention

    Taken from https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/

    This is Standard Attention Mechanism

    see also the facebook paper, access via http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2916.pdf
    see also Andrew NG's video, https://www.youtube.com/watch?v=quoGRI-1l0A

    # Input shape
        3D tensor with shape: `(batch size, steps / sequence size, features)`.
        Note: When dealing with variable length sequences, we require each sequence in same batch (or whole dataset)
        to be equal in length. So, we need tp padding shorter context sequences to the same length as the longest one
        in the batch before applying to this attention layer
    # Output (context_weighted_sum, weighted_input, weight)

    context_weighted_sum shape
        2D tensor with shape: `(batch size, features)`.
        The output can be then fed into FC and softmax layer for classification

    How to use:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN)
        The dimensions are inferred based on the output shape of the RNN.
    """
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(HierarchicalAttentionNet, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        # self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        """

        :param x: (batch_size, max steps, hidden_size)
        :param mask: (batch size, max steps)
            create a mask with true values in place of real values (i.e, 1) and zeros in pad values and use that mask before softmax
            so, no need to update weights for padded values

            Context of the maximum length will use all the attention weights, while shorter context will only use the first few.

            masking out setting is to negative infinity float('-inf') in [Vaswani 2017]

        :return: (context_weighted_sum, weighted_input, weight)
        """
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        #print("feature_dim: ", feature_dim)
        #print("self.weight shape: ", self.weight.shape)
        #print("self.weight: ", self.weight)
        #print("self.bias: ", self.bias)
        #print("batch input x shape: ", x.shape)
        # print("batch input x: ", x)
        #print("step_dim: ", step_dim)

        #matrix multiplication
        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        print("eij shape: ", eij.shape)
        #print("eij: ", eij)

        if self.bias:
            eij = eij + self.b

        # activation
        eij = torch.tanh(eij)

        print("weights (eij) shape before masking and softmax: ", eij.shape)
        if mask is not None:
            print("mask shape: ", mask.shape)
            print("mask: ", mask)

        if mask is not None:
            #a = a * mask
            # eij[~mask] = float("-inf") # this is inplace operation of mask filling, -inf can also be -1e32
            # use out-of-place version to fill mask value (inplace operation is not allowed for gradient computation)
            # eij = eij.masked_fill((1 - mask).byte(), float("-inf"))
            eij = eij.masked_fill(~mask, float("-inf"))
            # print("weights after masking: ", eij)

        # use softmax to predict a probability distribution (called attention weights) over the sequence elements
        #a = torch.exp(eij)
        #a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)
        a = torch.nn.functional.softmax(eij, dim=1)


        print("weights : ", a )
        print("attention weights (after softmax) shape: ", str(a.shape))
        print("two top attention weights from current batch: ")
        print("1st context: ", a[0])
        print("2nd context: ", a[1])

        weighted_input = x * torch.unsqueeze(a, -1)
        context_weighted_sum = torch.sum(weighted_input, 1)

        #print("context_weighted_sum: ", context_weighted_sum)
        # print("weighted_input shape: ", weighted_input.shape)
        return context_weighted_sum, weighted_input, a


class StructuredSelfAttention(nn.Module):
    """
    Note: experimental model and does not seems working as expected.

    LSTM + self attention

    Taken from https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/selfAttention.py

    https://github.com/kaushalshetty/Structured-Self-Attention

    Z. Lin, M. Feng, M. Yu, B. Xiang, B. Zhou, and Y. Bengio, "A Structured Self-attentive Sentence Embedding" pp. 1–15, 2017.

    Now we will use self attention mechanism to produce a matrix embedding of the input sentence in which every row represents an
    encoding of the inout sentence but giving an attention to a specific part of the sentence. We will use 30 such embedding of
    the input sentence and then finally we will concatenate all the 30 sentence embedding vectors and connect it to a fully
    connected layer of size 2000 which will be connected to the output layer of size 2 returning logits for our two classes i.e.,
    pos & neg.

     See also Andrew Ng's explain, https://www.youtube.com/watch?v=FMXUkEbjf9k

    Arguments
    ---------
    lstm_output = A tensor containing hidden states corresponding to each time step of the LSTM network.
            (batch_size, num_seq, 2*hidden_size)
    ---------
    Returns :
       1st param: average context embedding
       2nd param: Final Attention weight matrix for all the 30 different sentence embedding in which each of 30 embeddings give
              attention to different parts of the input sentence.
    Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)
                  attn_weight_matrix.size() = (batch_size, 30, num_seq)
    """
    # We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper
    # attention_unit = 350
    #hidden_size = 28
    # attention_hops = 56

    def __init__(self, input_dim, attention_unit = 350, attention_hops=30, **kwargs):
        """

        :param input_dim: LSTM hidden dim
        :param attention_unit: "d_a"
        :param attention_hops: "r"
        :param kwargs:
        """
        super(StructuredSelfAttention, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.attention_unit = attention_unit
        self.attention_hops = attention_hops
        self.W_s1 = torch.nn.Linear(input_dim, self.attention_unit)
        self.W_s1.bias.data.fill_(0)
        self.W_s2 = torch.nn.Linear(self.attention_unit, self.attention_hops)
        self.W_s2.bias.data.fill_(0)

        self.feature_dim = self.input_dim

    def forward(self, lstm_output: torch.FloatTensor, mask=None, if_concat: bool = False):
        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)

        print("attn_weight_matrix shape before softmax: ", attn_weight_matrix.shape)

        if mask is not None:
            raise ArithmeticError("Not supported yet!")
            #attn_weight_matrix = attn_weight_matrix.masked_fill((1 - mask).byte(), float("-inf"))

        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        context_embeddings = torch.bmm(attn_weight_matrix, lstm_output)
        if if_concat:
            # if we choose to concatenate multiple context representation (i.e., "r" or called attention hops)
            # the context embedding size will be attention_hops * lstm_hidden_dim
            concatenated_context_embeddings = context_embeddings.view(-1, context_embeddings.size()[1]*context_embeddings.size()[2])
            context_embeddings = concatenated_context_embeddings
        else:
            # we average all (i.e., attention_hops) the context representations
            avg_context_embeddings = torch.sum(context_embeddings,1)/self.attention_hops
            print("avg_context_embeddings shape: ", avg_context_embeddings.shape)
            context_embeddings = avg_context_embeddings
        # next: we sum up the LSTM hidden states H according to the weight provided by attn_weight_matrix to get a vector representation m of the input sentence
        # hidden_matrix = torch.bmm(attn_weight_matrix, lstm_output)
        # alternatively, matrix multiplication
        # context_embeddings = attn_weight_matrix@lstm_output
        # print("hidden matrix shape: ", hidden_matrix.shape)
        # This vector representation usually focuses on a specific component of the sentence, like a special set of related words or phrases.
        return context_embeddings, attn_weight_matrix


def test_elmo_with_attention():
    import os
    from data_loader import load_abs_path
    from embedding_layer import word_embedding_elmo

    elmo_credbank_model_path = load_abs_path(
        os.path.join(os.path.dirname(__file__), '..', "resource", "embedding", "elmo_model",
                     "elmo_credbank_2x4096_512_2048cnn_2xhighway_weights_10052019.hdf5"))
    elmo_options_file_path = load_abs_path(
        os.path.join(os.path.dirname(__file__), '..', "resource", "embedding", "elmo_model",
                     "elmo_2x4096_512_2048cnn_2xhighway_options.json"))

    sentence3= ["9/11", "sandy", "hook", "movie", "shooting", "boston", "bomb", "threats", "from", "n.",
                "korea", "and", "several", "other", "tragedies", "were", "all", "under", "the", "age", "of", "18"]

    fine_tuned_elmo = ElmoEmbedder(
        options_file=elmo_options_file_path,
        weight_file=elmo_credbank_model_path)

    avg_all_layer_sent_embedding = word_embedding_elmo(sentence3, fine_tuned_elmo)

    # print(avg_all_layer_sent_embedding)
    print("content avg ELMo embedding shape : ", avg_all_layer_sent_embedding.shape)
    assert avg_all_layer_sent_embedding.shape == (22, 1024)

    attention_layer = HierarchicalAttentionNet(1024, step_dim=22)
    maxlen = 200

    attention_weights = attention_layer.forward(torch.as_tensor(avg_all_layer_sent_embedding), maxlen)
    print("context attention weights shape: ", attention_weights.shape)
    assert attention_weights.shape == torch.Size([1, 1024])
    print(attention_weights)


def test_elmo_output_with_self_attention():
    import os
    from data_loader import load_abs_path
    import numpy as np

    elmo_credbank_model_path = load_abs_path(
        os.path.join(os.path.dirname(__file__), '..', "resource", "embedding", "elmo_model",
                     "elmo_credbank_2x4096_512_2048cnn_2xhighway_weights_10052019.hdf5"))
    elmo_options_file_path = load_abs_path(
        os.path.join(os.path.dirname(__file__), '..', "resource", "embedding", "elmo_model",
                     "elmo_2x4096_512_2048cnn_2xhighway_options.json"))

    sentence4 = ["i", "really", "enjoy", "Ashley", "and", "Ami", "salon", "she", "do", "a", "great", "job", "be",
                 "friendly", "and", "professional","I", "usually", "get", "my", "hair", "do", "when", "i", "go", "to",
                 "to", "MI", "because", "of", "the", "quality", "of", "the", "highlight", "and", "the", "price", "be",
                 "very", "affordable", "the", "highlight", "fantastic", "thank", "Ashley", "i", "highly", "recommend",
                 "you", "and", "ill", "be", "back"]
    fine_tuned_elmo = ElmoEmbedder(
        options_file=elmo_options_file_path,
        weight_file=elmo_credbank_model_path)
    sentence_vectors = fine_tuned_elmo.embed_sentence(sentence4)
    avg_all_layer_sent_embedding = np.mean(sentence_vectors, axis=0, dtype='float32')

    print("test with self-attentive model: ")
    self_attention_elmo_input = torch.stack([torch.as_tensor(avg_all_layer_sent_embedding)]).permute(1, 0, 2)
    # ELMo output.size() = (batch_size, num_seq, 2*hidden_size)
    self_attention_elmo_input = self_attention_elmo_input.permute(1, 0, 2)

    print("self_attention_elmo_input shape (batch_size, num_seq, 2*hidden_size) : ", self_attention_elmo_input.shape)

    self_attention_model = StructuredSelfAttention(1024)
    # print(self_attention_elmo_input)
    concatenated_context_embeddings, attn_weight_matrix = self_attention_model.forward(self_attention_elmo_input, if_concat=True)
    print("self attention weights (annotation A) of ELMo embedding shape (batch_size, r, num_seq): ", attn_weight_matrix.shape)
    print("attn_weight_matrix: ", attn_weight_matrix)
    # assert attn_weight_matrix.shape == torch.Size([22, 30, 1])

    # fc_input_tesnor = hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2])
    print(" concatenate the hidden_matrix  (b4 feeding into FC and softmax) shape:", concatenated_context_embeddings.shape)

    avg_context_embeddings, attn_weight_matrix = self_attention_model.forward(self_attention_elmo_input, if_concat=False)
    print(" averaged the hidden_matrix  (b4 feeding into FC and softmax) shape:", avg_context_embeddings.shape)


def test_context_metadata_lstm_attention_with_context():
    maxlen = 100
    context_metadata_encoder = torch.nn.LSTM(28, 28*2, num_layers=2, batch_first=True,bidirectional=False)

    tensor_seq = [torch.rand(5, 28),torch.rand(5, 28),torch.rand(5, 28),torch.rand(5, 28)]

    vary_lengths = torch.stack([torch.as_tensor(tensor_cxt_item.shape[0]) for tensor_cxt_item in tensor_seq])
    print("vary context length: ", vary_lengths)
    idxes = torch.arange(0,maxlen,out=torch.LongTensor(maxlen)).unsqueeze(0)
    mask = Variable((idxes<vary_lengths.unsqueeze(1)).float())
    mask = torch.as_tensor(mask, dtype=torch.uint8)

    print("mask: ", mask)

    # to apply AttentionWithContext layer, sequence need to be padded with maximum length
    padded_seq, len_tensor = torch.nn.utils.rnn.pad_packed_sequence(torch.nn.utils.rnn.pack_sequence(tensor_seq),batch_first=True,padding_value=0, total_length=maxlen)
    print("CM paddinged sequence shape: %s, len: %s" %(str(padded_seq.shape), str(len_tensor)))
    print(type(padded_seq))

    #seq_tensor = torch.stack(tensor_seq)
    seq_tensor = padded_seq

    lstm_output, (final_hidden_state, final_cell_state) = context_metadata_encoder.forward(seq_tensor)
    print("lstm output shape (batch size, seq size, dim)", lstm_output.shape)

    attention_layer = HierarchicalAttentionNet(56, maxlen)
    h_lstm_atten = attention_layer(lstm_output, mask)

    print("h_lstm_atten after applying attention with context shape : ", h_lstm_atten.shape)


def test_context_metadata_with_self_attention():
    context_metadata_encoder = torch.nn.LSTM(28, 28*2, num_layers=2, batch_first=True,bidirectional=False)
    seq_tensor = torch.stack([torch.rand(1, 28), torch.rand(1, 28), torch.rand(1, 28), torch.rand(1, 28)])
    cm_lstm_output, (final_hidden_state, final_cell_state) = context_metadata_encoder.forward(seq_tensor)
    print("lstm_output shape (seq_size, batch_size, feature_dim): ", cm_lstm_output.shape)

    cm_lstm_output = cm_lstm_output.permute(1, 0, 2)
    print("lstm_output shape after permuting(batch_size, num_seq, 2*hidden_size): ", cm_lstm_output.shape)
    assert cm_lstm_output.shape == torch.Size([1, 4, 56])

    self_attention_net = StructuredSelfAttention(28*2)
    cm_attn_weight_matrix = self_attention_net(cm_lstm_output)
    print("self attention weights (annotation A) of ELMo embedding shape (batch_size, r, num_seq): ", cm_attn_weight_matrix.shape)
    # assert attn_weight_matrix.shape == torch.Size([22, 30, 1])

    cm_hidden_matrix = torch.bmm(cm_attn_weight_matrix, cm_lstm_output)
    print("new attention weighted ELMo state shape (batch_size, r, 2*hidden_size): ", cm_hidden_matrix.shape)

    cm_fc_input_tensor = cm_hidden_matrix.view(-1, cm_hidden_matrix.size()[1] * cm_hidden_matrix.size()[2])
    print(" concatenate the hidden_matrix  (b4 feeding into FC and softmax) shape:", cm_fc_input_tensor.shape)
    assert cm_fc_input_tensor == torch.Size([1, 1680])


def init_context_lstm_hidden(batch_size, num_layers, hidden_size, cuda_device=-1, batch_first=True):
    """
    if not initialise hidden state for LSTM, it (Pytorch LSTM) will reinitialize hidden layer to zeros by default
    (which means it does not remember across time steps). It works but performance wont be as good.

    see https://pytorch.org/docs/master/nn.html#lstm

    :param batch_size:
    :return: (h_t,c_t), the weights are of the form (batch_size, lstm_layers, lstm_units) in batch first
    """
    if batch_first:
        hidden_a = torch.randn(batch_size, num_layers, hidden_size)
        hidden_b = torch.randn(batch_size, num_layers, hidden_size)
    else:
        hidden_a = torch.randn(num_layers, batch_size, hidden_size)
        hidden_b = torch.randn(num_layers, batch_size, hidden_size)

    if cuda_device != -1:
        hidden_a = hidden_a.cuda()
        hidden_b = hidden_b.cuda()

    hidden_a = Variable(hidden_a)
    hidden_b = Variable(hidden_b)

    return (hidden_a, hidden_b)


def test_dummy_context_content_lstm_attention_with_context():
    maxlen = 300
    context_content_encoder = torch.nn.LSTM(1024, 1024*2, num_layers=2, batch_first=True, bidirectional=False)

    tensor_seq = [torch.rand(5, 1024),torch.rand(259, 1024),torch.rand(105, 1024),torch.rand(5, 1024)]
    batch_size = len(tensor_seq)

    vary_lengths = torch.stack([torch.as_tensor(tensor_cxt_item.shape[0], dtype=torch.float) for tensor_cxt_item in tensor_seq])
    cxt_lengths_int = torch.stack([torch.as_tensor(tensor_cxt_item.shape[0], dtype=torch.int8) for tensor_cxt_item in tensor_seq])

    print("vary context length: ", vary_lengths)
    idxes = torch.arange(0,maxlen,out=torch.FloatTensor(maxlen)).unsqueeze(0)
    mask = idxes<vary_lengths.unsqueeze(1)
    # mask = Variable((mask).float())
    # mask = torch.as_tensor(mask, dtype=torch.uint8)
    print("type mask: ", type(mask))
    print("mask shape: ", mask.shape)

    # insert dummy item about global seq size
    tensor_seq.insert(0, torch.zeros(300, 1024))

    padded_seq = torch.nn.utils.rnn.pad_sequence(tensor_seq, batch_first=True)
    print("CM paddinged sequence shape: %s " %(str(padded_seq.shape)))
    print(type(padded_seq))

    padded_seq = padded_seq[1:]
    print("CM paddinged sequence shape after removing first dummy item: %s " %(str(padded_seq.shape)))
    print(padded_seq)

    #print("batch size: ", batch_size)
    seq_tensor = padded_seq
    # pass the tuple packed_seq_batch to the recurrent modules
    lstm_output, (final_hidden_state, final_cell_state) = context_content_encoder.forward(seq_tensor,
                                                                                          init_context_lstm_hidden(batch_size,
                                                                                                                   context_content_encoder.num_layers,
                                                                                                                   context_content_encoder.hidden_size,
                                                                                                                   batch_first=False))

    attention_layer = HierarchicalAttentionNet(context_content_encoder.hidden_size, maxlen)

    print("first record of lstm output: ", str(lstm_output[0].shape))
    print("attention_layer -> feature_dim: ", attention_layer.feature_dim)
    print("attention_layer -> step_dim: ", attention_layer.step_dim)

    h_lstm_atten, weighted_lstm_output, attention_weight = attention_layer(lstm_output, mask)

    print("h_lstm_atten after applying attention with context shape : ", h_lstm_atten.shape)

    print("1st entry of attentive lstm output: ")
    print(h_lstm_atten[0])

    print(" ======================== test layernorm =========================")
    from my_layer_norm import MyLayerNorm
    layer_norm_cxt_content_encoder = MyLayerNorm(attention_layer.feature_dim)
    normed_attention_output = layer_norm_cxt_content_encoder.forward(h_lstm_atten)
    print("normed_attention_output: ", normed_attention_output.shape)
    print("1st entry of normed attentive lstm output: ")
    print(normed_attention_output[0])

    print("======================== test Seq2Vec wrapper ====================")
    from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
    context_content_encoder_wrapper = PytorchSeq2VecWrapper(context_content_encoder)
    print("mask: ", mask)
    print("batch size: ", seq_tensor.shape[0])
    final_state_tensor = context_content_encoder_wrapper.forward(seq_tensor, mask,
                                                                 init_context_lstm_hidden(seq_tensor.shape[0], context_content_encoder.num_layers, context_content_encoder.hidden_size, batch_first=False))
    print("final_state_tensor: ", final_state_tensor)


def test_dummy_data_lstm_attention_with_context():
    import numpy as np

    np.random.seed(1)
    sample_no, sample_len = (4, 6)
    data = np.zeros((sample_no, sample_len), dtype=np.float32)
    seq_len = np.array([4, 1, 6, 3], dtype=np.int32)
    mask = np.arange(sample_len) < seq_len[:, None]
    data[~mask] = 1

    # generating the actual toy example
    np.random.seed(1)
    X = np.random.random(data.shape).round(1) * 2 + 3
    X = torch.from_numpy(X)
    X_len = torch.LongTensor([4, 1, 6, 3])  # length of each sequence

    annot = np.random.random(data.shape).round(1) * 2 + 3

    print("X: ",X)
    print("annot: ", annot)

    maxlen = X.size(1)
    mask = torch.arange(maxlen)[None, :] < X_len[:, None]
    print("mask: ", mask)

    maxlen = X.size(1)
    idx = torch.arange(maxlen).unsqueeze(0).expand(X.size())
    print("idx: ", idx)
    len_expanded = X_len.unsqueeze(1).expand(X.size())
    print("len_expanded: ", len_expanded)
    mask = idx < len_expanded
    print("mask: ", mask)

    X[~mask] = float('-inf')
    print("masked X: ", X)

    attention_weights = torch.softmax(X, dim=1)
    print("attention_weights: ", attention_weights)


def test_stacked_attention():
    print("test stacked attention")
    maxlen = 300

    # content embedding + attention
    print("============== content embedding + attention ================")
    from my_layer_norm import MyLayerNorm

    context_content_encoder = torch.nn.LSTM(1024, 1024*2, num_layers=2, batch_first=True, bidirectional=False)

    cc_tensor_seq = [torch.rand(5, 1024),torch.rand(259, 1024),torch.rand(105, 1024),torch.rand(5, 1024)]
    batch_size = len(cc_tensor_seq)
    ## padding and masking
    vary_lengths = torch.stack([torch.as_tensor(tensor_cxt_item.shape[0], dtype=torch.float) for tensor_cxt_item in cc_tensor_seq])
    cxt_lengths_int = torch.stack([torch.as_tensor(tensor_cxt_item.shape[0], dtype=torch.int8) for tensor_cxt_item in cc_tensor_seq])
    idxes = torch.arange(0,maxlen,out=torch.FloatTensor(maxlen)).unsqueeze(0)
    cc_mask = idxes<vary_lengths.unsqueeze(1)
    ### insert dummy item about global seq size
    cc_tensor_seq.insert(0, torch.zeros(maxlen, 1024))
    cc_padded_seq = torch.nn.utils.rnn.pad_sequence(cc_tensor_seq, batch_first=True)
    cc_padded_seq = cc_padded_seq[1:]
    #### LSTM embedding
    print("context_content_encoder.get_output_dim(): ", context_content_encoder.hidden_size)
    cc_lstm_output, (final_hidden_state, final_cell_state) = context_content_encoder.forward(cc_padded_seq,
                                                                                          init_context_lstm_hidden(batch_size,
                                                                                                                   context_content_encoder.num_layers,
                                                                                                                    context_content_encoder.hidden_size,
                                                                                                                   batch_first=False))
    cc_attention_layer = HierarchicalAttentionNet(context_content_encoder.hidden_size, maxlen)
    cc_attended_lstm_sum, cc_weighted_lstm_output, cc_attention_weight = cc_attention_layer(cc_lstm_output, cc_mask)

    cc_attention_layer_norm = MyLayerNorm((batch_size, maxlen, context_content_encoder.hidden_size))
    cc_weighted_lstm_output = cc_attention_layer_norm.forward(cc_weighted_lstm_output)

    print("attended CC lstm output shape: ", cc_weighted_lstm_output.shape)
    print(" =============================================================================")

    # metadata embedding + attention
    print("============== metadata embedding + attention ================")
    context_metadata_encoder = torch.nn.LSTM(28, 28*2, num_layers=2, batch_first=True,bidirectional=False)
    cm_tensor_seq = [torch.rand(5, 28),torch.rand(5, 28),torch.rand(5, 28),torch.rand(5, 28)]
    batch_size = len(cm_tensor_seq)
    ## padding and masking
    vary_lengths = torch.stack([torch.as_tensor(tensor_cxt_item.shape[0], dtype=torch.float) for tensor_cxt_item in cm_tensor_seq])
    cxt_lengths_int = torch.stack([torch.as_tensor(tensor_cxt_item.shape[0], dtype=torch.int8) for tensor_cxt_item in cm_tensor_seq])
    idxes = torch.arange(0,maxlen,out=torch.FloatTensor(maxlen)).unsqueeze(0)
    cm_mask = idxes<vary_lengths.unsqueeze(1)
    ### insert dummy item about global seq size
    cm_tensor_seq.insert(0, torch.zeros(maxlen, 28))
    cm_padded_seq = torch.nn.utils.rnn.pad_sequence(cm_tensor_seq, batch_first=True)
    cm_padded_seq = cm_padded_seq[1:]
    #### LSTM embedding
    cm_lstm_output, (final_hidden_state, final_cell_state) = context_metadata_encoder.forward(cm_padded_seq,
                                                                                             init_context_lstm_hidden(batch_size,
                                                                                                                      context_metadata_encoder.num_layers,
                                                                                                                      context_metadata_encoder.hidden_size,
                                                                                                                      batch_first=False))
    cm_attention_layer = HierarchicalAttentionNet(context_metadata_encoder.hidden_size, maxlen)
    cm_attended_lstm_sum, cm_weighted_lstm_output, cm_attention_weight = cm_attention_layer(cm_lstm_output, cm_mask)

    cm_attention_layer_norm = MyLayerNorm((batch_size, maxlen, context_metadata_encoder.hidden_size))
    cm_weighted_lstm_output = cm_attention_layer_norm.forward(cm_weighted_lstm_output)

    print("attended CM lstm output shape: ", cm_weighted_lstm_output.shape)
    print(" =============================================================================")
    print(" =================== CC + CM ============")
    context_embedding = torch.cat((cc_weighted_lstm_output, cm_weighted_lstm_output), 2)
    print(" context_embedding shape: ", context_embedding.shape)


    print(" ====================== 3rd attention on top of two attention layers ==========  ")

    concatenated_cxt_feature_dim = context_content_encoder.hidden_size + context_metadata_encoder.hidden_size

    context_attention_layer = HierarchicalAttentionNet(concatenated_cxt_feature_dim, maxlen)
    cxt_attended_lstm_sum, cxt_weighted_lstm_output, cxt_attention_weight = context_attention_layer(context_embedding, cm_mask)

    cxt_attention_layer_norm = MyLayerNorm(concatenated_cxt_feature_dim)
    cxt_attended_lstm_sum = cxt_attention_layer_norm.forward(cxt_attended_lstm_sum)

    print("cxt_attended_lstm_sum: ", cxt_attended_lstm_sum.shape)


if __name__ == '__main__':
    #test_elmo_with_attention()
    test_context_metadata_lstm_attention_with_context()
    #test_dummy_context_content_lstm_attention_with_context()
    #test_elmo_output_with_self_attention()
    #test_context_metadata_with_self_attention()
    #test_dummy_data_lstm_attention_with_context()
    #test_stacked_attention()
