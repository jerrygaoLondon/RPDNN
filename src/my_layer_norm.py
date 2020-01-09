import torch


class MyLayerNorm(torch.nn.Module):
    # pylint: disable=line-too-long
    """
    An implementation of `Layer Normalization
    <https://www.semanticscholar.org/paper/Layer-Normalization-Ba-Kiros/97fb4e3d45bb098e27e0071448b6152217bd35a5>`_ .

    Layer Normalization stabilises the training of deep neural networks by
    normalising the outputs of neurons from a particular layer. It computes:

    output = (gamma * (tensor - mean) / (std + eps)) + beta

    Parameters
    ----------
    dimension : ``int``, required.
        The dimension of the layer output to normalize.
    eps : ``float``, optional, (default = 1e-6)
        An epsilon to prevent dividing by zero in the case
        the layer has zero variance.

    Returns
    -------
    The normalized layer output.
    """
    def __init__(self,
                 dimension: int,
                 eps: torch.float32 = 1e-6) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(dimension))
        self.beta = torch.nn.Parameter(torch.zeros(dimension))
        self.eps = eps

    def forward(self, tensor: torch.Tensor, device=None):  # pylint: disable=arguments-differ
        mean = tensor.mean(-1, keepdim=True)
        std = tensor.std(-1, unbiased=False, keepdim=True)
        if torch.cuda.is_available() and device is not None:
            #print("mean type: [%s] mean device: [%s]" %(type(mean), mean.get_device()))
            #print("std device: ", std.get_device())
            #print("gamma device: ", self.gamma.get_device())
            #print("self.beta device: ", self.beta.get_device())
            #print("input tensor device: ", tensor.get_device())

            mean = mean.to(device=device)
            std = std.to(device=device)
            self.gamma = self.gamma.to(device=device)
            self.beta = self.beta.to(device=device)
            self.eps = torch.as_tensor(self.eps).cuda(device=device)
            # torch.FloatTensor(self.eps).cuda(device=device)
            tensor = tensor.to(device=device)

            #print("after moving all tensor parameters to a device to avoid RuntimeError 'arguments are located on different GPUs' .")
            #print("mean device: ", mean.get_device())
            #print("std device: ", std.get_device())
            #print("gamma device: ", self.gamma.get_device())
            #print("self.beta device: ", self.beta.get_device())
            #print("self.eps device: ", self.eps.get_device())
            #print("input tensor device: ", tensor.get_device())
        """
        print(" &&&&&&&&&&&&&&&&&&&&& debug start layer norm &&&&&&&&&&&&&&&&&&&&&&&&&& ")
        print("self.gamma shape: ", self.gamma.shape)
        print("self.gamma: ", self.gamma)
        print("mean: ", mean)
        print("std: ", std)
        print("self.eps: ", self.eps)
        print("self.beta: ", self.beta)
        print(" &&&&&&&&&&&&&&&&&&&&& debug end layer norm &&&&&&&&&&&&&&&&&&&&&&&&&& ")
        """
        r = self.gamma * (tensor - mean) / (std + self.eps) + self.beta
        # try with the solution to address NANs problem of stacked Layrnorm, mentioned in https://discuss.pytorch.org/t/nan-in-layer-normalization/13846/5
        # r.retain_grad() # does not work
        return r