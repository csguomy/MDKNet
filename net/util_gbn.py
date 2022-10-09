import torch
from torch import nn
from torch.nn import functional as F


"""
Code borrow from https://github.com/maheshkkumar/adacrowd/blob/master/adacrowd/models/adacrowd/blocks.py
"""

__constants__ = ['GuidedBatchNorm2d']

class GuidedBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(GuidedBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and \
            self.bias is not None, "Please assign GBN weights first"
        running_mean = self.running_mean
        running_var = self.running_var
        out = F.batch_norm(
            x, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


def assign_adaptive_params(gbn_params, model):
    assert gbn_params.dim() == 1
#     if gbn_params.dim() == 1:
#         gbn_params = gbn_params.unsqueeze(0)

    cur_bn_features = 0
    for m in model.modules():
        if m.__class__.__name__ in __constants__:
            # works for each gbn layer has different dims
            alphas = gbn_params[cur_bn_features : cur_bn_features + m.num_features]
            betas =  gbn_params[cur_bn_features + m.num_features: cur_bn_features + m.num_features * 2]
            m.bias = betas.contiguous().view(-1)
            m.weight = alphas.contiguous().view(-1)
            
            cur_bn_features += m.num_features * 2
            if gbn_params.size(0) > 2 * m.num_features:
                assert 1==2
                gbn_params = gbn_params[:2 * m.num_features]


def get_num_adaptive_params(model):
    # return the number of GBN parameters needed by the model
    num_gbn_params = 0
    for m in model.modules():
        if m.__class__.__name__ in __constants__:
            num_gbn_params += 2 * m.num_features
    return num_gbn_params

