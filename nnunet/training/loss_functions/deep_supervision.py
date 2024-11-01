#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from torch import nn
import torch

class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors  #深监督权重
        self.loss = loss

    def forward(self, x, y, w=None, w_1=None):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors
        w, w_1, e, m = w
        normal_weight = torch.ones_like(w).to(w.device) / x[0].shape[1]
        if e < m:
            w_nnu = 1 - e / m
            w_ada = 1 - w_nnu
        else:
            w_nnu = 0.
            w_ada = 1.
        #w_nnu = 0.
        #w_ada = 1.
        #l = weights[0] * (
        #        w_nnu * self.loss(x[0], y[0][:, 0:1], normal_weight[0]) + w_ada * self.loss(x[0], y[0][:, 1:2], w[0]))
        l = weights[0] * (
                w_nnu * self.loss(x[0], y[0], normal_weight[0], normal_weight[0]) + w_ada * self.loss(x[0], y[0], w[0], w_1[0]))
        for i in range(1, len(x)):
            if weights[i] != 0:
                #l += weights[i] * (
                #        w_nnu * self.loss(x[i], y[i][:, 0:1], normal_weight[i]) + w_ada * self.loss(x[i], y[i][:, 1:2],
                #                                                                                w[i]))
                l += weights[i] * (
                        w_nnu * self.loss(x[i], y[i], normal_weight[i], normal_weight[i]) + w_ada * self.loss(x[i], y[i], w[i], w_1[i]))
        return l


    def nnunet_forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l
