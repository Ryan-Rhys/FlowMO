# Author: Leo Klarner
"""
Molecule kernels for Gaussian Process Regression implemented in PyTorch.
"""

import torch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive


class Tanimoto(Kernel):
    """
    An implementation of the Tanimoto kernel (aka. Jaccard index)
    as a gpytorch kernel
    """

    def __init__(self, variance_constraint=None, **kwargs):
        super().__init__(**kwargs)
        # put positivity constraint on variance as default
        if variance_constraint is None:
            variance_constraint = Positive()
        # initialise variance parameter, potentially different for each batch
        # for batch-wise active learning
        self.register_parameter(name="raw_variance", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_constraint("raw_variance", variance_constraint)

    @property
    def variance(self):
        return self.raw_variance_constraint.transform(self.raw_variance)

    @variance.setter
    def variance(self, value):
        self._set_variance(value)

    def _set_variance(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_variance)
        self.initialize(raw_variance=self.raw_variance_constraint.inverse_transform(value))

    def forward(self, x1, x2, **params):
        """
        Compute the Tanimoto kernel matrix σ² * ((<x, y>) / ((||x||2)^2 + (||y||2)^2 - <x, y>))

        :param x1: N x D array
        :param x2: M x D array.
        :return: The kernel matrix of dimension N x M
        """
        # TODO: implement batched Tanimoto kernel calculation

        # calculate the squared L2-norm over the feature dimension of the x1 data tensor
        x1_norm = torch.unsqueeze(torch.sum(torch.square(x1), dim=-1), dim=-1)

        # check if both data tensors are identical
        if x1.size() == x2.size() and torch.equal(x1, x2):
            x2_norm = x1_norm
        else:
            # if both data tensors are not identical
            # calculate the squared L2-norm over the feature dimension of the x2 data tensor
            x2_norm = torch.unsqueeze(torch.sum(torch.square(x2), dim=-1), dim=-1)

        # calculate the matrix product of the data tensors
        cross_prod = torch.matmul(x1, torch.t(x2))

        denominator = torch.add(x1_norm, torch.t(x2_norm))
        denominator.sub_(cross_prod)

        return self.variance * torch.div(cross_prod, denominator)


class LazyTanimoto(Kernel):
    """
    An implementation of the Tanimoto kernel (aka. Jaccard index)
    as a gpytorch kernel using lazytensors
    """
    pass
