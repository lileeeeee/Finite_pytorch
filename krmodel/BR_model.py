from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
import krmodel.BR_data as BR_data
import krmodel.BR_layers as BR_layers
import numpy as np


# DNN with fully connected layers.
class DNN(nn.Module):
    def __init__(self, input_dim, n_out, depth, width_list, **kwargs):
        super(DNN, self).__init__(**kwargs)
        self.n_out = n_out
        self.depth = depth
        self.width_list = width_list
        self.hidden_layers = []
        for i in range(depth):
            self.hidden_layers.append(BR_layers.Linear(input_dim, width_list[i]))
        self.l_f = BR_layers.Linear(input_dim, n_out)

    def forward(self, inputs):
        x = inputs
        for i in range(self.depth):
            x = torch.relu(self.hidden_layers[i](x))
        x = self.l_f(x)

        return x

    def get_gradient(self, inputs):
        x = torch.Tensor(inputs)
        x.zero_grad()
        prediction = self.forward(x)
        prediction.backward()
        return x.grad


# DNN with the NN2 layers and actnorm layers.
class DNN_NN2(nn.Module):
    def __init__(self, input_dim, depth, width, n_out, **kwargs):
        super(DNN_NN2, self).__init__(**kwargs)
        self.n_out = n_out
        self.depth = depth
        self.width = width
        self.NN2_layers = []
        self.actnorm_layers = []
        for i in range(depth):
            self.NN2_layers.append(BR_layers.NN2(input_dim, width, width))
            self.actnorm_layers.append(BR_layers.actnorm(input_dim))
        self.l_NN2 = BR_layers.NN2(input_dim,width, n_out)

    def forward(self, inputs):
        x = inputs
        for i in range(self.depth):
            x = x + self.NN2_layers[i](x)
            x = self.actnorm_layers[i](x)
            x = torch.relu(x)
            # x = tf.nn.tanh(x)

        x = self.l_NN2(x)

        return x

    def get_gradient(self, inputs):
        x = torch.Tensor(inputs)
        x.zero_grad()
        prediction = self.forward(x)
        prediction.backward()
        return x.grad

    def actnorm_data_initialization(self):
        for i in range(self.depth):
            # self.actnorm_layers[i].data_init = False
            self.actnorm_layers[i].reset_data_initialization()


# invertible mapping based on real NVP
class IM_rNVP(nn.Module):
    def __init__(self, input_dim,n_depth, n_split_at, width=32, flow_coupling=0,
                 rotation=False, **kwargs):
        super(IM_rNVP, self).__init__(**kwargs)
        self.n_depth = n_depth
        self.n_split_at = n_split_at
        self.width = width
        self.flow_coupling = flow_coupling
        self.rotation = rotation

        # two affine coupling layers are needed for each update of the vector
        assert n_depth % 2 == 0

        # a rotation layer
        if rotation:
            self.W_LU = BR_layers.W_LU(input_dim)

        # stack affine coupling layers
        self.flow_mapping = BR_layers.flow_mapping(input_dim, n_depth,
                                                   n_split_at,
                                                   width=width,
                                                   flow_coupling=flow_coupling)

        # the prior distribution
        self.log_prior = BR_data.log_standard_Gaussian

    # computing the logarithm of the estimated pdf on the input data.
    def forward(self, inputs):
        objective = torch.zeros_like(inputs, dtype=torch.float32)[:, 0]
        objective = torch.reshape(objective, [-1, 1])

        z = inputs

        # unitary transformation
        if self.rotation:
            z, objective = self.W_LU(z, objective)

        # f(y) and log of jacobian
        z, objective = self.flow_mapping(z, objective)

        # logrithm of estimated pdf
        objective += self.log_prior(z)

        # add cross entropy to the loss function
        # CE = -tf.reduce_mean(objective)
        # self.add_loss(CE)
        return objective

    # return the first-order derivative wrt inputs
    def get_H1_regularization(self, inputs):
        x = torch.Tensor(inputs)
        x.zero_grad()
        objective = self.forward(x)
        objective.backward()
        fx = x.grad
        # fx = BR_data.flatten_sum(tf.square(fx*tf.exp(objective/2.0)))
        # fx = BR_data.flatten_sum(tf.square(fx*tf.exp(-objective/2.0)))
        fx = BR_data.flatten_sum(torch.square(fx))
        return torch.mean(fx)

    # mapping from data to prior
    def mapping_to_prior(self, inputs, logdet=None):
        z = inputs
        if logdet is not None:
            if self.rotation:
                z, logdet = self.W_LU(z, logdet)
            z, logdet = self.flow_mapping(z, logdet)
            return z, logdet
        else:
            if self.rotation:
                z = self.W_LU(z)
            z = self.flow_mapping(z)
            return z

    # mapping from prior to data subject to the estimated distribution
    def mapping_from_prior(self, inputs):
        z = self.flow_mapping(inputs, reverse=True)
        if self.rotation:
            z = self.W_LU(z, reverse=True)
        return z

    def actnorm_data_initialization(self):
        self.flow_mapping.actnorm_data_initialization()

    def WLU_data_initialization(self):
        self.W_LU.reset_data_initialization()

    # return samples from prior
    def draw_samples_from_prior(self, n_samples, n_dim):
        return torch.rand((n_samples, n_dim))


# invertible mapping based on real NVP and KR rearrangement and CDF inverse
class IM_rNVP_KR_CDF(nn.Module):
    def __init__(self, n_dim, n_step, n_depth,
                 n_width=32,
                 shrink_rate=1.0,
                 flow_coupling=0,
                 n_bins=16,
                 rotation=False,
                 **kwargs):
        super(IM_rNVP_KR_CDF, self).__init__(**kwargs)

        # two affine coupling layers are needed for each update of the vector
        assert n_depth % 2 == 0
        assert n_bins % 2 == 0

        self.n_dim = n_dim  # dimension of the data
        self.n_step = n_step  # step size for dimension reduction
        self.n_depth = n_depth  # depth for flow_mapping
        self.n_width = n_width
        self.n_bins = n_bins
        self.shrink_rate = shrink_rate
        self.flow_coupling = flow_coupling
        self.rotation = rotation

        # the number of filtering stages
        self.n_stage = n_dim // n_step
        if n_dim % n_step == 0:
            self.n_stage -= 1

        # the number of rotation layers
        self.n_rotation = self.n_stage

        if rotation:
            self.rotations = []
            for i in range(self.n_rotation):
                # rotate the coordinate system for a better representation of data
                self.rotations.append(BR_layers.W_LU(n_dim))

        # flow mapping with n_stage
        self.flow_mappings = []
        for i in range(self.n_stage):
            # flow_mapping given by such as real NVP
            n_split_at = n_dim - (i + 1) * n_step
            self.flow_mappings.append(BR_layers.flow_mapping(n_dim,
                                                             n_depth,
                                                             n_split_at,
                                                             n_width=n_width,
                                                             flow_coupling=flow_coupling))
            n_width = int(n_width * self.shrink_rate)

        # data will pass the squeezing layer at the end of each stage
        self.squeezing_layer = BR_layers.squeezing(n_dim, n_step)

        if self.n_bins > 0:
            self.cdf_layer = BR_layers.scale_and_CDF(n_dim, n_bins=n_bins)

        # the prior distribution is the Gaussian distribution
        self.log_prior = BR_data.log_standard_Gaussian

    # computing the logarithm of the estimated pdf on the input data.
    def forward(self, inputs):
        objective = torch.zeros_like(inputs, dtype=torch.float32)[:, 0]
        objective = torch.reshape(objective, [-1, 1])

        # f(y) and log of jacobian
        z, objective = self.mapping_to_prior(inputs, objective)

        # logrithm of estimated pdf
        objective += self.log_prior(z)

        return objective

    # mapping from data to prior
    def mapping_to_prior(self, inputs, logdet=None):
        z = inputs

        for i in range(self.n_stage):
            if logdet is not None:
                if self.rotation and i < self.n_rotation:
                    z, logdet = self.rotations[i](z, logdet)

                z, logdet = self.flow_mappings[i](z, logdet)
            else:
                if self.rotation and i < self.n_rotation:
                    z = self.rotations[i](z)

                z = self.flow_mappings[i](z)
            z = self.squeezing_layer(z)

        if self.n_bins > 0:
            z = self.cdf_layer(z, logdet)
            if logdet is not None:
                z, logdet = z

        if logdet is not None:
            return z, logdet
        else:
            return z

    # mapping from prior to data
    def mapping_from_prior(self, inputs):
        z = inputs
        if self.n_bins > 0:
            z = self.cdf_layer(z, reverse=True)

        for i in reversed(range(self.n_stage)):
            z = self.squeezing_layer(z, reverse=True)
            z = self.flow_mappings[i](z, reverse=True)

            if self.rotation and i < self.n_rotation:
                z = self.rotations[i](z, reverse=True)
        return z

    # data initialization for actnorm layers
    def actnorm_data_initialization(self):
        for i in range(self.n_stage):
            self.flow_mappings[i].actnorm_data_initialization()

    # return the first-order derivative wrt inputs
    def get_H1_regularization(self, inputs):
        x = torch.Tensor(inputs)
        x.zero_grad()
        objective = self.forward(x)
        objective.backward()
        fx = x.grad
        fx = BR_data.flatten_sum(torch.square(fx))
        return torch.mean(fx)

    # return samples from prior
    def draw_samples_from_prior(self, n_samples, n_dim):
        return torch.rand((n_samples, n_dim))
