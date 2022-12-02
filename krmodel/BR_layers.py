import torch
import torch.nn as nn
import torch.nn.init as init
import math

def entry_stop_gradients(target, mask):
    """
    mask specify which entries to be trained
    """
    mask_stop = ~mask
    mask = mask.float()
    mask_stop = mask_stop.float()

    return (mask_stop * target).requires_grad_(False) + mask * target

# This flow_mapping is for infinite support
# stacking actnorm, and affince coupling layers.
class flow_mapping(nn.Module):
  def __init__(self, input_dim, n_depth, n_split_at, n_width = 32, flow_coupling = 0, **kwargs):
    super(flow_mapping, self).__init__(**kwargs)
    self.n_depth = n_depth
    self.n_split_at = n_split_at
    self.n_width = n_width
    self.flow_coupling = flow_coupling

    # two affine coupling layers are needed for each update of the whole vector
    assert n_depth % 2 == 0

    self.n_length = input_dim
    self.scale_layers = []
    self.affine_layers = []

    sign = -1
    for i in range(self.n_depth):
      self.scale_layers.append(actnorm(input_dim))
      sign *= -1
      i_split_at = (self.n_split_at*sign + self.n_length) % self.n_length
      self.affine_layers.append(affine_coupling(input_dim,
                                                i_split_at,
                                                n_width=self.n_width,
                                                flow_coupling=self.flow_coupling))

  # the default setting mapping the given data to the prior distribution
  # without computing the jacobian.
  def forward(self, inputs, logdet=None, reverse=False):
    if not reverse:
      z = inputs
      for i in range(self.n_depth):
        z = self.scale_layers[i](z, logdet)
        if logdet is not None:
            z, logdet = z

        z = self.affine_layers[i](z, logdet)
        if logdet is not None:
            z, logdet = z

        z = torch.flip(z, [1])
    else:
      z = inputs

      for i in reversed(range(self.n_depth)):
        z = torch.flip(z, [1])

        z = self.affine_layers[i](z, logdet, reverse=True)
        if logdet is not None:
            z, logdet = z

        z = self.scale_layers[i](z, logdet, reverse=True)
        if logdet is not None:
            z, logdet = z

    if logdet is not None:
        return z, logdet
    return z

  def actnorm_data_initialization(self):
    for i in range(self.n_depth):
        self.scale_layers[i].reset_data_initialization()

class scale_and_CDF(nn.Module):
  def __init__(self, input_dim, n_bins=16, **kwargs):
    super(scale_and_CDF, self).__init__(**kwargs)
    self.n_bins = n_bins
    self.scale_layer = actnorm(input_dim)
    self.cdf_layer = CDF_quadratic(input_dim, self.n_bins)

  def forward(self, inputs, logdet=None, reverse=False):
    z = inputs
    if not reverse:
      z = self.scale_layer(z, logdet)
      if logdet is not None:
        z, logdet = z

      z = self.cdf_layer(z, logdet)
      if logdet is not None:
        z, logdet = z
    else:
      z = self.cdf_layer(z, logdet, reverse=True)
      if logdet is not None:
        z, logdet = z

      z = self.scale_layer(z, logdet, reverse=True)
      if logdet is not None:
        z, logdet = z

    if logdet is not None:
      return z, logdet

    return z

  def actnorm_data_initialization(self):
    self.scale_layer.reset_data_initialization()

# optimal linear mapping for least important dimension - rotation layer
class W_LU(nn.Module):
  def __init__(self, input_dim, **kwargs):
    super(W_LU, self).__init__(**kwargs)
    #self.data_init = True
    self.n_length = input_dim

    # lower- and upper-triangluar parts in one matrix.
    self.LU = nn.Parameter(torch.zeros(self.n_length, self.n_length))

    # identity matrix
    self.LU_init = torch.eye(self.n_length)
    # permutation - identity matrix
    #self.P = tf.range(self.n_length)
    # inverse of self.P
    #self.invP = tf.math.invert_permutation(self.P)

  def forward(self, inputs, logdet = None, reverse=False):
    x = inputs
    n_dim = x.shape[-1]

    # an initial rotation can be based on the PCA of initial data
    # if not self.data_init:
    #     # the covariance matrix
    #     cov = tfp.stats.covariance(x)
    #
    #     # eigen-value problem: e, eigenvalue; v, eigenvector
    #     e,v = tf.linalg.eigh(cov)
    #
    #     # order the eigenvalues in a decreasing order.
    #     e = e[::-1]
    #     v = v[:,::-1]
    #
    #     tf.print(e)
    #
    #     if e[0]/e[-1] > 5.0:
    #         # lu decomposition of W = v^T
    #         lu, p = tf.linalg.lu(tf.transpose(v))
    #
    #         # initialize LU_init and P
    #         self.LU_init = lu
    #         self.P = p
    #         self.invP = tf.math.invert_permutation(p)
    #
    #     self.data_init = True

    # invP*L*U*x
    LU = self.LU_init + self.LU

    # upper-triangular matrix
    U = torch.triu(LU)

    # diagonal line
    U_diag = torch.diag(U)

    # trainable mask for U
    U_mask = (torch.triu(torch.ones([n_dim, n_dim])) >= 1)
    U = entry_stop_gradients(U, U_mask)

    # lower-triangular matrix
    I = torch.eye(self.n_length)
    L = torch.tril(I+LU)-torch.diag(LU)

    # trainable mask for L
    L_mask = (torch.tril(torch.ones([n_dim, n_dim])) - torch.diag(torch.ones([n_dim, n_dim])) >= 1)
    L = entry_stop_gradients(L, L_mask)

    if not reverse:
        x = torch.transpose(x)
        x = torch.matmul(U,x)
        x = torch.matmul(L,x)
        #x = tf.gather(x, self.invP)
        x = torch.transpose(x)
    else:
        x = torch.transpose(x)
        #x = tf.gather(x, self.P)
        x = torch.matmul(torch.inverse(L), x)
        x = torch.matmul(torch.inverse(U), x)

        #It seems that triangular_solve does not do backpropagation in tensorflow.
        #x = tf.linalg.triangular_solve(L, x, lower=True)
        #x = tf.linalg.triangular_solve(U, x, lower=False)
        x = torch.transpose(x)

    if logdet is not None:
        dlogdet = torch.sum(torch.log(torch.abs(U_diag)))
        if reverse:
            dlogdet *= -1.0
        return x, logdet + dlogdet

    return x


class actnorm(nn.Module):
  def __init__(self, input_dim, scale = 1.0, logscale_factor = 3.0, **kwargs):
    super(actnorm, self).__init__(**kwargs)
    self.scale = scale
    self.logscale_factor = logscale_factor

    self.data_init = True

    self.n_length = input_dim
    self.b = nn.Parameter(torch.zeros(1, self.n_length, dtype=torch.float32))

    self.b_init = nn.Parameter(torch.zeros(1, self.n_length, dtype=torch.float32), requires_grad=False)

    self.logs = nn.Parameter(torch.zeros(1, self.n_length, dtype=torch.float32))

    self.logs_init = nn.Parameter(torch.zeros(1, self.n_length, dtype=torch.float32), requires_grad=False)


  def forward(self, inputs, logdet = None, reverse = False):
    # data initialization
    # by default, no data initialization is implemented.
    if not self.data_init:
        x_mean = torch.mean(inputs, [0], keepdim=True)
        x_var = torch.mean(torch.square(inputs-x_mean), [0], keepdim=True)

        self.b_init = nn.Parameter(-x_mean)
        self.logs_init = nn.Parameter(torch.log(self.scale/(torch.sqrt(x_var)+1e-6))/self.logscale_factor)

        self.data_init = True

    if not reverse:
      x = inputs + (self.b + self.b_init)
      #x = x * tf.exp(self.logs + self.logs_init)
      x = x * torch.exp(torch.clip(self.logs + self.logs_init, -5., 5.))
    else:
      #x = inputs * tf.exp(-self.logs - self.logs_init)
      x = inputs * torch.exp(-torch.clip(self.logs + self.logs_init, -5., 5.))
      x = x - (self.b + self.b_init)

    if logdet is not None:
      #dlogdet = tf.reduce_sum(self.logs + self.logs_init)
      dlogdet = torch.sum(torch.clip(self.logs + self.logs_init, -5., 5.))
      if reverse:
        dlogdet *= -1
      return x, logdet + dlogdet

    return x

  def reset_data_initialization(self):
    self.data_init = False



# affine coupling layer
class affine_coupling(nn.Module):
  def __init__(self, input_dim, n_split_at, n_width = 32, flow_coupling = 1, **kwargs):
    super(affine_coupling, self).__init__(**kwargs)
    # partition as [:n_split_at] and [n_split_at:]
    self.n_split_at = n_split_at
    self.flow_coupling = flow_coupling
    self.n_width = n_width

    n_length = input_dim
    if self.flow_coupling == 0:
      self.f = NN2(n_length, self.n_width, n_length-self.n_split_at, flow_coupling)
    elif self.flow_coupling == 1:
      self.f = NN2(n_length, self.n_width, (n_length-self.n_split_at)*2, flow_coupling)
    else:
      raise Exception()
    self.log_gamma  = nn.Parameter(torch.zeros(1, n_length-self.n_split_at, dtype=torch.float32))


  # the default setting performs a mapping of the data
  # without computing the jacobian
  def forward(self, inputs, logdet=None, reverse=False):
    z = inputs
    n_split_at = self.n_split_at

    alpha = 0.6

    if not reverse:
      z1 = z[:,:n_split_at]
      z2 = z[:,n_split_at:]

      if self.flow_coupling == 0:
        shift = self.f(z1)
        shift = torch.exp(self.log_gamma)*torch.tanh(shift)
        z2 += shift
      elif self.flow_coupling == 1:
        h = self.f(z1)
        shift = h[:,0::2]

        # orignal real NVP
        #scale = tf.nn.sigmoid(h[:,1::2]+2.0)
        #z2 += shift
        #z2 *= scale
        #if logdet is not None:
        #  logdet += tf.reduce_sum(tf.math.log(scale), axis=[1], keepdims=True)

        # resnet-like trick
        # we suppressed both the scale and the shift.
        scale = alpha*torch.tanh(h[:,1::2])
        shift = torch.exp(self.log_gamma)*torch.tanh(shift)
        #shift = tf.exp(tf.clip_by_value(self.log_gamma, -5.0, 5.0))*tf.nn.tanh(shift)
        z2 = z2 + scale * z2 + shift
        if logdet is not None:
           logdet += torch.sum(torch.log(scale + torch.ones_like(scale)),axis=[1], keepdim=True)
      else:
        raise Exception()

      z = torch.concat([z1, z2], 1)
    else:
      z1 = z[:,:n_split_at]
      z2 = z[:,n_split_at:]

      if self.flow_coupling == 0:
        shift = self.f(z1)
        shift = torch.exp(self.log_gamma)*torch.tanh(shift)
        z2 -= shift
      elif self.flow_coupling == 1:
        h = self.f(z1)
        shift = h[:,0::2]

        # original real NVP
        #scale = tf.nn.sigmoid(h[:,1::2]+2.0)
        #z2 /= scale
        #z2 -= shift
        #if logdet is not None:
        #  logdet -= tf.reduce_sum(tf.math.log(scale), axis=[1], keepdims=True)

        # resnet-like trick

        # we suppressed both the scale and the shift.
        scale = alpha*torch.tanh(h[:,1::2])
        #shift = tf.exp(self.log_gamma)*tf.nn.tanh(shift)
        shift = torch.exp(torch.clip(self.log_gamma, -5.0, 5.0))*torch.tanh(shift)
        z2 = (z2 - shift) / (torch.ones_like(scale) + scale)
        if logdet is not None:
           logdet -= torch.sum(torch.log(scale + torch.ones_like(scale)),
                                   axis=[1], keepdim=True)
      else:
        raise Exception()

      z = torch.concat([z1, z2], 1)

    if logdet is not None:
        return z, logdet

    return z

# squeezing layer - KR rearrangement
class squeezing(nn.Module):
    def __init__(self, n_dim, n_cut=1, **kwargs):
        super(squeezing, self).__init__(**kwargs)
        self.n_dim = n_dim
        self.n_cut = n_cut
        self.x = None

    def forward(self, inputs, reverse=False):
        z = inputs
        n_length = z.size()[-1]

        if not reverse:
            if n_length < self.n_cut:
                raise Exception()

            if self.n_dim == n_length:
                if self.n_dim > 2 * self.n_cut:
                    if self.x is not None:
                        raise Exception()
                    else:
                        self.x = z[:, (n_length - self.n_cut):]
                        z = z[:, :(n_length - self.n_cut)]
                else:
                    self.x = None
            elif (n_length - self.n_cut) <= self.n_cut:
                z = torch.concat([z, self.x], 1)
                self.x = None
            else:
                cut = z[:, (n_length - self.n_cut):]
                self.x = torch.concat([cut, self.x], 1)
                z = z[:, :(n_length - self.n_cut)]
        else:
            if self.n_dim == n_length:
                n_start = self.n_dim % self.n_cut
                if n_start == 0:
                    n_start += self.n_cut
                self.x = z[:, n_start:]
                z = z[:, :n_start]

            x_length = self.x.size()[-1]
            if x_length < self.n_cut:
                raise Exception()

            cut = self.x[:, :self.n_cut]
            z = torch.concat([z, cut], 1)

            if (x_length - self.n_cut) == 0:
                self.x = None
            else:
                self.x = self.x[:, self.n_cut:]
        return z


# one linear layer with default width 32.
class Linear(nn.Module):
  def __init__(self, input_dim, n_width=32, **kwargs):
    super(Linear, self).__init__(**kwargs)
    self.n_width = n_width
    self.w = nn.Parameter(torch.empty((int(input_dim), self.n_width), dtype=torch.float32))
    init.xavier_normal_(self.w)
    self.b = nn.Parameter(torch.empty((self.n_width,), dtype=torch.float32))
    init.zeros_(self.b)

  def forward(self, inputs):
    return torch.matmul(inputs, self.w) + self.b

# two-hidden-layer neural network
class NN2(nn.Module):
    def __init__(self, input_dim, n_width=32, n_out=None, flow_coupling=1, **kwargs):
        super(NN2, self).__init__(**kwargs)
        self.n_width = n_width
        self.n_out = n_out
        if flow_coupling == 0:
            self.l_1 = Linear(n_out, self.n_width)
        elif flow_coupling == 1:
            self.l_1 = Linear(n_out/2, self.n_width)
        else:
            raise Exception()
        self.l_2 = Linear(self.n_width, self.n_width)

        n_out = self.n_out or int(input_dim)
        self.l_f = Linear(self.n_width, n_out)

    def forward(self, inputs):
        # relu with low regularity

        x = nn.functional.relu(self.l_1(inputs))
        x = nn.functional.relu(self.l_2(x))

        # tanh with high regularity
        #x = tf.nn.tanh(self.l_1(inputs))
        #x = tf.nn.tanh(self.l_2(x))

        x = self.l_f(x)

        return x

# affine linear mapping from a bounded domain [lb, hb] to [0,1]^d
class Affine_linear_mapping(nn.Module):
    def __init__(self, lb, hb, **kwargs):
        super(Affine_linear_mapping, self).__init__(**kwargs)
        self.lb = lb
        self.hb = hb

    def forward(self, inputs, logdet=None, reverse=False):
        x = inputs
        n_dim = x.shape[-1]
        # mapping from [lb, hb] to [0,1]^d for PDE: y = (x-l) / (h - l)
        if not reverse:
            x = x / (self.hb - self.lb)
            x = x - self.lb / (self.hb - self.lb)
        else:
            x = x + self.lb / (self.hb - self.lb)
            x = x * (self.hb - self.lb)

        if logdet is not None:
            dlogdet = n_dim * torch.log(1.0/(self.hb-self.lb)) * torch.reshape(torch.ones_like(x[:,0], dtype=torch.float32), [-1,1])
            if reverse:
                dlogdet *= -1.0
            return x, logdet + dlogdet

        return x


class Logistic_mapping(nn.Module):
    """
    Logistic mapping, (-inf, inf) --> (0, 1):
    y = (tanh(x/2) + 1) / 2 = e^x/(1 + e^x)
    derivate: dy/dx = y* (1-y)
    inverse: x = log(y) - log(1-y)

    For PDE, data to prior direction: [a,b] ---> (-inf, inf)
    So we need to use an affine linear mapping first and then use logistic mapping
    """

    def __init__(self, **kwargs):
        super(Logistic_mapping, self).__init__(**kwargs)
        self.s = 2.0

    # the direction of this mapping is not related to the flow
    # direction between the data and the prior
    def forward(self, inputs, logdet=None, reverse=False):
        x = inputs

        if not reverse:
            x = torch.clip(x, 1.0e-10, 1.0 - 1.0e-10)
            tp1 = torch.log(x)
            tp2 = torch.log(1 - x)
            x = self.s / 2.0 * (tp1 - tp2)
            if logdet is not None:
                tp = torch.log(self.s / 2.0) - tp1 - tp2
                dlogdet = torch.sum(tp, axis=[1], keepdim=True)
                return x, logdet + dlogdet

        else:
            x = (torch.tanh(x / (self.s_init + self.s)) + 1.0) / 2.0

            if logdet is not None:
                x = torch.clip(x, 1.0e-10, 1.0 - 1.0e-10)
                tp = torch.log(x) + torch.log(1 - x) + torch.log(2.0 / self.s)
                dlogdet = torch.sum(tp, axis=[1], keepdim=True)
                return x, logdet + dlogdet

        return x


## bounded support mapping: logistic mapping from (-inf, inf) to (0,1), affine linear (lb,hb) to (0,1)
## this is for resample of PDE domain
class Bounded_support_mapping(nn.Module):
    def __init__(self, input_dim, lb, hb, **kwargs):
        super(Bounded_support_mapping, self).__init__(**kwargs)
        self.lb = lb
        self.hb = hb
        self.n_length = input_dim
        self.logistic_layer = Logistic_mapping()
        self.affine_linear_layer = Affine_linear_mapping(self.lb, self.hb)

    def forward(self, inputs, logdet=None, reverse=False):
        x = inputs

        if not reverse:
            x = self.affine_linear_layer(x, logdet)
            if logdet is not None:
                x, logdet = x
            x = self.logistic_layer(x, logdet)
            if logdet is not None:
                x, logdet = x

        else:
            x = self.logistic_layer(x, logdet, reverse=True)
            if logdet is not None:
                x, logdet = x
            x = self.affine_linear_layer(x, logdet, reverse=True)
            if logdet is not None:
                x, logdet = x

        if logdet is not None:
            return x, logdet

        return x


# mapping defined by a piecewise quadratic cumulative distribution function (CDF)
# Assume that each dimension has a compact support [0,1]
# CDF(x) maps [0,1] to [0,1], where the prior uniform distribution is defined.
# Since x is defined on (-inf,+inf), we only consider a CDF() mapping from
# the interval [-bound, bound] to [-bound, bound], and leave alone other points.
# The reason we do not consider a mapping from (-inf,inf) to (0,1) is the
# singularity induced by the mapping.
class CDF_quadratic(nn.Module):
    def __init__(self, input_dim, n_bins, r=1.2, bound=30.0, beta=1e-8, **kwargs):
        super(CDF_quadratic, self).__init__(**kwargs)

        assert n_bins % 2 == 0

        self.n_bins = n_bins

        # generate a nonuniform mesh symmetric to zero,
        # and increasing by ratio r away from zero.
        self.bound = bound
        self.r = r
        self.beta = beta

        m = n_bins/2
        x1L = bound*(r-1.0)/(math.pow(r, m)-1.0)

        index = torch.reshape(torch.arange(0, self.n_bins + 1, dtype=torch.float32),(-1,1))
        index -= m
        xr = torch.where(index>=0, (1.-torch.pow(r, index))/(1.-r),
                      (1.-torch.pow(r,torch.abs(index)))/(1.-r))
        xr = torch.where(index>=0, x1L*xr, -x1L*xr)
        xr = torch.reshape(xr,(-1,1))
        xr = (xr + bound)/2.0/bound
        self.x1L = x1L/2.0/bound
        self.mesh = torch.concat([torch.reshape(torch.Tensor([0.0]),(-1,1)), torch.reshape(xr[1:-1,0],(-1,1)), torch.reshape(torch.Tensor([1.0]),(-1,1))],0)
        self.elmt_size = torch.reshape(self.mesh[1:] - self.mesh[:-1],(-1,1))
        self.n_length = input_dim

        self.p = nn.Parameter(torch.zeros(self.n_bins - 1, self.n_length, dtype=torch.float32))

    def forward(self, inputs, logdet=None, reverse=False):
        # normalize the PDF
        self._pdf_normalize()

        x = inputs
        if not reverse:
            # for the interval [-a,a]
            # rescale such points in [-bound, bound] will be mapped to [0,1]
            x = (x + self.bound) / 2.0 / self.bound

            # cdf mapping
            x = self._cdf(x, logdet)
            if logdet is not None:
                x, logdet = x

            # maps [0,1] back to [-bound, bound]
            x = x * 2.0 * self.bound - self.bound

            #for the interval (a,inf)
            x = torch.where( x > self.bound, self.beta*(x-self.bound)+self.bound, x)
            if logdet is not None:
                dlognet = x
                dlogdet = torch.where(dlognet > self.bound, self.beta, 1.0)
                dlogdet = torch.sum(torch.log(dlogdet), axis=[1], keepdim=True)
                logdet += dlogdet

            #for the interval (-inf,a)
            x = torch.where( x < -self.bound, self.beta*(x+self.bound)-self.bound, x)
            if logdet is not None:
                dlognet = x
                dlogdet = torch.where(dlognet < -self.bound, self.beta, 1.0)
                dlogdet = torch.sum(torch.log(dlogdet), axis=[1], keepdim=True)
                logdet += dlogdet
        else:
            # for the interval [-a,a]
            # rescale such points in [-bound, bound] will be mapped to [0,1]
            x = (x + self.bound) / 2.0 / self.bound

            # cdf mapping
            x = self._cdf_inv(x, logdet)
            if logdet is not None:
                x, logdet = x

            # maps [0,1] back to [-bound, bound]
            x = x * 2.0 * self.bound - self.bound

            #for the interval (a,inf)
            x = torch.where( x > self.bound, (x-self.bound)/self.beta+self.bound, x)
            if logdet is not None:
                dlognet = x
                dlogdet = torch.where(dlognet > self.bound, 1.0/self.beta, 1.0)
                dlogdet = torch.sum(torch.log(dlogdet), axis=[1], keepdim=True)
                logdet += dlogdet

            #for the interval (-inf,a)
            x = torch.where( x < -self.bound, (x+self.bound)/self.beta-self.bound, x)
            if logdet is not None:
                dlognet = x
                dlogdet = torch.where(dlognet < -self.bound, 1.0/self.beta, 1.0)
                dlogdet = torch.sum(torch.log(dlogdet), axis=[1], keepdim=True)
                logdet += dlogdet

        if logdet is not None:
            return x, logdet

        return x

    # normalize the piecewise representation of pdf
    def _pdf_normalize(self):
        # peicewise pdf
        p0 = torch.ones((1, self.n_length), dtype=torch.float32) * self.beta
        self.pdf = p0
        px = torch.exp(self.p) * (self.elmt_size[:-1] + self.elmt_size[1:]) / 2.0
        px = (1.0 - (self.elmt_size[0] + self.elmt_size[-1]) * self.beta / 2.0) / torch.sum(px, 0, keepdim=True)
        px = px * torch.exp(self.p)
        self.pdf = torch.concat([self.pdf, px], 0)
        self.pdf = torch.concat([self.pdf, p0], 0)

        # probability in each element
        cell = (self.pdf[:-1, :] + self.pdf[1:, :]) / 2.0 * self.elmt_size
        # CDF - contribution from previous elements.
        r_zeros = torch.zeros((1, self.n_length), dtype=torch.float32)
        self.F_ref = r_zeros
        for i in range(1, self.n_bins):
            tp = torch.sum(cell[:i, :], 0, keepdim=True)
            self.F_ref = torch.concat([self.F_ref, tp], 0)

    # the cdf is a piecewise quadratic function.
    def _cdf(self, x, logdet=None):
        """"""""
        xr = torch.tile(self.mesh, [1, self.n_length])
        k_ind = torch.searchsorted(torch.transpose(xr,1,0), torch.transpose(x,1,0), side='right')
        k_ind = torch.transpose(k_ind,1,0)
        k_ind = k_ind.type(torch.LongTensor)
        k_ind -= 1


        cover = torch.where(k_ind * (k_ind - self.n_bins + 1) <= 0, 1.0, 0.0)

        k_ind = torch.where(k_ind < 0, 0, k_ind)
        k_ind = torch.where(k_ind > (self.n_bins - 1), self.n_bins - 1, k_ind)

        v1 = torch.reshape(torch.gather(self.pdf[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.n_length):
            tp = torch.reshape(torch.gather(self.pdf[:, i], 0, k_ind[:, i]), (-1, 1))
            v1 = torch.concat([v1, tp], 1)

        v2 = torch.reshape(torch.gather(self.pdf[:, 0],  0, k_ind[:, 0] + 1), (-1, 1))
        for i in range(1, self.n_length):
            tp = torch.reshape(torch.gather(self.pdf[:, i], 0, k_ind[:, i] + 1), (-1, 1))
            v2 = torch.concat([v2, tp], 1)

        xmodi = torch.reshape(x[:, 0] - torch.gather(self.mesh[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.n_length):
            tp = torch.reshape(x[:, i] - torch.gather(self.mesh[:, 0], 0, k_ind[:, i]), (-1, 1))
            xmodi = torch.concat([xmodi, tp], 1)

        h_list = torch.reshape(torch.gather(self.elmt_size[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.n_length):
            tp = torch.reshape(torch.gather(self.elmt_size[:, 0], 0, k_ind[:, i]), (-1, 1))
            h_list = torch.concat([h_list, tp], 1)

        F_pre = torch.reshape(torch.gather(self.F_ref[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.n_length):
            tp = torch.reshape(torch.gather(self.F_ref[:, i], 0, k_ind[:, i]), (-1, 1))
            F_pre = torch.concat([F_pre, tp], 1)

        y = torch.where(cover > 0, F_pre + xmodi ** 2 / 2.0 * (v2 - v1) / h_list + xmodi * v1, x)

        if logdet is not None:
            dlogdet = torch.where(cover > 0, xmodi * (v2 - v1) / h_list + v1, 1.0)
            dlogdet = torch.sum(torch.log(dlogdet), axis=[1], keepdim=True)
            return y, logdet + dlogdet

        return y

    # inverse of the cdf
    def _cdf_inv(self, y, logdet=None):
        xr = torch.tile(self.mesh, [1, self.n_length])
        yr1 = self._cdf(xr)

        p0 = torch.zeros((1, self.n_length), dtype=torch.float32)
        p1 = torch.ones((1, self.n_length), dtype=torch.float32)
        yr = torch.concat([p0, yr1[1:-1, :], p1], 0)

        k_ind = torch.searchsorted(torch.transpose(yr), torch.transpose(y), side='right')
        k_ind = torch.transpose(k_ind)
        k_ind = k_ind.type(torch.IntTensor)
        k_ind -= 1

        cover = torch.where(k_ind * (k_ind - self.n_bins + 1) <= 0, 1.0, 0.0)

        k_ind = torch.where(k_ind < 0, 0, k_ind)
        k_ind = torch.where(k_ind > (self.n_bins - 1), self.n_bins - 1, k_ind)

        c_cover = torch.reshape(cover[:, 0], (-1, 1))
        v1 = torch.where(c_cover > 0, torch.reshape(torch.gather(self.pdf[:, 0], 0, k_ind[:, 0]), (-1, 1)), -1.0)
        for i in range(1, self.n_length):
            c_cover = torch.reshape(cover[:, i], (-1, 1))
            tp = torch.where(c_cover > 0, torch.reshape(torch.gather(self.pdf[:, i], 0, k_ind[:, i]), (-1, 1)), -1.0)
            v1 = torch.concat([v1, tp], 1)

        c_cover = torch.reshape(cover[:, 0], (-1, 1))
        v2 = torch.where(c_cover > 0, torch.reshape(torch.gather(self.pdf[:, 0], 0, k_ind[:, 0] + 1), (-1, 1)), -2.0)
        for i in range(1, self.n_length):
            c_cover = torch.reshape(cover[:, i], (-1, 1))
            tp = torch.where(c_cover > 0, torch.reshape(torch.gather(self.pdf[:, i], 0, k_ind[:, i] + 1), (-1, 1)), -2.0)
            v2 = torch.concat([v2, tp], 1)

        ys = torch.reshape(y[:, 0] - torch.gather(yr[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.n_length):
            tp = torch.reshape(y[:, i] - torch.gather(yr[:, i], 0, k_ind[:, i]), (-1, 1))
            ys = torch.concat([ys, tp], 1)

        xs = torch.reshape(torch.gather(xr[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.n_length):
            tp = torch.reshape(torch.gather(xr[:, i], 0, k_ind[:, i]), (-1, 1))
            xs = torch.concat([xs, tp], 1)

        h_list = torch.reshape(torch.gather(self.elmt_size[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.n_length):
            tp = torch.reshape(torch.gather(self.elmt_size[:, 0], 0, k_ind[:, i]), (-1, 1))
            h_list = torch.concat([h_list, tp], 1)

        h_s = (v2 - v1) / h_list
        tp = v1 * v1 + 2.0 * ys * h_s
        tp = torch.sqrt(tp) + v1
        tp = 2.0 * ys / tp
        tp += xs

        x = torch.where(cover > 0, tp, y)

        if logdet is not None:
            tp = 2.0 * ys * h_s
            tp += v1 * v1
            tp = 1.0 / torch.sqrt(tp)

            dlogdet = torch.where(cover > 0, tp, 1.0)
            dlogdet = torch.sum(torch.log(dlogdet), axis=[1], keepdim=True)
            return x, logdet + dlogdet

        return x