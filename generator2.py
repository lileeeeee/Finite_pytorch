import argparse
import random
import numpy as np
import math


from torch.autograd import Variable
from scipy.integrate import dblquad
import torch.nn as nn
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--d_epoch', type=int, default=5000,help='The number discriminator updates during one epoch of generator')
parser.add_argument('--g_epoch', type=int, default=50, help='The number generator updates during one epoch of krnet')
parser.add_argument("--n_dim", type=int, default=10, help='The number of random dimension.')
parser.add_argument('--sample', type=int, default=50, help='Train data size.')
parser.add_argument('--test_sample', type=int, default=100000, help='Test data size.')
parser.add_argument('--c', type=float, default=1, help='The covariance is (1+c)*I')
parser.add_argument("--g_lr", type=float, default=0.01, help='Base generator learning rate.')
parser.add_argument("--d_lr", type=float, default=0.01, help='Base discriminator learning rate.')
opt = parser.parse_args()
print(opt)


cuda = True if torch.cuda.is_available() else False

device = torch.device("mps")

if torch.cuda.is_available():
    device = torch.device("cuda")

class Generator(nn.Module):
    def __init__(self, **kwargs):
        super(Generator, self).__init__(**kwargs)

        self.model = nn.Sequential(
            nn.Linear(opt.n_dim - 1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, opt.n_dim - 1),
        )

    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.n_dim, 64),
            nn.LeakyReLU(0.8, inplace=True),
            nn.Linear(64, 256),
            nn.LeakyReLU(0.8, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.8, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.8, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        validity = self.model(x)
        return validity

class test_Discriminator(nn.Module):
    def __init__(self):
        super(test_Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(0.8, inplace=True),
            nn.Linear(64, 256),
            nn.LeakyReLU(0.8, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.8, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.8, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        validity = self.model(x)
        return validity

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)


if cuda:
    generator.cuda()
    discriminator.cuda()

def gen_n_dim_gaussian(n_dim, lam = 1, n_sample = 100000):
    mean = np.zeros(n_dim)
    cov = np.identity(n_dim)
    return np.random.multivariate_normal(mean, lam * cov, n_sample)

def gen_g_data(exp, cov, n_sample=100000):
    ori = np.random.multivariate_normal(exp, cov, n_sample)
    return ori

def inv_sphere_proj(raw_point, n_dim, n_train, radius):
    """inverse stereographic projection
    raw_point: torch.Tensor
    the original point

    n_dim: int
    the dimension of the original data

    n_train: int
    the number of points to be projected

    radius:
    the radius of the sphere
    """

    res = []
    for i in range(n_train):
        tmp = []
        normal = torch.sum(torch.square(raw_point[i]))
        for j in range(n_dim):
            tmp.append(torch.sum((2 * raw_point[i][j] / (normal + 1) * radius)))
        tmp.append(torch.sum(((normal - 1) / (normal + 1) * radius)))
        res.append(torch.stack(tmp))
    res = torch.stack(res)
    return res

def get_d_loss(net, real, fake):
    # return -tf.reduce_mean(real) - tf.reduce_mean(1. - fake)
    return -torch.mean(torch.log(net(real) + 1e-10) + torch.log(1. - net(fake) + 1e-10))

def get_g_loss(fake):
    # return tf.reduce_mean(1. - fake)
    return -torch.mean(torch.log(fake + 1e-10))

def d_real_loss(real, fake):
    """
    calculate the expectation of
        real + 1 - fake

    :param real:
    :param fake:
    :return:
    """
    return torch.mean(real) + torch.mean(1. - fake)

def d_metric(net, ori, contrast):
    # get the probabilities
    ori = net(ori)
    contrast = net(contrast)

    # get D^
    ori = torch.where(ori < 0.5, 0., 1.)
    contrast = torch.where(contrast < 0.5, 0., 1.)

    # print calculated TVD
    metric = d_real_loss(ori, contrast) - 1
    # tf.print("TVD-ori", metric, "TVD-cal", a, "delta", tf.abs(metric - a))
    return metric

def g_metric_base(c, n_dim):
    """
    TV distance 解析解

    c:

    n_dim:
        dimension of the data
    """
    r_sq = n_dim * (1 + c) * np.log(1 + c) / c
    return torch.igamma(torch.tensor(n_dim) / 2, torch.tensor(r_sq) / 2) - torch.igamma(torch.tensor(n_dim) / 2, torch.tensor(r_sq) / ((1 + c) * 2))

def gauss(x, sigma):
    return np.exp(-np.dot(np.dot(x, np.linalg.inv(sigma)), x) / 2) / (2 * np.pi) / np.sqrt(np.linalg.det(sigma))

def cal_numerical_TVD(sigma, i, j, c):
    sig = np.array([[sigma[i][i], sigma[i][j]], [sigma[j][i], sigma[j][j]]])
    print(sig)
    val, err = dblquad(lambda y, x: np.abs(gauss(np.array([x, y]), sig) - gauss(np.array([x, y]), c * sig)) / 2,
                       -np.inf, np.inf, -np.inf, np.inf)
    return val

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.g_lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.d_lr)


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

print(g_metric_base(opt.c, opt.n_dim).item())

# ----------
#  Training
# ----------
u_G = np.random.random(opt.n_dim)
assert (u_G == 0).sum() != 10
radius = math.sqrt((u_G * u_G).sum())
tmp = np.random.random((opt.n_dim, opt.n_dim))
tmp = np.dot(tmp, tmp.transpose())
sig_G = tmp + tmp.T + 4 * np.identity(opt.n_dim)
print(u_G, sig_G)

ori_data = Variable(Tensor(gen_g_data(u_G, sig_G, opt.sample))).to(device)

# Sample noise as generator input
z = Variable(Tensor(gen_n_dim_gaussian(opt.n_dim - 1, 1, 1)))
z = torch.tile(z, (opt.sample, 1)).to(device)
for g_epoch in range(opt.g_epoch):

    print("---------------g_epoch",g_epoch,"--------------------")
    optimizer_G.zero_grad()
    sphPoints = generator(z)
    sphPoints = inv_sphere_proj(sphPoints, opt.n_dim - 1, opt.sample, radius)
    print(sphPoints[0])
    fake_data = sphPoints + Variable(Tensor(np.random.multivariate_normal(np.zeros(opt.n_dim), (opt.c + 1) * sig_G, opt.sample))).to(device)
    # print(fake_data)

    # Loss measures generator's ability to fool the discriminator

    # ---------------------
    #  Train Discriminator
    # ---------------------
    prev_metric = 1000.
    for d_epoch in range(opt.d_epoch):
        optimizer_D.zero_grad()

        d_loss = get_d_loss(discriminator, ori_data, fake_data.detach())
        d_loss.backward()
        optimizer_D.step()
        train_metric = d_metric(discriminator, ori_data, fake_data)
        print(train_metric)
        if (torch.abs_((train_metric - prev_metric)) < 1e-4):
            print(train_metric)
            break
        prev_metric = train_metric

    # -----------------
    #  Train Generator
    # -----------------
    g_loss = get_g_loss(discriminator(fake_data))
    g_loss.backward()
    optimizer_G.step()

point = inv_sphere_proj(generator(z), opt.n_dim - 1, 1, radius)
point = torch.tile(point, (opt.test_sample, 1))
print("ug", u_G, "point", point[0].detach().cpu().numpy(), "norm", torch.sqrt(torch.sum(torch.square(Tensor(u_G - point[0].detach().cpu().numpy())))).numpy())
# --------------------------
# Test
# --------------------------
print("------------------------------Test-result-----------------------------------------")
for test in range(6):
    dim_1 = random.randint(0, 9)
    dim_2 = random.randint(0, 9)
    while dim_1 == dim_2:
        dim_2 = random.randint(0, 9)
    TVD_num = cal_numerical_TVD(sig_G, dim_1, dim_2, opt.c + 1)
    print("trial", test, "dim", dim_1, dim_2, "TVD", TVD_num)
    test_dis = test_Discriminator().to(device)
    optimizer_testD = torch.optim.Adam(test_dis.parameters(), lr=opt.d_lr)
    for i in range(50):
        test_ori_data = Variable(Tensor(gen_g_data(u_G, sig_G, opt.test_sample))).to(device)
        test_fake_data = point + Variable(Tensor(np.random.multivariate_normal(np.zeros(opt.n_dim), (opt.c + 1) * sig_G, opt.test_sample))).to(device)

        test_ori = torch.cat((test_ori_data[:, dim_1:dim_1 + 1], test_ori_data[:, dim_2:dim_2 + 1]), dim=1)
        test_fake = torch.cat((test_fake_data[:, dim_1:dim_1 + 1], test_fake_data[:, dim_2:dim_2 + 1]), dim=1)
        # print(test_ori, test_fake)

        prev_metric = 1000.
        for d_epoch in range(opt.d_epoch):
            optimizer_testD.zero_grad()

            d_loss = get_d_loss(test_dis, test_ori, test_fake.detach())
            d_loss.backward()
            optimizer_testD.step()
            train_metric = d_metric(test_dis, test_ori, test_fake)
            if (torch.abs_((train_metric - prev_metric)) < 1e-4):
                print(train_metric.item())
                break
            prev_metric = train_metric

