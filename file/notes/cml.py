import torch
import collections
import numpy as np
from torch import nn
from torch.nn.functional import mse_loss

class Data(object):
    # for storing subsamples
    def __init__(self, y, z, x):
        self.y = y
        self.z = z
        self.x = x

class DNN(nn.Module):
    # basic model
    def __init__(self, d_in, d_out, hid_arch):
        super(DNN, self).__init__()
        modules = []
        modules.append(('layer_in', nn.Linear(d_in, hid_arch[1])))
        modules.append(('act_0',nn.ReLU()))
        for i in range(hid_arch[0]):
            modules.append((f'layer_{i}', nn.Linear(hid_arch[1], hid_arch[1])))
            modules.append((f'act_{i}', nn.ReLU()))
        modules.append((f'layer_out', nn.Linear(hid_arch[1], d_out)))
        self.net = nn.Sequential(collections.OrderedDict(modules))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                
    def forward(self, x):
        return self.net(x)

def c_dnn(data, hid_arch=(20, 20), n_epoch=1000, lr=0.01):
    # first stage
    y, z, x = data.y, data.z, data.x
    d_in = x.shape[1]
    d_out = z.shape[1] + 1
    model = DNN(d_in, d_out, hid_arch)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_store = []
    for i in range(n_epoch):
        optimizer.zero_grad()
        ab = model(x)
        alpha, beta = ab[:, 0, None], ab[:, 1:]
        y_pred = alpha + (beta*z).sum(-1, keepdim=True)
        loss = mse_loss(y_pred, y, reduction='mean')
        loss.backward()
        optimizer.step()
        loss_store.append(loss.item())
    y_pred = alpha + (beta*z).sum(-1, keepdim=True)
    # using "sum" trick to compute gradients
    loss = 0.5*mse_loss(y_pred, y, reduction='sum')
    abg = torch.autograd.grad(loss, ab, retain_graph=True, create_graph=True)
    abg_sum = abg[0].sum(0)
    h = []
    for i in range(abg[0].shape[1]):
        h.append(torch.autograd.grad(abg_sum[i], ab, retain_graph=True)[0])
    hess = torch.cat(h, -1)
    return {'model': model, 'ab': ab, 'loss_store': loss_store, 'hess': hess}

def lin_dnn(y, x, hid_arch, n_epoch, lr=0.01):
    # linear regression y = f(x) + e, f(x) is approximated by a DNN
    d_in = x.shape[1]
    d_out = y.shape[1]
    model = DNN(d_in, d_out, hid_arch)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_store = []
    for i in range(n_epoch):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = mse_loss(y_pred, y, reduction='mean')
        loss.backward()
        optimizer.step()
        loss_store.append(loss.item())
    return {'model': model, 'y_pred': y_pred, 'loss_store': loss_store}

def proj_hess(data, model, hid_arch=(5, 5), n_epoch=200):
    # second stage
    y, z, x = data.y, data.z, data.x
    d_in = x.shape[1]
    d_z = z.shape[1]
    d_out = d_z + 1
    ab = model(x)
    alpha, beta = ab[:, 0, None], ab[:, 1:]
    alpha.retain_grad()
    beta.retain_grad()
    ab.retain_grad()
    y_pred = alpha + (beta*z).sum(-1, keepdim=True)
    # "sum" trick
    loss = 0.5*mse_loss(y_pred, y, reduction='sum')
    abg = torch.autograd.grad(loss, ab, retain_graph=True, create_graph=True)
    abg_sum = abg[0].sum(0)
    h = []
    for i in range(abg[0].shape[1]):
        h.append(torch.autograd.grad(abg_sum[i], ab, retain_graph=True)[0])
    hess = torch.cat(h, -1)
    # projects Hessian of loss function on to x
    p_hess = []
    for i in range(hess.shape[1]):
        Hy = hess[:, i, None]
        p_hess.append(lin_dnn(Hy, x, hid_arch, n_epoch))
    return p_hess

# compute IF for each split and stack
# what statistic are we interested in
# here we care about ATE = (E[H]=E(CATE)=E(ab[:,1]))
def stat(ab, data=None):
    return ab[:, 1, None]

def proc_res(data, model, proj, stat):
    # thrid stage
    y, z, x = data.y, data.z, data.x
    d_in = x.shape[1]
    d_z = z.shape[1]
    d_out = d_z + 1
    ab = model['model'](x)
    alpha, beta = ab[:, 0, None], ab[:, 1:]
    alpha.retain_grad()
    beta.retain_grad()
    ab.retain_grad()
    y_pred = alpha + (beta*z).sum(-1, keepdim=True)
    # "sum" trick
    loss = 0.5*mse_loss(y_pred, y, reduction='sum')
    abg = torch.autograd.grad(loss, ab, retain_graph=True, create_graph=True)[0].detach().numpy()
    hess_list = []
    for i in range(len(proj)):
        hess_list.append(proj[i]['model'](x))
    hess = torch.cat(hess_list, 1).detach().numpy()
    hi = stat(ab)
    hab = torch.autograd.grad(hi.sum(), ab)[0].detach().numpy()
    V = np.apply_along_axis(lambda x: np.linalg.pinv(x.reshape(2,2)), 1, hess)
    plugin = hi.detach().numpy()
    auto_if = plugin + np.einsum('...ij,...jk,...kh->...ih', hab[:, None, :], V, abg[:, :, None])[:, 0, :]
    return {'auto.if': auto_if, 'plugin': plugin}