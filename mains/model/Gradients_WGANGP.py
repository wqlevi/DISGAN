import torch
from torch import nn


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1, 1) # create 5D tensor for coefficient \lambda as mixing dataset
    if torch.cuda.is_available():
        alpha = alpha.cuda()
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size())
    if torch.cuda.is_available():
        fake = fake.cuda()
        
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def get_grads_D_WAN(net):
    top = 0
    bottom = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            # Hardcoded param name, subject to change of the network
            if name == 'model.0.weight':
                top = param.grad.abs().mean()
                #print (name + str(param.grad))
            # Hardcoded param name, subject to change of the network
            if name == 'model.23.weight':
                bottom = param.grad.abs().mean()
                #print (name + str(param.grad))
    return top, bottom

def get_grads_G(net):
    top = 0
    bottom = 0
    #torch.set_printoptions(precision=10)
    #torch.set_printoptions(threshold=50000)
    for name, param in net.named_parameters():
        if param.requires_grad:
            # Hardcoded param name, subject to change of the network
            if name == 'conv1.weight':
                top = param.grad.abs().mean()
                #print (name + str(param.grad))
            # Hardcoded param name, subject to change of the network
            if name == 'conv3.2.weight':
                bottom = param.grad.abs().mean()
                #print (name + str(param.grad))
    return top, bottom
