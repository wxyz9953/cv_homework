import torch.nn as nn
from torch.nn import init
import torch


class ResBlock(nn.Module):
    def __init__(self, in_features):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(in_features, in_features, 3),
                                   nn.InstanceNorm2d(in_features),
                                   nn.ReLU(inplace=True),
                                   nn.ReflectionPad2d(1),
                                   nn.Conv2d(in_features, in_features, 3),
                                   nn.InstanceNorm2d(in_features))

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels=input_nc, out_channels=64, kernel_size=7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_residual_blocks):
            model += [ResBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, kernel_size=7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        model = [nn.Conv2d(in_channels=input_nc, out_channels=64, kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def define_Gen(input_nc, output_nc, netG, gpu_ids=[0]):
    gen_net = None
    if netG == 'resnet_9blocks':
        gen_net = Generator(input_nc, output_nc, n_residual_blocks=9)
    elif netG == 'resnet_6blocks':
        gen_net = Generator(input_nc, output_nc, n_residual_blocks=6)

    return init_network(gen_net, gpu_ids)


def define_Dis(input_nc, gpu_ids=[0]):
    dis_net = Discriminator(input_nc)
    return init_network(dis_net, gpu_ids)


def init_weights(net, gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)
    net.apply(init_func)


def init_network(net, gpu_ids=[]):
    if torch.cuda.is_available():
        if len(gpu_ids) > 0:
            assert (torch.cuda.is_available())
            net.cuda(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)
        init_weights(net)
    return net


def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad
