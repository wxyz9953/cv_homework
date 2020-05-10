import itertools

import os
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
from networks import define_Gen, define_Dis, set_grad
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader


class cycleGAN(object):
    def __init__(self, args):

        self.Gab = define_Gen(input_nc=3, output_nc=3, netG=args.gen_net, gpu_ids=args.gpu_ids)
        self.Gba = define_Gen(input_nc=3, output_nc=3, netG=args.gen_net, gpu_ids=args.gpu_ids)
        self.Da = define_Dis(input_nc=3, gpu_ids=args.gpu_ids)
        self.Db = define_Dis(input_nc=3, gpu_ids=args.gpu_ids)

        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gab.parameters(), self.Gba.parameters()), lr=args.lr,
                                            betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Da.parameters(), self.Db.parameters()), lr=args.lr,
                                            betas=(0.5, 0.999))

        self.g_lr_scheduler = lr_scheduler.LambdaLR(self.g_optimizer,
                                                    lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)
        self.d_lr_scheduler = lr_scheduler.LambdaLR(self.d_optimizer,
                                                    lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)

        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
        self.start_epoch = ckpt['epoch']
        self.Da.load_state_dict(ckpt['Da'])
        self.Db.load_state_dict(ckpt['Db'])
        self.Gab.load_state_dict(ckpt['Gab'])
        self.Gba.load_state_dict(ckpt['Gba'])
        self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
        self.g_optimizer.load_state_dict(ckpt['g_optimizer'])

    def train(self, args):
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.Resize((args.load_height, args.load_width)),
             transforms.RandomCrop((args.crop_height, args.crop_width)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        dataset_dirs = utils.get_traindata_link(args.dataset_dir)

        a_loader = DataLoader(dsets.ImageFolder(dataset_dirs['trainA'], transform=transform),
                              batch_size=args.batch_size, shuffle=True, num_workers=4)
        b_loader = DataLoader(dsets.ImageFolder(dataset_dirs['trainB'], transform=transform),
                              batch_size=args.batch_size, shuffle=True, num_workers=4)

        a_fake_sample = utils.Sample_from_Pool()
        b_fake_sample = utils.Sample_from_Pool()

        for epoch in range(self.start_epoch, args.epochs):

            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

            for i, (a_real, b_real) in enumerate(zip(a_loader, b_loader)):
                set_grad([self.Da, self.Db], False)
                self.g_optimizer.zero_grad()

                a_real = a_real[0]
                b_real = b_real[0]
                a_real, b_real = utils.cuda([a_real, b_real])

                a_fake = self.Gab(b_real)
                b_fake = self.Gba(a_real)

                a_recon = self.Gab(b_fake)
                b_recon = self.Gba(a_fake)

                a_idt = self.Gab(a_real)
                b_idt = self.Gba(b_real)

                a_idt_loss = self.L1(a_idt, a_real) * 5.0
                b_idt_loss = self.L1(b_idt, b_real) * 5.0

                a_fake_dis = self.Da(a_fake)
                b_fake_dis = self.Db(b_fake)

                real_label = utils.cuda(torch.ones(a_fake_dis.size()))

                a_gen_loss = self.MSE(a_fake_dis, real_label)
                b_gen_loss = self.MSE(b_fake_dis, real_label)

                a_cycle_loss = self.L1(a_recon, a_real) * 10.0
                b_cycle_loss = self.L1(b_recon, b_real) * 10.0

                gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

                gen_loss.backward()
                self.g_optimizer.step()

                set_grad([self.Da, self.Db], True)
                self.d_optimizer.zero_grad()

                a_fake = torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0])
                b_fake = torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0])
                a_fake, b_fake = utils.cuda([a_fake, b_fake])

                a_real_dis = self.Da(a_real)
                a_fake_dis = self.Da(a_fake)
                b_real_dis = self.Db(b_real)
                b_fake_dis = self.Db(b_fake)
                real_label = utils.cuda(torch.ones(a_real_dis.size()))
                fake_label = utils.cuda(torch.zeros(a_fake_dis.size()))

                a_dis_real_loss = self.MSE(a_real_dis, real_label)
                a_dis_fake_loss = self.MSE(a_fake_dis, fake_label)
                b_dis_real_loss = self.MSE(b_real_dis, real_label)
                b_dis_fake_loss = self.MSE(b_fake_dis, fake_label)

                a_dis_loss = (a_dis_real_loss + a_dis_fake_loss) * 0.5
                b_dis_loss = (b_dis_real_loss + b_dis_fake_loss) * 0.5

                a_dis_loss.backward()
                b_dis_loss.backward()
                self.d_optimizer.step()

                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" %
                      (epoch, i + 1, min(len(a_loader), len(b_loader)),
                       gen_loss, a_dis_loss + b_dis_loss))

            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Da': self.Da.state_dict(),
                                   'Db': self.Db.state_dict(),
                                   'Gab': self.Gab.state_dict(),
                                   'Gba': self.Gba.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict()},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))

            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()
