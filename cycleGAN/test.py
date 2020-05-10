import os
import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
from networks import define_Gen
from torch.utils.data import DataLoader


def test(args):
    transform = transforms.Compose(
        [transforms.Resize((args.crop_height, args.crop_width)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    dataset_dirs = utils.get_testdata_link(args.dataset_dir)

    a_test_data = dsets.ImageFolder(dataset_dirs['testA'], transform=transform)
    b_test_data = dsets.ImageFolder(dataset_dirs['testB'], transform=transform)

    a_test_loader = DataLoader(a_test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    b_test_loader = DataLoader(b_test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    Gab = define_Gen(input_nc=3, output_nc=3, netG=args.gen_net, gpu_ids=args.gpu_ids)
    Gba = define_Gen(input_nc=3, output_nc=3, netG=args.gen_net, gpu_ids=args.gpu_ids)

    ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
    Gab.load_state_dict(ckpt['Gab'])
    Gba.load_state_dict(ckpt['Gba'])

    a_real_test = iter(a_test_loader).next()[0]
    b_real_test = iter(b_test_loader).next()[0]
    a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])

    Gab.eval()
    Gba.eval()

    with torch.no_grad():
        a_fake_test = Gab(b_real_test)
        b_fake_test = Gba(a_real_test)
        a_recon_test = Gab(b_fake_test)
        b_recon_test = Gba(a_fake_test)

    pic = (torch.cat([a_real_test, b_fake_test, a_recon_test, b_real_test, a_fake_test, b_recon_test],
                     dim=0).data + 1) / 2.0

    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    torchvision.utils.save_image(pic, args.results_dir + '/sample.jpg', nrow=3)
