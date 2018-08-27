import os
import glob
import time
import numpy as np
import torch
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from unet import UNet
from fcn import *
from utils import *
from dataset import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'

fcns = {
    64: FCN_64_11,
    112: FCN_112_11,
    224: FCN_224_11,
}


def save_imgs(imgs, idxs=[0], save_path=None):
    assert save_path
    n = len(imgs)
    
    for idx in idxs:
        plt.figure(figsize=(20, 5))
        for i, (k, v) in enumerate(imgs.items()):
            ax = plt.subplot(1, n, i+1)
            if k[0] == 'x':
                ax.imshow(convert_tensor_to_PIL(v[idx]), cmap='gray')
            else:
                sns.heatmap(
                    v[idx][0].data.cpu().numpy(), vmin=0, vmax=1, 
                    cmap='gray', square=True, cbar=False)
            ax.set_title(k)
            ax.set_xticks([])
            ax.set_yticks([])    
    plt.savefig(save_path)


def train(D, G, D_optimizer, G_optimizer, criterion, train_loader, num_epochs, gamma, save_path):    
    # Training
    history = {'D_loss': [], 'G_loss': []}

    D.train()
    G.train()
    D_best_loss = np.inf
    G_best_loss = np.inf

    for epoch in range(num_epochs):
        D_running_loss = 0
        G_running_loss = 0
        _D_real_loss = 0
        _D_fake_loss = 0

        start = time.time()
        for x_real in train_loader:
            D.zero_grad()
            G.zero_grad()
            D_optimizer.zero_grad()
            G_optimizer.zero_grad()

            x_real = x_real.to(device) 

            ### Update Discriminator ###
            # real
            D_real = D(x_real)
            y_real = torch.ones(D_real.size()).to(device)
            y_fake = torch.zeros(D_real.size()).to(device)
            D_real_loss = criterion(D_real, y_real)

            # fake
            eta = torch.randn(x_real.size()).to(device)
            x_noise = x_real + gamma * eta / 127.5
            x_fake = G(x_noise)
            D_fake = D(x_fake.detach())  # detach for computational speed
            D_fake_loss = criterion(D_fake, y_fake)

            # update params
            D_loss = D_real_loss + D_fake_loss
            D_loss.backward()
            D_optimizer.step()
            D_running_loss += D_loss.item()
            # for debug
            _D_real_loss += D_real_loss.item()
            _D_fake_loss += D_fake_loss.item()

            ### Update Generator ###
            eta = torch.randn(x_real.size()).to(device)
            x_noise = x_real + gamma * eta / 127.5
            x_fake = G(x_noise)
            D_fake = D(x_fake)

            # update params
            G_loss = criterion(D_fake, y_real)
            G_loss.backward()
            G_optimizer.step()
            G_running_loss += G_loss.item()

        elapsed_time = (time.time() - start) / 60

        D_running_loss /= len(train_loader)
        G_running_loss /= len(train_loader)
        history['D_loss'].append(D_running_loss)
        history['G_loss'].append(G_running_loss)

        torch.save(D.state_dict(), save_path+'D_{:03d}.pth'.format(epoch))
        torch.save(G.state_dict(), save_path+'G_{:03d}.pth'.format(epoch))

        if D_running_loss < D_best_loss:
            D_best_loss = D_running_loss
            torch.save(D.state_dict(), save_path+'D.pth'.format(epoch))
        if G_running_loss < G_best_loss:
            G_best_loss = G_running_loss
            torch.save(G.state_dict(), save_path+'G.pth')

        print('Epoch {}: {:.1f}min, D_loss: {:.6f}, G_loss: {:.6f}'.format(
            epoch, elapsed_time, D_running_loss, G_running_loss))

        imgs = {
            'x_real': tanh2sigmoid(x_real),
            'x_noise': tanh2sigmoid(x_noise),
            'D_real': D_real,
            'x_fake': tanh2sigmoid(x_fake),
            'D_fake': D_fake,
        }
        idxs = np.random.randint(0, x_real.size(0), [1])
        save_imgs(imgs, idxs, save_path=save_path+'{:03d}.png'.format(epoch))

    torch.save(history, save_path+'history.pth')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--optimizer', type=str, default='Adam', 
                        choices=['Adam', 'SGD'])
    parser.add_argument('--g_lr', type=float, default=2e-4)
    parser.add_argument('--d_lr', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=0.4)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--input_size', type=int, default=112, 
                        choices=[64, 112, 224])
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['IR-MNIST', 'UCSDped1', 'UCSDped2'])
    
    args = parser.parse_args()
    
    DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(DIR, 'outs/' + args.dataset)
    save_path = os.path.join(SAVE_DIR, 'gamma{:.1f}_depth{}/'.format(args.gamma, args.depth))
    os.makedirs(save_path, exist_ok=True)
    print('save_path:', save_path)
    
    if args.dataset == 'IR-MNIST':
        DATA_DIR = os.path.join(DIR, 'data/IR-MNIST/')
        TRAIN_PATTERN = os.path.join(DATA_DIR, 'Train Samples/*.jpg')
        rgb = False
        n_channel = 1
    else:
        DATA_DIR = os.path.join(DIR, 'data/UCSD_processed/' + args.dataset)
        TRAIN_PATTERN = os.path.join(DATA_DIR, 'Train/*.png')
        rgb = True
        n_channel = 3
    train_paths = glob.glob(TRAIN_PATTERN)
    
    train_dataset = Dataset(
        train_paths, (args.input_size, args.input_size), rgb=rgb)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    G = UNet(n_channel, n_channel, depth=args.depth).to(device)
    D = fcns[args.input_size](n_channel).to(device)
    
    G_optimizer = get_optimizer(args.optimizer, G.parameters(), args.g_lr, args.momentum)
    D_optimizer = get_optimizer(args.optimizer, D.parameters(), args.d_lr, args.momentum)

    criterion = l2_BCE    
    
    print('start training')
    train(D, G, D_optimizer, G_optimizer, criterion, train_loader, args.num_epochs, args.gamma, save_path)

    
if __name__ == '__main__':
    main()
