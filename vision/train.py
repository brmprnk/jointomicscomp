import os
import sys
from datetime import datetime

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F

from model import MVAE
import datasets as datasets

import numpy as np
import matplotlib.pyplot as plt

N_MODALITIES = 3


def elbo_loss(recon_image, image, recon_gray, gray, mu, logvar, annealing_factor=1.):
    BCE = 0
    if recon_image is not None and image is not None:
        # recon_image, image = recon_image.view(-1, 3 * 64 * 64), image.view(-1, 3 * 64 * 64)
        image_BCE = torch.sum(binary_cross_entropy_with_logits(recon_image, image), dim=1)
        BCE += image_BCE

    if recon_gray is not None and gray is not None:
        # recon_gray, gray = recon_gray.view(-1, 1 * 64 * 64), gray.view(-1, 1 * 64 * 64)
        gray_BCE = torch.sum(binary_cross_entropy_with_logits(recon_gray, gray), dim=1)
        BCE += gray_BCE

    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    # NOTE: we use lambda_i = 1 for all i since each modality is roughly equal
    ELBO = torch.mean(BCE / float(N_MODALITIES) + annealing_factor * KLD)
    return ELBO


def binary_cross_entropy_with_logits(input, target):
    """Sigmoid Activation + Binary Cross Entropy

    @param input: torch.Tensor (size N)
    @param target: torch.Tensor (size N)
    @return loss: torch.Tensor (size N)
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(
            target.size(), input.size()))

    return (torch.clamp(input, 0) - input * target
            + torch.log(1 + torch.exp(-torch.abs(input))))


def loss_function(recon_rna, rna, recon_gcn, gcn, recon_dna, dna, mu, log_var, kld_weight) -> dict:
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}

    :return:
    """
    # Reconstruction loss
    recons_loss = 0
    if recon_rna is not None and rna is not None:
        recons_loss += F.mse_loss(recon_rna, rna)
    if recon_gcn is not None and gcn is not None:
        recons_loss += F.mse_loss(recon_gcn, gcn)
    if recon_dna is not None and dna is not None:
        recons_loss += F.mse_loss(recon_dna, dna)

    recons_loss /= float(N_MODALITIES)  # Account for number of modalities
    # print("recons_loss", recons_loss)

    # KLD Loss
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    # print("kld loss", kld_loss)

    # Loss
    loss = recons_loss + kld_weight * kld_loss
    return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}


def reconstruction_loss_function(key, loss_meter, recon_rna, rna, recon_gcn, gcn, recon_dna, dna) -> None:
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}

    :return:
    """
    if key == "rna":
        rna_recon_rna = F.mse_loss(recon_rna, rna)
        rna_recon_gcn = F.mse_loss(recon_gcn, gcn)
        rna_recon_dna = F.mse_loss(recon_dna, dna)
        loss_meter.modality_loss(rna_recon_rna.detach().numpy(), "rna_rna")
        loss_meter.modality_loss(rna_recon_gcn.detach().numpy(), "rna_gcn")
        loss_meter.modality_loss(rna_recon_dna.detach().numpy(), "rna_dna")

    if key == "gcn":
        gcn_recon_gcn = F.mse_loss(recon_gcn, gcn)
        gcn_recon_rna = F.mse_loss(recon_rna, rna)
        gcn_recon_dna = F.mse_loss(recon_dna, dna)
        loss_meter.modality_loss(gcn_recon_gcn.detach().numpy(), "gcn_gcn")
        loss_meter.modality_loss(gcn_recon_rna.detach().numpy(), "gcn_rna")
        loss_meter.modality_loss(gcn_recon_dna.detach().numpy(), "gcn_dna")

    if key == "dna":
        dna_recon_dna = F.mse_loss(recon_dna, dna)
        dna_recon_rna = F.mse_loss(recon_rna, rna)
        dna_recon_gcn = F.mse_loss(recon_gcn, gcn)
        loss_meter.modality_loss(dna_recon_dna.detach().numpy(), "dna_dna")
        loss_meter.modality_loss(dna_recon_rna.detach().numpy(), "dna_rna")
        loss_meter.modality_loss(dna_recon_gcn.detach().numpy(), "dna_gcn")

    return


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name
        self.epochs = 50
        self.values = []
        self.reconstruct_losses = {
            "average" : [],
            "rna_rna" : [],
            "gcn_gcn" : [],
            "dna_dna" : [],
            "rna_gcn" : [],
            "rna_dna" : [],
            "gcn_rna": [],
            "gcn_dna": [],
            "dna_rna": [],
            "dna_gcn": [],
        }

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.values.append(self.avg.item())

    def modality_loss(self, val, key):
        self.reconstruct_losses[key].append(val)


def save_checkpoint(state, lowest_loss, save_dir):
    """
    Saves a Pytorch model's state, and also saves it to a separate object if it is the best model (lowest loss) thus far

    @param state:       Python dictionary containing the model's state
    @param lowest_loss: Boolean check if the current checkpoint has had the best (lowest) loss so far
    @param save_dir:      String of the folder to save the model to
    @param filename:    String of the model's output name
    @return: None
    """
    # Save checkpoint
    # torch.save(state, os.path.join(save_dir, filename))

    # If this is the best checkpoint (lowest loss) thus far, copy this model to a file named model_best
    if lowest_loss:
        print("Best epoch thus far (lowest loss) --> Saving to model_best")
        torch.save(state, os.path.join(save_dir, 'best_model.pth.tar'))


def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = MVAE(checkpoint['latent_dim'])
    model.load_state_dict(checkpoint['state_dict'])
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default="",
                        help='Name of the experiment being conducted for saving purposes')
    parser.add_argument('--n-latents', type=int, default=128,
                        help='size of the latent embedding (default: 250)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 50)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--annealing-epochs', type=int, default=2, metavar='N',
                        help='number of epochs to anneal KL for [default: 20]')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # random seed
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Current Time for output files
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")

    save_dir = './results/PoE {} {}'.format(args.experiment, dt_string)
    os.makedirs(save_dir)

    # Fetch Datasets
    tcga_data = datasets.TCGAData(save_dir=save_dir)
    train_dataset = tcga_data.get_data_partition("train")
    val_dataset = tcga_data.get_data_partition("val")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)  # (1 batch)

    total_batches = len(train_loader)
    total_val_batches = len(val_loader)  # Should be 1

    # Setup and log model
    model = MVAE(args.n_latents, use_cuda=args.cuda)

    # Log Data shape, input arguments and model
    model_file = open("{}/Model {}.txt".format(save_dir, dt_string), "a")
    model_file.write("Running at {}\n".format(dt_string))
    model_file.write("Input shape : {}, 3000\n".format(len(train_loader.dataset)))
    model_file.write("Input args : {}\n".format(args))
    model_file.write("PoE Model : {}".format(model))
    model_file.close()

    # Preparation for training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.cuda:
        model.cuda()

    def train(epoch, train_loss_meter, train_recon_loss_meter, train_kld_loss_meter):
        model.train()

        progress_bar = tqdm(total=len(train_loader))
        for batch_idx, (rna, gcn, dna) in enumerate(train_loader):

            # if epoch < args.annealing_epochs:
            #     # compute the KL annealing factor for the current mini-batch in the current epoch
            #     annealing_factor = (float(batch_idx + (epoch - 1) * total_batches + 1) /
            #                         float(args.annealing_epochs * total_batches))
            # else:
            #     # by default the KL annealing factor is unity
            #     annealing_factor = 1.0

            # refresh the optimizer
            optimizer.zero_grad()

            kld_weight = len(rna) / len(train_loader.dataset) # Account for the minibatch samples from the dataset

            # compute reconstructions using all the modalities
            (joint_recon_rna, joint_recon_gcn, joint_recon_dna, joint_mu, joint_logvar) = model(rna, gcn, dna)

            # compute reconstructions using each of the individual modalities
            (rna_recon_rna, rna_recon_gcn, rna_recon_dna, rna_mu, rna_logvar) = model(rna=rna)

            (gcn_recon_rna, gcn_recon_gcn, gcn_recon_dna, gcn_mu, gcn_logvar) = model(gcn=gcn)

            (dna_recon_rna, dna_recon_gcn, dna_recon_dna, dna_mu, dna_logvar) = model(dna=dna)

            # Compute joint train loss
            joint_train_loss = loss_function(joint_recon_rna, rna,
                                             joint_recon_gcn, gcn,
                                             joint_recon_dna, dna,
                                             joint_mu, joint_logvar, kld_weight)

            # compute loss with single modal inputs
            rna_train_loss = loss_function(rna_recon_rna, rna,
                                           rna_recon_gcn, gcn,
                                           rna_recon_dna, dna,
                                           rna_mu, rna_logvar, kld_weight)

            gcn_train_loss = loss_function(gcn_recon_rna, rna,
                                           gcn_recon_gcn, gcn,
                                           gcn_recon_dna, dna,
                                           gcn_mu, gcn_logvar, kld_weight)

            dna_train_loss = loss_function(dna_recon_rna, rna,
                                           dna_recon_gcn, gcn,
                                           dna_recon_dna, dna,
                                           dna_mu, dna_logvar, kld_weight)

            train_loss = joint_train_loss['loss'] + rna_train_loss['loss'] + gcn_train_loss['loss'] + dna_train_loss['loss']
            train_recon_loss = joint_train_loss['Reconstruction_Loss'] + rna_train_loss['Reconstruction_Loss'] + gcn_train_loss['Reconstruction_Loss'] + dna_train_loss['Reconstruction_Loss']
            train_kld_loss = joint_train_loss['KLD'] + rna_train_loss['KLD'] + gcn_train_loss['KLD'] + dna_train_loss['KLD']

            train_loss_meter.update(train_loss, len(rna))
            train_recon_loss_meter.update(train_recon_loss, len(rna))
            train_kld_loss_meter.update(train_kld_loss, len(rna))

            # ELBO
            # # compute joint loss
            # joint_train_loss = elbo_loss(joint_recon_rna, rna,
            #                              joint_recon_gcn, gcn,
            #                              joint_mu, joint_logvar,
            #                              annealing_factor=annealing_factor)
            #
            # # compute loss with single modal inputs
            # rna_train_loss = elbo_loss(rna_recon_rna, rna,
            #                              rna_recon_gcn, gcn,
            #                              rna_mu, rna_logvar,
            #                              annealing_factor=annealing_factor)
            #
            # gcn_train_loss = elbo_loss(gcn_recon_rna, rna,
            #                             gcn_recon_gcn, gcn,
            #                             gcn_mu, gcn_logvar,
            #                             annealing_factor=annealing_factor)
            #
            # train_loss = joint_train_loss + rna_train_loss + gcn_train_loss
            #
            # train_loss_meter.update(train_loss.item(), len(rna))

            # compute and take gradient step
            train_loss.backward()
            optimizer.step()

            # if batch_idx % args.log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(rna), len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader), train_loss_meter.avg))

            progress_bar.update()

        progress_bar.close()
        print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, train_loss_meter.avg))
        print('====> Epoch: {}\tReconstruction Loss: {:.4f}'.format(epoch, train_recon_loss_meter.avg))
        print('====> Epoch: {}\tKLD Loss: {:.4f}'.format(epoch, train_kld_loss_meter.avg))


    def test(epoch, val_loss_meter, val_recon_loss_meter, val_kld_loss_meter):
        model.eval()
        test_loss = 0

        for batch_idx, (rna, gcn, dna) in enumerate(val_loader):

            #
            # if epoch < args.annealing_epochs:
            #     # compute the KL annealing factor for the current mini-batch in the current epoch
            #     annealing_factor = (float(batch_idx + (epoch - 1) * total_batches + 1) /
            #                         float(args.annealing_epochs * total_batches))
            # else:
            #     # by default the KL annealing factor is unity
            #     annealing_factor = 1.0

            # compute reconstructions using each of the individual modalities (for results)
            (rna_recon_rna, rna_recon_gcn, rna_recon_dna, rna_mu, rna_logvar) = model(rna=rna)

            (gcn_recon_rna, gcn_recon_gcn, gcn_recon_dna, gcn_mu, gcn_logvar) = model(gcn=gcn)

            (dna_recon_rna, dna_recon_gcn, dna_recon_dna, dna_mu, dna_logvar) = model(dna=dna)

            reconstruction_loss_function("rna", val_recon_loss_meter, rna_recon_rna, rna,
                                         rna_recon_gcn, gcn,
                                         rna_recon_dna, dna)

            reconstruction_loss_function("gcn", val_recon_loss_meter, gcn_recon_rna, rna,
                                         gcn_recon_gcn, gcn,
                                         gcn_recon_dna, dna)

            reconstruction_loss_function("dna", val_recon_loss_meter, dna_recon_rna, rna,
                                         dna_recon_gcn, gcn,
                                         dna_recon_dna, dna)

            # for ease, only compute the joint loss in validation
            (joint_recon_rna, joint_recon_gcn, joint_recon_dna, joint_mu, joint_logvar) = model(rna, gcn)

            kld_weight = len(rna) / len(val_loader.dataset)  # Account for the minibatch samples from the dataset

            # Compute joint loss
            joint_test_loss = loss_function(joint_recon_rna, rna,
                                            joint_recon_gcn, gcn,
                                            joint_recon_dna, dna,
                                            joint_mu, joint_logvar, kld_weight)

            val_loss_meter.update(joint_test_loss['loss'], len(rna))
            val_recon_loss_meter.update(joint_test_loss['Reconstruction_Loss'], len(rna))
            val_kld_loss_meter.update(joint_test_loss['KLD'], len(rna))

            test_loss += joint_test_loss['loss']

        print('====> Test Loss: {:.4f}'.format(test_loss))
        print('====> Test Loss: {}\tLoss: {:.4f}'.format(epoch, val_loss_meter.avg))
        print('====> Test Loss: {}\tReconstruction Loss: {:.4f}'.format(epoch, val_recon_loss_meter.avg))
        print('====> Test Loss: {}\tKLD Loss: {:.4f}'.format(epoch, val_kld_loss_meter.avg))
        return val_loss_meter.avg


    train_loss_meter = AverageMeter("Loss")
    train_recon_loss_meter = AverageMeter("Reconstruction Loss")
    train_kld_loss_meter = AverageMeter("KLD Loss")

    val_loss_meter = AverageMeter("Validation Loss")
    val_recon_loss_meter = AverageMeter("Validation Reconstruction Loss")
    val_kld_loss_meter = AverageMeter("Validation KLD Loss")
    for epoch in range(1, args.epochs + 1):
        train(epoch, train_loss_meter, train_recon_loss_meter, train_kld_loss_meter)
        latest_loss = test(epoch, val_loss_meter, val_recon_loss_meter, val_kld_loss_meter)

        # save the best model and current model
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_loss': latest_loss,
            'latent_dim': args.n_latents,
            'optimizer': optimizer.state_dict(),
        }, True, save_dir)

    # Save all reconstruction losses
    modal = ['rna', 'gcn', 'dna']
    for modal1 in modal:
        for modal2 in modal:
            key = "{}_{}".format(modal1, modal2)
            np.save("{}/Recon array {}.npy".format(save_dir, key),
                    np.array(val_recon_loss_meter.reconstruct_losses[key]))

    # Do some plotting
    x_axis = [*range(1, args.epochs + 1)]

    # Loss
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    plt.plot(x_axis, train_loss_meter.values[::total_batches], marker='.', color='tab:purple')
    for x, y in zip(x_axis, train_loss_meter.values[::total_batches]):
        if x == 1 or x % 25 == 0:
            ax1.annotate('  ({}, {:.4f})'.format(x, y), xy=(x, y), textcoords='data', fontsize=9)
    plt.title("Product of Experts: RNA-seq and GCN (gistic2)\nLatent space : {}, LR: {}".format(args.n_latents, args.lr))
    plt.xlabel("Epochs")
    plt.ylabel("Average Training Loss")
    plt.savefig("{}/Loss {}.png".format(save_dir, dt_string), dpi=400)

    # Reconstruction Loss
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    plt.plot(x_axis, train_recon_loss_meter.values[::total_batches], marker='.', color='tab:orange')
    for x, y in zip(x_axis, train_recon_loss_meter.values[::total_batches]):
        if x == 1 or x % 25 == 0:
            ax2.annotate('  ({}, {:.4f})'.format(x, y), xy=(x, y), textcoords='data', fontsize=9)
    plt.title("Product of Experts: RNA-seq and GCN (gistic2)\nLatent space : {}, LR: {}".format(args.n_latents, args.lr))
    plt.xlabel("Epochs")
    plt.ylabel("Average Training Reconstruction Loss")
    plt.savefig("{}/Recon Loss {}.png".format(save_dir, dt_string), dpi=400)

    # KLD Loss
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111)
    plt.plot(x_axis, train_kld_loss_meter.values[::total_batches], marker='.', color='tab:red')
    for x, y in zip(x_axis, train_kld_loss_meter.values[::total_batches]):
        if x == 1 or x % 25 == 0:
            ax3.annotate('  ({}, {:.4f})'.format(x, y), xy=(x, y), textcoords='data', fontsize=9)
    plt.title("Product of Experts: RNA-seq and GCN (gistic2)\nLatent space : {}, LR: {}".format(args.n_latents, args.lr))
    plt.xlabel("Epochs")
    plt.ylabel("Average Training KLD Loss")
    plt.savefig("{}/KLD Loss {}.png".format(save_dir, dt_string), dpi=400)

    # Validation Loss
    fig4 = plt.figure(4)
    ax4 = fig4.add_subplot(111)
    plt.plot(x_axis, val_loss_meter.values[::total_val_batches], marker='.', color='tab:purple')
    for x, y in zip(x_axis, val_loss_meter.values[::total_val_batches]):
        if x == 1 or x % 25 == 0:
            ax4.annotate('  ({}, {:.4f})'.format(x, y), xy=(x, y), textcoords='data', fontsize=9)
    plt.title("Product of Experts: RNA-seq and GCN (gistic2)\nLatent space : {}, LR: {}".format(args.n_latents, args.lr))
    plt.xlabel("Epochs")
    plt.ylabel("Average Validation Loss")
    plt.savefig("{}/Validation Loss {}.png".format(save_dir, dt_string), dpi=400)

    # Validation Reconstruction Loss
    fig5 = plt.figure(5)
    ax5 = fig5.add_subplot(111)
    plt.plot(x_axis, val_recon_loss_meter.values[::total_val_batches], marker='.', color='tab:orange')
    for x, y in zip(x_axis, val_recon_loss_meter.values[::total_val_batches]):
        if x == 1 or x % 25 == 0:
            ax5.annotate('  ({}, {:.4f})'.format(x, y), xy=(x, y), textcoords='data', fontsize=9)
    plt.title("Product of Experts: RNA-seq and GCN (gistic2)\nLatent space : {}, LR: {}".format(args.n_latents, args.lr))
    plt.xlabel("Epochs")
    plt.ylabel("Average Validation Reconstruction Loss")
    plt.savefig("{}/Validation Recon Loss {}.png".format(save_dir, dt_string), dpi=400)

    # Validation KLD Loss
    fig6 = plt.figure(6)
    ax6 = fig6.add_subplot(111)
    plt.plot(x_axis, val_kld_loss_meter.values[::total_val_batches], marker='.', color='tab:red')
    for x, y in zip(x_axis, val_kld_loss_meter.values[::total_val_batches]):
        if x == 1 or x % 25 == 0:
            ax6.annotate('  ({}, {:.4f})'.format(x, y), xy=(x, y), textcoords='data', fontsize=9)
    plt.title("Product of Experts: RNA-seq and GCN (gistic2)\nLatent space : {}, LR: {}".format(args.n_latents, args.lr))
    plt.xlabel("Epochs")
    plt.ylabel("Average Validation KLD Loss")
    plt.savefig("{}/Validation KLD Loss {}.png".format(save_dir, dt_string), dpi=400)
