import os
from datetime import datetime

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F

from src.MVAE.model import MVAE
import src.MVAE.datasets as datasets
import src.util.logger as logger

import numpy as np

N_MODALITIES = 2


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


def loss_function(recon_ge, GE, recon_me, ME, mu, log_var, kld_weight) -> dict:
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}

    :return:
    """
    # Reconstruction loss
    recons_loss = 0
    if recon_ge is not None and GE is not None:
        recons_loss += F.mse_loss(recon_ge, GE)
    if recon_me is not None and ME is not None:
        recons_loss += F.mse_loss(recon_me, ME)

    recons_loss /= float(N_MODALITIES)  # Account for number of modalities

    # KLD Loss
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

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
        ge_recon_ge = F.mse_loss(recon_rna, rna)
        ge_recon_me = F.mse_loss(recon_gcn, gcn)
        rna_recon_dna = F.mse_loss(recon_dna, dna)
        loss_meter.modality_loss(ge_recon_ge.detach().numpy(), "rna_rna")
        loss_meter.modality_loss(ge_recon_me.detach().numpy(), "rna_gcn")
        loss_meter.modality_loss(rna_recon_dna.detach().numpy(), "rna_dna")

    if key == "gcn":
        me_recon_me = F.mse_loss(recon_gcn, gcn)
        me_recon_ge = F.mse_loss(recon_rna, rna)
        gcn_recon_dna = F.mse_loss(recon_dna, dna)
        loss_meter.modality_loss(me_recon_me.detach().numpy(), "gcn_gcn")
        loss_meter.modality_loss(me_recon_ge.detach().numpy(), "gcn_rna")
        loss_meter.modality_loss(gcn_recon_dna.detach().numpy(), "gcn_dna")

    return


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name
        self.epochs = 100
        self.values = []
        self.reconstruct_losses = {
            "average": [],
            "ge_ge": [],
            "me_me": [],
            "ge_me": [],
            "me_ge": [],
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
    @return: None
    """
    # Save checkpoint
    # torch.save(state, os.path.join(save_dir, filename))

    # If this is the best checkpoint (lowest loss) thus far, copy this model to a file named model_best
    if lowest_loss:
        print("Final epoch --> Saving to model_best")
        torch.save(state, os.path.join(save_dir, 'best_model.pth.tar'))


def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)

    trained_model = MVAE(use_mixture=checkpoint['use_mixture'], latent_dim=checkpoint['latent_dim'])
    trained_model.load_state_dict(checkpoint['state_dict'])
    return trained_model, checkpoint


def run(args) -> None:

    args['cuda'] = args['cuda'] and torch.cuda.is_available()

    # random seed
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args['random_seed'])
    np.random.seed(args['random_seed'])

    save_dir = os.path.join(args['save_dir'], '{}'.format('MoE' if args['mixture'] else 'PoE'))
    os.makedirs(save_dir)

    # Fetch Datasets

    tcga_data = datasets.TCGAData(args, save_dir=save_dir)
    train_dataset = tcga_data.get_data_partition("train")
    val_dataset = tcga_data.get_data_partition("val")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)  # (1 batch)

    total_batches = len(train_loader)
    total_val_batches = len(val_loader)  # Should be 1

    # Setup and log model
    model = MVAE(use_mixture=args['mixture'], latent_dim=args['latent_dim'], use_cuda=args['cuda'])

    # Log Data shape, input arguments and model
    model_file = open("{}/MVAE {} Model.txt".format(save_dir, 'MoE' if args['mixture'] else 'PoE'), "a")
    model_file.write("Running at {}\n".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    model_file.write("Input shape : {}, 5000\n".format(len(train_loader.dataset)))
    model_file.write("Input args : {}\n".format(args))
    model_file.write("PoE Model : {}".format(model))
    model_file.close()

    # Preparation for training
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    if args['cuda']:
        model.cuda()

    def train(epoch, train_loss_meter, train_recon_loss_meter, train_kld_loss_meter):
        model.train()

        progress_bar = tqdm(total=len(train_loader))
        for batch_idx, (GE, ME) in enumerate(train_loader):

            # refresh the optimizer
            optimizer.zero_grad()

            kld_weight = len(GE) / len(train_loader.dataset)  # Account for the minibatch samples from the dataset

            # compute reconstructions using all the modalities
            (joint_recon_rna, joint_recon_gcn, joint_mu, joint_logvar) = model(GE, ME)

            # compute reconstructions using each of the individual modalities
            (ge_recon_ge, ge_recon_me, ge_mu, ge_logvar) = model(ge=GE)

            (me_recon_ge, me_recon_me, me_mu, me_logvar) = model(me=ME)

            # Compute joint train loss
            joint_train_loss = loss_function(joint_recon_rna, GE,
                                             joint_recon_gcn, ME,
                                             joint_mu, joint_logvar, kld_weight)

            # compute loss with single modal inputs
            ge_train_loss = loss_function(ge_recon_ge, GE,
                                          ge_recon_me, ME,
                                          ge_mu, ge_logvar, kld_weight)

            me_train_loss = loss_function(me_recon_ge, GE,
                                          me_recon_me, ME,
                                          me_mu, me_logvar, kld_weight)

            train_loss = joint_train_loss['loss'] + ge_train_loss['loss'] + me_train_loss['loss']
            train_recon_loss = joint_train_loss['Reconstruction_Loss'] + ge_train_loss['Reconstruction_Loss'] + me_train_loss['Reconstruction_Loss']
            train_kld_loss = joint_train_loss['KLD'] + ge_train_loss['KLD'] + me_train_loss['KLD']

            train_loss_meter.update(train_loss, len(GE))
            train_recon_loss_meter.update(train_recon_loss, len(GE))
            train_kld_loss_meter.update(train_kld_loss, len(GE))

            # compute and take gradient step
            train_loss.backward()
            optimizer.step()

            progress_bar.update()

        progress_bar.close()
        if epoch % args['log_interval'] == 0:
            print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, train_loss_meter.avg))
            print('====> Epoch: {}\tReconstruction Loss: {:.4f}'.format(epoch, train_recon_loss_meter.avg))
            print('====> Epoch: {}\tKLD Loss: {:.4f}'.format(epoch, train_kld_loss_meter.avg))


    def test(epoch, val_loss_meter, val_recon_loss_meter, val_kld_loss_meter):
        model.eval()
        test_loss = 0

        for batch_idx, (GE, ME) in enumerate(val_loader):

            # for ease, only compute the joint loss in validation
            (joint_recon_ge, joint_recon_me, joint_mu, joint_logvar) = model(GE, ME)

            kld_weight = len(GE) / len(val_loader.dataset)  # Account for the minibatch samples from the dataset

            # Compute joint loss
            joint_test_loss = loss_function(joint_recon_ge, GE,
                                            joint_recon_me, ME,
                                            joint_mu, joint_logvar, kld_weight)

            val_loss_meter.update(joint_test_loss['loss'], len(GE))
            val_recon_loss_meter.update(joint_test_loss['Reconstruction_Loss'], len(GE))
            val_kld_loss_meter.update(joint_test_loss['KLD'], len(GE))

            test_loss += joint_test_loss['loss']

        if epoch % args['log_interval'] == 0:
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
    for epoch in range(1, args['epochs'] + 1):
        train(epoch, train_loss_meter, train_recon_loss_meter, train_kld_loss_meter)
        latest_loss = test(epoch, val_loss_meter, val_recon_loss_meter, val_kld_loss_meter)

        # Save the last model
        if epoch == args['epochs']:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_loss': latest_loss,
                'latent_dim': args['latent_dim'],
                'epochs': args['epochs'],
                'lr': args['lr'],
                'batch_size': args['batch_size'],
                'use_mixture': args['mixture'],
                'optimizer': optimizer.state_dict(),
            }, True, save_dir)

    print(model)

    if args['task'] == 2:
        print(model)
        logger.success("Extract z1 and z2 for classification of {}".format(args['ctype']))
        # Test sets are stratified data from cancer type into stages
        GEtrainctype = np.load(args['x_ctype_train_file'])
        GEvalidctype = np.load(args['x_ctype_valid_file'])
        MEtrainctype = np.load(args['y_ctype_train_file'])
        MEvalidctype = np.load(args['y_ctype_valid_file'])

        dataExtract1 = np.float32(np.vstack((GEtrainctype, GEvalidctype, np.load(args['x_ctype_test_file']))))
        dataExtract2 = np.float32(np.vstack((MEtrainctype, MEvalidctype, np.load(args['y_ctype_test_file']))))

        datasetExtract = datasets.TCGADataset(dataExtract1, dataExtract2)

        extract_loader = torch.utils.data.DataLoader(datasetExtract, batch_size=dataExtract1.shape[0], shuffle=False)

        for batch_idx, (GE, ME) in enumerate(extract_loader):
            z = model.extract(GE, ME)
            z = z.detach().numpy()

            np.save("{}/task2_z.npy".format(save_dir), z)

        # Extract Z from all data from the chosen cancer type
        # Do predictions separately


    if args['plot']:
        # Only import here to save time importing matplotlib only when required
        from src.util.MVAE_plotting import LossPlotter

        plotter = LossPlotter(args, save_dir)
        plotter.plot_training_losses(train_loss_meter, train_recon_loss_meter, train_kld_loss_meter, total_batches)
        plotter.plot_validation_loss(val_loss_meter, val_recon_loss_meter, val_kld_loss_meter, total_val_batches)
