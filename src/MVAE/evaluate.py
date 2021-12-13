import os.path

import torch

from sklearn.metrics import mean_squared_error
from src.MVAE.model import MVAE
import src.MVAE.datasets as datasets


def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)

    trained_model = MVAE(use_mixture=checkpoint['use_mixture'], latent_dim=checkpoint['latent_dim'])
    trained_model.load_state_dict(checkpoint['state_dict'])
    return trained_model, checkpoint


def predict_loss(recon_data, input_data):
    """
    Computes mean squared error between reconstructed data (in Tensor form) and actual input data

    @param recon_data:
    @param input_data:
    @return: int MSE
    """
    return mean_squared_error(input_data, recon_data)


def predict(args: dict) -> None:
    save_dir = os.path.dirname(args['pre_trained'])

    print("-----   Loading Trained Model   -----")
    model, checkpoint = load_checkpoint(args['pre_trained'], use_cuda=False)
    model.eval()

    all_data = datasets.TCGAData(args, save_dir=save_dir)

    impute_dataset = all_data.get_data_partition("test")

    # 1 batch (whole test set)
    impute_loader = torch.utils.data.DataLoader(impute_dataset, batch_size=len(impute_dataset), shuffle=False)

    omic1_from_joint, omic2_from_joint, \
    omic1_from_omic1, omic2_from_omic1, \
    omic1_from_omic2, omic2_from_omic2 = impute(model, impute_loader)

    # Output results
    result_dir = os.path.dirname(args['pre_trained'])
    # Current Time from model for output files

    print("ME from GE ||| {}\n".format(predict_loss(omic2_from_omic1, impute_dataset.omic2_data)))
    print("GE from ME ||| {}\n".format(predict_loss(omic1_from_omic2, impute_dataset.omic1_data)))

    print("GE from GE (unimodal) ||| {}\n".format(predict_loss(omic1_from_omic1, impute_dataset.omic1_data)))
    print("ME from ME (unimodal) ||| {}\n".format(predict_loss(omic2_from_omic2, impute_dataset.omic2_data)))


def impute(model, data_loader, use_cuda=False):
    # From both omics
    for batch_idx, (omic1, omic2) in enumerate(data_loader):

        if use_cuda:
            omic1 = omic1.cuda()
            omic2 = omic2.cuda()

        (omic1_recon_joint, omic2_recon_joint, joint_mu, joint_logvar) = model(omic1=omic1, omic2=omic2)

        if use_cuda:
            omic1_from_joint = omic1_recon_joint.detach().cpu().numpy()
            omic2_from_joint = omic2_recon_joint.detach().cpu().numpy()
        else:
            omic1_from_joint = omic1_recon_joint.detach().numpy()
            omic2_from_joint = omic2_recon_joint.detach().numpy()


    # From omic1 only
    for batch_idx, (omic1, omic2) in enumerate(data_loader):

        if use_cuda:
            omic1 = omic1.cuda()
            omic2 = omic2.cuda()

        (omic1_recon_ge, omic1_recon_me, omic1_mu, omic1_logvar) = model(omic1=omic1)

        if use_cuda:
            omic1_from_omic1 = omic1_recon_ge.detach().cpu().numpy()
            omic2_from_omic1 = omic1_recon_me.detach().cpu().numpy()
        else:
            omic1_from_omic1 = omic1_recon_ge.detach().numpy()
            omic2_from_omic1 = omic1_recon_me.detach().numpy()

    # From omic2 only
    for batch_idx, (omic1, omic2) in enumerate(data_loader):

        if use_cuda:
            omic1 = omic1.cuda()
            omic2 = omic2.cuda()

        (omic2_recon_ge, omic2_recon_me, omic2_mu, omic2_logvar) = model(omic2=omic2)

        if use_cuda:
            omic1_from_omic2 = omic2_recon_ge.detach().cpu().numpy()
            omic2_from_omic2 = omic2_recon_me.detach().cpu().numpy()
        else:
            omic1_from_omic2 = omic2_recon_ge.detach().numpy()
            omic2_from_omic2 = omic2_recon_me.detach().numpy()

    return omic1_from_joint, omic2_from_joint, omic1_from_omic1, omic2_from_omic1, omic1_from_omic2, omic2_from_omic2

