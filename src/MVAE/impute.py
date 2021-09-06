import os.path

import torch

from sklearn.metrics import mean_squared_error
from src.MVAE.train import load_checkpoint
import src.MVAE.datasets as datasets


def predict_loss(recon_data, input_data):
    """
    Computes mean squared error between reconstructed data (in Tensor form) and actual input data

    @param recon_data:
    @param input_data:
    @return: int MSE
    """
    return mean_squared_error(input_data, recon_data)


def predict(args: dict) -> None:
    save_dir = "/Users/bram/jointomicscomp/results/experiment 06-09-2021 14:36:06/MoE"

    print("-----   Loading Trained Model   -----")
    model, checkpoint = load_checkpoint(args['pre_trained'], use_cuda=False)
    model.eval()

    tcga_data = datasets.TCGAData(args, save_dir=save_dir, indices_path=os.path.dirname(args['pre_trained']))

    predict_dataset = tcga_data.get_data_partition("predict")

    # 1 batch for prediction
    predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=len(predict_dataset), shuffle=False)

    for batch_idx, (ge, me) in enumerate(predict_loader):
        (ge_recon_ge, ge_recon_me, ge_mu, ge_logvar) = model(ge=ge)

        ge_predicted_ge = ge_recon_ge.detach().numpy()
        ge_predicted_me = ge_recon_me.detach().numpy()

    for batch_idx, (ge, me) in enumerate(predict_loader):
        (me_recon_ge, me_recon_me, me_mu, me_logvar) = model(me=me)

        me_predicted_ge = me_recon_ge.detach().numpy()
        me_predicted_me = me_recon_me.detach().numpy()

    # Output results
    result_dir = os.path.dirname(args['pre_trained'])
    # Current Time from model for output files

    file1 = open("{}/predict_results.txt".format(result_dir), "w")
    file1.write("ME from GE ||| {}\n".format(predict_loss(ge_predicted_me, predict_dataset.me_data)))
    file1.write("GE from ME ||| {}\n".format(predict_loss(me_predicted_ge, predict_dataset.ge_data)))

    file1.write("GE from GE (unimodal) ||| {}\n".format(predict_loss(ge_predicted_ge, predict_dataset.ge_data)))
    file1.write("ME from ME (unimodal) ||| {}\n".format(predict_loss(me_predicted_me, predict_dataset.me_data)))
    file1.close()
