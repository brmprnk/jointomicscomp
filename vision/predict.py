import torch

from sklearn.metrics import mean_squared_error
from train import load_checkpoint
import datasets
import numpy as np
import pandas as pd


def predict_loss(recon_data, input_data):
    """
    Computes mean squared error between reconstructed data (in Tensor form) and actual input data

    @param recon_data:
    @param input_data:
    @return: int MSE
    """
    return mean_squared_error(input_data, recon_data)


if __name__ == "__main__":

    print("-----   Loading Trained Model   -----")
    model_path = "/Users/bram/multimodal-vae-public-master/vision/results/PoE cancer3 14-06-2021 09:09:39/best_model.pth.tar"
    model = load_checkpoint(model_path, use_cuda=False)
    model.eval()

    indices_path = "/Users/bram/multimodal-vae-public-master/vision/results/PoE cancer3 14-06-2021 09:09:39"
    tcga_data = datasets.TCGAData(indices_path=indices_path)

    predict_dataset = tcga_data.get_data_partition("predict")

    # 1 batch for prediction
    predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=len(predict_dataset), shuffle=False)

    # FOR 2 Predictions
    for batch_idx, (rna, gcn, dna) in enumerate(predict_loader):
        (rna_recon_rna, rna_recon_gcn, rna_recon_dna, joint_mu, joint_logvar) = model(rna=rna)

        rna_predicted_rna = rna_recon_rna.detach().numpy()
        rna_predicted_gcn = rna_recon_rna.detach().numpy()
        rna_predicted_dna = rna_recon_rna.detach().numpy()

    for batch_idx, (rna, gcn, dna) in enumerate(predict_loader):
        (gcn_recon_rna, gcn_recon_gcn, gcn_recon_dna, joint_mu, joint_logvar) = model(gcn=gcn)

        gcn_predicted_rna = gcn_recon_rna.detach().numpy()
        gcn_predicted_gcn = gcn_recon_rna.detach().numpy()
        gcn_predicted_dna = gcn_recon_rna.detach().numpy()

    for batch_idx, (rna, gcn, dna) in enumerate(predict_loader):
        (dna_recon_rna, dna_recon_gcn, dna_recon_dna, joint_mu, joint_logvar) = model(dna=dna)

        dna_predicted_rna = dna_recon_rna.detach().numpy()
        dna_predicted_gcn = dna_recon_gcn.detach().numpy()
        dna_predicted_dna = dna_recon_dna.detach().numpy()

    # rna_loss = predict_loss(predicted_rna, predict_dataset.rna_data)
    # gcn_loss = predict_loss(predicted_gcn, predict_dataset.gcn_data)
    # dna_loss = predict_loss(predicted_dna, predict_dataset.dna_data)

    # print("rna loss = ", rna_loss)
    # print("gcn loss = ", gcn_loss)
    # print("dna loss = ", dna_loss)

    # Output results
    result_dir = indices_path
    # Current Time from model for output files
    dt_string = "temp"

    # file3 = open("{}/3.txt".format(result_dir), "a")
    # file3.write("RNA Joint at {} ||| {}\n".format(dt_string, rna_loss))
    # file3.write("GCN Joint at {} ||| {}\n".format(dt_string, gcn_loss))
    # file3.write("DNA Joint at {} ||| {}\n\n".format(dt_string, dna_loss))
    # file3.close()
    #
    # file2 = open("{}/2.txt".format(result_dir), "a")
    # file2.write("RNA from others at {} ||| {}\n".format(dt_string, rna_loss))
    # file2.write("GCN from others at {} ||| {}\n".format(dt_string, gcn_loss))
    # file2.write("DNA from others at {} ||| {}\n\n".format(dt_string, dna_loss))
    # file2.close()
    #
    file1 = open("{}/predict_results.txt".format(result_dir), "a")
    file1.write("GCN from RNA at {} ||| {}\n".format(dt_string, predict_loss(rna_predicted_gcn, predict_dataset.gcn_data)))
    file1.write("DNA from RNA at {} ||| {}\n".format(dt_string, predict_loss(rna_predicted_dna, predict_dataset.dna_data)))
    file1.write("RNA from GCN at {} ||| {}\n".format(dt_string, predict_loss(gcn_predicted_rna, predict_dataset.rna_data)))
    file1.write("DNA from GCN at {} ||| {}\n".format(dt_string, predict_loss(gcn_predicted_dna, predict_dataset.dna_data)))
    file1.write("RNA from DNA at {} ||| {}\n".format(dt_string, predict_loss(dna_predicted_rna, predict_dataset.rna_data)))
    file1.write("GCN from DNA at {} ||| {}\n".format(dt_string, predict_loss(dna_predicted_gcn, predict_dataset.gcn_data)))

    file1.write("RNA from RNA (unimodal) at {} ||| {}\n".format(dt_string, predict_loss(rna_predicted_rna, predict_dataset.rna_data)))
    file1.write("GCN from GCN (unimodal) at {} ||| {}\n".format(dt_string, predict_loss(gcn_predicted_gcn, predict_dataset.gcn_data)))
    file1.write("DNA from DNA (unimodal) at {} ||| {}\n".format(dt_string, predict_loss(dna_predicted_dna, predict_dataset.dna_data)))
    file1.close()
