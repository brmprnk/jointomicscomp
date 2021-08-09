import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from nets import *
from model import train, extract
from sklearn.model_selection import train_test_split

import sys

def str2bool(value):
	return value.lower() == 'true'


parser = argparse.ArgumentParser(description='')
# Learning hyperparameters
parser.add_argument('--dataset',		dest='dataset',           default='GE') # 'GE' / 'ME'
parser.add_argument('--phase',          dest='phase',           default='train') # 'train' / 'test' / 'extract'
parser.add_argument('--batch_size',     dest='batch_size',      type=int,       default=128)
parser.add_argument('--num_epochs',     dest='num_epochs',      type=int,       default=200)
parser.add_argument('--enc_lr',        dest='encoder_lr',         type=float,     default=0.0005)
parser.add_argument('--dec_lr',        dest='decoder_lr',         type=float,     default=0.0005)
parser.add_argument('--loss_function',  dest='loss_function',        type=str,  default='mse')
parser.add_argument('--distribution',  dest='distribution',        type=str,  default='beta')
parser.add_argument('--dropout_probability',  dest='dropout_probability',        type=float,  default=0.0)
parser.add_argument('--use_batch_norm',  dest='use_batch_norm',        type=str2bool,  default=False)
parser.add_argument('--optimizer',  dest='optimizer',        type=str,  default='Adam')

# Architecture hyperparameters
parser.add_argument('--net_type',       dest='net_type',        default='ae')
parser.add_argument('--enc_hidden_dim',   dest='enc_hidden_dim',    type=str,       default='100')
parser.add_argument('--dec_hidden_dim',   dest='dec_hidden_dim',    type=str,       default='')

parser.add_argument('--enc_last_activation',   dest='enc_lastActivation',    type=str,       default='relu')
parser.add_argument('--enc_output_scale',   dest='enc_outputScale',    type=float,       default=1.)

parser.add_argument('--vae_beta_start_value', dest='beta_start_value', type=float, default=1.)
parser.add_argument('--vae_beta_end_value', dest='beta_end_value', type=float, default=1.)
parser.add_argument('--vae_beta_start_epoch', dest='beta_start_epoch', type=int, default=1000000000000000)
parser.add_argument('--vae_beta_niter', dest='beta_niter', type=int, default=1)

parser.add_argument("--validation_fraction", dest='validation_fraction', type=float, default=0.1)
# parser.add_argument("--checkpoint-every", type=int, default=50, help="Frequency of model checkpointing (in epochs).")
# parser.add_argument("--backup-every", type=int, default=5, help="Frequency of model backups (in epochs).")


# Training directories and files
parser.add_argument('--model_dir',      dest='model_dir',       default='models/GE/ae/')

# Test directories and files
parser.add_argument('--model_file',     dest='model_file',      default='models/ae/checkpoint/model_epoch40.pth.tar')
parser.add_argument('--test_file',      dest='test_file',       default='data/test.names')
parser.add_argument('--emb_save_file',  dest='emb_save_file',   default='models/ae/embeddings.pkl')
parser.add_argument('--split_random_seed', dest='seed',    type=int,     default=1)
args = parser.parse_args()


# Check cuda availability
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("[*] Selected device: ", device)


npyData = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/datasets/pancancer/' + args.dataset + '.npy')

input_dim = npyData.shape[1]


encoder_layers = [int(k) for k in args.enc_hidden_dim.split('-')]

if len(args.dec_hidden_dim) == 0:
	decoder_layers = []
else:
	decoder_layers = [int(k) for k in args.dec_hidden_dim.split('-')]



torch.manual_seed(42)

# Initialize network model
if args.net_type == 'ae':
	net = AutoEncoder(input_dim, encoder_layers, decoder_layers, args.loss_function, args.use_batch_norm, args.dropout_probability, args.optimizer, args.encoder_lr, args.decoder_lr, args.enc_lastActivation, args.enc_outputScale).to(device)

elif args.net_type == 'lae':
	net = LikelihoodAutoEncoder(input_dim, encoder_layers, decoder_layers, args.distribution, args.use_batch_norm, args.dropout_probability, args.optimizer, args.encoder_lr, args.decoder_lr, args.enc_lastActivation, args.enc_outputScale).to(device)

elif args.net_type == 'vae':
	net = VariationalAutoEncoder(input_dim, encoder_layers, decoder_layers, args.loss_function, args.use_batch_norm, args.dropout_probability, args.optimizer, args.encoder_lr, args.decoder_lr, args.enc_lastActivation, args.enc_outputScale, args.beta_start_value, args.beta_end_value, args.beta_niter, args.beta_start_epoch).to(device)

elif args.net_type == 'lvae':
	raise NotImplementedError()
	net = LikelihoodVariationalAutoencoder(dec_hidden_dim=decoder_layers, decoder_lr=args.decoder_lr, distribution=args.distribution, beta_start_value=args.beta_start_value, beta_end_value=args.beta_end_value, beta_n_iterations=args.beta_niter, beta_start_iteration=args.beta_start_epoch, use_batch_norm=args.use_batch_norm, dropoutP=args.dropout_probability, encoder_lr=args.encoder_lr, input_dim=input_dim, enc_hidden_dim=encoder_layers).to(device)

else:
	raise NotImplementedError('[!] Unknown network type, try "ae", "lae", "lvae", "vae".')


net = net.double()


print("[*] Initialize model successfully, network type: %s." % args.net_type)
print(net)
print("[*] Number of model parameters:")
print(sum(p.numel() for p in net.parameters() if p.requires_grad))


# Training phase
if args.phase == 'train':
	# Create directories for checkpoint, sample and logs files
	ckpt_dir = args.model_dir + '/checkpoint'
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)
	logs_dir = args.model_dir + '/logs'

	# Data loading
	print("[*] Loading training and validation data...")
	dataTrain, dataValidation = train_test_split(npyData, test_size=args.validation_fraction, shuffle=True, random_state=args.seed)
	dataTrain = torch.tensor(dataTrain, device=device)
	dataValidation = torch.tensor(dataValidation, device=device)

	datasetTrain = TensorDataset(dataTrain)
	datasetValidation = TensorDataset(dataValidation)

	train_loader = DataLoader(datasetTrain, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)

	train_loader_eval = DataLoader(datasetTrain, batch_size=dataTrain.shape[0], shuffle=False, num_workers=0, drop_last=False)

	valid_loader = DataLoader(datasetValidation, batch_size=dataValidation.shape[0], shuffle=False, num_workers=0, drop_last=False)

	# Training and validation

	train(device=device, net=net, num_epochs=args.num_epochs, train_loader=train_loader, train_loader_eval=train_loader_eval, valid_loader=valid_loader,
	ckpt_dir=ckpt_dir, logs_dir=logs_dir, save_step=5)

	'''
# get embeddings
elif args.phase == 'extract':

	# Data loading
	print("\n[*] Loading test data %s." % args.test_file)

	extract_set = Mydataset(args.test_file, args.feats_dir)
	extract_loader = DataLoader(extract_set, batch_size=1, shuffle=True, num_workers=1, drop_last=True)

	# Embedding extractor
	extract(device=device, net=net, model_file=args.model_file, names_file=args.test_file, loader=extract_loader, save_file=args.emb_save_file)
	'''

else:
	raise NotImplementedError('[!] Unknown phase')
