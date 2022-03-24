import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, Beta
import torch.optim as optimizer_module
import math
from tensorboardX import SummaryWriter

# several utils

# utility function to initialize an optimizer from its name
def init_optimizer(optimizer_name, params):
	assert hasattr(optimizer_module, optimizer_name)
	OptimizerClass = getattr(optimizer_module, optimizer_name)
	return OptimizerClass(params)


def identity(x):
	return x


########################################################################################
# basic components

# fully-connected layers
class FullyConnectedModule(nn.Module):
	def __init__(self, input_dim, hidden_dim=[100], use_batch_norm=False, dropoutP=0.0, lastActivation='none', outputScale=1.):
		super(FullyConnectedModule, self).__init__()

		self.z_dim = hidden_dim[-1]

		in_neurons = [input_dim] + hidden_dim[:-1]

		if use_batch_norm:
			encode_layers = [nn.Sequential(nn.Linear(in_d, out_d), nn.BatchNorm1d(out_d)) for (in_d, out_d) in zip(in_neurons, hidden_dim)]
		else:
			encode_layers = [nn.Linear(in_d, out_d) for (in_d, out_d) in zip(in_neurons, hidden_dim)]

		self.num_layers = len(encode_layers)
		self.encode_layers = nn.ModuleList(encode_layers)
		self.drop = nn.Dropout(p=dropoutP)


		if lastActivation == 'relu':
			self.lastActivation = F.relu
		elif lastActivation == 'sigmoid':
			self.lastActivation = torch.sigmoid
		elif lastActivation == 'tanh':
			self.lastActivation = torch.tanh
		elif lastActivation == 'none':
			self.lastActivation = None
		else:
			raise NotImplementedError('Use \'relu\' for encoder or \'sigmoid\' for decoder')

		self.outputScale = outputScale


	def forward(self, x):

		for i, layer in enumerate(self.encode_layers):
			try:
				x = layer(self.drop(x))
			except RuntimeError:
				# thrown by IWAE when batch normalization is used, due to shape mismatch
				assert len(x.shape) == 3
				assert len(layer) == 2
				# pass through linear layer and switch the last to dims
				x = layer[0](self.drop(x)).transpose(1,2)
				# pass through batchnorm now that dims match and revert to original shape
				x = layer[1](x).transpose(2,1)


			if i < self.num_layers - 1:
				x = F.relu(x)
			else:
				if self.lastActivation is not None:
					x = self.lastActivation(x)

		return x * self.outputScale


class ProbabilisticFullyConnectedModule(FullyConnectedModule):
	def __init__(self, input_dim, hidden_dim=[100], distribution='normal', use_batch_norm=False, dropoutP=0.0, lastActivation='none', outputScale=1.):

		self.distribution = distribution
		if self.distribution != 'cbernoulli':
			newlist = [h for h in hidden_dim]
			newlist[-1] = newlist[-1] * 2
			hidden_dim = newlist

		super(ProbabilisticFullyConnectedModule, self).__init__(input_dim, hidden_dim, use_batch_norm, dropoutP, lastActivation, outputScale)


		if self.distribution == 'normal':
			self.p1Transform = identity
			self.p2Transform = torch.exp
			self.D = torch.distributions.Normal
		elif self.distribution == 'beta':
			self.p1Transform = torch.exp
			self.p2Transform = torch.exp
			self.D = torch.distributions.Beta
		elif self.distribution == 'lognormal':
			self.p1Transform = identity
			self.p2Transform = torch.exp
			self.D = torch.distributions.LogNormal

		elif self.distribution == 'cbernoulli':
			self.D = torch.distributions.ContinuousBernoulli
		elif self.distribution == 'laplace':
			self.p1Transform = identity
			self.p2Transform = torch.exp
			self.D = torch.distributions.Laplace
		else:
			raise NotImplementedError('%s not supported. Use: \'normal\' or \'beta\'' % self.distribution)

		if self.distribution != 'cbernoulli':
			self.z_dim = self.z_dim // 2

	def forward(self, x):
		z = super().forward(x)

		if self.distribution != 'cbernoulli':
			p1 = self.p1Transform(z[:, :self.z_dim])
			p2 = self.p2Transform(z[:, self.z_dim:])

			return Independent(self.D(p1, p2), 0)
		else:
			if self.lastActivation is None:
				return Independent(self.D(logits=z), 0)
			elif self.lastActivation == torch.sigmoid:
				return Independent(self.D(probs=z), 0)

			else:
				raise NotImplementedError('Wrong activation for lamda parameter of ContinuousBernoulli')



# base class
class RepresentationLearner(nn.Module):
	def __init__(self, input_dim, enc_hidden_dim=[100], use_batch_norm=False, dropoutP=0.0, optimizer_name='Adam', encoder_lr=1e-4, probabilistic_encoder=False, enc_lastActivation='none', enc_outputScale=1., enc_distribution='normal'):

		super(RepresentationLearner, self).__init__()

		self._epoch = 0
		self.loss_items = {}

		self._input_dim = input_dim
		self._dropoutP = dropoutP
		self._use_batch_norm = use_batch_norm

		self.z_dim = enc_hidden_dim[-1]

		# Intialization of the encoder
		if probabilistic_encoder:
			self.encoder = ProbabilisticFullyConnectedModule(input_dim, enc_hidden_dim, enc_distribution, use_batch_norm, dropoutP, enc_lastActivation, enc_outputScale)
		else:
			self.encoder = FullyConnectedModule(input_dim, enc_hidden_dim, use_batch_norm, dropoutP, enc_lastActivation, enc_outputScale)

		self.opt = init_optimizer(optimizer_name, [
			{'params': self.encoder.parameters(), 'lr': encoder_lr},
		])


	def increment_epoch(self):
		self._epoch += 1

	def get_device(self):
		return list(self.parameters())[0].device



	def save(self, model_path):
		items_to_save = self._get_items_to_store()
		items_to_save['epoch'] = self.epoch

		# Save the model and increment the checkpoint count
		torch.save(items_to_save, model_path)

	def load(self, model_path):
		items_to_load = torch.load(model_path)
		for key, value in items_to_load.items():
			assert hasattr(self, key)
			attribute = getattr(self, key)

			# Load the state dictionary for the stored modules and optimizers
			if isinstance(attribute, nn.Module) or isinstance(attribute, Optimizer):
				attribute.load_state_dict(value)

				# Move the optimizer parameters to the same correct device.
				# see https://github.com/pytorch/pytorch/issues/2830 for further details
				if isinstance(attribute, Optimizer):
					device = list(value['state'].values())[0]['exp_avg'].device # Hack to identify the device
					for state in attribute.state.values():
						for k, v in state.items():
							if isinstance(v, torch.Tensor):
								state[k] = v.to(device)

			# Otherwise just copy the value
			else:
				setattr(self, key, value)


	def _get_items_to_store(self):
		items_to_store = dict()

		# store the encoder and optimizer parameters
		items_to_store['encoder'] = self.encoder.state_dict()
		items_to_store['opt'] = self.opt.state_dict()

		return items_to_store

	def _train_step(self, data):
		loss = self.compute_loss(data)

		self.opt.zero_grad()
		loss.backward()
		self.opt.step()

	def compute_loss(self, data):
		raise NotImplemented

	def encode(self, data):

		z = self.encoder(data)
		if isinstance(self.encoder, FullyConnectedModule):
			return z
		else:
			return z.mean


class MultiOmicRepresentationLearner(RepresentationLearner):
	def __init__(self, input_dim1, input_dim2, enc_hidden_dim=[100], use_batch_norm=False, dropoutP=0.0, optimizer_name='Adam', probabilistic_encoder=False, encoder1_lr=1e-4, enc1_lastActivation='none', enc1_outputScale=1., encoder2_lr=1e-4, enc2_lastActivation='none', enc2_outputScale=1., enc_distribution='normal'):
		super(MultiOmicRepresentationLearner, self).__init__(input_dim1, enc_hidden_dim, use_batch_norm, dropoutP, optimizer_name, encoder1_lr, probabilistic_encoder, enc1_lastActivation, enc1_outputScale, enc_distribution)


		# Intialization of 2nd encoder
		if probabilistic_encoder:
			self.encoder2 = ProbabilisticFullyConnectedModule(input_dim2, enc_hidden_dim, enc_distribution, use_batch_norm, dropoutP, enc2_lastActivation, enc2_outputScale)
		else:
			self.encoder2 = FullyConnectedModule(input_dim2, enc_hidden_dim, use_batch_norm, dropoutP, enc2_lastActivation, enc2_outputScale)

		self.opt.add_param_group({'params': self.encoder2.parameters(), 'lr': encoder2_lr})

	def encode(self, x1, x2):
		z1 = self.encoder(x1)
		z2 = self.encoder2(x2)

		return z1, z2


class VariationalAutoEncoder(RepresentationLearner):

	def __init__(self, input_dim, enc_hidden_dim=[100], dec_hidden_dim=[], likelihood='normal', use_batch_norm=False, dropoutP=0.0, optimizer_name='Adam', encoder_lr=1e-4, decoder_lr=1e-4, enc_lastActivation='relu', enc_outputScale=1., beta=1.):
		super(VariationalAutoEncoder, self).__init__(input_dim, enc_hidden_dim, use_batch_norm, dropoutP, optimizer_name, encoder_lr, True, enc_lastActivation, enc_outputScale)

		# Intialization of the decoder and loss function

		self.decoder = ProbabilisticFullyConnectedModule(self.z_dim, dec_hidden_dim + [self._input_dim], likelihood, self._use_batch_norm, self._dropoutP, lastActivation='none')


		self.opt.add_param_group({'params': self.decoder.parameters(), 'lr': decoder_lr})
		self.beta = beta


	def compute_loss(self, x):

		z = self.encoder(x)
		zmean = z.mean
		zsigma = z.stddev

		x_hat = self.decoder(z.rsample())

		reconstruction_loss = - torch.sum(torch.mean(d.log_prob(x), 1))
		kl = -0.5 * torch.sum(torch.mean(1 + torch.log(zsigma ** 2) - (zmean ** 2) - (zsigma ** 2), 1))

		b = self.beta

		loss = reconstruction_loss + b * kl

		return loss

	def encode(self, data):
		# return only the mean
		return super().encode(data).mean

	def evaluate(self, x):
		metrics = dict()
		with torch.no_grad():
			z = self.encoder(x)
			zmean = z.mean
			zsigma = z.stddev

			x_hat = self.decoder(z.rsample())

			metrics['LL'] = torch.sum(x_hat.log_prob(x)).item()
			metrics['mse'] = torch.mean(torch.sum(torch.nn.MSELoss(reduction='none')(x_hat.mean, x), 1)).item()

			kl = -0.5 * torch.sum(torch.mean(1 + torch.log(zsigma ** 2) - (zmean ** 2) - (zsigma ** 2), 1))
			metrics['KL'] = kl.item()

			b = self.beta

			metrics['b*KL'] = metrics['KL'] * b

			metrics['loss'] = metrics['b*KL'] - metrics['LL']


		return metrics



class CrossGeneratingVariationalAutoencoder(MultiOmicRepresentationLearner):
	def __init__(self, input_dim1, input_dim2, enc_hidden_dim=[100], dec_hidden_dim=[], likelihood1='normal', likelihood2='normal', use_batch_norm=False, dropoutP=0.0, optimizer_name='Adam', encoder1_lr=1e-4, decoder1_lr=1e-4, enc1_lastActivation='none', enc1_outputScale=1., encoder2_lr=1e-4, decoder2_lr=1e-4, enc2_lastActivation='none', enc2_outputScale=1., enc_distribution='normal', beta=1.0, zconstraintCoef=1.0, crossPenaltyCoef=1.0):
		super(CrossGeneratingVariationalAutoencoder, self).__init__(input_dim1, input_dim2, enc_hidden_dim, use_batch_norm, dropoutP, optimizer_name, True, encoder1_lr, enc1_lastActivation, enc1_outputScale, encoder2_lr, enc2_lastActivation, enc2_outputScale, enc_distribution)

		self.decoder = ProbabilisticFullyConnectedModule(self.z_dim, dec_hidden_dim + [input_dim1], distribution=likelihood1, use_batch_norm=self._use_batch_norm, dropoutP=self._dropoutP, lastActivation='none', outputScale=1.0)
		self.decoder2 = ProbabilisticFullyConnectedModule(self.z_dim, dec_hidden_dim + [input_dim2], distribution=likelihood2, use_batch_norm=self._use_batch_norm, dropoutP=self._dropoutP, lastActivation='none', outputScale=1.0)


		self.opt.add_param_group({'params': self.decoder.parameters(), 'lr': decoder1_lr})
		self.opt.add_param_group({'params': self.decoder2.parameters(), 'lr': decoder2_lr})

		self.zconstraintCoef = zconstraintCoef
		self.crossPenaltyCoef = crossPenaltyCoef
		self.beta = beta


	def compute_loss(self, x1, x2):
		#qz_x in MoE
		z1 = self.encoder(x1)
		z1mean = z1.mean
		z1std = z1.stddev
		z1sample = z1.rsample()

		x1_hat = self.decoder(z1sample)

		z2 = self.encoder2(x2)
		z2mean = z2.mean
		z2std = z2.stddev
		z2sample = z2.rsample()

		x2_hat = self.decoder2(z2sample)

		cross1_hat = self.decoder(z2sample)
		cross2_hat = self.decoder2(z1sample)


		rec1 = - torch.sum(torch.mean(x1_hat.log_prob(x1), 1))
		cross_rec1 = - torch.mean(torch.sum(cross1_hat.log_prob(x1), 1))


		rec2 = - torch.sum(torch.mean(x2_hat.log_prob(x2), 1))
		cross_rec2 = - torch.sum(torch.mean(cross2_hat.log_prob(x2), 1))


		# KL divergence
		kl1 = -0.5 * torch.sum(torch.mean(1 + torch.log(z1std ** 2) - (z1mean ** 2) - (z1std ** 2), 1))
		kl2 = -0.5 * torch.sum(torch.mean(1 + torch.log(z2std ** 2) - (z2mean ** 2) - (z2std ** 2), 1))


		# wasserstein-2 distance between two z's
		similarityConstraint = nn.MSELoss(reduction='none')(z1mean, z2mean) + nn.MSELoss(reduction='none')(z1std, z2std)
		similarityConstraint = torch.sum(torch.mean(similarityConstraint, 1))

		loss = rec1 + rec2 + self.beta * (kl1 + kl2) + self.crossPenaltyCoef * (cross_rec1 + cross_rec2) + self.zconstraintCoef * similarityConstraint
		return loss


	def evaluate(self, x1, x2):
		metrics = dict()
		with torch.no_grad():
			# encode, sample, decode
			z1 = self.encoder(x1)
			z1mean = z1.mean
			z1std = z1.stddev
			z1sample = z1.rsample()

			x1_hat = self.decoder(z1sample)

			z2 = self.encoder2(x2)
			z2mean = z2.mean
			z2std = z2.stddev
			z2sample = z2.rsample()

			x2_hat = self.decoder2(z2sample)

			# decode from the other sample
			cross1_hat = self.decoder(z2sample)
			cross2_hat = self.decoder2(z1sample)

			# similarity between 2 z's
			metrics['z-L2'] = nn.MSELoss(reduction='none')(z1sample, z2sample)
			metrics['z-L2'] = torch.mean(torch.sum(metrics['z-L2'], 1)).item()

			metrics['z-L1'] = nn.L1Loss(reduction='none')(z1sample, z2sample)
			metrics['z-L1'] = torch.mean(torch.sum(metrics['z-L1'], 1)).item()
			metrics['z-cos'] = torch.mean(1 - torch.cosine_similarity(z1sample,z2sample)).item()

			w2 = nn.MSELoss(reduction='none')(z1mean, z2mean) + nn.MSELoss(reduction='none')(z1std, z2std)
			metrics['z-W2'] = torch.mean(torch.sum(w2, 1)).item()
			similarityConstraint = metrics['z-W2']


			# KL divergence between each z and N(0,I)
			kl1 = -0.5 * torch.mean(torch.sum(1 + torch.log(z1std ** 2) - (z1mean ** 2) - (z1std ** 2), 1))
			kl2 = -0.5 * torch.mean(torch.sum(1 + torch.log(z2std ** 2) - (z2mean ** 2) - (z2std ** 2), 1))

			metrics['KL/1'] = kl1.item()
			metrics['KL/2'] = kl2.item()


			# likelihood/loss
			metrics['LL/1'] = torch.mean(torch.sum(x1_hat.log_prob(x1), 1)).item()
			metrics['cross-LL/1'] = torch.mean(torch.sum(cross1_hat.log_prob(x1), 1)).item()

			x1_hat = x1_hat.mean
			cross1_hat = cross1_hat.mean
			l1 = -metrics['LL/1']
			cl1 = -metrics['cross-LL/1']


			metrics['LL/2'] = torch.sum(x2_hat.log_prob(x2)).item()
			metrics['cross-LL/2'] = torch.mean(torch.sum(cross2_hat.log_prob(x2), 1)).item()

			x2_hat = x2_hat.mean
			cross2_hat = cross2_hat.mean
			l2 = -metrics['LL/2']
			cl2 = -metrics['cross-LL/2']


			metrics['mse/1'] = torch.mean(torch.sum(torch.nn.MSELoss(reduction='none')(x1_hat, x1), 1)).item()
			metrics['mse/2'] = torch.mean(torch.sum(torch.nn.MSELoss(reduction='none')(x2_hat, x2), 1)).item()

			metrics['cross-mse/1'] = torch.mean(torch.sum(torch.nn.MSELoss(reduction='none')(cross1_hat, x1), 1)).item()
			metrics['cross-mse/2'] = torch.mean(torch.sum(torch.nn.MSELoss(reduction='none')(cross2_hat, x2), 1)).item()

		metrics['loss'] = l1 + l2 + self.crossPenaltyCoef * (cl1 + cl2) + self.zconstraintCoef * similarityConstraint + self.beta * (metrics['KL/1'] + metrics['KL/2'])

		return metrics

	def encode(self, x1, x2):
		z1, z2 = super().encode(x1, x2)
		return z1.mean, z2.mean

	def embedAndReconstruct(self, x1, x2):
		with torch.no_grad():
			z1, z2 = self.encode(x1, x2)

			x1_hat = self.decoder(z1)
			x2_hat = self.decoder2(z2)

			x1_cross_hat = self.decoder(z2)
			x2_cross_hat = self.decoder2(z1)

			return z1, z2, x1_hat, x2_hat, x1_cross_hat, x2_cross_hat




# Auxiliary network for mutual information estimation
# change the default to only one hidden layer
# in the future, make it possible to choose architecture (like OmicsEncoder)

class MIEstimator(FullyConnectedModule):
	def __init__(self, input_dim, arch, use_batch_norm, dropoutP, lastActivation):
		super(MIEstimator, self).__init__(input_dim, arch, use_batch_norm, dropoutP, lastActivation)


	# Gradient for JSD mutual information estimation and EB-based estimation
	def forward(self, x1, x2):
		pos = super(MIEstimator, self).forward(torch.cat([x1, x2], 1))  # Positive Samples
		neg = super(MIEstimator, self).forward(torch.cat([torch.roll(x1, 1, 0), x2], 1))  # Negative Samples

		print(pos.shape, neg.shape)

		return -F.softplus(-pos).mean() - F.softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1



class MVIB(MultiOmicRepresentationLearner):
	def __init__(self, input_dim1, input_dim2, enc_hidden_dim=[100], mi_net_arch=[100, 1], use_batch_norm=False, dropoutP=0.0, optimizer_name='Adam', encoder1_lr=1e-4, enc1_lastActivation='none', enc1_outputScale=1., encoder2_lr=1e-4, enc2_lastActivation='none', enc2_outputScale=1., mi_net_lr=1e-4, beta=1.0):
		super().__init__(input_dim1, input_dim2, enc_hidden_dim, use_batch_norm, dropoutP, optimizer_name, True, encoder1_lr, enc1_lastActivation, enc1_outputScale, encoder2_lr, enc2_lastActivation, enc2_outputScale)

		if mi_net_arch[-1] != 1:
			mi_net_arch.append(1)

		self.beta = beta
		self.MInet = MIEstimator(2 * self.z_dim, mi_net_arch, self._use_batch_norm, self._dropoutP, lastActivation='none')

		self.opt.add_param_group({'params': self.MInet.parameters(), 'lr': mi_net_lr})


	def compute_loss(self, x1, x2):
		p_z1_given_v1 = self.encoder(x1)
		p_z2_given_v2 = self.encoder2(x2)

		z1 = p_z1_given_v1.rsample()
		z2 = p_z2_given_v2.rsample()


		# Symmetrized Kullback-Leibler divergence
		kl_1_2 = p_z1_given_v1.log_prob(z1) - p_z2_given_v2.log_prob(z1)
		kl_2_1 = p_z2_given_v2.log_prob(z2) - p_z1_given_v1.log_prob(z2)
		skl = (kl_1_2 + kl_2_1) / 2.

		skl = torch.mean(torch.sum(skl, 1))

		# Mutual information estimation
		mi_gradient, mi_estimation = self.MInet(z1, z2)
		# mi_gradient = mi_gradient.mean()
		# mi_estimation = mi_estimation.mean()

		loss = - mi_gradient + self.beta * skl


		return loss

	def evaluate(self, x1, x2):
		metrics = dict()
		with torch.no_grad():
			p_z1_given_v1 = self.encoder(x1)
			p_z2_given_v2 = self.encoder2(x2)

			z1mean = p_z1_given_v1.mean
			z2mean = p_z2_given_v2.mean
			z1std = p_z1_given_v1.stddev
			z2std = p_z2_given_v2.stddev
			z1var = p_z1_given_v1.variance
			z2var = p_z2_given_v2.variance

			# compute the KL in closed form
			metrics['KL_1_2'] = torch.mean(torch.sum(torch.log(z2std / z1std) + 0.5 * (z1var + (z1mean - z2mean) ** 2) / z2var - 0.5, 1))
			metrics['KL_2_1'] = torch.mean(torch.sum(torch.log(z1std / z2std) + 0.5 * (z2var + (z2mean - z1mean) ** 2) / z1var - 0.5, 1))

			metrics['KL_1_2'] = metrics['KL_1_2'].item()
			metrics['KL_2_1'] = metrics['KL_2_1'].item()

			metrics['SKL'] = 0.5 * (metrics['KL_1_2'] + metrics['KL_2_1'])

			# using the means here
			migrad, miest = self.MInet(z1mean, z2mean)

			metrics['MI_grad'] = migrad.item()
			metrics['MI_est'] = miest.item()

			metrics['loss'] = - metrics['MI_grad'] + self.beta * metrics['SKL']

			metrics['z-L2'] = nn.MSELoss(reduction='none')(z1mean, z2mean)
			metrics['z-L2'] = torch.mean(torch.sum(metrics['z-L2'], 1)).item()

			metrics['z-L1'] = nn.L1Loss(reduction='none')(z1mean, z2mean)
			metrics['z-L1'] = torch.mean(torch.sum(metrics['z-L1'], 1)).item()
			metrics['z-cos'] = torch.mean(1 - torch.cosine_similarity(z1mean, z2mean)).item()

		return metrics


	def embedAndReconstruct(self, x1, x2):
		with torch.no_grad():
			z1, z2 = super().encode(x1, x2)
			return z1.mean, z2.mean



class MLP(nn.Module):
	# 2-layer perceptron
	def __init__(self, input_dim, hidden_dim, n_classes):
		super(MLP, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim  = hidden_dim
		self.n_classes = n_classes

		self.hidden_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim))
		self.output_layer = torch.nn.Linear(hidden_dim, n_classes)

	def forward(self, x):
		hidden = F.relu(self.hidden_layer(x))
		y_raw = self.output_layer(hidden)

		return y_raw

	def predict(self, x):
		with torch.no_grad():
			y_raw = self.forward(x)
			return nn.Softmax(dim=1)(y_raw)

	def optimize(self, num_epochs, lr, train_loader, valid_loader, ckpt_dir, logs_dir, early_stopping, weights=None):
		device = torch.device('cuda:0')
		if weights is None:
			weights = torch.ones(self.n_classes)

		weights = weights.double().to(device)
		bestValLoss = 1e100

		loss_fun = nn.CrossEntropyLoss(weight=weights)

		tf_logger = SummaryWriter(logs_dir)
		opt = init_optimizer('Adam', [{'params': self.parameters(), 'lr': lr}])

		# Start training phase
		print("[*] Start training...")
		# Training epochs
		for epoch in range(num_epochs):
			self.train()

			print("[*] Epoch %d..." % (epoch + 1))
			for x, y in train_loader:
				with torch.set_grad_enabled(True):  # no need to specify 'requires_grad' in tensors
					opt.zero_grad()
					y_pred = self.forward(x[0].double().to(device))
					current_loss = loss_fun(y_pred, y[0].to(device))

					# Backward pass and optimize
					current_loss.backward()
					opt.step()


			with torch.no_grad():
				trainingLoss = 0.
				validationLoss = 0.

				for x, y in train_loader:
					y_pred = self.forward(x[0].double().to(device))
					trainingLoss += loss_fun(y_pred, y[0].to(device)).item() * x[0].shape[0]

				trainingLoss /= len(train_loader.dataset)

				for x, y in valid_loader:
					y_pred = self.forward(x[0].double().to(device))
					validationLoss += loss_fun(y_pred, y[0].to(device)).item()  * x[0].shape[0]

				validationLoss /= len(valid_loader.dataset)


				tf_logger.add_scalar('train loss', trainingLoss, epoch + 1)
				tf_logger.add_scalar('valid loss', validationLoss, epoch + 1)


			# Save last model
			state = {'epoch': epoch + 1, 'state_dict': self.state_dict(), 'optimizer': opt.state_dict()}
			torch.save(state, ckpt_dir + '/model_last.pth.tar')

			if validationLoss < bestValLoss:
				torch.save(state, ckpt_dir + '/model_best.pth.tar')
				bestValLoss = validationLoss


			if (epoch + 1) % 10 == 0:
				print("[*] Saving model epoch %d..." % (epoch + 1))
				torch.save(state, ckpt_dir + '/model_epoch%d.pth.tar' % (epoch + 1))

			early_stopping(validationLoss)

			# Stop training when not improving
			if early_stopping.early_stop:
				break

		print("[*] Finish training.")





if __name__ == '__main__':
	model = MLP(5, 3, 3)
	device = torch.device('cuda:0')

	model = model.double().to(device)

	x = torch.rand(20, 5)
	y = torch.zeros(20)
	y[:10] = 1
	y[-5:] = 2
