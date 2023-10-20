import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, Beta
import torch.optim as optimizer_module
import math
from tensorboardX import SummaryWriter
import src.util.mydistributions as mydistributions
import pyro.distributions.zero_inflated as pyroZI
# several utils

# utility function to initialize an optimizer from its name
def init_optimizer(optimizer_name, params):
	assert hasattr(optimizer_module, optimizer_name)
	OptimizerClass = getattr(optimizer_module, optimizer_name)
	return OptimizerClass(params)


def identity(x):
	return x


def getPointEstimate(d):
	if type(d) is torch.distributions.categorical.Categorical:
		return torch.argmax(d.probs, -1)

	return d.mean


########################################################################################
# basic components

# fully-connected layers
class FullyConnectedModule(nn.Module):
	def __init__(self, input_dim, hidden_dim=[100], use_batch_norm=False, dropoutP=0.0, lastActivation='none'):
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

		return x


class ProbabilisticFullyConnectedModule(FullyConnectedModule):
	def __init__(self, input_dim, hidden_dim=[100], distribution='normal', use_batch_norm=False, dropoutP=0.0, lastActivation='none', n_categories=None, log_input=False):

		self.input_dim = input_dim
		self.n_categories = n_categories
		self.distribution = distribution
		self.log_input = log_input

		if self.distribution == 'categorical':
			assert n_categories is not None
			newlist = [h for h in hidden_dim]
			newlist[-1] *= n_categories
			hidden_dim = newlist

		elif self.distribution != 'cbernoulli' and self.distribution != 'bernoulli' and self.distribution != 'poisson':
			newlist = [h for h in hidden_dim]
			newlist[-1] = newlist[-1] * 2
			hidden_dim = newlist


		super(ProbabilisticFullyConnectedModule, self).__init__(input_dim, hidden_dim, use_batch_norm, dropoutP, lastActivation)


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
		elif self.distribution == 'categorical':
			self.D = torch.distributions.Categorical
		elif self.distribution == 'cbernoulli':
			self.D = torch.distributions.ContinuousBernoulli
		elif self.distribution == 'bernoulli':
			self.D = torch.distributions.Bernoulli
			self.p1Transform = identity
		elif self.distribution == 'poisson':
			self.p1Transform = torch.exp
			self.D = torch.distributions.Poisson
		elif self.distribution == 'laplace':
			self.p1Transform = identity
			self.p2Transform = torch.exp
			self.D = torch.distributions.Laplace
		elif self.distribution == 'zip':
			self.p1Transform = torch.exp
			self.p2Transform = torch.nn.Sigmoid()
			self.D = pyroZI.ZeroInflatedPoisson
		elif self.distribution == 'nb':
			self.p1Transform = torch.exp
			self.p2Transform = torch.nn.Sigmoid()
			self.D = torch.distributions.NegativeBinomial
		else:
			raise NotImplementedError('%s not supported. Use: \'normal\' or \'beta\'' % self.distribution)

		if self.distribution != 'cbernoulli' and self.distribution != 'bernoulli' and self.distribution != 'poisson':
			if self.distribution != 'categorical':
				self.z_dim = self.z_dim // 2
			else:
				self.z_dim = self.z_dim // self.n_categories

	def forward(self, x):
		if self.log_input:
			x = torch.log(x+1)

		z = super().forward(x)

		#if self.distribution != 'cbernoulli' and self.distribution != 'categorical'and self.distribution != 'poisson' and self.distribution != 'poisson':
		if self.distribution not in {'cbernoulli', 'bernoulli', 'categorical', 'poisson', 'zip'}:
			p1 = self.p1Transform(z[:, :self.z_dim])
			p2 = self.p2Transform(z[:, self.z_dim:])

			return Independent(self.D(p1, p2), 0)
		else:
			if self.distribution == 'poisson':
				p1 = self.p1Transform(z)
				return Independent(self.D(p1), 0)

			if self.distribution == 'zip':
				p1 = self.p1Transform(z[:, :self.z_dim])
				p2 = self.p2Transform(z[:, self.z_dim:])

				return Independent(self.D(p1, gate=p2), 0)

			if self.distribution == 'categorical':
				z = torch.reshape(z, (-1, self.z_dim, self.n_categories))

			if self.lastActivation is None:
				#return Independent(self.D(logits=z), 0)
				return self.D(logits=z)
			elif self.lastActivation == torch.sigmoid:
				#return Independent(self.D(probs=z), 0)
				return self.D(probs=z)

			else:
				raise NotImplementedError('Wrong activation for lamda parameter of ContinuousBernoulli')


class SCVIdecoder(FullyConnectedModule):
	def __init__(self, input_dim, hidden_dim=[100], distribution='nb', use_batch_norm=False, dropoutP=0.0):

		self.input_dim = input_dim
		self.distribution = distribution
		self.Ngenes = hidden_dim[-1]
		# if self.distribution == 'nb':
		# 	newlist = [h for h in hidden_dim]
		# 	newlist[-1] *= n_categories
		# 	hidden_dim = newlist

		if self.distribution == 'zinb' or self.distribution == 'nbm':
			newlist = [h for h in hidden_dim]
			newlist[-1] = newlist[-1] * 2
			hidden_dim = newlist
		elif self.distribution != 'nb':
			raise NotImplementedError

		super(SCVIdecoder, self).__init__(input_dim, hidden_dim, use_batch_norm, dropoutP, 'none')

		if self.distribution == 'nb':
			self.p1Transform = torch.exp
			self.p2Transform = None
			self.D = mydistributions.NegativeBinomial
		elif self.distribution == 'nbm':
			self.p1Transform = torch.exp
			self.p2Transform = None
			self.D = mydistributions.NegativeBinomialMixture
		elif self.distribution == 'zinb':
			self.p1Transform = torch.exp
			self.p2Transform = None
			self.D = mydistributions.ZeroInflatedNegativeBinomial

		else:
			raise NotImplementedError('%s not supported. Use: \'nb\', \'nbm\' or \'zinb\'' % self.distribution)

		if self.distribution != 'nb':
			self.z_dim = self.z_dim // 2

		# initialization log(phi) ~ N(-2, 0.2) --> phi in range ~[0.05, 0.30]
		self.phi = torch.nn.Parameter(torch.normal(-2*torch.ones(self.z_dim), 0.2*torch.ones(self.z_dim)), requires_grad=True)

		if self.distribution == 'nbm':
			self.protein_background_beta = torch.nn.Parameter(torch.normal(torch.zeros(self.z_dim), torch.ones(self.z_dim)))


	def forward(self, x):
		z = super().forward(x)

		if self.distribution == 'nb':
			mean = self.p1Transform(z)
			return Independent(self.D(mean, torch.exp(self.phi)), 0)

		elif self.distribution == 'zinb':
			mean = self.p1Transform(z[:, :self.z_dim])
			logitP = z[:, self.z_dim:]
			return Independent(self.D(mean, torch.exp(self.phi), logitP), 0)

		elif self.distribution == 'nbm':
			mean1 = torch.exp(self.protein_background_beta)

			alpha = 1 + torch.exp(z[:, :self.z_dim])
			mean2 = mean1 * alpha
			logitP = z[:, self.z_dim:]

			return Independent(self.D(mean1, mean2, torch.exp(self.phi), logitP), 0)



		else:
			raise NotImplementedError


# base class
class RepresentationLearner(nn.Module):
	def __init__(self, input_dim, enc_hidden_dim=[100], use_batch_norm=False, dropoutP=0.0, optimizer_name='Adam', encoder_lr=1e-4, probabilistic_encoder=True, enc_distribution='normal'):

		super(RepresentationLearner, self).__init__()

		self._epoch = 0
		self.loss_items = {}

		self._input_dim = input_dim
		self._dropoutP = dropoutP
		self._use_batch_norm = use_batch_norm

		self.z_dim = enc_hidden_dim[-1]

		# Intialization of the encoder
		if probabilistic_encoder:
			self.encoder = ProbabilisticFullyConnectedModule(input_dim, enc_hidden_dim, enc_distribution, use_batch_norm, dropoutP, 'none')
		else:
			self.encoder = FullyConnectedModule(input_dim, enc_hidden_dim, use_batch_norm, dropoutP, 'none')

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


class MultiOmicRepresentationLearner(nn.Module):
	def __init__(self, input_dims, enc_hidden_dim=[100], use_batch_norm=False, dropoutP=0.0, optimizer_name='Adam', probabilistic_encoder=True, encoder_lr=1e-4, enc_distribution='normal', cpu=False, log_input=False):
		# check input arguments
		super(MultiOmicRepresentationLearner, self).__init__()

		if cpu:
			self.device = torch.device('cpu')
		else:
			self.device = torch.device('cuda:0')

		self.n_modalities = len(input_dims)

		if type(encoder_lr) == float:
			encoder_lr = [encoder_lr] * self.n_modalities

		if type(log_input) is bool:
			log_inputs = [log_input] * self.n_modalities
		else:
			assert type(log_input) is list
			log_inputs = log_input

		self._epoch = 0
		self.loss_items = {}

		self._input_dims = input_dims
		self._dropoutP = dropoutP
		self._use_batch_norm = use_batch_norm

		self.z_dim = enc_hidden_dim[-1]

		# Intialization of the encoder
		if probabilistic_encoder:
			self.encoders = [ProbabilisticFullyConnectedModule(input_dim, enc_hidden_dim, enc_distribution, use_batch_norm, dropoutP, 'none', n_categories=None, log_input=log_input).double().to(self.device) for input_dim, log_input in zip(input_dims, log_inputs)]
		else:
			self.encoders = [FullyConnectedModule(input_dim, enc_hidden_dim, use_batch_norm, dropoutP, 'none').double().to(self.device) for input_dim in input_dims]


		for i, (enc, lr) in enumerate(zip(self.encoders, encoder_lr)):
			if i == 0:
				self.opt = init_optimizer(optimizer_name, [{'params': enc.parameters(), 'lr': lr}])
			else:
				self.opt.add_param_group({'params': enc.parameters(), 'lr': lr})


	def increment_epoch(self):
		self._epoch += 1

	def get_device(self):
		return list(self.parameters())[0].device


	def encode(self, x):
		z = [enc(xi) for enc, xi in zip(self.encoders, x)]

		return z


class VariationalAutoEncoder(RepresentationLearner):

	def __init__(self, input_dim, enc_hidden_dim=[100], dec_hidden_dim=[], likelihood='normal', use_batch_norm=False, dropoutP=0.0, optimizer_name='Adam', encoder_lr=1e-4, decoder_lr=1e-4, beta=1.):
		super(VariationalAutoEncoder, self).__init__(input_dim, enc_hidden_dim, use_batch_norm, dropoutP, optimizer_name, encoder_lr)

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
	def __init__(self, input_dims, enc_hidden_dim=[100], dec_hidden_dim=[], likelihoods='normal', use_batch_norm=False, dropoutP=0.0, optimizer_name='Adam', encoder_lr=1e-4, decoder_lr=1e-4, enc_distribution='normal', beta=1.0, zconstraintCoef=1.0, crossPenaltyCoef=1.0, n_categories=None, log_input=False):
		super(CrossGeneratingVariationalAutoencoder, self).__init__(input_dims, enc_hidden_dim, use_batch_norm, dropoutP, optimizer_name, True, encoder_lr, enc_distribution, log_input=log_input)

		if type(likelihoods) == str:
			likelihoods = [likelihoods] * self.n_modalities


		if 'categorical' in likelihoods:
			assert n_categories is not None
		else:
			n_categories = [None for _ in range(self.n_modalities)]

		#self.decoders = [ProbabilisticFullyConnectedModule(self.z_dim, dec_hidden_dim + [input_dim1], distribution=ll, use_batch_norm=self._use_batch_norm, dropoutP=self._dropoutP, lastActivation='none', n_categories=n_categories1).double().to(self.device) for input_dim1, ll, n_categories1 in zip(input_dims, likelihoods, n_categories)]
		self.decoders = []
		for input_dim1, ll, n_categories1 in zip(input_dims, likelihoods, n_categories):
			if ll not in {'nb', 'zinb', 'nbm'}:
				self.decoders.append(ProbabilisticFullyConnectedModule(self.z_dim, dec_hidden_dim + [input_dim1], distribution=ll, use_batch_norm=self._use_batch_norm, dropoutP=self._dropoutP, lastActivation='none', n_categories=n_categories1).double().to(self.device))
			else:
				#input_dim, hidden_dim=[100], distribution='nb', use_batch_norm=False, dropoutP=0.0
				self.decoders.append(SCVIdecoder(self.z_dim, dec_hidden_dim + [input_dim1], distribution=ll, use_batch_norm=self._use_batch_norm, dropoutP=self._dropoutP).double().to(self.device))

		if type(decoder_lr) == float:
			decoder_lr = [decoder_lr] * self.n_modalities

		for lr, dec in zip(decoder_lr, self.decoders):
			self.opt.add_param_group({'params': dec.parameters(), 'lr': lr})

		self.zconstraintCoef = zconstraintCoef
		self.crossPenaltyCoef = crossPenaltyCoef
		self.beta = beta

	def eval(self):
		self.training = False
		for enc in self.encoders:
			enc.eval()
		for dec in self.decoders:
			dec.eval()

	def train(self):
		self.training = True
		for enc in self.encoders:
			enc.train()
		for dec in self.decoders:
			dec.train()



	def compute_loss(self, x):

		# encode all modalities
		z = [enc(xi) for xi, enc in zip(x, self.encoders)]
		# sample from each z
		zsample = [zi.rsample() for zi in z]


		zmean = [zi.mean for zi in z]
		zstd = [zi.stddev for zi in z]

		# decode each modality from all z's
		x_hat = [[dec(zi) for dec in self.decoders] for zi in zsample]

		# calculate loss
		loss = 0.0
		for i in range(self.n_modalities):
			for j in range(self.n_modalities):
				if i == j:
					loss -= torch.sum(torch.mean(x_hat[i][j].log_prob(x[j]), 1))
				else:
					loss -= torch.sum(torch.mean(x_hat[i][j].log_prob(x[j]), 1)) * self.crossPenaltyCoef

					if i > j:
						# wasserstein has to be calculated only once for each pair of modalities
						similarityConstraint = nn.MSELoss(reduction='none')(zmean[i], zmean[j]) + nn.MSELoss(reduction='none')(zstd[i], zstd[j])
						similarityConstraint = torch.sum(torch.mean(similarityConstraint, 1))
						loss += self.zconstraintCoef * similarityConstraint

			# KL divergence to prior for each modality's z
			loss -= 0.5 * self.beta * torch.sum(torch.mean(1 + torch.log(zstd[i] ** 2) - (zmean[i] ** 2) - (zstd[i] ** 2), 1))


		return loss


	def evaluate(self, x):
		metrics = dict()
		with torch.no_grad():
			# encode all modalities
			z = [enc(xi) for xi, enc in zip(x, self.encoders)]
			# sample from each z
			zsample = [zi.rsample() for zi in z]

			zmean = [zi.mean for zi in z]
			zstd = [zi.stddev for zi in z]

			# decode each modality from all z's
			x_hat = [[dec(zi) for dec in self.decoders] for zi in zmean]

			loss = 0.0
			for i in range(self.n_modalities):
				klkey = 'KL/%d' % (i+1)
				for j in range(self.n_modalities):
					llkey = 'LL%d/%d' % (j+1, i+1)
					msekey = 'MSE%d/%d' % (j+1, i+1)
					w2key = 'z-W2/%d-%d' % (i+1, j+1)
					l2key = 'z-L2/%d-%d' % (i+1, j+1)
					coskey = 'z-cos/%d-%d' % (i+1, j+1)

					metrics[llkey] = torch.mean(torch.sum(x_hat[i][j].log_prob(x[j]), 1)).item()
					metrics[msekey] = torch.mean(torch.sum(torch.nn.MSELoss(reduction='none')(getPointEstimate(x_hat[i][j]), x[j]), 1)).item()

					if i == j:
						loss -= metrics[llkey]
					else:
						loss -= metrics[llkey] * self.crossPenaltyCoef

						if i > j:
							# wasserstein has to be calculated only once for each pair of modalities
							similarityConstraint = nn.MSELoss(reduction='none')(zmean[i], zmean[j]) + nn.MSELoss(reduction='none')(zstd[i], zstd[j])
							similarityConstraint = torch.mean(torch.sum(similarityConstraint, 1)).item()

							metrics[w2key] = similarityConstraint
							loss += self.zconstraintCoef * similarityConstraint

							l2 = nn.MSELoss(reduction='none')(zmean[i], zmean[j])
							metrics[l2key] = torch.mean(torch.sum(l2, 1)).item()

							metrics[coskey] = torch.mean(1 - torch.cosine_similarity(zmean[i],zmean[j])).item()


				# KL divergence to prior for each modality's z
				metrics[klkey] = -0.5 * torch.mean(torch.sum(1 + torch.log(zstd[i] ** 2) - (zmean[i] ** 2) - (zstd[i] ** 2), 1)).item()
				loss += self.beta * metrics[klkey]


		metrics['loss'] = loss

		return metrics


	def reconstructionPerDataPoint(self, x):
		metrics = dict()
		with torch.no_grad():
			# encode all modalities
			z = [enc(xi) for xi, enc in zip(x, self.encoders)]

			zmean = [zi.mean for zi in z]

			# decode each modality from all z's
			x_hat = [[dec(zi) for dec in self.decoders] for zi in zmean]

			for i in range(self.n_modalities):
				for j in range(self.n_modalities):
					llkey = 'LL%d/%d' % (j+1, i+1)

					metrics[llkey] = torch.sum(x_hat[i][j].log_prob(x[j]), 1)

		return metrics



	def encode(self, x):
		z = super().encode(x)
		return [zi.mean for zi in z]

	def embedAndReconstruct(self, x):
		with torch.no_grad():
			z = self.encode(x)

			x_hat = [[dec(zi) for dec in self.decoders] for zi in z]

			return z, x_hat

class ConcatenatedVariationalAutoencoder(MultiOmicRepresentationLearner):
	def __init__(self, input_dims, enc_hidden_dim=[100], dec_hidden_dim=[], likelihoods='normal', use_batch_norm=False, dropoutP=0.0, optimizer_name='Adam', encoder_lr=1e-4, decoder_lr=1e-4, enc_distribution='normal', beta=1.0, n_categories=None, log_input=False):
		super(ConcatenatedVariationalAutoencoder, self).__init__([sum(input_dims)], enc_hidden_dim, use_batch_norm, dropoutP, optimizer_name, True, encoder_lr, enc_distribution, log_input=log_input)

		self.n_modalities = len(input_dims)
		if type(likelihoods) == str:
			likelihoods = [likelihoods] * self.n_modalities

		if 'categorical' in likelihoods:
			assert n_categories is not None
		else:
			n_categories = [None for _ in range(self.n_modalities)]

		#self.decoders = [ProbabilisticFullyConnectedModule(self.z_dim, dec_hidden_dim + [input_dim1], distribution=ll, use_batch_norm=self._use_batch_norm, dropoutP=self._dropoutP, lastActivation='none', n_categories=n_categories1).double().to(self.device) for input_dim1, ll, n_categories1 in zip(input_dims, likelihoods, n_categories)]
		self.decoders = []
		for input_dim1, ll, n_categories1 in zip(input_dims, likelihoods, n_categories):
			if ll not in {'nb', 'zinb', 'nbm'}:
				self.decoders.append(ProbabilisticFullyConnectedModule(self.z_dim, dec_hidden_dim + [input_dim1], distribution=ll, use_batch_norm=self._use_batch_norm, dropoutP=self._dropoutP, lastActivation='none', n_categories=n_categories1).double().to(self.device))
			else:
				#input_dim, hidden_dim=[100], distribution='nb', use_batch_norm=False, dropoutP=0.0
				self.decoders.append(SCVIdecoder(self.z_dim, dec_hidden_dim + [input_dim1], distribution=ll, use_batch_norm=self._use_batch_norm, dropoutP=self._dropoutP).double().to(self.device))


		if type(decoder_lr) == float:
			decoder_lr = [decoder_lr] * self.n_modalities

		for lr, dec in zip(decoder_lr, self.decoders):
			self.opt.add_param_group({'params': dec.parameters(), 'lr': lr})

		self.beta = beta

	def eval(self):
		self.training = False
		for enc in self.encoders:
			enc.eval()
		for dec in self.decoders:
			dec.eval()

	def train(self):
		self.training = True
		for enc in self.encoders:
			enc.train()
		for dec in self.decoders:
			dec.train()


	def compute_loss(self, x):

		# encode all modalities
		z = self.encoders[0](torch.cat(x,axis=1))
		# sample from each z
		zsample = z.rsample()


		zmean = z.mean
		zstd = z.stddev

		# decode each modality from all z's
		x_hat = [dec(zsample) for dec in self.decoders]

		# calculate loss
		loss = 0.0
		for i in range(self.n_modalities):
			loss -= torch.sum(torch.mean(x_hat[i].log_prob(x[i]), 1))


		# KL divergence to prior
		loss -= 0.5 * self.beta * torch.sum(torch.mean(1 + torch.log(zstd ** 2) - (zmean ** 2) - (zstd ** 2), 1))


		return loss

	def evaluate(self, x):

		metrics = dict()
		with torch.no_grad():

			# encode all modalities
			z = self.encoders[0](torch.cat(x,axis=1))
			# sample from each z
			zmean = z.mean
			zstd = z.stddev

			# decode each modality from all z's
			x_hat = [dec(zmean) for dec in self.decoders]

			# calculate loss
			loss = 0.0
			for i in range(self.n_modalities):
				# evaluate reconstruction of each modality from all data
				llkey = 'LL%d' % (i+1)
				msekey = 'MSE%d' % (i+1)

				metrics[llkey] = torch.mean(torch.sum(x_hat[i].log_prob(x[i]), 1)).item()
				metrics[msekey] = torch.mean(torch.sum(torch.nn.MSELoss(reduction='none')(getPointEstimate(x_hat[i]), x[i]), 1)).item()

				loss -= metrics[llkey]

				# evaluate reconstruction of each modality from one modality
				y = [xi if j == i else torch.zeros(xi.shape).double().to(self.device) for j, xi in enumerate(x)]
				zz = self.encoders[0](torch.cat(y, axis=1))
				zzmean = zz.mean
				zzstd = zz.stddev
				xx_hat = [dec(zzmean) for dec in self.decoders]

				for j in range(self.n_modalities):
					llkey = 'LL%d/%d' % (j+1, i+1)
					msekey = 'MSE%d/%d' % (j+1, i+1)

					metrics[llkey] = torch.mean(torch.sum(xx_hat[j].log_prob(x[j]), 1)).item()
					metrics[msekey] = torch.mean(torch.sum(torch.nn.MSELoss(reduction='none')(getPointEstimate(x_hat[j]), x[j]), 1)).item()


			metrics['KL'] = -0.5 * self.beta * torch.sum(torch.mean(1 + torch.log(zstd ** 2) - (zmean ** 2) - (zstd ** 2), 1)).item()
			loss += metrics['KL']

			metrics['loss'] = loss

		return metrics

	def reconstructionPerDataPoint(self, x):
		metrics = dict()
		with torch.no_grad():
			# encode all modalities
			z = self.encoders[0](torch.cat(x,axis=1))
			# sample from each z
			zmean = z.mean
			zstd = z.stddev

			# decode each modality from all z's
			x_hat = [dec(zmean) for dec in self.decoders]

			for i in range(self.n_modalities):
				# evaluate reconstruction of each modality from one modality
				y = [xi if j == i else torch.zeros(xi.shape).double().to(self.device) for j, xi in enumerate(x)]
				zz = self.encoders[0](torch.cat(y, axis=1))
				zzmean = zz.mean
				zzstd = zz.stddev
				xx_hat = [dec(zzmean) for dec in self.decoders]

				for j in range(self.n_modalities):
					llkey = 'LL%d/%d' % (j+1, i+1)

					metrics[llkey] = torch.sum(xx_hat[j].log_prob(x[j]), 1)




		return metrics






	def encode(self, x):
		z = self.encoders[0](torch.cat(x,axis=1))
		return z.mean

	def embedAndReconstruct(self, x):
		with torch.no_grad():
			z = self.encode(x)

			x_hat = [dec(z) for dec in self.decoders]

			return z, x_hat



# Auxiliary network for mutual information estimation

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
	def __init__(self, input_dim1, input_dim2, enc_hidden_dim=[100], mi_net_arch=[100, 1], use_batch_norm=False, dropoutP=0.0, optimizer_name='Adam', encoder1_lr=1e-4, enc1_lastActivation='none', encoder2_lr=1e-4, enc2_lastActivation='none', mi_net_lr=1e-4, beta=1.0):
		super().__init__(input_dim1, input_dim2, enc_hidden_dim, use_batch_norm, dropoutP, optimizer_name, True, encoder1_lr, enc1_lastActivation, encoder2_lr, enc2_lastActivation)

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


class OmicRegressor(ProbabilisticFullyConnectedModule):
	def __init__(self, input_dim, output_dim, distribution='normal', optimizer_name='Adam', lr=0.0001, n_categories=None):
		super(OmicRegressor, self).__init__(input_dim, [output_dim], distribution=distribution, use_batch_norm=False, dropoutP=0.0, lastActivation='none', n_categories=n_categories)

		self.opt = init_optimizer(optimizer_name, [
			{'params': self.parameters(), 'lr': lr},
		])

	def compute_loss(self, x):

		x0, y0 = x
		yhat = self.forward(x0)
		ll = yhat.log_prob(y0)

		loss = - torch.sum(torch.mean(ll, 1))


		return loss

	def evaluate(self, x):
		metrics = dict()
		with torch.no_grad():
			x0, y0 = x
			yhat = self.forward(x0)
			ll = yhat.log_prob(y0)

			metrics['loss'] = - torch.mean(torch.sum(ll, 1))
			#metrics['MSE'] = torch.mean(torch.sum(torch.nn.MSELoss(reduction='none')(getPointEstimate(yhat), y0), 1)).item()

		return metrics


class OmicRegressorSCVI(SCVIdecoder):
	def __init__(self, input_dim, output_dim, distribution='nb', optimizer_name='Adam', lr=0.0001, use_batch_norm=False, log_input=False):
		#print(use_batch_norm)
		super(OmicRegressorSCVI, self).__init__(input_dim, [output_dim], distribution=distribution, use_batch_norm=use_batch_norm, dropoutP=0.0)

		self.opt = init_optimizer(optimizer_name, [
			{'params': self.parameters(), 'lr': lr},
		])
		self.log_input = log_input

	def compute_loss(self, x):

		x0, y0 = x

		yhat = self.forward(x0)
		ll = yhat.log_prob(y0)

		loss = - torch.sum(torch.mean(ll, 1))


		return loss

	def evaluate(self, x):
		metrics = dict()
		with torch.no_grad():
			x0, y0 = x

			yhat = self.forward(x0)
			ll = yhat.log_prob(y0)

			metrics['loss'] = - torch.mean(torch.sum(ll, 1))
			#metrics['MSE'] = torch.mean(torch.sum(torch.nn.MSELoss(reduction='none')(getPointEstimate(yhat), y0), 1)).item()

		return metrics

	def forward(self, x):
		if self.log_input:
			x = torch.log(x+1)
		return super(OmicRegressorSCVI, self).forward(x)


if __name__ == '__main__':

	device = torch.device('cuda:0')

	model = ConcatenatedVariationalAutoencoder([5, 3], enc_hidden_dim=[2, 2], dec_hidden_dim=[2], likelihoods=['nb', 'bernoulli'], use_batch_norm=False, dropoutP=0.1, optimizer_name='Adam', encoder_lr=1e-4, decoder_lr=1e-4, enc_distribution='normal', beta=1.0, n_categories=[None, None], log_input=[True, False])
	#model = ConcatenatedVariationalAutoencoder([5, 3, 4], enc_hidden_dim=[2, 2], dec_hidden_dim=[2], likelihoods=['normal', 'nb', 'categorical'], use_batch_norm=False, dropoutP=0.0, optimizer_name='Adam', encoder_lr=1e-4, decoder_lr=1e-4, enc_distribution='normal', beta=1.0, n_categories=[None, None, 5])
	#model = CrossGeneratingVariationalAutoencoder([5, 3, 4], enc_hidden_dim=[2, 2], dec_hidden_dim=[2], likelihoods=['normal', 'nb', 'categorical'], use_batch_norm=False, dropoutP=0.0, optimizer_name='Adam', encoder_lr=1e-4, decoder_lr=1e-4, enc_distribution='normal', beta=1.0, n_categories=[None, None, 5])
	model = model.double().to(device)
	# model = SCVIdecoder(32, hidden_dim=[128,5000], distribution='nb', use_batch_norm=True, dropoutP=0.1).to(device)
	# model2 = SCVIdecoder(32, hidden_dim=[128,5000], distribution='zinb', use_batch_norm=True, dropoutP=0.1).to(device)

	sys.exit(0)
	x1 = torch.rand(20, 5).to(device)
	x2 = torch.rand(20, 3).to(device)
	x3 = torch.randint(0, 5, (20, 4)).to(device)

	# dec = model2(x1)

	x = [x1.double(), x2.double(), x3.double()]
	#x = [x1.double(), x2.double()]

	metrics = model.evaluate(x)
	print(metrics)
	model.opt.zero_grad()
	current_loss = model.compute_loss(x)
	current_loss.backward()
	model.opt.step()
	metrics = model.evaluate(x)
	print(metrics)
