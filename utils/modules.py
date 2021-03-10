import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus


# custom encoder for omics data of unknown size
# also allows choosing 0 or 1 hidden layer

class OmicsEncoder(nn.Module):
	def __init__(self, input_dim, z_dim, hidden_dim=[]):
		super(OmicsEncoder, self).__init__()

		self.z_dim = z_dim

		in_neurons = [input_dim] + hidden_dim
		out_neurons = hidden_dim + [2 * z_dim]

		encode_layers = []
		# Vanilla MLP
		for i, (in_d, out_d) in enumerate(zip(in_neurons, out_neurons)):
			encode_layers.append(nn.Linear(in_d, out_d))

		self.num_layers = len(encode_layers)
		self.encode_layers = encode_layers

	def forward(self, x):

		for i, layer in enumerate(self.encode_layers):
			x = layer(x)
			if i < self.num_layers - 1:
				x = F.relu(x)

		mu, sigma = x[:, :self.z_dim], x[:, self.z_dim:]
		sigma = softplus(sigma) + 1e-7  # Make sigma always positive

		return Independent(Normal(loc=mu, scale=sigma), 1)  # Return a factorized Normal distribution





# Encoder architecture
class Encoder(nn.Module):
	def __init__(self, z_dim):
		super(Encoder, self).__init__()

		self.z_dim = z_dim

		# Vanilla MLP
		self.net = nn.Sequential(
			nn.Linear(28 * 28, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 1024),
			nn.ReLU(True),
			nn.Linear(1024, z_dim * 2),
		)

	def forward(self, x):
		x = x.view(x.size(0), -1)  # Flatten the input
		params = self.net(x)

		mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
		sigma = softplus(sigma) + 1e-7  # Make sigma always positive

		return Independent(Normal(loc=mu, scale=sigma), 1)  # Return a factorized Normal distribution


class Decoder(nn.Module):
	def __init__(self, z_dim, scale=0.39894):
		super(Decoder, self).__init__()

		self.z_dim = z_dim
		self.scale = scale

		# Vanilla MLP
		self.net = nn.Sequential(
			nn.Linear(z_dim, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 28 * 28)
		)

	def forward(self, z):
		x = self.net(z)
		return Independent(Normal(loc=x, scale=self.scale), 1)


# Auxiliary network for mutual information estimation
# change the default to only one hidden layer
# in the future, make it possible to choose architecture (like OmicsEncoder)

class MIEstimator(nn.Module):
	def __init__(self, size1, size2):
		super(MIEstimator, self).__init__()

		# Vanilla MLP
		self.net = nn.Sequential(
			nn.Linear(size1 + size2, 1500),
			nn.ReLU(True),
			nn.Linear(1500, 1),
		)

	# Gradient for JSD mutual information estimation and EB-based estimation
	def forward(self, x1, x2):
		pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
		neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
		return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1
