from training.multiview_infomax import MVInfoMaxTrainer
from training.base import MultiModalRepresentationTrainer
from utils.schedulers import ExponentialScheduler
from utils.modules import MIEstimator



####################
# OmicsMIB Trainer #
####################

# the same as the MIBTrainer, but uses
# two different encoders with different sizes and parameters

class OmicsMIBTrainer(MultiModalRepresentationTrainer):
	def __init__(self, miest_lr, beta_start_value=1e-3, beta_end_value=1,
				 beta_n_iterations=100000, beta_start_iteration=50000, **params):
		# The neural networks architectures and initialization procedure is analogous to Multi-View InfoMax
		super(OmicsMIBTrainer, self).__init__(**params)

		# Initialization of the mutual information estimation network
		self.mi_estimator = MIEstimator(self.z_dim, self.z_dim)

		# Adding the parameters of the estimator to the optimizer
		self.opt.add_param_group(
			{'params': self.mi_estimator.parameters(), 'lr': miest_lr}
		)


		# Definition of the scheduler to update the value of the regularization coefficient beta over time
		self.beta_scheduler = ExponentialScheduler(start_value=beta_start_value, end_value=beta_end_value,
												   n_iterations=beta_n_iterations, start_iteration=beta_start_iteration)


	def _get_items_to_store(self):
		items_to_store = super(OmicsMIBTrainer, self)._get_items_to_store()

		# Add the mutual information estimator parameters to items_to_store
		items_to_store['mi_estimator'] = self.mi_estimator.state_dict()
		return items_to_store



	def _compute_loss(self, data):
		# Read the two views v1 and v2 and ignore the label y
		v1, v2, _ = data

		# Encode a batch of data
		p_z1_given_v1 = self.encoder_v1(v1)
		p_z2_given_v2 = self.encoder_v2(v2)

		# Sample from the posteriors with reparametrization
		z1 = p_z1_given_v1.rsample()
		z2 = p_z2_given_v2.rsample()

		# Mutual information estimation
		mi_gradient, mi_estimation = self.mi_estimator(z1, z2)
		mi_gradient = mi_gradient.mean()
		mi_estimation = mi_estimation.mean()

		# Symmetrized Kullback-Leibler divergence
		kl_1_2 = p_z1_given_v1.log_prob(z1) - p_z2_given_v2.log_prob(z1)
		kl_2_1 = p_z2_given_v2.log_prob(z2) - p_z1_given_v1.log_prob(z2)
		skl = (kl_1_2 + kl_2_1).mean() / 2.

		# Update the value of beta according to the policy
		beta = self.beta_scheduler(self.iterations)

		# Logging the components
		self._add_loss_item('loss/I_z1_z2', mi_estimation.item())
		self._add_loss_item('loss/SKL_z1_z2', skl.item())
		self._add_loss_item('loss/beta', beta)

		# Computing the loss function
		loss = - mi_gradient + beta * skl

		return loss
