import os
import yaml
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.MVIB.utils.data import MultiOmicsDataset, SingleOmicsDataset
from src.MVIB.utils.evaluation import evaluate, split
from src.MVIB import training as training_module

def run(args: dict) -> None:

	save_dir = os.path.join(args['save_dir'], '{}'.format('MVIB'))
	os.makedirs(save_dir)

	parser.add_argument("--data-dir", type=str, default='.', help="Root path for the datasets.")
	parser.add_argument("--no-logging", action="store_true", help="Disable tensorboard logging")
	parser.add_argument("--overwrite", action="store_true",
						help="Force the over-writing of the previous experiment in the specified directory.")
	parser.add_argument("--device", type=str, default="cuda:0",
						help="Device on which the experiment is executed (as for tensor.device). Specify 'cpu' to "
							 "force execution on CPU.")
	parser.add_argument("--num-workers", type=int, default=8,
						help="Number of CPU threads used during the data loading procedure.")
	parser.add_argument("--batch-size", type=int, default=64, help="Batch size used for the experiments.")
	parser.add_argument("--load-model-file", type=str, default=None,
						help="Checkpoint to load for the experiments. Note that the specified configuration file needs "
							 "to be compatible with the checkpoint.")
	parser.add_argument("--checkpoint-every", type=int, default=50, help="Frequency of model checkpointing (in epochs).")
	parser.add_argument("--backup-every", type=int, default=5, help="Frequency of model backups (in epochs).")
	parser.add_argument("--evaluate-every", type=int, default=5, help="Frequency of model evaluation.")
	parser.add_argument("--epochs", type=int, default=1000, help="Total number of training epochs")



	logging = not args.no_logging
	save_dir = args.save_dir
	data_dir = args.data_dir
	config_file = args.config_file
	overwrite = args.overwrite
	device = args['cuda']
	num_workers = args.num_workers
	batch_size = args.batch_size
	load_model_file = args.load_model_file
	checkpoint_every = args.checkpoint_every
	backup_every = args.backup_every
	evaluate_every = args.evaluate_every
	epochs = args.epochs

	# Check if the experiment directory already contains a model
	pretrained = os.path.isfile(os.path.join(save_dir, 'model.pt')) \
				 and os.path.isfile(os.path.join(save_dir, 'config.yml'))


	if pretrained and not (config_file is None) and not overwrite:
		raise Exception("The experiment directory %s already contains a trained model, please specify a different "
						"experiment directory or remove the --config-file option to resume training or use the --overwrite"
						"flag to force overwriting")

	resume_training = pretrained and not overwrite


	if resume_training:
		load_model_file = os.path.join(save_dir, 'model.pt')
		config_file = os.path.join(save_dir, 'config.yml')

	if logging:
		from torch.utils.tensorboard import SummaryWriter
		writer = SummaryWriter(log_dir=save_dir)
	else:
		os.makedirs(save_dir, exist_ok=True)
		writer = None

	# Load the configuration file
	with open(config_file, 'r') as file:
		config = yaml.safe_load(file)

	# Copy it to the experiment folder
	with open(os.path.join(save_dir, 'config.yml'), 'w') as file:
		yaml.dump(config, file)

	# Instantiating the trainer according to the specified configuration
	TrainerClass = getattr(training_module, config['trainer'])
	trainer = TrainerClass(writer=writer, **config['params'])

	# Resume the training if specified
	if load_model_file:
		trainer.load(load_model_file)

	# Moving the models to the specified device
	trainer.to(device)
	#
	trainer.double()

	###########
	# Dataset #
	###########
	dataset_suffix = '_train.npy'
	train_set = SingleOmicsDataset(data_dir + 'view1' + dataset_suffix, data_dir + 'y' + dataset_suffix)
	dataset_suffix = '_test.npy'
	test_set = SingleOmicsDataset(data_dir + 'view1' + dataset_suffix, data_dir + 'y' + dataset_suffix)


	dataset_suffix = '_train.npy'
	mv_train_set = MultiOmicsDataset(data_dir + 'view1' + dataset_suffix, data_dir + 'view2' + dataset_suffix, data_dir + 'y' + dataset_suffix )

	dataset_suffix = '_test.npy'
	mv_test_set = MultiOmicsDataset(data_dir + 'view1' + dataset_suffix, data_dir + 'view2' + dataset_suffix, data_dir + 'y' + dataset_suffix )


	# Initialization of the data loader
	train_loader = DataLoader(mv_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

	# Select a subset 100 samples (10 for each per label)
	train_subset = split(train_set, 100, 'Balanced')

	##########

	checkpoint_count = 1

	for epoch in tqdm(range(epochs)):
		for data in tqdm(train_loader):
			trainer.train_step(data)

		if epoch % evaluate_every == 0:
			# Compute train and test_accuracy of a logistic regression
			train_accuracy, test_accuracy = evaluate(encoder=trainer.encoder_v1, train_on=train_subset, test_on=test_set,
													 device=device)
			if not (writer is None):
				writer.add_scalar(tag='evaluation/train_accuracy', scalar_value=train_accuracy, global_step=trainer.iterations)
				writer.add_scalar(tag='evaluation/test_accuracy', scalar_value=test_accuracy, global_step=trainer.iterations)

			tqdm.write('Train Accuracy: %f' % train_accuracy)
			tqdm.write('Test Accuracy: %f' % test_accuracy)

		if epoch % checkpoint_every == 0:
			tqdm.write('Storing model checkpoint')
			while os.path.isfile(os.path.join(save_dir, 'checkpoint_%d.pt' % checkpoint_count)):
				checkpoint_count += 1

			trainer.save(os.path.join(save_dir, 'checkpoint_%d.pt' % checkpoint_count))
			checkpoint_count += 1

		if epoch % backup_every == 0:
			tqdm.write('Updating the model backup')
			trainer.save(os.path.join(save_dir, 'model.pt'))
