import numpy as np
import torch
from torch.utils.data import Subset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


class EmbeddedDataset:
	BLOCK_SIZE = 256

	def __init__(self, base_dataset, encoder1, encoder2, device='cpu'):
		encoder1 = encoder1.to(device)
		encoder2 = encoder2.to(device)
		self.means1, self.means2, self.target = self._embed(encoder1, encoder2, base_dataset, device)

	def _embed(self, encoder1, encoder2, dataset, device):
		encoder1.eval()
		encoder2.eval()

		data_loader = torch.utils.data.DataLoader(
			dataset,
			batch_size=self.BLOCK_SIZE,
			shuffle=False)

		ys = []
		reps1 = []
		reps2 = []
		with torch.no_grad():
			# list of omic1, omic2, y
			for omics_data in data_loader:
				x = omics_data[0].to(device)
				y = omics_data[1].to(device)
				labels = omics_data[2].to(device)

				p_z1_given_x = encoder1(x)
				p_z2_given_y = encoder2(y)

				reps1.append(p_z1_given_x.mean.detach())
				reps2.append(p_z2_given_y.mean.detach())
				ys.append(labels)

			ys = torch.cat(ys, 0)

		return reps1, reps2, ys

	def __getitem__(self, index):
		y = self.target[index]
		x1 = self.means1[index // self.BLOCK_SIZE][index % self.BLOCK_SIZE]
		x2 = self.means2[index // self.BLOCK_SIZE][index % self.BLOCK_SIZE]

		return x1, x2, y

	def __len__(self):
		return self.target.size(0)


def split(dataset, size, split_type):
	if split_type == 'Random':
		data_split, _ = torch.utils.data.random_split(dataset, [size, len(dataset) - size])
	elif split_type == 'Balanced':
		class_ids = {}
		for idx, (_, y) in enumerate(dataset):
			if isinstance(y, torch.Tensor):
				y = y.item()
			if y not in class_ids:
				class_ids[y] = []
			class_ids[y].append(idx)

		ids_per_class = size // len(class_ids)

		selected_ids = []

		for ids in class_ids.values():
			selected_ids += list(np.random.choice(ids, min(ids_per_class, len(ids)), replace=False))
		data_split = Subset(dataset, selected_ids)

	return data_split


def build_matrix(dataset):
	data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

	x1s = []
	x2s = []
	ys = []

	for omics_data in data_loader:
		x1s.append(omics_data[0])
		x2s.append(omics_data[1])
		ys.append(omics_data[2])

	x1s = torch.cat(x1s, 0)
	x2s = torch.cat(x2s, 0)
	ys = torch.cat(ys, 0)

	if x1s.is_cuda:
		x1s = x1s.cpu()
	if x2s.is_cuda:
		x2s = x1s.cpu()
	if ys.is_cuda:
		ys = ys.cpu()

	return x1s.data.numpy(), x2s.data.numpy(), ys.data.numpy()


def evaluate(encoder1, encoder2, train_on, test_on, device):
	embedded_train = EmbeddedDataset(train_on, encoder1, encoder2, device=device)
	embedded_test = EmbeddedDataset(test_on, encoder1, encoder2, device=device)
	return train_and_evaluate_linear_model(embedded_train, embedded_test)


def train_and_evaluate_linear_model_from_matrices(x_train, y_train, solver='saga', multi_class='multinomial', tol=.1, C=10):
	model = LogisticRegression(solver=solver, multi_class=multi_class, tol=tol, C=C)
	model.fit(x_train, y_train)
	return model


def train_and_evaluate_linear_model(train_set, test_set, solver='saga', multi_class='multinomial', tol=.1, C=10):
	x1_train, x2_train, y_train = build_matrix(train_set)
	x1_test, x2_test, y_test = build_matrix(test_set)

	scaler = MinMaxScaler()

	x1_train = scaler.fit_transform(x1_train)
	x2_train = scaler.fit_transform(x2_train)
	x1_test = scaler.transform(x1_test)
	x2_test = scaler.transform(x2_test)

	model = LogisticRegression(solver=solver, multi_class=multi_class, tol=tol, C=C)
	model.fit(x1_train, y_train)

	test1_accuracy = model.score(x1_test, y_test)
	train1_accuracy = model.score(x1_train, y_train)

	model = LogisticRegression(solver=solver, multi_class=multi_class, tol=tol, C=C)
	model.fit(x2_train, y_train)

	test2_accuracy = model.score(x2_test, y_test)
	train2_accuracy = model.score(x2_train, y_train)

	# Now what do we do here?
	train_accuracy = (train1_accuracy + train2_accuracy) / 2
	test_accuracy = (test1_accuracy + test2_accuracy) / 2

	return train_accuracy, test_accuracy
