import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, TensorDataset
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np


class MultiOmicsDataset:
	def __init__(self, filename1, filename2, filenameLabels):
		# load the three data matrices and make them datasets
		self.d1 = torch.tensor(np.load(filename1))
		self.d2 = torch.tensor(np.load(filename2))
		self.y = torch.tensor(np.load(filenameLabels))

		# store number of items and make sure size matches in all types
		self.length = len(self.d1)
		assert len(self.d2) == self.length
		assert len(self.y) == self.length

	def __getitem__(self, index):
		# get the index-th element of the 3 matrices
		return self.d1[index], self.d2[index], self.y[index]

	def __len__(self):
		return self.length



# Wrapper to create Multi-View datasets starting from 1 view and augmentation
class AugmentedDataset(Dataset):
	def __init__(self, dataset, augmentation, transform=None, target_transform=None, apply_same=False):
		assert hasattr(augmentation, '__call__')

		self.dataset = dataset
		self.augmentation = augmentation
		self.transform = transform
		self.target_transform = target_transform
		self.to_tensor = transforms.ToTensor()
		self.apply_same = apply_same

	def __getitem__(self, index):
		x, y = self.dataset[index]

		v_1 = self.augmentation(x)

		if self.apply_same:
			v_2 = v_1
		else:
			v_2 = self.augmentation(x)

		if self.transform is not None:
			v_1 = self.transform(v_1)
			v_2 = self.transform(v_2)

		if self.target_transform is not None:
			y = self.target_transform(y)

		return v_1, v_2, y

	def __len__(self):
		return len(self.dataset)



# Transform which randomly corrupts pixels with a given probabiliy
class PixelCorruption(object):
	MODALITIES = ['flip', 'drop']

	def __init__(self, p, min=0, max=1, mode='drop'):
		super(PixelCorruption, self).__init__()

		assert mode in self.MODALITIES

		self.p = p
		self.min = min
		self.max = max
		self.mode = mode

	def __call__(self, im):
		if isinstance(im, Image.Image) or isinstance(im, np.ndarray):
			im = F.to_tensor(im)

		if self.p < 1.0:
			mask = torch.bernoulli(torch.zeros(im.size(1), im.size(2)) + 1. - self.p).bool()
		else:
			mask = torch.zeros(im.size(1), im.size(2)).bool()

		if len(im.size())>2:
			mask = mask.unsqueeze(0).repeat(im.size(0),1,1)

		if self.mode == 'flip':
			im[mask] = self.max - im[mask]
		elif self.mode == 'drop':
			im[mask] = self.min

		return im
