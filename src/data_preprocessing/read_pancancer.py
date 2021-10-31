import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from myplots import *
import pickle
import sys

def cancerShortNames():

	name2code = {'Acute Myeloid Leukemia': 'LAML',
	'Adrenocortical carcinoma': 'ACC',
	'Bladder Urothelial Carcinoma': 'BLCA',
	'Brain Lower Grade Glioma': 'LGG',
	'Breast invasive carcinoma': 'BRCA',
	'Cervical squamous cell carcinoma and endocervical adenocarcinoma': 'CESC',
	'Cholangiocarcinoma': 'CHOL',
	'Colon adenocarcinoma': 'COAD',
	'Esophageal carcinoma': 'ESCA',
	'Esophageal carcinoma ': 'ESCA',
	'Glioblastoma multiforme': 'GBM',
	'Head and Neck squamous cell carcinoma': 'HNSC',
	'Kidney Chromophobe': 'KICH',
	'Kidney renal clear cell carcinoma': 'KIRC',
	'Kidney renal papillary cell carcinoma': 'KIRP',
	'Liver hepatocellular carcinoma': 'LIHC',
	'Lung adenocarcinoma': 'LUAD',
	'Lung squamous cell carcinoma': 'LUSC',
	'Lymphoid Neoplasm Diffuse Large B-cell Lymphoma': 'DLBC',
	'Mesothelioma': 'MESO',
	'Ovarian serous cystadenocarcinoma': 'OV',
	'Pancreatic adenocarcinoma': 'PAAD',
	'Pheochromocytoma and Paraganglioma': 'PCPG',
	'Prostate adenocarcinoma': 'PRAD',
	'Rectum adenocarcinoma': 'READ',
	'Sarcoma': 'SARC',
	'Skin Cutaneous Melanoma': 'SKCM',
	'Stomach adenocarcinoma': 'STAD',
	'Testicular Germ Cell Tumors': 'TGCT',
	'Thymoma': 'THYM',
	'Thyroid carcinoma': 'THCA',
	'Uterine Carcinosarcoma': 'UCS',
	'Uterine Corpus Endometrial Carcinoma': 'UCEC',
	'Uveal Melanoma': 'UVM',
	}

	name2number = {'Acute Myeloid Leukemia': 0,
	'Adrenocortical carcinoma': 1,
	'Bladder Urothelial Carcinoma': 2,
	'Brain Lower Grade Glioma': 3,
	'Breast invasive carcinoma': 4,
	'Cervical squamous cell carcinoma and endocervical adenocarcinoma': 5,
	'Cholangiocarcinoma': 6,
	'Colon adenocarcinoma': 7,
	'Esophageal carcinoma': 8,
	'Esophageal carcinoma ': 9,
	'Glioblastoma multiforme': 10,
	'Head and Neck squamous cell carcinoma': 11,
	'Kidney Chromophobe': 12,
	'Kidney renal clear cell carcinoma': 13,
	'Kidney renal papillary cell carcinoma': 14,
	'Liver hepatocellular carcinoma': 15,
	'Lung adenocarcinoma': 16,
	'Lung squamous cell carcinoma': 17,
	'Lymphoid Neoplasm Diffuse Large B-cell Lymphoma': 18,
	'Mesothelioma': 19,
	'Ovarian serous cystadenocarcinoma': 20,
	'Pancreatic adenocarcinoma': 21,
	'Pheochromocytoma and Paraganglioma': 22,
	'Prostate adenocarcinoma': 23,
	'Rectum adenocarcinoma': 24,
	'Sarcoma': 25,
	'Skin Cutaneous Melanoma': 26,
	'Stomach adenocarcinoma': 27,
	'Testicular Germ Cell Tumors': 28,
	'Thymoma': 29,
	'Thyroid carcinoma': 30,
	'Uterine Carcinosarcoma': 31,
	'Uterine Corpus Endometrial Carcinoma': 32,
	'Uveal Melanoma': 33,
	}

	classNames = []
	for i, k in enumerate(name2number):
		assert name2number[k] == i
		classNames.append(name2code[k])


	return name2code, name2number, classNames


def readME(from_file=False, Ngenes=5000, geneCriterion='pmad', scaling=None, save=False, infile=None, outfile=None):
	"""
	works for the data downloaded from tcga, manually merged

	from_file: 		boolean, if false, read & pre-process 'raw' data, else load from infile. If true, Ngenes, geneCriterion, scaling and save are ignored
	Ngenes:			int, number of top most variant genes to keep
	geneCriterion:	str, currently only pmad is support (mean absolute deviation from the mean)
	scaling:		str, 'none', 'min-max', 'z-score'
	save:			boolean, whether to save after preprocessing
	infile:			str, location of stored data matrix (if from_file == True)
	outfile:		str, location to save data frame (if save == True)

	"""
	if from_file:
		assert infile is not None
		return pd.read_csv(infile, header=0, index_col=0)

	data = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/methylation_tss.npy')
	genes = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/methylation_tss_2kbp_regions.npy', allow_pickle=True)

	samples = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/methylation_sampleNames.npy', allow_pickle=True)

	df = pd.DataFrame(data=data, index=samples, columns=genes)

	# remove genes with only missing values
	delCol = df.columns[df.isna().sum(axis=0) == df.shape[0]]
	df.drop(delCol, axis=1, inplace=True)

	# replace remaining missing values with 0
	df.fillna(0, inplace=True)

	if geneCriterion == 'pmad':
		mad_genes = df.mad(axis=0).sort_values(ascending=False)
	else:
		raise NotImplementedError

	selectedGenes = mad_genes.iloc[:Ngenes, ].index
	df = df[selectedGenes]

	if scaling is not None:
		if scaling == 'min-max':
			scaler = MinMaxScaler()
		else:
			assert scaling == 'z-score'
			scaler = StandardScaler()

		scaledData = scaler.fit_transform(df)

		df = pd.DataFrame(scaledData, columns=df.columns, index=df.index)

	if save:
		df.to_csv(outfile, header=True, index=True)

	return df


def readRNA(from_file=False, Ngenes=5000, geneCriterion='pmad', scaling='min-max', save=False, infile=None, outfile=None):
	"""
	works for the dataset downloaded from Xena

	from_file: 		boolean, if false, read & pre-process raw data, else load from infile. If true, Ngenes, geneCriterion, scaling and save are ignored
	Ngenes:			int, number of top most variant genes to keep
	geneCriterion:	str, currently only pmad is support (mean absolute deviation from the mean)
	scaling:		str, 'none', 'min-max', 'z-score'
	save:			boolean, whether to save after preprocessing
	infile:			str, location of stored data matrix (if from_file == True)
	outfile:		str, location to save data frame (if save == True)

	"""

	if from_file:
		assert infile is not None
		return pd.read_csv(infile, header=0, index_col=0)

	# read data and replace missing values with 0
	data = pd.read_table('../data/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena', index_col=0).fillna(0)

	if geneCriterion == 'pmad':
		mad_genes = data.mad(axis=1).sort_values(ascending=False)
	else:
		raise NotImplementedError

	selectedGenes = mad_genes.iloc[:Ngenes, ].index

	data = data.loc[selectedGenes].T

	if scaling is not None:
		if scaling == 'min-max':
			scaler = MinMaxScaler()
		else:
			assert scaling == 'z-score'
			scaler = StandardScaler()

		scaledData = scaler.fit_transform(data)

		data = pd.DataFrame(scaledData, columns=data.columns, index=data.index)

	if save:
		data.to_csv(outfile, header=True, index=True)


	return data



def getTss2cancerType(tss):
	tss2c = dict()
	with open('/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/meta-data/codeTables/tissueSourceSite.tsv') as f:
		for line in f:
			tssCurrent, _, ct, _ = line.split('\t')
			assert tssCurrent not in tss2c
			tss2c[tssCurrent] = ct

	cancertype = []
	for t in tss:
		cancertype.append(tss2c[t])

	return tss2c, np.array(cancertype)


def mergeGE_GCN_ME(fileGE='/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/rna-pancancer-5000-minmax.csv', fileME='/tudelft.net/staff-bulk/ewi/insy/DBL/smakrod/lb/tcga-download/data/ME-pancancer-5000.csv'):
	GE = readRNA(from_file=True, infile=fileGE)
	ME = readME(from_file=True, infile=fileME)

	# collect labels (cancer type and tumor vs normal)
	tss = np.empty(GE.shape[0], dtype='U2')
	tumorNormal = np.empty(GE.shape[0], dtype='U4')
	patientID = np.empty(GE.shape[0], dtype='U20')

	for i, sample in enumerate(GE.index):
		tcga, tss[i], pat, tumorNormal[i] = sample.split('-')
		patientID[i] = tcga + '-' + tss[i] + '_' + pat

	tss2cancerType, cancerType = getTss2cancerType(tss)
	nameLong2short, nameLong2y, classNames = cancerShortNames()

	y = np.array([nameLong2y[c] for c in cancerType])

	# methylation names have an extra aliquot id
	# if you can't uniquely match remove everything, otherwise assume they match
	geID = list(GE.index)
	meID = list(ME.index)

	meIDwithoutLast = [m[:-1] for m in meID]
	unique_meID, counts = np.unique(meIDwithoutLast, return_counts=True)

	unique_meID = unique_meID[counts == 1]


	geID2row = dict()
	for i,m in enumerate(geID):
		geID2row[m] = i

	meID2row = dict()
	for i,m in enumerate(meID):
		meID2row[m] = i

	commonSamples = [m for m in unique_meID if m in geID2row]

	GE2 = np.zeros((len(commonSamples), GE.shape[1]))
	ME2 = np.zeros((len(commonSamples), ME.shape[1]))
	y2 = np.zeros(len(commonSamples), int)

	for i, s in enumerate(commonSamples):
		GE2[i] = GE.loc[s]
		y2[i] = y[geID2row[s]]

		try:
			ME2[i] = ME.loc[s + 'A']
		except KeyError:
			try:
				ME2[i] = ME.loc[s + 'B']
			except KeyError:
				try:
					ME2[i] = ME.loc[s + 'C']
				except KeyError:
					sys.exit(0)


	#class 8 ESCA is not present
	del classNames[8]
	y2[y2>7] = y2[y2>7] - 1


	return GE2, ME2, y2, commonSamples, classNames






if __name__ == '__main__':
	GE, ME, y, samples, cancerTypeNames = mergeGE_ME()
	dir = sys.argv[1]
	np.save(dir + 'GE.npy', GE)
	np.save(dir + 'ME.npy', ME)
	np.save(dir + 'cancerType.npy', y)
	np.save(dir + 'sampleNames.npy', samples)
	np.save(dir + 'cancerTypes.npy', cancerTypeNames)

	'''
	data = readRNA(from_file=True, infile='../data/rna-pancancer-5000-minmax.csv')
	tss = np.empty(data.shape[0], dtype='U2')
	tumorNormal = np.empty(data.shape[0], dtype='U4')
	patientID = np.empty(data.shape[0], dtype='U20')


	for i, sample in enumerate(data.index):
		tcga, tss[i], pat, tumorNormal[i] = sample.split('-')
		patientID[i] = tcga + '-' + tss[i] + '_' + pat

	tss2cancerType, cancerType = getTss2cancerType(tss)
	nameLong2short, nameLong2y, classNames = cancerShortNames()

	y = np.array([nameLodng2y[c] for c in cancerType])

	#class 8 ESCA is not present
	del classNames[8]
	y[y>7] = y[y>7] - 1


	X = np.array(data)

	dataME = readME(scaling='none', save=True, outfile='../data/ME-pancancer-5000.csv')

	sys.exit(0)

	for i in range(X.shape[0]):
		with open('../data/datasets/expression/' + data.index[i] + '.pkl', 'wb') as f:
			sd = {'x': X[i], 'y': y[i]}
			pickle.dump(sd, f)


	with open('../data/GE_pancancer_full.pkl', 'wb') as f:
		dd = {'X': X, 'y': y, 'classes': classNames, 'sampleNames': np.array(data.index)}
		pickle.dump(dd, f)


	figp = plt.figure()
	ax = figp.add_subplot(111)
	pca, ax = dimRedPlot(X, method='pca', y=y, classNames=classNames, ax=ax, showDensity=False)


	figt = plt.figure()
	ax = figt.add_subplot(111)
	tsne, ax = dimRedPlot(X, method='tsne', y=y, classNames=classNames, ax=ax, showDensity=False)


	figu = plt.figure()
	ax = figu.add_subplot(111)
	umap, ax = dimRedPlot(X, method='umap', y=y, classNames=classNames, ax=ax, showDensity=False)

	figp.savefig('pancancer_GE_pca.png', dpi=500)
	figt.savefig('pancancer_GE_tsne.png', dpi=500)
	figu.savefig('pancancer_GE_umap.png', dpi=500)
	'''
