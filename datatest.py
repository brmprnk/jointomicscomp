#SPLIT for TASK 1/3

#variable y contains cancer type/cell type
 

from sklearn.model_selection import StratifiedShuffleSplit

split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.1)

trainValidInd, testInd = split1.split(X, y)
 

Xtest = X[testInd]

ytest = y[testInd]
 

XtrainValid = X[trainValidInd]

ytrainValid = y[trainValidInd]
 

split2 = StratifiedShuffleSplit(n_splits=1, test_size=1/9)

trainInd, validInd = split2.split(XtrainValid, ytrainValid)
 

Xtrain = XtrainValid[trainInd]

ytrain = ytrainValid[trainInd]
 

Xvalid = XtrainValid[validInd]

yvalid = ytrainValid[validInd]


#SPLIT for TASK 2

#variable y contains cancer type, variable stage contains cancer stage
 

ctype = 'BRCA'
 

Xtype = X[y == ctype]

Xrest = X[y != ctype]

yrest = y[y != ctype]
 

split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.1)

trainValidInd, testInd = split1.split(Xtype, stage)
 

Xtest = Xtype[testInd]

stagetest = stage[testInd]
 

XtrainValid = X[trainValidInd]

stagetrainValid = stage[trainValidInd]
 

split2 = StratifiedShuffleSplit(n_splits=1, test_size=1/9)

trainInd, validInd = split1.split(XtrainValid, stagetrainValid)
 

Xtrain = XtrainValid[trainInd]

stagetrain = stagetrainValid[trainInd]
 

Xvalid = XtrainValid[validInd]

stagevalid = stagetrainValid[validInd]


splitRest = StratifiedShuffleSplit(n_splits=1, test_size=1/9)

trainInd, validInd = split3.split(Xrest, yrest)
 

Xresttrain = Xrest[trainInd]

Xrestvalid = Xrest[validInd]
 

# could be vstack, i always confuse these too

XtrainAll = np.hstack((Xtrain, Xresttrain))

XvalidnAll = np.hstack((Xvalid, Xrestvalid))

XtestAll = np.hstack((Xtest, Xrestest))


