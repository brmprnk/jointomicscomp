from scipy.stats import norm, nbinom, lognorm, poisson
import numpy as np
from scipy.optimize import minimize

def pntomv(n, p):
    m = n * (1 - p) / p
    var = m / p

    return m, var

def mvtonp(m, var):
    p = m / var
    n = (m ** 2) / (var - m)

    return n, p


def NLLnbinom(x, data):
    # data 1-dimensional (vector)
    res = -np.mean(nbinom.logpmf(data, x[0], x[1], loc=0))
    return res


def NLLzip(x, data):
    # data 1-dimensional (vector),
    # x: p and lambda

    indZ = np.where(data == 0)[0]


    likelihood = (1 - x[0]) * poisson.pmf(data, x[1], loc=0)
    likelihood[indZ] += x[0]

    res = - np.mean(np.log(likelihood))

    #res = -np.mean(nbinom.logpmf(data, x[0], x[1], loc=0))
    #print(res)
    return res




def testDistributions(xtrain, xtest):

    logxtrain = np.log(xtrain+1)
    logxtest = np.log(xtest+1)

    ll = {'train': dict(), 'test': dict()}

    res = minimize(NLLnbinom, mvtonp(np.mean(xtrain), np.var(xtrain)), args=xtrain, jac='3-point', bounds=[(1e-8, 1e5), (1e-8, 1)])

    ll['train']['nb'] = -NLLnbinom(res.x, xtrain)
    ll['test']['nb'] = -NLLnbinom(res.x, xtest)


    ############################################
    res = minimize(NLLzip,  [np.mean(xtrain == 0), np.mean(xtrain[xtrain > 0])], args=xtrain, jac='3-point', bounds=[(1e-8, 1e5), (1e-8, 1)])

    ll['train']['zip'] = -NLLzip(res.x, xtrain)
    ll['test']['zip'] = -NLLzip(res.x, xtest)


    ############################################
    ninit, pinit = mvtonp(np.mean(logxtrain), np.var(logxtrain))

    if ninit < 1e-8:
        ninit = 0.001
    if pinit < 1e-8 or pinit > 0.99999999999:
        pinit = 0.5


    res = minimize(NLLnbinom, [ninit, pinit], args=logxtrain, jac='3-point', bounds=[(1e-8, 1e5), (1e-8, 1)])
    #print(res)

    # nest, pest = res.x
    # print('%.5f \t%.5f' % (n, nest))
    # print('%.5f \t%.5f' % (p, pest))

    ll['train']['nbonlog'] = -NLLnbinom(res.x, logxtrain)
    ll['test']['nbonlog'] = -NLLnbinom(res.x, logxtest)


    ############################################
    res = norm.fit(logxtrain)

    ll['train']['normonlog'] = np.mean(norm.logpdf(logxtrain, res[0], res[1]))
    ll['test']['normonlog'] = np.mean(norm.logpdf(logxtest, res[0], res[1]))

    ############################################
    res = lognorm.fit(xtrain+1)

    ll['train']['lognorm'] = np.mean(lognorm.logpdf(xtrain + 1, res[0], res[1]))
    ll['test']['lognorm'] = np.mean(lognorm.logpdf(xtest + 1, res[0], res[1]))

    return ll



data = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/pbmc_multimodal_RNA_5000MAD.npy')

trainInd = np.load('/tudelft.net/staff-bulk/ewi/insy/DBL/bpronk/jointomicscomp/data/CELL/trainInd.npy')

data = data[trainInd]

# lm = 2
# disp = 0.5
#
# m = np.exp(lm)
# var = m + ((m ** 2) / disp)
#
#
# p = m / var
# n = (m ** 2) / (var - m)
#
#
# x = nbinom.rvs(n, p, loc=0, size=N)
#

loglikes = np.zeros((data.shape[1], 5))
distros = ['lognorm', 'normonlog', 'nbonlog', 'nb', 'zip']

for ind in range(data.shape[1]):
	print(ind)
	x = np.round(np.exp(data[:,ind]) - 1)

	xtrain = x[:10000]
	xtest = x[10000:]

	loglike = testDistributions(xtrain, xtest)
	for j, d in enumerate(distros):
		loglikes[ind, j] = loglike['test'][d]


#------------------
# p = 0.01
# l = 3
#
# x = np.zeros(10000)
#
# for i in range(10000):
#     if np.random.rand() < p:
#         x[i] = poisson.rvs(l, size=1)
#     else:
#         x[i] = 0
#
# pinit = np.mean(x > 0)
# linit = np.mean(x[x>0])
# res = minimize(NLLzip, [pinit, linit], args=x, jac='2-point', bounds=[(1e-8, 1), (1e-8, 1e5)])
# print(res)
#-------------------

#for k in loglike:
#   print(k)
#    for l in loglike[k]:
#        print(l, loglike[k][l])
