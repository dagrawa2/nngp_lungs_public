import numpy as np
import scipy.linalg as sp_la
import scipy.special as special

def ARC(trues, preds, confs):
	indices = np.argsort(confs)
	c = np.equal(trues, preds)[indices]
	rejs = np.arange(indices.shape[0]+1)/indices.shape[0]
	accs = []
	for i in range(indices.shape[0]):
		accs.append( np.mean(c[i:]) )
	accs = np.array(accs + [1])
	return rejs, accs

def ARC_L1_score(rejs, accs):
	A_oracle = accs[0] - accs[0]*np.log(accs[0])
	A_arc = (rejs[1:]-rejs[:-1]).dot(accs[:-1]+accs[1:])/2
	return A_oracle - A_arc

def RC(trues, preds, confs, n_bins=10):
	c = np.equal(trues, preds)
	indices = np.arange(len(c))
	pnts = np.linspace(np.min(c), 1, n_bins)
	avg_confs = []
	accs = []
	for p_0, p_1 in zip(pnts[:-1], pnts[1:]):
		inds = indices[np.logical_and(p_0<=confs, confs<=p_1)]
		if len(inds) > 0:
			avg_confs.append(np.mean(confs[inds]))
			accs.append(np.mean(c[inds]))
	return np.array(avg_confs), np.array(accs)

def ECE(confs, accs):
	return np.mean(np.abs(confs-accs))


def Phi(x):
	return 1/2*(1+special.erf(x/np.sqrt(2)))

def vote(means, cov, n_samples):
	sums = np.zeros(means.shape)
	cov_sqrt = sp_la.sqrtm(np.matrix(cov))
	for i in range(n_samples//100):
		samples = means[:,:,np.newaxis] + np.einsum("ij,jkl->ikl", cov_sqrt, np.random.normal(size=(means.shape[0], means.shape[1], 100)))
#		_, update = np.unique(np.argmax(samples, axis=1), axis=1, return_counts=True)
#		sums += update
		samples = np.argmax(samples, axis=1)
		for j in range(means.shape[1]):
			sums[:,j] += np.sum(samples==j, axis=1)
	return sums/n_samples
