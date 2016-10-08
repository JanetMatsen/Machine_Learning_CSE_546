#!/bin/python
import scipy.sparse as sp
import numpy as np

####
# This is a quick walkthrough to help you understand the operations in scipy.sparse
####

# construct a sparse array, here we simply construct it from dense array
A = np.arange(12).reshape(3,4)
print A
X = sp.csc_matrix(A)

w = np.ones(4)

#  matrix vector multiplication
y = X.dot(w)
print y

#
# dot product between i-th column of S and g
#
i = 0
g = np.ones(3)
# r1 = dot(X[:,i], g), because X takes matrix syntax, we need to do it in this way
r1 = X[:,i].T.dot(g)
print r1
#
# This is how you can get dot(X[:,i], X[:,i]) in csc_matix
#
r2 = X[:,i].T.dot(X[:,i])[0,0]
print r2


#-------------------------------------------
# This is an alternative way to hack into data structure of csc_matrix, 
# the materials before this should be sufficient for your implemenation
# see http://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_.28CSC_or_CCS.29

# r1 = dot(X[:,i], g) = sum_j X[j,i] * g[j]
start, end = X.indptr[i], X.indptr[i+1]
r1 = np.sum(X.data[start:end] * g[ X.indices[start:end] ])
print r1

# r2 = dot(X[:,i], X[:,i]) = sum_j X[j,i]^2
r2 = np.sum(X.data[start:end]**2)
print r2

