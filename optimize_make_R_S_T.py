import numpy as np
import time


XY = np.load("X_AND_Y_100_MODEL.npz")

X= XY["X"]
Y= XY["Y"]

print(X.shape)
print(Y.shape)

"""
X shape --> nf x p x n_o, i,j,k
Y shape --> nf x p*n_i x n_o, i,l,k

R is X.H @ X, R shape --> p x p x n_o, j,j,k

S is X.H @ Y, S shape --> p x p*n_in x n_o, j,l,k

T is Y.H @ Y, T shape --> p*n_in x p*n_in x n_o, l,l,k

"""

timeTranspose = time.time()
XH = np.transpose(X.conj(),(1,0,2))
YH = np.transpose(Y.conj(),(1,0,2))
print("Time Taken for Hermitian Transpose: " + str(time.time()-timeTranspose))


print(XH.shape)
print(YH.shape)


timeR=time.time()

R = np.zeros((100,100,12))
for i in range(12):
    R[:,:,i] = np.real(np .dot(XH[:,:,i],X[:,:,i]))

print("Time Taken for R calculation: " + str(time.time() - timeR))
print(R.shape)

S = np.zeros((100,1200,12))
timeS = time.time()
for i in range(12):
    S[:,:,i] = np.real(np.dot(XH[:,:,i],Y[:,:,i]))

# S = np.einsum("ijk,jlk->ilk",XH,Y,optimize='optimal')
print("Time Taken for S calculation: " + str(time.time()-timeS))

T = np.zeros((1200,1200,12))

timeT = time.time()
for i in range(12):
    T[:,:,i] = np.real(np.dot(YH[:,:,i],Y[:,:,i]))
print("Time Taken for T calculation: " + str(time.time()-timeT))