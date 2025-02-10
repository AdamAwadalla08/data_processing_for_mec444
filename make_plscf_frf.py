""" This script is to "step-by-step" the coding of making the FRF rational polynomial estimate.

MIMO.npz file structure keys:
            M: Mass matrix shape (2,2)
            C: Damping matrix shape (2,2)
            K: Stiffness matrix shape (2,2)
            wn: Natural frequency vector shape (2,)
            zn: Damping ration vector shape (2,)
            phi: Modeshape vector shape (2, 2)
            ws: Frequency vector shape (nlines)
            H: Exact FRF of system (compliance by default) shape (2, 2, nlines)
            t: Time vector shape (T,),
            u: Forcing input shape (2, T)
            d: Displacement output shape (2, T)
            v: velocity output shape (2, T)
            a: acceleration output shape (2, T)

"""


import numpy as np
import modal_analysis as ma
import matplotlib.pyplot as plt
from utils import *

order = 10

testdata = np.load('fixed_data/MIMO.npz')
delta_t = testdata['t'][1]-testdata['t'][0]


omega = ma.make_polynomial_basis_fcn(order,testdata['ws'],sampling_rate=delta_t)

weighting = ma.frequency_dependent_weighting(testdata['H'])
X,Y = ma.make_X_and_Y(omega,weighting,testdata['H'])
R,S,T = ma.make_RST_optimized(X,Y)
M = ma.make_M_matrix(R,S,T)

alpha = ma.make_LSQ_alpha(M,2)
beta = ma.make_LSQ_beta(alpha,R,S)
companion = ma.make_companion_Matrix(alpha,2)
Poles,_ = ma.make_time_poles_and_participation_factors(companion)
wn,_ = ma.pLSCF_poles_to_modal(Poles)

print(wn)



A = np.zeros((2,2,len(testdata['ws']))).astype(np.complex128)


alpha2 = np.reshape(alpha,(order+1,2,2))

# print(A.shape)
# print(alpha.shape)
# print(omega.shape)
# print(alpha2.shape)

for bin in range(A.shape[-1]):
    for i in range(alpha2.shape[0]):    
        A[:,:,bin] = A[:,:,bin] + alpha2[i]*omega[bin,i]

B = np.zeros((2,2,len(testdata['ws']))).astype(np.complex128)

print(beta.shape)
for bin in range(B.shape[-1]):
    for output in range(2):
        for i in range(beta.shape[0]):
            B[output,:,bin] = B[output,:,bin]+omega[bin,i]*beta[i,:,output]


H = np.zeros((2,2,len(testdata['ws'])),dtype=np.complex128)
for output in range(2):
    for bin in range(A.shape[-1]):
        H[output,:,bin] = B[output,:,bin] @ np.linalg.inv(A[:,:,bin])

freqs = testdata['ws']
print(freqs[1]-freqs[0])
plt.plot(freqs,normalize_psd(np.abs(H[0,0]),0.0075))
plt.plot(freqs,weighting[:,0])

# plt.semilogy(np.abs(H[0,1]))
# plt.semilogy(np.abs(H[1,0]))
# plt.semilogy(np.abs(H[1,1]))
plt.plot(freqs,np.abs(testdata['H'][0,0]))
# plt.semilogy(np.abs(testdata['H'][0,1]))
# plt.semilogy(np.abs(testdata['H'][1,0]))
# plt.semilogy(np.abs(testdata['H'][1,1]))
plt.vlines(wn,0,1)
plt.show()
