import numpy as np
import modal_analysis as ma
import matplotlib.pyplot as plt
import time

""" 
.NPZ DATA STRUCTURE:

    ["Name"]: str # e.g. healthy, damage 1, damage 2 etc.

    ["freqs"]: np.ndarray # frequency vector

    ["refPSD"]: np.ndarray # PSD from Hammer

    ["respPSD"]: np.ndarray # PSDs from Accels N_out x N_in x N_f
    
    ["FRF"]: np.ndarray # frf matrix N_out x N_in x N_f

    ["Coherence"]: np.ndarray # coherence N_out x N_in x N_f

    ["cross_PSD"]: np.ndarray # Cross powers N_out x N_in x N_f 

"""

FIXTURE_MIMO_SYSTEM = np.load('MIMO.npz')
delta_t = FIXTURE_MIMO_SYSTEM['t'][1]-FIXTURE_MIMO_SYSTEM['t'][0]
freqs = FIXTURE_MIMO_SYSTEM['ws']
frf = FIXTURE_MIMO_SYSTEM['H']

print(FIXTURE_MIMO_SYSTEM['wn']/(2*np.pi))
def polymax_test(frf,freqs,delta_t,order):
    polybasis = ma.make_polynomial_basis_fcn(order,freqs,sampling_rate=delta_t)
    weighting = ma.frequency_dependent_weighting(frf)

    X,Y = ma.make_X_and_Y(polybasis,weighting,frf)
    R,S,T=ma.make_RST_optimized(X,Y)
    M = ma.M_MATRIX(R,S,T)
    alpha = ma.LSQ_ALPHA(M,2)
    compmat = ma.make_companion_Matrix(alpha,2)
    poles,_ = ma.make_time_poles_and_participation_factors(compmat)
    poles = ma.basic_stability(poles)
    poles = ma.unphysical_mode_filer(poles)
    wn,zn = ma.poles_to_modal(poles)
    return wn,zn

# for i in range(1,20):
#     wn,_ = polymax_test(frf,freqs,delta_t,i)
#     print("Omega1 errors: ")
#     print(wn-FIXTURE_MIMO_SYSTEM['wn'][0]/(2*np.pi))
#     print("Omega2 errors: ")
#     print(wn-FIXTURE_MIMO_SYSTEM['wn'][1]/(2*np.pi))
# print(delta_t*400)
# print(200*2*np.pi*(1/133))
fig,ax1 = plt.subplots()
ax1.semilogy(freqs*2*np.pi,np.abs(frf[1,1]))
ax2 = ax1.twinx()


for i in range(1,30):
    try:
        wn,_ = polymax_test(frf,freqs,delta_t,order=i)
        ax2.scatter(wn,np.tile(i,len(wn)))
    except np.linalg.LinAlgError: continue

ax2.vlines(FIXTURE_MIMO_SYSTEM['wn'],0,20,linestyles='-.',colors='black')
plt.xlim(0,freqs[-1]*2*np.pi)
plt.show()










# Healthy_Data= np.load("fixed_data/Healthy.npz")

# a,b,c = ma.plscf_bootleg(20,Healthy_Data)
# np.savez("100_MODEL_ORDER_POLYMAX_SOLUTION_ATTEMPT",WN = a, ZETA =b, MPFM = c)

# # loaded_sol = np.load("100_MODEL_ORDER_POLYMAX_SOLUTION_ATTEMPT.npz")
# # wn = loaded_sol["WN"]
# # print(wn[0:55])



# # samp_freq = 8192
# # time1 = time.time()
# # data = Healthy_Data
# # model_order = 99


# # poly_basis = ma.make_polynomial_basis_fcn(model_order, data["freqs"][:-2], sampling_frequency=samp_freq)
# # time2 = time.time()

# # weighting_fn = ma.frequency_dependent_weighting(data["FRF"][:, :, :-2])
# # time3 = time.time()

# # X, Y = ma.make_X_and_Y(poly_basis, weighting_fn, data["FRF"][:, :, :-2])
# # time4 = time.time()

# # p = model_order+1
# # N_out = data["FRF"].shape[0]
# # N_in = data["FRF"].shape[1]
# # print(X.shape)

# # R,S,T = rst.make_RST(X,Y)

# # print(R.shape)
# # print(R[0,0,0])

# # print(S.shape)
# # print(S[0,0,0])

# # print(T.shape)
# # print(T[0,0,0])



# # a,b,c = plscf_bootleg(5,Healthy_Data)

# # print(a)

# # plt.plot(Healthy_Data['freqs'],np.abs(Healthy_Data['FRF'][0,0]))
# # plt.show()
