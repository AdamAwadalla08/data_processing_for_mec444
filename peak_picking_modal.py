import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy.signal import find_peaks

"""
Due to implementation failure, the use of the poly-reference Least Squres Complex Frequency Domain method, pLSCF, or PolyMAX,
has been abandonned due to time constraints.
"""


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


# First Step, loading relevant data

Healthy = np.load('fixed_data/Healthy.npz')
dmg1 = np.load('fixed_data/Damage1.npz')
dmg2 = np.load('fixed_data/Damage2.npz')
dmg3 = np.load('fixed_data/Damage3.npz')
dmg4 = np.load('fixed_data/Damage4.npz')
# freqs = Healthy['freqs']
# frf = Healthy["FRF"]
# psd = Healthy['respPSD']
# coh = Healthy['Coherence']
# # Plot FRFs of most Relevant points.


# FRF + Coherence of point at damange 1

# figs, axs = plt.subplots(2,1,sharex=True)
# axs[0].semilogy(Healthy['freqs'],np.abs(Healthy['FRF'][8,8]),linewidth=0.5,label='Healthy')
# axs[0].semilogy(Healthy['freqs'],np.abs(dmg1['FRF'][8,8]),linewidth=0.5,label='Damage State 1')
# axs[0].semilogy(Healthy['freqs'],np.abs(dmg2['FRF'][8,8]),linewidth=0.5,label='Damage State 2')
# axs[0].semilogy(Healthy['freqs'],np.abs(dmg3['FRF'][8,8]),linewidth=0.5,label='Damage State 3')
# axs[0].semilogy(Healthy['freqs'],np.abs(dmg4['FRF'][8,8]),linewidth=0.5,label='Damage State 4')
# axs[1].plot(Healthy['freqs'],Healthy['Coherence'][8,8],linewidth=0.5)
# axs[1].plot(Healthy['freqs'],dmg1['Coherence'][8,8],linewidth=0.5)
# axs[1].plot(Healthy['freqs'],dmg2['Coherence'][8,8],linewidth=0.5)
# axs[1].plot(Healthy['freqs'],dmg3['Coherence'][8,8],linewidth=0.5)
# axs[1].plot(Healthy['freqs'],dmg4['Coherence'][8,8],linewidth=0.5)
# plt.xlim(0,512)
# axs[0].set_ylabel("|H(f,9,9)|")
# axs[0].legend()
# axs[1].set_ylabel("γ(f,9,9)")
# plt.xlabel("Frequency (Hz)")
# figs.suptitle("FRF Magnitute, and Coherence of Output at Point 3,1 to input at Point 3,1")
# plt.show()


# fig,ax1 = plt.subplots()
# plt.semilogy(Healthy['freqs'],np.abs(Healthy['FRF'][0,1]),linewidth=0.7,label='H12')
# plt.semilogy(Healthy['freqs'],np.abs(Healthy['FRF'][1,0]),linewidth=0.7,label='H21')
# plt.xlim(0,512)
# plt.legend()
# plt.xlabel('Frequency (Hz)')
# plt.show()

_,ax1 = plt.subplots()
ax1.semilogy(Healthy['freqs'],np.abs(Healthy['FRF'][0,0]),linewidth=0.7,color='orange',label='FRF')
ax2 = ax1.twinx()
ax2.semilogy(Healthy['freqs'],Healthy['refPSD'][0],linewidth=0.7,label='PSD',color = '#1f77b4')
plt.xlim(0,512)
ax1.set_ylabel("FRF")
ax1.tick_params(axis='y', colors='orange')
ax2.tick_params(axis='y',colors='#1f77b4')
ax2.set_ylabel("PSD")
ax1.set_xlabel("Frequency (Hz)")
plt.show()





# pks,_ = find_peaks(np.abs(np.sum(frf[0],0))[0:4095],[1,70],prominence=1.25)
# # print(pks)
# plt.semilogy(freqs,np.abs(frf[8,8]))
# plt.semilogy(freqs, np.abs(frf[4,4]))
# plt.semilogy(freqs,np.abs(frf[0,0]))
# plt.xlim([0,256])
# plt.show()

# plt.semilogy(freqs,np.abs(np.sum(frf[0],0)).T,linewidth=0.75)
# plt.xlim((0,1024))
# plt.vlines(freqs[pks],0.005,80,linewidth=0.7,linestyles='-.',colors='black')
# plt.show()





# Plot Power Spectral Densities Normalized



# Plot difference between reciprocal FRF parts to try and make a measure of linearity

