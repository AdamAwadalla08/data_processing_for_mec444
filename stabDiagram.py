import numpy as np
import matplotlib.pyplot as plt
import modal_analysis as ma
import math
model_order_range = range(10,20)
Data = np.load("fixed_data/Healthy.npz")

# wn,_,_ = ma.plscf_bootleg(30,Data)
# print(wn.shape)
# wn = np.reshape(wn,(12,30))
# print(wn)
fix,ax1 = plt.subplots()
ax1.semilogy(Data["freqs"]/(2*np.pi),np.abs(Data["respPSD"][8,8]),linewidth=0.75)
ax2 = ax1.twinx()

for i in model_order_range:
    try:
        wn,_,_ = ma.plscf_bootleg(i,Data)
        ax2.scatter(wn*2*math.pi,np.tile(i,len(wn)),marker='x',linewidths=0.8)
    except np.linalg.LinAlgError:
        continue
plt.xlim(0,512/(2*np.pi))
plt.show()