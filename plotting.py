import numpy as np
import matplotlib.pyplot as plt


Healthy= np.load("fixed_data/Healthy.npz")
Damage1 = np.load("fixed_data/Damage1.npz")
Damage2 = np.load("fixed_data/Damage2.npz")
Damage3 = np.load("fixed_data/Damage3.npz")
Damage4 = np.load("fixed_data/Damage4.npz")

# loaded_sol = np.load("100_MODEL_ORDER_POLYMAX_SOLUTION_ATTEMPT.npz")
# print(loaded_sol["WN"][0:50])
def normalize_psd(PSD,delta_f):
    total_power = np.sum(PSD)*delta_f
    return PSD/total_power
print(Healthy["respPSD"].shape)

# plt.semilogy(Healthy["freqs"],np.abs(Healthy["respPSD"][0,0]))
plt.plot(Healthy["freqs"],normalize_psd(Healthy["respPSD"][4,4],0.25),linewidth=0.6,label="Hlt")
plt.plot(Healthy["freqs"],normalize_psd(Damage1["respPSD"][4,4],0.25),linewidth=0.6,label="dmg1")
plt.plot(Healthy["freqs"],normalize_psd(Damage2["respPSD"][4,4],0.25),linewidth=0.6,label="dmg2")
plt.plot(Healthy["freqs"],normalize_psd(Damage3["respPSD"][4,4],0.25),linewidth=0.6,label="dmg3")
plt.plot(Healthy["freqs"],normalize_psd(Damage4["respPSD"][4,4],0.25),linewidth=0.6,label="dmg4")
plt.xlim(0,512)
plt.legend()
plt.show()