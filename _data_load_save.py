"""
DO NOT RUN UNNECESSARILY!!!

Saves all data to numpy z-compressed files (.npz)

"""

import numpy as np
import read_unv as ru

Healthy_Data = ru.format_state("full_tests_new.unv","Healthy State",4104,16384)
Damage1_Data = ru.format_state("full_tests_new.unv","Damage State 1",4104,16384)
Damage2_Data = ru.format_state("full_tests_new.unv","Damage State 2",4104,16384)
Damage3_Data = ru.format_state("full_tests_latest.unv","Damage State 3",6840,16384)
Damage4_Data = ru.format_state("full_tests_latest.unv","Damage State 4",6840,16384)

np.savez("fixed_data/Healthy.npz",**Healthy_Data._asdict())
np.savez("fixed_data/Damage1.npz",**Damage1_Data._asdict())
np.savez("fixed_data/Damage2.npz",**Damage2_Data._asdict())
np.savez("fixed_data/Damage3.npz",**Damage3_Data._asdict())
np.savez("fixed_data/Damage4.npz",**Damage4_Data._asdict())
