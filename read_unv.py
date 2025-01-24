import os
import sys
import subprocess
from typing import NamedTuple

# installs venv for examiner who might've not used python previously

def setup_venv():
    if not os.path.exists('.venv'):
        subprocess.check_call([sys.executable,'-m','venv','.venv'])
    else:
        print("Virtual environment exists")
    
    activate_bat = os.path.join('.venv','Scripts','activate.bat')
    if not os.path.exists(activate_bat):
        sys.exit(1)
        print("pls check you cloned repo correctly")

    required_libs = 'required.txt'
    if not os.path.exists(required_libs):
        sys.exit(1)
        print("pls check you cloned repo correctly")

    

# setup_venv()
# subprocess.check_call([os.path.join('venv', 'Scripts', 'pip'),'install','-r','requires.txt'])
# ensures libraries are installed correctly for this script and others

try:

    import numpy as np  
    import pyuff as UNV
    import matplotlib.pyplot # for plotting.py
    import pandas # for data visualization in jupyter notebooks


except ImportError as error_lib:
    print(f"Error importing: {error_lib}")
    sys.exit(1)

# SCRIPT LOGIC


def reverse_dict(dict:dict):
    return {value: key for key, value in dict.items()}

class State(NamedTuple):
    Name: str # e.g. healthy, damage 1, damage 2 etc.

    freqs: np.ndarray # frequency vector

    refPSD: np.ndarray # PSD from Hammer

    respPSD: np.ndarray # PSDs from Accels N_out x N_in x N_f
    
    FRF: np.ndarray # frf matrix N_out x N_in x N_f

    Coherence: np.ndarray # coherence N_out x N_in x N_f

    cross_PSD:np.ndarray # Cross powers N_out x N_in x N_f 

class input_args(NamedTuple):
    State: str
    Type: str
    tap_location: dict
    measure_location: dict


# parsed_file = UNV.UFF("full_tests.unv")
# data = parsed_file.read_sets()
# print(data[2].keys())
# print(data[2]['id4'].split()[-1].split("_"))



def fetch_test_state(data: dict, index:int = 2):
    state_dict = {
        'hlth': "Healthy State",
        'dmg1': "Damage State 1",
        'dmg2': "Damage State 2",
        'dmg3': "Damage State 3",
        'dmg4': "Damage State 4",
    }

    return state_dict[data[index]['id4'].split()[-1].split("_")[0].replace("\"","")]


def fetch_test_location(data:dict,index:int=2):
    init_string = data[index]['id4'].split()[-1].split("_")[-1].replace("\"","")
    return {
        "Beam": int(init_string[-2]),
        "Accelerometer": int(init_string[-1])
        }

def fetch_function_type(data: dict, index: int):


    function_type_dict = {
        0: "N/A", "0": "N/A",
        1: "Time", "1": "Time",
        2: "Auto Spectrum", "2": "Auto Spectrum",
        3: "Cross Power", "3": "Cross Power",
        4: "FRF", "4": "FRF",
        5: "Transmissibility", "5": "Transmissibility",
        6: "Coherence", "6": "Coherence",
        7: "Cross Correlation", "7": "Cross Correlation",
        9: "PSD", "9": "PSD",
        10: "ESD", "10": "ESD",
        11: "PDF", "11": "PDF",
        12: "Spectrum", "12": "Spectrum"
    }

    return function_type_dict[data[index]['func_type']]


def fetch_test_data(data: list, idx1: int, idx2: int, inputs: NamedTuple):

    desired_func_type = inputs.Type
    desired_state = inputs.State
    desired_tap_location = inputs.tap_location
    test_data = []

    for i in range(idx1,idx2) :
        func_type = fetch_function_type(data,i)
        state = fetch_test_state(data,i)
        tap_location = fetch_test_location(data,i)
        
        if (
            func_type == desired_func_type and 
            state == desired_state and 
            tap_location == desired_tap_location
        ):
            test_data.append((data[i]['id1'],data[i]['id4'],data[i]['x'],data[i]['data']))
            continue

    return test_data





# input_test = input_args(
#                 tap_location = {
#                     'Beam': 1,
#                     'Accelerometer': 1}, 
#                 State = "Healthy State", 
#                 Type = "PSD",
#                 measure_location={
#                     "Beam": 1,
#                     "Accelerometer": 1
#                 }
#     )


def load_state(filename:str, state: str, n_total_tests:int):
    parsed_file = UNV.UFF(filename)
    format_types = parsed_file.get_set_types()

    where_58 = np.where(format_types == 58)[0]
    idx1 = where_58[0]
    idx2 = idx1 + n_total_tests
    raw_data = parsed_file.read_sets()
    func_list= [ "Coherence","Cross Power", "PSD","FRF"]
    state_data= []
    for beam in range(1,4):
        for accel in range(1,5):
            for type in func_list:

                try:
                    inputs = input_args(
                        State=state,
                        Type = type,
                        measure_location=None,
                        tap_location={"Beam": beam, "Accelerometer": accel}
                    )
                    result = fetch_test_data(raw_data,idx1,idx2,inputs=inputs)
                    if result:
                        func = result
                        state_data.append(
                            {
                                "Location": inputs.tap_location,
                                "Data" : func
                            }
                        )
                        continue

                except KeyError:
                    print("Error at input args:" + str(inputs.tap_location["Beam"]) + str(inputs.tap_location["Accelerometer"]))
                    continue
    return state_data



# healthy_data = load_state("full_tests.unv","Healthy State",1764)
# print(type(healthy_data[0]['Data'][0][3][0]))

"""
freqs: 1xNf float64
refPSD: 1xNf float64
respPSD: NoxNixNf float64
FRF: NoxNixNf complex128
Coherence: NoxNixNf complex128
Cross Power: NoxNixNf complex128

output data structure from load state:

list of dicts:

Beam 1 accel 1 all functions: functions as func_list = ["Coherence", "Cross Power", "PSD", "FRF"]
progress in accel
then progress in beams

dict type in list:

"location": "beam":whatevs, "Accelerometer": whatevs

"data": tuple
data[0]: id1 from the unv file which is the title, say FRF for A11/Hammer
data[1]: id4 from unv file, which is a more comprehensive title, say   Record   49 of section "Section1", run "hlth_tap11"
data[2]: freq vector
data[3]: data vector

"""


def format_state(filename: str, state:str,n_total_tests:int,spectral_lines:int):
    state_data = load_state(filename,state,n_total_tests)

    freqs = state_data[0]['Data'][0][2]
    coherence = np.zeros((12,12,spectral_lines+2))
    frf = np.zeros((12,12,spectral_lines+2)).astype(np.complex128)
    outPSD = np.zeros((12,12,spectral_lines+2))
    inpsd = np.zeros((12,spectral_lines+2))
    cpsd = np.zeros((12,12,spectral_lines+2)).astype(np.complex128)

    for input in range(0,12):
        inpsd[input] = np.real(state_data[(input*4)+2]['Data'][-1][3])
        for output in range(0,12):

            coherence[output,input] = np.real(state_data[(input)*4]['Data'][-1*(output+1)][3])
            cpsd[output,input] = state_data[(input*4)+1]['Data'][-1*(output+1)][3]
            outPSD[output,input] = np.real(state_data[(input*4)+2]['Data'][-1*(output+2)][3])
            frf[output,input] = state_data[(input*4)+3]['Data'][-1*(output+1)][3]

    return State(

        Name=state,
        freqs=freqs,
        refPSD=inpsd,
        respPSD=outPSD,
        FRF= frf,
        Coherence = coherence,
        cross_PSD=cpsd

    )


#inshallah_works = format_state("full_tests.unv","Healthy State",1764)
# works