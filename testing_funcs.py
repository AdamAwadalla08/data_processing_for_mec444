import numpy as np
import read_unv as funcs


def debug_load_state(filename:str,state:str,n_exp_tests:int):
    out_data = funcs.load_state(filename=filename,state=state,n_total_tests=n_exp_tests)
    with open("out.txt", 'w') as file:
        for entry in out_data:
            file.write(f"{entry['Location']}\n")
            file.write(f"{entry['Data']}\n")
    pass

debug_load_state("full_tests.unv","Healthy State",1764)


# with open("output_file.txt", "w") as file:
#     for row in var_mat:
#         # Convert each row to a string with values separated by spaces
#         row_str = " ".join(map(str, row))
#         file.write("["+ row_str + "] \n \n")

# w = 1 /np.sqrt(var_mat[0:1024,:])

# with open("output_file2.txt", "w") as file:
#     for row in w:
#         row_str = " ".join(map(str, row))
#         file.write("["+ row_str + "] \n \n")
