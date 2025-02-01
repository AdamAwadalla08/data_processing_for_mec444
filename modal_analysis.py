import numpy as np
import read_unv as ru
import time
import utils


def make_polynomial_basis_fcn(polynomial_order : int, frequency_vector : np.ndarray,sampling_frequency:float=None, sampling_rate:float=None):
    """Function that creates a Polynomial Basis Function matrix
 
    Args:
        polynomial_order (int): Order of the polynomial created. i.e. if 2 then P(x) = a0 * x^0  +  a1 * x^1 + a2 * x^2
        in the case of pLSCF, Ω(w) = P(e^{jwΔt})

        frequency_vector (np.ndarray): vector of frequencies measured or simulated, can be hz or rads-1. 
        MUST be either a row or column vector/1D Array

    Returns:
       Polynomial basis function matrix.
    """
    
    # makes dt for either case.
    if sampling_rate is None:
        dt = 1/sampling_frequency
    elif sampling_frequency is None:
        dt = sampling_rate
    else:
        dt = sampling_rate # if both are inputted then sampling rate is favoured. 
        # if neither are inputted then the function just doesn't work
    
    freq = frequency_vector
    p_end = polynomial_order
    # a matrix with rows exhibiting the same value in each column, where columns are 1,2,3 ... p
    p_matrix = np.arange(p_end+1) # arange excludes stop value
    p_matrix = np.tile(p_matrix,(len(freq),1)) #repeats it for the number of frequencies
    #   multiply each row by the frequency vector, should be row vector shaped
    #   multiply by obtained delta t
    #   multiply by j
    p_matrix = (np.dot(np.diag(freq),p_matrix))*dt*1j
    #   exp of that matrix to give Omega
    return np.exp(-1*p_matrix)


def frequency_dependent_weighting(MIMO_transfer_function: np.ndarray):
    """frequency_dependent_weighting: physically speaking, this function is a frequency-domain function, which gives higher 
    weight in the least-squres estimate to the less variant parts of each output.

    from [insert paper reference]:
    W_output = 1/sqrt(var(H_output)) or = 1/std(H_output)
    the variance for each output at a given frequency bin for all inputs is calculated and inverted.

    Args:
        MIMO_transfer_function (np.ndarray): Multi-Input-Multi-Output Transfer function, generally should be N_outputs x N_inputs x Freq_Bins


    Returns:
        np.ndarray of weighting calculated weighting function.
    """
    # Assuming FRF matrix is N_out x N_out x N_freq


    reshaped_mimo_tf = np.transpose(MIMO_transfer_function, (0,2,1)) # This makes it into N_out x N_f x N_in
    variance_matrix = np.var(reshaped_mimo_tf,axis=2,ddof=1).T # Makes the variance

    return 1 / np.sqrt(variance_matrix)
    


def make_X_and_Y(polynomial_basis_function: np.ndarray, weighting_function: np.ndarray,MIMO_FRF: np.ndarray):
    """ Function to calculate X and Y tensors.
    X is Freq_bins x (polynomial_order+1) x N_output
    Y is Freq_bins x N_input*(polynomialorder+1) x N_output

    [insert Peeters Paper + Brandon Review as reference, and insert equation numbers as well cuz you can't put them in here]

    Args:
        polynomial_basis_function (np.ndarray): Output of function: make_polynomial_basis_fcn
        weighting_function (np.ndarray): Output of function: frequency_dependent_weighting
        MIMO_FRF (np.ndarray): Multi-Input-Multi-Output Frequency Response Function.

    Returns:
        Tuple (X,Y) calculated.
    """


# X calculation: 

# Array broascasting as is needed for calculation, as each have only 1 shared dimension, and X has the dimensions from both.
    broadcasted_w = weighting_function[:,np.newaxis,:]  
    broadcasted_polybasisfcn = polynomial_basis_function[:,:,np.newaxis]

    X = broadcasted_polybasisfcn*broadcasted_w

# Y calculation:

    N_out = MIMO_FRF.shape[0]
    N_in = MIMO_FRF.shape[1]
    N_f = MIMO_FRF.shape[2]
    poly_order = np.shape(polynomial_basis_function)[1] #   *** not accurate, it's p+1 not just p

    Y = np.zeros((N_f,N_in*poly_order,N_out)).astype(np.complex128)

    # Reshaping the FRF to a dimension easier to work with for this specific calculation
    reshape_frf_for_kronecker = np.transpose(MIMO_FRF,(2,1,0)) # becomes N_f x N_in x N_out

    initial_product = []
    # From the paper, 2 things multiplied, then kronecker product with the FRF basically
    for i in range(N_out):
        temp_var = -weighting_function[:,i][:,np.newaxis]*polynomial_basis_function
        initial_product.append(temp_var)

    initial_product = np.transpose(initial_product,(1,2,0))

    #does kronecker product

    for i in range(0,N_out):
        for j in range(0,N_f):
            kronecker_product = np.kron(initial_product[j,:,i],reshape_frf_for_kronecker[j,:,i])
            Y[j,:,i] = kronecker_product



    return X,Y



def make_RST_optimized(X,Y):
    """Makes R, S, and T tensors for least-squares estimate (probably most cruicial function for least squares estimate)

    R and T, are the hermitian gram matrices, (Gram Matrix with Complex Conjugation), of X and Y respectively. 
    While S, is the hermitian gram matrix of Y and X together
    [ref papers again]

    Args:
        X (Tensor, np.ndarray): Outputs from make_X_and_Y
        Y (Tensor, np.ndarray): ~~~~~~~~~~~~~~~~~~~~~~~~~~

    Returns:
        tuple: R,S,T matrices
    """
    XH = np.transpose(X.conj(),(1,0,2)) # Transposes without touching the output dimension and does complex conjugation
    YH = np.transpose(Y.conj(),(1,0,2))

    p = X.shape[1] # Technically is p+1 but just for the sake of matrix shape.
    N_out = X.shape[2] # Test should assert that N_out is same for X and Y, but probably is.
    N_in = Y.shape[1] // p # Y is freq_bins x N_in x(p+1)xN_ouput

    # Supposedly a real number

    R = np.zeros((p, p, N_out), dtype=np.float64)
    S = np.zeros((p, N_in*p, N_out), dtype=np.float64)
    T = np.zeros((N_in*p, N_in*p, N_out), dtype=np.float64)

    # Memory optimized calculation, uses numpy.dot, which uses BLAS compiled in C. same as using @

    for i in range(N_out):
        R[:,:,i] = np.real(np.dot(XH[:,:,i],X[:,:,i]))
        S[:,:,i] = np.real(np.dot(XH[:,:,i],Y[:,:,i]))
        T[:,:,i] = np.real(np.dot(YH[:,:,i],Y[:,:,i]))

    return R, S, T


# I think the errors start here.

def make_M_matrix(R:np.ndarray,S:np.ndarray,T:np.ndarray):
    """Makes M matrix, which is the result of minimizing the linearized least squres cost function.
    [ref paper]

    the matrix M is defined as the sum of all the slices of this value:

    (T + S^T R^-1 S)_output. where the slices are the corresponding output.

    Args:
        R (np.ndarray): shape (p+1) x (p+1) x N_outputs
        S (np.ndarray): shape (p+1) x N_inputs *(p+1) x N_outputs
        T (np.ndarray):  N_inputs * (p+1) x N_inputs *(p+1) x N_outputs

    Returns:
       M matrix
    """
    N_out =R.shape[-1]
    M = np.zeros((T.shape[0],T.shape[1]))

    for i in range(N_out):
        M = M + (T[:,:,i] + (S[:,:,i].T) @ np.linalg.inv(R[:,:,i]) @ S[:,:,i] )
    
    return 2*M






def make_LSQ_alpha(M: np.ndarray, N_in: int):
    """Makes the least squares estimate of alpha

    Args:
        M (np.ndarray): _description_
        N_in (int): _description_

    Returns:
        _type_: _description_
    """
    polynomial_order = M.shape[0] // N_in 

    alpha = np.zeros((N_in * (polynomial_order), N_in))

    alpha[-N_in:, :] = np.eye(N_in)

    
    M_lhs = M[0:N_in * (polynomial_order - 1), 0: N_in * (polynomial_order - 1)]
    M_rhs = M[0:N_in * (polynomial_order - 1), N_in * (polynomial_order - 1):N_in * polynomial_order]

    # alpha[:-N_in, :] = np.linalg.solve(M_lhs, M_rhs)
    alpha[:-N_in, :] = np.dot(np.linalg.inv(M_lhs),M_rhs)
    return alpha






def make_LSQ_beta(alpha,R,S):

    N_out = R.shape[-1]
    N_in = alpha.shape[-1]
    polynomial_order = alpha.shape[0]//N_in

    beta = np.zeros((polynomial_order,N_in,N_out))
    for i in range(N_out):
        beta[:,:,i] = -np.linalg.inv(R[:,:,i]) @ S[:,:,i] @ alpha

    return beta







def make_companion_Matrix(alpha:np.ndarray,N_in:int):

    alpha_to_p_minus1 = alpha[:-N_in,:]
    mp = alpha_to_p_minus1.shape[0]
    p = mp//N_in
    companion = np.zeros((mp,mp))

    for i in range(p-1):
        companion[N_in * i:N_in * (i + 1), N_in * (i + 1):N_in * (i + 2)] = np.eye(N_in)

    for i in range(p):
        companion[mp-N_in:mp, i*N_in:(i+1)*N_in] = -alpha[i*N_in:(i+1)*N_in, :].T
    
    return companion





def make_time_poles_and_participation_factors(companion_matrix: np.ndarray):
    return np.linalg.eig(companion_matrix)

def basic_stability(poles):
    return poles[np.real(poles) <= 0]



def poles_to_modal(poles):

    wn = np.abs(poles) 
    dr = -np.real(poles) / wn
    
    return wn, dr


def unphysical_mode_filer(poles):

    _,zn = poles_to_modal(poles)
    
    return poles[zn >=0 ]


def make_polynomial_FRF(alpha,beta,polynomial_basis):
    """
    A is N_in x N_in by N_f

    B is N_out x N_in by N_f

    polynomial_basis is p+1 x N_f



    H is B @ inv(A)

    



    """
    N_f = polynomial_basis.shape[0]
    poly_order = polynomial_basis.shape[1]
    N_in = alpha.shape[0] // poly_order
    N_out = beta.shape[-1]

    A = np.zeros((N_in,N_in,N_f))

    B = np.zeros((N_out,N_in,N_f))
    
    H = np.zeros((N_out,N_in,N_f))

    for i in range(poly_order):
        A+= polynomial_basis[:]
    pass

def pLSCF_poles_to_modal(eigenvalues):
    eigenvalues2 = basic_stability(eigenvalues)
    eigenvalues3 = unphysical_mode_filer(eigenvalues2)
    return poles_to_modal(eigenvalues3)



def plscf_bootleg(model_order: int, data: dict):

    mimo_normed_psds = np.zeros((12,12,16386))
    for i in range(12):
        for j in range(12):
            mimo_normed_psds[i,j] = utils.normalize_psd(data["respPSD"][i,j],0.25)

    samp_freq = 8192

    poly_basis = make_polynomial_basis_fcn(model_order, data["freqs"][:-2], sampling_frequency=samp_freq)

    weighting_fn = frequency_dependent_weighting(mimo_normed_psds[:, :, :-2])
    # weighting_fn = np.ones(weighting_fn.shape,np.float64)

    X, Y = make_X_and_Y(poly_basis, weighting_fn, mimo_normed_psds[:, :,:-2])

    R, S, T = make_RST_optimized(X, Y)

    M = make_M_matrix(R, S, T)

    alpha = make_LSQ_alpha(M, 12)

    beta = make_LSQ_beta(alpha, R, S)

    companion = make_companion_Matrix(alpha, 12)

    eig_val, eig_mat = np.linalg.eig(companion)
    eig_val = basic_stability(eig_val)
    eig_val = unphysical_mode_filer(eig_val)
    wn, zeta = poles_to_modal(eig_val)

    return wn, zeta, eig_mat



def plscf_bootleg_plus_timing(model_order: int, data: dict):


    samp_freq = 8192
    time1 = time.time()

    poly_basis = make_polynomial_basis_fcn(model_order, data["freqs"][2:2050], sampling_frequency=samp_freq)
    time2 = time.time()
    
    weighting_fn = frequency_dependent_weighting(data["FRF"][:, :, 2:2050])
    time3 = time.time()

    X, Y = make_X_and_Y(poly_basis, weighting_fn, data["FRF"][:, :, 2:2050])
    time4 = time.time()

    R, S, T = make_RST_optimized(X,Y)
    time5 = time.time()

    M = make_M_matrix(R, S, T)
    time6 = time.time()

    alpha = make_LSQ_alpha(M, 12)
    time7 = time.time()

    beta = make_LSQ_beta(alpha, R, S)
    time8 = time.time()

    companion = make_companion_Matrix(alpha, 12)
    time9 = time.time()

    eig_val, eigs_mat = np.linalg.eig(companion)
    time10 = time.time()

    wn, zeta = poles_to_modal(eig_val)
    time11 = time.time()


    print("Time for polynomail basis: " + str(time2-time1))
    print("Time for Frequency dependent weighting: "+ str(time3-time2))
    print("Time for X,Y: "+ str(time4-time3))
    print("Time for R, S, T: " + str(time5-time4))
    print("Time for M matrix: "+ str(time6-time5))
    print("Time for alpha: "+ str(time7-time6))
    print("Time for beta: "+ str(time8-time7))
    print("Time for companion matrix: "+ str(time9-time8))
    print("Time for eigendecomposition: "+ str(time10-time9))
    print("Time for poles to modal: "+ str(time11-time10))
    print("Total Elapsed Time: " + str(time11-time1))


    return wn, zeta, eigs_mat

































# print(np.shape(polybasistest))
# print(np.shape(w))

# expanded_polybasistest = polybasistest[:,:,np.newaxis]
# expanded_w = w[:,np.newaxis,:]

# print(np.shape(expanded_polybasistest))
# print(np.shape(expanded_w))

# X = expanded_polybasistest*expanded_w

# print(np.shape(X))

# reshape_frf_kronecker = np.transpose(Healthy_Data.FRF,(2,1,0))[0:-2] # N_f x N_in x N_out
# print(np.shape(reshape_frf_kronecker))








# Y = []

# for i in range(0,12):
#     initial_product =  -w[:,i][:,np.newaxis]*polybasistest
#     for j in range(0,1024):
#         kronecker_product = np.kron(initial_product[j,:],reshape_frf_kronecker[j,:,i])


# meow = []        
# for i in range(12):
#     aaa = -w[:,i][:,np.newaxis]*polybasistest
#     meow.append(aaa)
    
# meow = np.transpose(meow,(1,2,0))
# print(np.shape(meow))

# Y = np.zeros((1024,240,12)).astype(np.complex128)
# for i in range(12):
#     for j in range(1024):
#         kronecker_product = np.kron(meow[j,:,i],reshape_frf_kronecker[j,:,i])
#         Y[j,:,i]=kronecker_product




# with open("output_file4.txt", "w") as file:
#     for i in range(Y.shape[0]):  # Iterate over the first dimension
#         file.write(f"Slice {i}:\n")  # Write the slice index
#         np.savetxt(file, Y[i, :, :], fmt="%.6f", delimiter="\t")  # Write each 2D slice
#         file.write("\n")


# Healthy_Data = ru.format_state("full_tests_new.unv","Healthy State",4104,16384)

# fs = (Healthy_Data.freqs[1]-Healthy_Data.freqs[0]) * (len(Healthy_Data.freqs)-2)*2

# frf = Healthy_Data.FRF[:,:,0:-2]


# polybasistest =  make_polynomial_basis_fcn(20,Healthy_Data.freqs,sampling_frequency=fs)

# polybasistest = polybasistest[0:-2]

# w =  frequency_dependent_weighting(frf)

# X,Y =  make_X_and_Y(polynomial_basis_function=polybasistest,weighting_function=w,MIMO_FRF=frf)

# # print(X.shape)
# # print(Y.shape)


# R,S,T =  make_R_S_T(X,Y)
# # print(R.shape)
# # print(S.shape)
# # print(T.shape)
    
# M =  M_MATRIX(R,S,T)


# alpha =  make_LSQ_alpha(M,12)

# beta =  make_LSQ_beta(alpha,R,S)


# comp_mat =  make_companion_Matrix(alpha,12)
# # print(comp_mat)



# test1,test2 = make_time_poles_and_participation_factors(comp_mat)

# # print(test1.shape)
# # print(test2.shape)

# wn,zeta = poles_to_modal(test1)

# print(wn)
# print(zeta)


# a,b,c = plscf_bootleg(20,data=Healthy_Data)

# print(a)
# print(b)
# #   bismillah 
