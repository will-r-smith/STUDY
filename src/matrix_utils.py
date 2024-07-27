import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.linalg import svd, inv


import torch
import torch.nn as nn
from copy import deepcopy
import json


def norms(original_mat, approx_mat):
    # Calculate the Frobenius norm
    frobenius_diff = torch.norm(original_mat - approx_mat, p='fro')
    frobenius_original = torch.norm(original_mat, p='fro')
    frobenius_ratio = (frobenius_diff / frobenius_original).item()

    return frobenius_diff, frobenius_ratio


# Helper functions for abs weight pruning
def sorted_mat(matrix):
    temp = list(abs(matrix).flatten())
    temp.sort()
    return temp


def prune(matrix, mat_sort, to_prune):
    if to_prune != 0:
        alpha = mat_sort[int(to_prune * 0.1 * len(mat_sort))]
        matrix[abs(matrix) <= alpha] = 0
    return matrix


def rank(matrix):
    np_matrix = np.array(matrix)
    return np.linalg.matrix_rank(np_matrix)/min(list(np_matrix.shape))


# What percentage can be pruned by weight
def sparsity(matrix, alpha):
    abs_matrix = abs(matrix)
    filtered_matrix = abs_matrix[abs_matrix < alpha]
    return len(filtered_matrix)/matrix.size


def viz_rank_change(rank_list,name):
    fig = plt.figure()
    plt.plot(rank_list)
    plt.savefig(name)





def do_lr(model, name, weight, k):
    niter = 20

    assert weight.ndim == 2

    max_rank = min(weight.shape[0], weight.shape[1])
    desired_rank = int(max_rank * k)

    results = torch.svd_lowrank(weight, q=desired_rank, niter=niter)

    U = results[0].clone().detach().requires_grad_(True).to(weight.dtype)
    S = torch.diag(results[1]).clone().detach().requires_grad_(True).to(weight.dtype)
    Vt = results[2].T.clone().detach().requires_grad_(True).to(weight.dtype)

    # Create a valid parameter name by replacing periods
    param_name_base = name.replace('.', '_')

    # Get the layer object where the parameters belong
    layer = model
    for part in name.split('.')[:-1]:
        layer = getattr(layer, part)

    setattr(layer, f"{param_name_base}_U", nn.Parameter(U))
    setattr(layer, f"{param_name_base}_S", nn.Parameter(S))
    setattr(layer, f"{param_name_base}_Vt", nn.Parameter(Vt))

    def low_rank_forward(self, input):
        U = getattr(self, f"{param_name_base}_U")
        S = getattr(self, f"{param_name_base}_S")
        Vt = getattr(self, f"{param_name_base}_Vt")

        # Ensure all tensors are of the same dtype
        dtype = input.dtype
        U = U.to(dtype)
        S = S.to(dtype)
        Vt = Vt.to(dtype)
        
        weight_approx = U @ S @ Vt
        return nn.functional.linear(input, weight_approx)

    # Save the original forward method
    original_forward = layer.forward

    # Override the forward method for the layer
    layer.forward = low_rank_forward.__get__(layer, type(layer))
    layer._original_forward = original_forward  # Save the original forward method if needed

    weight_approx = results[0] @ torch.diag(results[1]) @ results[2].T

    return model, weight_approx, [getattr(layer, f"{param_name_base}_U"), getattr(layer, f"{param_name_base}_S"), getattr(layer, f"{param_name_base}_Vt")]



# Function to project a given matrix onto the MM* class
def do_mm(weight):
    """
    Project a given weight matrix onto the class of MM* matrices using the Monarch parametrization.
    This implementation handles the general case for rectangular matrices.
    
    orig_shape = weight.shape

    b = int(orig_shape[0] / orig_shape[1])
    b = int(np.sqrt(orig_shape[0]))
    print(b)


    blocks = int(orig_shape[0] / b)

    pad_cols = int(orig_shape[0] - orig_shape[1])

    padded_weight = F.pad(weight, (0, int(pad_cols), 0, 0), mode='constant', value=0)
    
    padded_size = padded_weight.shape

    print(padded_size)

    # Create block matrices P(b,n) and P^T(b,n)
    P = torch.zeros(padded_size)
    for i in range(blocks):
        for j in range(blocks):
            P[i * b: (i + 1) * b, j * b: (j + 1) * b] = torch.eye(b)

    PT = P.T
    P_weight = P @ padded_weight @ PT

    print("check1")

    # Step 1: Compute F(i,j) matrices
    F_matrices = torch.zeros((blocks, blocks, b, b))
    for i in range(blocks):
        for j in range(blocks):
            Mfi1 = P_weight[i * b: (i + 1) * b, 0: b]
            Mfij = P_weight[i * b: (i + 1) * b, j * b: (j + 1) * b]
            Mf1j = P_weight[0: b, j * b: (j + 1) * b]
            Mf11 = P_weight[0: b, 0: b]
            F_matrices[i, j] = inv(Mfi1) @ Mfij @ inv(Mf1j) @ Mf11

    # Step 2: Compute C_hat_1 by simultaneous diagonalization
    C_hat_1 = torch.eye(b)
    for i in range(blocks):
        for j in range(blocks):
            C_hat_1 = C_hat_1 @ torch.linalg.eigh(F_matrices[i, j])[1]

    # Step 3: Compute A_hat_i matrices
    A_hat = torch.zeros((blocks, b, b))
    for i in range(blocks):
        Mfi1 = P_weight[i * b: (i + 1) * b, 0: b]
        A_hat[i] = Mfi1 @ inv(C_hat_1)

    print("check4")

    # Step 4: Compute C_hat_j matrices
    C_hat = torch.zeros((blocks, b, b))
    C_hat[0] = C_hat_1
    for j in range(1, blocks):
        Mf1j = P_weight[0: b, j * b: (j + 1) * b]
        C_hat[j] = inv(A_hat[0]) @ Mf1j

    print("check5")

    # Step 5: Compute D_hat_ij matrices
    D_hat = torch.zeros((blocks, blocks, b, b))
    for i in range(blocks):
        for j in range(blocks):
            Mfij = P_weight[i * b: (i + 1) * b, j * b: (j + 1) * b]
            D_hat[i, j] = inv(A_hat[i]) @ Mfij @ inv(C_hat[j])

    print("check6")

    # Reconstruct the Monarch matrix from A_hat and C_hat
    Monarch_matrix = torch.zeros(padded_size)
    for i in range(blocks):
        for j in range(blocks):
            Monarch_matrix[i * b: (i + 1) * b, j * b: (j + 1) * b] = A_hat[i] @ D_hat[i, j] @ C_hat[j]

    # Extract the relevant portion of the matrix
    weight_approx = Monarch_matrix[:orig_shape[0], :orig_shape[1]]
    weight_approx = torch.nn.Parameter(weight_approx)


    return weight_approx

    """

