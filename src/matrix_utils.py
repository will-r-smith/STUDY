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




def do_psm(model, name, weight, reduction_rate, num_matrices=3, num_interpolation_steps=20):
    """
    Applies a sparse matrix approximation to the given weight matrix.
    
    Args:
    - model: The model containing the layer to be approximated.
    - name: The name of the parameter (layer) to be approximated.
    - weight: The weight matrix to be approximated.
    - reduction_rate: The reduction rate specifying the fraction of parameters to retain.
    - num_matrices: The number of matrices in the sparse factorization.
    - num_interpolation_steps: Number of interpolation steps for approximation.
    
    Returns:
    - model: The model with the sparse approximation applied.
    - approx_weight: The approximated weight matrix.
    - trainable_params: List of trainable parameters in the approximated matrix.
    """

    # Convert weight matrix to numpy for approximation
    weight_np = weight.detach().cpu().numpy()

    # Define the sparse matrix approximator
    approximator = PSMApproximatorWrapper(num_interpolation_steps=num_interpolation_steps, max_nb_matrices=num_matrices)

    # Approximate the weight matrix
    approx_res = approximator.approximate(weight_np, nb_params_share=reduction_rate)

    # Extract the approximated dense matrix
    approx_weight_np = approx_res["approx_mat_dense"]

    # Convert the approximated weight back to a PyTorch tensor
    approx_weight = torch.tensor(approx_weight_np, dtype=weight.dtype, device=weight.device)

    # Convert sparse factors to PyTorch tensors with gradients enabled
    sparse_factors = [
        torch.tensor(sparse_factor.toarray(), dtype=weight.dtype, device=weight.device).requires_grad_(True)
        for sparse_factor in approx_res["faust_approximation"]
    ]

    # Create a valid parameter name by replacing periods
    param_name_base = name.replace('.', '_')

    # Get the layer object where the parameters belong
    layer = model
    for part in name.split('.')[:-1]:
        layer = getattr(layer, part)

    # Set sparse factors as model parameters
    for i, sparse_factor in enumerate(sparse_factors):
        setattr(layer, f"{param_name_base}_sparse_{i}", nn.Parameter(sparse_factor))

    def sparse_forward(self, input):
        # Reconstruct the weight matrix from sparse factors
        res = getattr(self, f"{param_name_base}_sparse_0")
        for i in range(1, len(sparse_factors)):
            res = res @ getattr(self, f"{param_name_base}_sparse_{i}")
        return nn.functional.linear(input, res)

    # Save the original forward method
    original_forward = layer.forward

    # Override the forward method for the layer
    layer.forward = sparse_forward.__get__(layer, type(layer))
    layer._original_forward = original_forward  # Save the original forward method if needed

    return model, approx_weight, sparse_factors




import abc
import numpy as np

from scipy.optimize import minimize_scalar

from pyfaust.fact import hierarchical
from pyfaust.proj import sp
from pyfaust.factparams import ParamsHierarchical, StoppingCriterion



class Approximator:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def approximate(self, optim_mat: np.ndarray, nb_params_share: float) -> dict:
        pass

    @abc.abstractmethod
    def get_name(self) -> str:
        pass




class PSMApproximator(Approximator):
    def __init__(self, nb_matrices: int, linear_nb_nonzero_elements_distribution: bool, max_last_mat_param_share=0.9):
        self.nb_matrices = nb_matrices
        self.linear_nb_nonzero_elements_distribution = linear_nb_nonzero_elements_distribution
        self.max_last_mat_param_share = max_last_mat_param_share

    def approximate(self, optim_mat: np.ndarray, nb_params_share: float, num_interpolation_steps=9):
        self.nnz_share = nb_params_share

        optim_mat64 = optim_mat.astype("float64")
        nb_nonzero_elements = int(optim_mat64.size * self.nnz_share)
        best_approximation = None
        for last_mat_param_share in np.linspace(0.1, self.max_last_mat_param_share, num=num_interpolation_steps):
            last_nb_nonzero_elements = int(last_mat_param_share * nb_nonzero_elements)
            res_dict = self.faust_approximation(weights=optim_mat64, last_nb_nonzero_elements=last_nb_nonzero_elements, total_nb_nonzero_elements=nb_nonzero_elements)
            
            if not (res_dict is None):
                norm = np.linalg.norm(optim_mat64 - res_dict["approx_mat_dense"], ord="fro")

                if best_approximation is None or best_approximation["objective_function_result"] > norm:
                    res_dict["objective_function_result"] = norm
                    best_approximation = res_dict
        
        return best_approximation

    def get_name(self):
        name = "PSMApproximator_" + str(self.nb_matrices) + "F_"
        if self.linear_nb_nonzero_elements_distribution:
            name += "linear"
        else:
            name += "nonlinear"
        return name

    def get_nb_elements_list(self) -> list:
        if self.nb_matrices == 2:
            nb_nonzero_elements_list = [self.total_nb_nonzero_elements-self.last_nb_nonzero_elements, self.last_nb_nonzero_elements]
        else:
            # NOTE if linear is false, then the residual shaping is done exponentially
            k = self.nb_matrices - 1
            if self.linear_nb_nonzero_elements_distribution:
                alpha = (2 * self.total_nb_nonzero_elements) / (k+1) - self.last_nb_nonzero_elements
                m = (self.last_nb_nonzero_elements - alpha) / k
                nb_nonzero_elements_list = [int(alpha + m*mat_i) for mat_i in range(self.nb_matrices)]
            else:
                optim_res = minimize_scalar(self.exponential_distribution_optimization_fun, bounds=(1, self.total_nb_nonzero_elements), method='bounded')
                alpha = optim_res.x
                m = self.alpha_to_m(alpha)
                nb_nonzero_elements_list = [int(alpha * np.exp(m * mat_i)) for mat_i in range(self.nb_matrices)]
        
        # Due to rounding errors, there might be minimal errors in the number elements list
        nb_exceeded_elements = sum(nb_nonzero_elements_list) - self.total_nb_nonzero_elements
        while nb_exceeded_elements > 0:
            nb_elements_pos_to_alter = nb_nonzero_elements_list.index(max(nb_nonzero_elements_list))
            nb_nonzero_elements_list[nb_elements_pos_to_alter] -= nb_exceeded_elements
            nb_exceeded_elements = sum(nb_nonzero_elements_list) - self.total_nb_nonzero_elements

        assert len(nb_nonzero_elements_list) == self.nb_matrices, "Length of the nb_nonzero_elements_list does not match the expected length"
        assert nb_nonzero_elements_list[-1] <= self.last_nb_nonzero_elements, "The last entry in the nb_nonzero_elements_list does not match the expected last nb of nonzero elements"
        assert sum(nb_nonzero_elements_list) <= self.total_nb_nonzero_elements, "The sum over the element distribution deviates from the expected total number of elements"

        if any([tmp < 0 for tmp in nb_nonzero_elements_list]):
            nb_nonzero_elements_list = [0 for _ in range(self.nb_matrices)]

        return nb_nonzero_elements_list

    def alpha_to_m(self, alpha):
        k = self.nb_matrices - 1
        arg = self.last_nb_nonzero_elements / alpha
        if arg > 1e-6:
            return np.log(arg) / k
        else:
            return -14 / k

    def exponential_distribution_optimization_fun(self, x):
        nb_nonzero_elements_integral = 0
        for mat_i in range(self.nb_matrices):
            nb_nonzero_elements_integral += x * np.exp(self.alpha_to_m(x) * mat_i)
        return np.abs(nb_nonzero_elements_integral - self.total_nb_nonzero_elements)

    def get_matrices_shapes_list(self, target_mat_shape: np.ndarray):
        max_shape_dim = max(target_mat_shape)
        matrices_shapes_list = [(target_mat_shape[0], max_shape_dim)] \
            + [(max_shape_dim, max_shape_dim) for _ in range(self.nb_matrices - 2)] \
            + [(max_shape_dim, target_mat_shape[1])]
        return matrices_shapes_list

    def faust_factorization_to_dense(self, faust_matrices: list, nb_nonzero_elements_list=None): 
        # The nb_nonzero_elements_list is only used to check if the number nonzero elements are right
        res = faust_matrices[0]
        assert nb_nonzero_elements_list is None or len(res.data) <= nb_nonzero_elements_list[0], "Number of nonzero elements is wrong"
        for mat_i, mat in enumerate(faust_matrices[1:]):
            assert nb_nonzero_elements_list is None or len(mat.data) <= nb_nonzero_elements_list[mat_i+1], "Number of nonzero elements is wrong"
            res = res @ mat
        if not isinstance(res, np.ndarray):
            res = res.todense()
        return res

    def faust_approximation(self, weights: np.ndarray, last_nb_nonzero_elements: int, total_nb_nonzero_elements: int):
        self.total_nb_nonzero_elements = total_nb_nonzero_elements
        self.last_nb_nonzero_elements = last_nb_nonzero_elements

        nb_elements_distribution = self.get_nb_elements_list()
        matrices_shapes_list = self.get_matrices_shapes_list(target_mat_shape=weights.shape)

        try:
            # Set constrains
            # The constrains define the constrains for non zero elements per factor
            # Details : https://faustgrp.gitlabpages.inria.fr/faust/last-doc/html/constraint.png
            # https://faust.inria.fr/api-doc/
            # NOTE That the last residual is then taken as the last matrix entry of the matrices_shapes_list
            fact_cons = []
            res_cons = []
            for factor_i, (factor_shape, factor_nb_nonzero_elements) in enumerate(zip(matrices_shapes_list[:-1], nb_elements_distribution[:-1])):
                fact_cons.append(sp(factor_shape, factor_nb_nonzero_elements, normalized=True))
                nb_residual_elements = nb_elements_distribution[factor_i + 1]
                residual_shape = (factor_shape[1], weights.shape[1])
                res_cons.append(sp(residual_shape, nb_residual_elements, normalized=True))

            # Set stopping criteria
            local_stopping_criteria = StoppingCriterion()
            global_stopping_criteria = StoppingCriterion()

            param = ParamsHierarchical(fact_cons,
                                    res_cons,
                                    local_stopping_criteria,
                                    global_stopping_criteria)

            approximation = hierarchical(weights, param, backend=2016)
            sparse_factors = [approximation.factors(i) for i in range(self.nb_matrices)]

            res_dict = dict()
            res_dict["type"] = "HierarchicalFaust"
            res_dict["faust_approximation"] = sparse_factors
            res_dict["approx_mat_dense"] = self.faust_factorization_to_dense(sparse_factors, nb_nonzero_elements_list=nb_elements_distribution)
            res_dict["nb_parameters"] = sum(nb_elements_distribution)
            res_dict["linear_nb_nonzero_elements_distribution"] = self.linear_nb_nonzero_elements_distribution
            res_dict["nnz_share"] = self.nnz_share
            res_dict["nb_matrices"] = self.nb_matrices
            return res_dict
        except Exception as e:
            print(e)
            res_dict = dict()
            res_dict["type"] = "HierarchicalFaust"
            res_dict["faust_approximation"] = None
            res_dict["approx_mat_dense"] = np.zeros_like(weights)
            res_dict["nb_parameters"] = 0
            res_dict["linear_nb_nonzero_elements_distribution"] = self.linear_nb_nonzero_elements_distribution
            res_dict["nnz_share"] = self.nnz_share
            res_dict["nb_matrices"] = self.nb_matrices
            return res_dict
        





class PSMApproximatorWrapper(Approximator):
    def __init__(self, num_interpolation_steps=17, only_linear_distribution=True, max_nb_matrices=3, max_last_mat_param_share=0.9):
        self.num_interpolation_steps = num_interpolation_steps
        if only_linear_distribution:
            self.linear_nb_nonzero_elements_distribution_values = [True]
        else:
            self.linear_nb_nonzero_elements_distribution_values = [True, False]
        self.max_nb_matrices = max_nb_matrices
        self.max_last_mat_param_share = max_last_mat_param_share

    def get_name(self):
        name = "PSMApproximatorWrapper"
        return name

    def approximate(self, optim_mat: np.ndarray, nb_params_share: float):
        optim_mat64 = optim_mat.astype("float64")
        best_approximation = None
        nb_matrices_list = list(range(2, self.max_nb_matrices+1))
        for nb_matrices in nb_matrices_list:
            for linear_nb_nonzero_elements_distribution in self.linear_nb_nonzero_elements_distribution_values:
                approximator = PSMApproximator(nb_matrices=nb_matrices, linear_nb_nonzero_elements_distribution=linear_nb_nonzero_elements_distribution)
                res_dict = approximator.approximate(optim_mat=optim_mat, nb_params_share=nb_params_share, num_interpolation_steps=self.num_interpolation_steps)
                
                if res_dict is not None:
                    norm = np.linalg.norm(optim_mat64 - res_dict["approx_mat_dense"], ord="fro")

                    if best_approximation is None or best_approximation["objective_function_result"] > norm:
                        res_dict["objective_function_result"] = norm
                        best_approximation = res_dict
        
        return best_approximation 
    




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

