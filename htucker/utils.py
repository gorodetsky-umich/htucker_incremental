"""Utility functions for Hierarchical Tucker decomposition."""

import numpy as np
from math import ceil
from warnings import warn


class NotFoundError(Exception):
    """Exception raised when a required tensor is not found."""
    
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def create_permutations(dims):
    """Creates permutations to compute the matricizations.
    
    Args:
        dims (list or int): Dimensions or number of dimensions
        
    Returns:
        list: List of permutation arrays
    """
    permutations = []
    if isinstance(dims, int):
        # If dims is an integer, treat it as number of dimensions
        dimensionList = list(range(dims))
    else:
        # Otherwise, use the provided list
        dimensionList = dims.copy()
        
    for dim in dimensionList:
        copyDimensions = dimensionList.copy()
        firstDim = copyDimensions.pop(copyDimensions.index(dim))
        permutations.append([firstDim] + copyDimensions)
    return permutations


def split_dimensions(tensor, dims=None):
    """Split dimensions for hierarchical decomposition.
    
    Args:
        tensor (ndarray): Input tensor
        dims (list, optional): Dimension specification for splitting
        
    Returns:
        ndarray or tuple: Split tensor or two lists containing split dimensions
    """
    if dims is None:
        # Original functionality - split a list of dimensions
        n_dims = len(tensor)
        return tensor[:ceil(n_dims/2)], tensor[ceil(n_dims/2):]
    else:
        # New functionality - reshape tensor based on dimension specification
        if not isinstance(dims, list) or not all(isinstance(d, list) for d in dims):
            raise ValueError("dims must be a list of lists specifying the new shape")
            
        # Calculate the new shape from the dimensions specification
        new_shape = [len(dim_group) for dim_group in dims]
        for i, dim_group in enumerate(dims):
            new_shape[i] = np.prod([tensor.shape[d] for d in dim_group]).astype(int)
        
        # Reshape the tensor
        return tensor.reshape(new_shape)


def mode_n_unfolding(tensor, mode):
    """Computes mode-n unfolding/matricization of a tensor in the sense of Kolda&Bader.
    
    Args:
        tensor (ndarray): Input tensor
        mode (int): Mode to unfold along (0-indexed)
        
    Returns:
        ndarray: Unfolded tensor
    """
    nDims = len(tensor.shape)
    dims = [dim for dim in range(nDims)]
    modeIdx = dims.pop(mode)
    dims = [modeIdx] + dims
    tensor = tensor.transpose(dims)
    return tensor.reshape(tensor.shape[0], -1, order='F')


def mode_n_product(tensor, matrix, mode):
    """Compute the n-mode product of a tensor and a matrix.
    
    Args:
        tensor (ndarray): Input tensor
        matrix (ndarray): Matrix to multiply with
        mode (int): Mode along which to multiply
        
    Returns:
        ndarray: Result of the n-mode product
    """
    dims = [idx for idx in range(len(tensor.shape) + len(matrix.shape) - 2)]
    tensor_ax, matrix_ax = mode, 1  # Mode axis of tensor, second axis of matrix
    dims.pop(tensor_ax)
    dims.append(tensor_ax)
    tensor = np.tensordot(tensor, matrix, axes=([tensor_ax], [matrix_ax]))
    tensor = tensor.transpose(np.argsort(dims).tolist())
    return tensor


def convert_to_base2(num, width=None):
    """Convert a number to its binary representation as a list.
    
    Args:
        num (int): Input number
        width (int, optional): Width of the binary representation
        
    Returns:
        list: Binary representation as a list of integers
    """
    binary = bin(num)[2:]  # convert to binary string
    if width is not None:
        binary = binary.zfill(width)  # pad with leading zeros to specified width
    else:
        binary = binary.zfill(3)  # default padding to 3 digits
    binary_list = [int(bit) for bit in binary]
    return binary_list
