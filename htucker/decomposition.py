"""Decomposition methods for Hierarchical Tucker."""

import numpy as np

from htucker.utils import create_permutations, mode_n_unfolding


def truncated_svd(
    a, truncation_threshold=None, full_matrices=True, compute_uv=True, hermitian=False
):
    """Compute truncated SVD with error control.

    Args:
        a (ndarray): Input matrix
        truncation_threshold (float, optional): Threshold for truncation
        full_matrices (bool): Whether to compute full matrices
        compute_uv (bool): Whether to compute U and V
        hermitian (bool): Whether the input matrix is hermitian

    Returns:
        list: [u, s, v] - Left singular vectors, singular values, and right singular vectors
    """
    try:
        [u, s, v] = np.linalg.svd(
            a, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian
        )
    except np.linalg.LinAlgError:
        # Fallback to QR decomposition if SVD fails
        q, r = np.linalg.qr(a)
        [u, s, v] = np.linalg.svd(
            r, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian
        )
        u = q @ u

    if truncation_threshold is None:
        return [u, s, v]

    trunc = sum(s >= truncation_threshold)
    u = u[:, :trunc]
    s = s[:trunc]
    v = v[:trunc, :]

    return [u, s, v]


def hosvd(tensor, rtol=None, tol=None, threshold=1e-8, norm=None, dimensions=None):
    """Higher-Order Singular Value Decomposition.

    Args:
        tensor (ndarray): Input tensor
        rtol (float, optional): Relative tolerance
        tol (float, optional): Absolute tolerance
        threshold (float): Threshold for truncated SVD
        norm (float, optional): Norm of tensor
        dimensions (int, optional): Number of dimensions

    Returns:
        tuple: (core_tensor, left_singular_vectors)
    """
    if tol is not None:
        tolerance = tol
    else:
        tolerance = 1e-8

    if (tol is None) and (rtol is not None):
        tensor_norm = np.linalg.norm(tensor)
        tolerance = tensor_norm * rtol

    # if dimensions is None:
    #     ndims = len(tensor.shape)
    # else:
    #     ndims = dimensions

    if len(tensor.shape) == 2:
        [u, s, v] = truncated_svd(tensor, truncation_threshold=threshold, full_matrices=False)
        u = u[:, np.cumsum((s**2)[::-1])[::-1] > (tolerance) ** 2]
        v = v[np.cumsum((s**2)[::-1])[::-1] > (tolerance) ** 2, :]
        s = s[np.cumsum((s**2)[::-1])[::-1] > (tolerance) ** 2]
        return np.diag(s), [u, v.T]

    permutations = create_permutations(len(tensor.shape))

    leftSingularVectors = []
    singularValues = []

    # Compute SVDs for each unfolding
    for dim, _perm in enumerate(permutations):
        tempTensor = mode_n_unfolding(tensor, dim)
        [u, s, v] = truncated_svd(tempTensor, truncation_threshold=threshold, full_matrices=False)
        leftSingularVectors.append(u[:, np.cumsum((s**2)[::-1])[::-1] > (tolerance) ** 2])
        singularValues.append(s[np.cumsum((s**2)[::-1])[::-1] > (tolerance) ** 2])

    # Apply tensor multiplication with transposed singular vectors
    for dim, u in enumerate(leftSingularVectors):
        tensorShape = list(tensor.shape)
        currentIndices = list(range(1, len(tensorShape)))
        currentIndices = currentIndices[:dim] + [0] + currentIndices[dim:]
        tensor = np.tensordot(u.T, tensor, axes=(1, dim)).transpose(currentIndices)

    return tensor, leftSingularVectors


def hosvd_only_for_dimensions(
    tensor,
    tol=None,
    rtol=None,
    threshold=1e-8,
    norm=None,
    dims=None,
    batch_dimension=None,
    contract=False,
):
    """HOSVD for specified dimensions only.

    Args:
        tensor (ndarray): Input tensor
        tol (float, optional): Absolute tolerance
        rtol (float, optional): Relative tolerance
        threshold (float): Threshold for truncated SVD
        norm (float, optional): Norm of tensor
        dims (list, optional): Dimensions to compute HOSVD for
        batch_dimension (int, optional): Batch dimension
        contract (bool): Whether to contract results

    Returns:
        list: Left singular vectors or (contracted tensor,
              left singular vectors) if contract is True
    """
    if dims is None:
        dims = list(range(len(tensor.shape)))

    if (batch_dimension is not None) and (batch_dimension in dims):
        # if batch_dimension in dims:
        dims.remove(batch_dimension)

    if (tol is None) and (rtol is not None):
        if norm is None:
            norm = np.linalg.norm(tensor)
        tol = norm * rtol
    elif (tol is None) and (rtol is None):
        tol = 1e-8
    elif tol is not None:
        pass
    else:
        raise ValueError("Invalid tolerance configuration")

    if len(tensor.shape) == 2:
        [u, s, v] = truncated_svd(tensor, truncation_threshold=threshold, full_matrices=False)
        u = u[:, np.cumsum((s**2)[::-1])[::-1] > (tol) ** 2]
        v = v[np.cumsum((s**2)[::-1])[::-1] > (tol) ** 2, :]
        s = s[np.cumsum((s**2)[::-1])[::-1] > (tol) ** 2]
        return [u, v.T]

    leftSingularVectors = []
    for dimension in dims:
        [u, s, v] = truncated_svd(
            mode_n_unfolding(tensor, dimension), truncation_threshold=threshold, full_matrices=False
        )
        leftSingularVectors.append(u[:, np.cumsum((s**2)[::-1])[::-1] > (tol) ** 2])

    if contract:
        for dimension, lsv in zip(dims, leftSingularVectors):
            dimension_flip = list(range(len(tensor.shape) - 1))
            dimension_flip.insert(dimension, len(tensor.shape) - 1)
            tensor = np.tensordot(tensor, lsv, axes=[dimension, 0]).transpose(dimension_flip)
        return tensor, leftSingularVectors

    return leftSingularVectors
