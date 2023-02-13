"""Hierarchical Tucker."""


import numpy as np


import htucker as ht

__all__ = [
     "HTucker",
]


class HTucker:

    # harded for 4d first
    def __init__(self):
        self.leaves = [None]*4
        self.transfer_nodes = [None]*2
        self.root = None


    def compress(self, tensor):

        # split tensor
        mat_tensor = np.reshape(tensor, (tensor.shape[0]*tensor.shape[1],
                                         tensor.shape[2]*tensor.shape[3]), order='F')

        [u, s, v2] = np.linalg.svd(mat_tensor, full_matrices=True)

        v = np.dot(np.diag(s), v2)

        # u is n1n2 x r5
        # v is n3n4 x r6
        
        # need an HOSVD tucker of u (and v) this (look at kolda paper)
        
def hosvd(tensor):
    
    # ndims = len(tensor.shape)
    ndims = 3
    ndims = 4

    # Need to find a generalized way to compute the permutations later!
    permutations = [
        (0,1,2),
        (1,0,2),
        (2,0,1)
        ]
    permutations = [
        (0,1,2,3),
        (1,0,2,3),
        (2,0,1,3),
        (3,0,1,2)
        ]
    
    # Found it. :)
    permutations=createPermutations(ndims)

    leftSingularVectors=[]
    singularValues=[]
    for dim , perm in enumerate(permutations):
        # print(dim,perm)
        tempTensor=tensor.transpose(perm).reshape(tensor.shape[dim],-1)
        [u, s, _] = np.linalg.svd(tempTensor,full_matrices=False)
        # Automatic rank truncation, can later be replaced with the deltaSVD function
        leftSingularVectors.append(u[:,:sum(s>=1e-8)])
        singularValues.append(s)

    
    for dim , u in enumerate(leftSingularVectors):
        # print(dim,u.shape,tensor.shape)
        tensorShape=list(tensor.shape)
        currentIndices=list(range(1,len(tensorShape)))
        currentIndices=currentIndices[:dim]+[0]+currentIndices[dim:]
        tensor=np.tensordot(u.T, tensor, axes=(1, dim)).transpose(currentIndices)
    # tensor = np.einsum('ij,kl,mn,op,ikmo->jlnp',leftSingularVectors[0],leftSingularVectors[1],leftSingularVectors[2],leftSingularVectors[3],tensor)

    return tensor , leftSingularVectors

        

        
