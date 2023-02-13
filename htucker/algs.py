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
                                         tensor.shape[2]*tensor.shape[3]))

        [u, s, v2] = np.linalg.svd(mat_tensor, full_matrices=True)

        v = np.dot(np.diag(s), v2)

        # u is n1n2 x r5
        # v is n3n4 x r6
        
        # need an HOSVD tucker of u (and v) this (look at kolda paper)
        
def hosvd(tensor):
    
    # ndims = len(tensor.shape)
    ndims = 4

    # Need to find a generalized way to compute the permutations later!
    permutations = [
        (0,1,2,3),
        (1,0,2,3),
        (2,0,1,3),
        (3,0,1,2)
        ]
        
    leftSingularVectors=[]
    for dim , perm in enumerate(permutations):
        tempTensor=tensor.transpose(perm).reshape(tensor.shape[dim],-1)
        u, _, _ = np.linalg.svd(tempTensor,full_matrices=False)
        leftSingularVectors.append(u)
    
    for dim , u in enumerate(leftSingularVectors):
        tensor=np.tensordot(tensor, u.T, axes=(dim,0))

    return tensor , leftSingularVectors

        

        
