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
        
        u=u[:,:sum(s>=1e-8)]
        v2=v2[:sum(s>=1e-8),:]
        s=s[:sum(s>=1e-8)]
        v = np.dot(np.diag(s), v2)

        # u is n1n2 x r5
        # v is n3n4 x r6

        tucU = hosvd(u.reshape(tensor.shape[0],tensor.shape[1],-1, order='F'))
        tucV = hosvd(v.reshape(-1, tensor.shape[2],tensor.shape[3], order='F'))

        
        # need an HOSVD tucker of u (and v) this (look at kolda paper)
        
def hosvd(tensor):
    
    # ndims = len(tensor.shape)
    ndims = 3
    ndims = 4
    ndims=len(tensor.shape)

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

def createPermutations(nDims):
    # Creates permutations to compute the matricizations
    permutations=[]
    dimensionList=list(range(nDims))
    for dim in dimensionList:
        copyDimensions=dimensionList.copy()
        firstDim=copyDimensions.pop(dim)
        permutations.append([firstDim]+copyDimensions)
    return permutations

def modeNUnfolding(tensor,mode):
    # Computes mode-n unfolding/matricization of a tensor in the sense of Kolda&Bader
    # Assumes the mode is given in 0 indexed format
    nDims = len(tensor.shape)
    dims = [dim for dim in range(nDims)]
    modeIdx = dims.pop(mode)
    dims=[modeIdx]+dims
    tensor=tensor.transpose(dims)
    return tensor.reshape(tensor.shape[0],-1,order='F')

# def matrixTensorProduct(matrix,tensor,axes):
    
#     ax1,ax2=axes

#     matrixShape=list(matrix.shape)
#     tensorShape=list(tensor.shape)
#     assert(matrixShape[ax1]==tensorShape[ax2])

#     productShape=tensorShape.copy()

#     tensorIndices=list(range(len(tensorShape)))

#     order=tensorIndices.pop(ax2)
#     tensorIndices=[order]+tensorIndices

#     # tensor=tensor.transpose(tensorIndices).reshape(tensorShape[ax2],-1,order='F')
#     tensor=np.tensordot(matrix,tensor,axes=[ax1,ax2])
#     currentIndices=list(range(1,len(tensorShape)))
#     currentIndices=currentIndices[:ax2]+[0]+currentIndices[ax2:]


#     return None

        

        
