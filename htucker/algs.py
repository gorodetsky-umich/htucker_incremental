"""Hierarchical Tucker."""


import numpy as np


import htucker as ht

__all__ = [
    "HTucker",
    "hosvd",
]

def truncated_svd(a, truncation_tolerance=None, full_matrices=True, compute_uv=True, hermitian=False):

    [u, s, v] = np.linalg.svd(a,
                              full_matrices=full_matrices,
                              compute_uv=compute_uv,
                              hermitian=False)
    if truncation_tolerance == None:
        return [u, s, v]

    trunc = sum(s>=truncation_tolerance)
    u=u[:,:trunc]
    s=s[:trunc]
    v=v[:trunc,:]

    return [u, s, v]
  
class TuckerCore:
    # Object for tucker cores. Planning to use it to create the recursive graph structure
    def __init__(self,core=None, parent=None) -> None:
        self.parent=parent
        self.core=core
        self.children=[]
        self.dims=[]
        self.ranks=[]

        if parent is not None:
            self._isroot=True
        else:
            self._isroot=False

    def get_ranks(self):
        if self.core is not None:
            self.ranks=list(self.core.shape)
        
class HTucker:

    # harded for 4d first
    def __init__(self):
        self.leaves = [None]*4
        self.transfer_nodes = [None]*2
        self.root = None

    def initialize(self,tensor):
        self.original_shape = list(tensor.shape)
        self.leaves = [None]*len(self.originalShape)
        self.transfer_nodes = [None]*(len(self.originalShape)-2) #Root node not included here
        self.root = None
        

    def split_dimensions(dims):
        n_dims=len(dims)
        return dims[:n_dims//2],dims[n_dims//2:]

    def compress(self, tensor):
        # TODO: Replace initial SVD with HOSVD
        # TODO: Create a structure for the HT
        # TODO: Make compress() function general for n-dimensional tensors
        
        dims=list(tensor.shape)

        # initial split for the tensor
        left,right=split_dimensions(dims)
        
        self.root=TuckerCore()

                


        mat_tensor = np.reshape(tensor, (tensor.shape[0]*tensor.shape[1],
                                         tensor.shape[2]*tensor.shape[3]), order='F')

        # [u, s, v] = np.linalg.svd(mat_tensor, full_matrices=False)
        [u, s, v] = truncated_svd(mat_tensor, 1e-8, full_matrices=False)
        
        # u=u[:,:sum(s>=1e-8)]
        # u=u[:,:sum(s>=1e-8)]
        # u=u[:,:sum(s>=1e-8)]
        # v2=v2[:sum(s>=1e-8),:]
        # s=s[:sum(s>=1e-8)]
        # v = np.dot(np.diag(s), v2)

        # u is n1n2 x r5
        # v is n3n4 x r6

        [core_l, lsv_l] = hosvd(u.reshape(tensor.shape[0],tensor.shape[1],-1, order='F'))
        [core_r, lsv_r] = hosvd(v.reshape(-1, tensor.shape[2],tensor.shape[3], order='F'))

        # need an HOSVD tucker of u (and v) this (look at kolda paper)
        core_l = np.einsum('ijk,lk->ijl', core_l, lsv_l[-1])
        core_r = np.einsum('ijk,li->ljk', core_r, lsv_r[0])

        # top = np.eye(u.shape[1])
        top = np.diag(s)

        return (lsv_l[0], lsv_l[1], lsv_r[1], lsv_r[2], core_l, core_r, top)
        
        
        
def hosvd(tensor):
    ndims=len(tensor.shape)
    permutations=createPermutations(ndims)

    leftSingularVectors=[]
    singularValues=[]

    # May replace this with a combination of mode-n unfolding and truncated svd
    for dim , perm in enumerate(permutations):
        # print(dim,perm)
        tempTensor=tensor.transpose(perm).reshape(tensor.shape[dim],-1)

        # [u, s, v] = np.linalg.svd(tempTensor,full_matrices=False)
        [u, s, v] = truncated_svd(tempTensor,truncation_tolerance=1e-8,full_matrices=False)

        # Automatic rank truncation, can later be replaced with the deltaSVD function
        leftSingularVectors.append(u)
        # singularValues.append(s)

    
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

        

        
