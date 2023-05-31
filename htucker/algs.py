"""Hierarchical Tucker."""


import numpy as np
import htucker as ht

from math import ceil


__all__ = [
    "HTucker",
    "hosvd",
    "truncated_svd",
    "create_permutations",
    "split_dimensions",
    "mode_n_unfolding"
]

  
class TuckerCore:
    # Object for tucker cores. Planning to use it to create the recursive graph structure
    def __init__(self,core=None, parent=None, dims=None, idx=None) -> None:
        self.parent=parent
        self.core=core
        self.left=None
        self.right=None
        self.dims=dims
        self.core_idx=idx
        self.ranks=[]

        if parent is None:
            self._isroot=True
        else:
            self._isroot=False

        self._isexpanded=False

    def get_ranks(self):
        if self.core is not None:
            self.ranks=list(self.core.shape)
    
    def contract_children(self):
        _ldims=len(self.left.dims)+1
        _rdims=len(self.right.dims)+1
        
        left_str = ''.join([chr(idx) for idx in range(97,97+_ldims)])
        right_str = ''.join([chr(idx) for idx in range(97+_ldims,97+_ldims+_rdims)])
        if self._isroot:
            core_str = left_str[-1]+right_str[-1]
        else:    
            core_str = left_str[-1]+right_str[-1]+chr(97+_ldims+_rdims)

        result_str = core_str.replace(left_str[-1],left_str[:-1])
        result_str = result_str.replace(right_str[-1],right_str[:-1])
        self.core = np.einsum(
            ','.join([left_str,right_str,core_str])+'->'+result_str,
            self.left.core,self.right.core,self.core
            )
        pass

    @property
    def shape(self):
        return self.core.shape
class TuckerLeaf:
    def __init__(self, matrix=None, parent=None, dims=None, idx=None) -> None:
        self.parent=parent
        self.core=matrix #Refactoring this as core for consistency with TuckerCore
        self.dims=dims
        self.leaf_idx=idx
        if matrix is not None: self.rank=matrix.shape[1]
        
class HTucker:

    # harded for 4d first
    def __init__(self):
        self._leaf_count=4
        self.leaves = [None]*self._leaf_count
        self.transfer_nodes = [None]*(self._leaf_count-2)
        self.root = None
        self.nodes2Expand=[]
        self._iscompressed=False

    def initialize(self,tensor):
        self.original_shape = list(tensor.shape)
        self._leaf_count=len(self.original_shape)
        self.leaves = [None]*self._leaf_count
        self.transfer_nodes = [None]*(self._leaf_count-2) #Root node not included here
        self.root = None
        self.nodes2Expand = []
        self._iscompressed = False

        
    def compress(self, tensor, isroot=False):
        # TODO: Replace initial SVD with HOSVD -> Done, Requires testing
        # TODO: Create a structure for the HT -> 
        # TODO: Make compress() function general for n-dimensional tensors -> Done, need to check imbalanced trees (n=5)
        # TODO: Write a reconstruct() function.
        assert(self._iscompressed is False)
        _leaf_counter=0
        _node_counter=0

        #this if check looks unnecessary
        if self.root is None: isroot=True

        
        dims=list(tensor.shape)

        # initial split for the tensor
        left,right=split_dimensions(dims)
        
        self.root=TuckerCore(dims=dims)

        self.root.left=TuckerCore(parent=self.root, dims=left)
        self.root.right=TuckerCore(parent=self.root, dims=right)
        # Reshape initial tensor into a matrix for the first splt
        tensor=tensor.reshape(np.prod(left),np.prod(right), order='F')
        
        self.root.core, self.root.left.core, self.root.right.core = hosvd(tensor)
        # The reshapings below might be unnecessary, will look into those
        # self.root.left.core = self.root.left.core.reshape(left+[-1],order='F')
        # self.root.right.core = self.root.right.core.reshape(right+[-1],order='F')
        self.root.get_ranks()

        self.nodes2Expand.append(self.root.left)
        self.nodes2Expand.append(self.root.right)

        while self.nodes2Expand:

            node=self.nodes2Expand.pop(0)
            # if len(node.dims)==1:
            #     print(node.dims)
            #     continue
            left,right=split_dimensions(node.dims)



            node.core = node.core.reshape((np.prod(left),np.prod(right),-1), order='F')
            node.core, [lsv1, lsv2, lsv3] = hosvd(node.core)
            # Contract the third leaf with the tucker core for now
            node.core = np.einsum('ijk,lk->ijl',node.core,lsv3)
            node._isexpanded = True
            node.core_idx = _node_counter
            self.transfer_nodes[_node_counter] = node
            _node_counter += 1
            
            if len(left)==1:
                # i.e we have a leaf
                node.left=TuckerLeaf(matrix=lsv1,parent=node, dims=left, idx=_leaf_counter)
                self.leaves[_leaf_counter]=node.left
                _leaf_counter+=1
            else:
                node.left=TuckerCore(core=lsv1, parent=node, dims=left)
                self.nodes2Expand.append(node.left)

            if len(right)==1:
                node.right=TuckerLeaf(matrix=lsv2,parent=node, dims=left, idx=_leaf_counter)
                self.leaves[_leaf_counter]=node.right
                _leaf_counter+=1
            else:
                node.right=TuckerCore(parent=node, dims=right)
        
        self._iscompressed=True
        return None
    
    def reconstruct(self):
        # The strategy is to start from the last core and work the way up to the root.
        assert(self._iscompressed)
        while self.transfer_nodes:
            node=self.transfer_nodes.pop(-1)
            node.contract_children()
            
        self.root.contract_children()

        self._iscompressed=False
        return None

    def compress_sanity_check(self,tensor):
        # Commenting out below for now, might be needed later for checking

        mat_tensor = np.reshape(tensor, (tensor.shape[0]*tensor.shape[1],
                                         tensor.shape[2]*tensor.shape[3]), order='F')

        [u, s, v] = truncated_svd(mat_tensor, 1e-8, full_matrices=False)
        [core_l, lsv_l] = hosvd(u.reshape(tensor.shape[0],tensor.shape[1],-1, order='F'))
        [core_r, lsv_r] = hosvd(v.reshape(-1, tensor.shape[2],tensor.shape[3], order='F'))

        # need an HOSVD tucker of u (and v) this (look at kolda paper)
        # print("\n",core_l)
        # print("\n",lsv_l[-1])
        core_l = np.einsum('ijk,lk->ijl', core_l, lsv_l[-1])
        core_r = np.einsum('ijk,li->ljk', core_r, lsv_r[0])
        # print("\n",core_l)
        # print("\n",core_r)
        top = np.diag(s)
        return (lsv_l[0], lsv_l[1], lsv_r[1], lsv_r[2], core_l, core_r, top)


        
        
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
        
def hosvd(tensor):
    ndims=len(tensor.shape)

    if ndims == 2:
        [u, s, v] = truncated_svd(tensor, truncation_tolerance=1e-8, full_matrices=False)
        return np.diag(s), u, v.T

    permutations=create_permutations(ndims)

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

def create_permutations(nDims):
    # Creates permutations to compute the matricizations
    permutations=[]
    dimensionList=list(range(nDims))
    for dim in dimensionList:
        copyDimensions=dimensionList.copy()
        firstDim=copyDimensions.pop(dim)
        permutations.append([firstDim]+copyDimensions)
    return permutations

def split_dimensions(dims):
        n_dims=len(dims)
        return dims[:ceil(n_dims/2)],dims[ceil(n_dims/2):]

def mode_n_unfolding(tensor,mode):
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


class Tree:
    def __init__(self) -> None:
        self.root = None
        self._depth = 0
        self._size = 0
        self._leafCount = 0
        self._nodeCount = 0
        self._leaves = []
        self._level_items = None 

    def findNode(self, node, key):
        if (node is None) or (node.val == key):
            return node
        for child in node.children:
            return_node = self.findNode(child, key)
            if return_node:
                return return_node
        return None

    def isEmpty(self):
        return self._size == 0

    def initializeTree(self, vals):
        # Initalizes the tree
        if self.root is None:
            if type(vals) is list:
                self.root = vals
            else:
                raise TypeError(f"Type: {type(vals)} is not known!")
        else:
            warn("Root node already implemented! Doing nothing.")

    def insertNode(self, val, parent=None, dim_index=None):
        newNode = Node(val)
        newNode._dimension_index=dim_index
        if parent is None: # No parent is given, i.e. Root node
            self.root = newNode
            self._depth = 1
            self._size = 1
            newNode._level = 0
            newNode.adjust_ranks()
        elif type(parent) is Node: # Parent is given directly as a node object
            parent.children.append(newNode)
            parent._propagated = True
            # parent._ranks+=[None]
            self._size += 1
            newNode.parent = parent
            newNode._level = parent._level+1
            parent.adjust_ranks()
        else: # Key/dimensions of the parent is given as input
            parentNode = self.findNode(self.root, parent)
            if not (parentNode):
                raise NotFoundError(f"No parent was found for parent name: {parent}")
            parentNode.children.append(newNode)
            parentNode._propagated = True
            # parentNode._ranks+=[None]
            self._size += 1
            newNode.parent = parentNode
            newNode._level = parentNode._level+1
            parentNode.adjust_ranks()
        if len(val)==1:
            newNode._isleaf = True
            newNode.adjust_ranks()
            self._leaves.append(newNode)
            self._leafCount+=1

    def get_max_depth(self):
        self._depth = 0
        for leaf in self._leaves:
            depth=0
            node = leaf
            while node.parent is not None:
                depth += 1
                node = node.parent
            if depth > self._depth:
                self._depth = depth
        return None
                
    def get_items_from_level(self):
        self._level_items=[]
        for _ in range(self._depth+1):
            self._level_items.append([])
        # for depth,items in enumerate(level_items):
        nodes2expand=[self.root]
        while nodes2expand:
            node = nodes2expand.pop(0)
            nodes2expand.extend(node.children)
            self._level_items[node._level].append(node)

    def toList(self):
        # Returns a list from the tree
        return None


def createDimensionTree(inp, numSplits, minSplitSize):
    if type(inp) is np.ndarray:
        dims = np.array(inp.shape)
    elif type(inp) is tuple or list:
        dims = np.array(inp)  # NOQA
    else:
        raise TypeError(f"Type: {type(inp)} is unsupported!!")
    dimensionTree = Tree()
    dimensionTree.insertNode(dims.tolist())
    print(np.array(dimensionTree.root.val))
    dimensionTree.root._dimension_index = [idx for idx,_ in enumerate(dimensionTree.root.val)]
    nodes2expand = []
    nodes2expand.append(dimensionTree.root.val.copy())
    while nodes2expand:
        # print(leaves)
        node2expand = nodes2expand.pop(0)
        node = dimensionTree.findNode(dimensionTree.root, node2expand)
        dim_split=np.array_split(np.array(node.val), numSplits)
        idx_split=np.array_split(np.array(node._dimension_index), numSplits)
        if (not node._propagated) and (len(node.val) > minSplitSize + 1):
            # for split in [data[x:x+10] for x in xrange(0, len(data), 10)]:
            for dims,indices in zip(dim_split,idx_split): # place zip here
                print(dims)
                # tree.insertNode(split,node.val)
                # leaves.append(split)
                dimensionTree.insertNode(dims.tolist(), node.val,dim_index=indices.tolist())
                nodes2expand.append(dims.tolist())
        elif (not node._propagated) and (len(node.val) > minSplitSize):
            # i.e. the node is a leaf
            print(node.val)
            for dims,indices in zip(dim_split,idx_split): # place zip here
                dimensionTree.insertNode(dims.tolist(), node.val, dim_index=indices.tolist())
    dimensionTree.get_max_depth()
    dimensionTree._nodeCount = dimensionTree._size-dimensionTree._leafCount-1 #last -1 is to subtract root node
    return dimensionTree


#     return None

        

        
