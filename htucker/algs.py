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


class NotFoundError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
  
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
        self.children = []

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
            self.left.core,self.right.core,self.core,optimize=True
            )
        pass

    def contract_children_dimension_tree(self):
        dimensions=[]
        matrices=[]
        for chld in self.children:
            if type(chld) is ht.TuckerLeaf:
                dimensions.append(2)
            else:
                dimensions.append(len(chld.dims)+1)
            matrices.append(chld.core)
        matrices.append(self.core)
        strings=[]
        core_string=""
        last_char=97
        for dims in dimensions:
            strings.append(''.join([chr(idx) for idx in range(last_char,last_char+dims)]))
            last_char+=dims
            core_string += strings[-1][-1]
        core_string+=chr(last_char)
        result_string=""
        for stri in strings:
            result_string += stri[:-1]
        result_string+=core_string[-1]
        if self.parent is not None:
            pass
        else: #We are contracting the root node
            # We need to adjust the einstein summation strings for root node
            result_string, core_string = result_string[:-1],core_string[:-1]
        self.core = eval(
            "np.einsum("+
            "'"+
            ",".join([
                ','.join(strings+[core_string])+'->'+result_string+"'",
                ",".join([f"matrices[{idx}]" for idx in range(len(matrices))]),
                'optimize=True']
            )+
            ")"
        )

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
    @property
    def shape(self):
        return self.core.shape
        
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

        
    def compress_root2leaf(self, tensor=None, isroot=False): # isroot flag is subject to remove
        # TODO: Replace initial SVD with HOSVD -> Done, Requires testing
        # TODO: Create a structure for the HT -> 
        # TODO: Make compress() function general for n-dimensional tensors -> Done, need to check imbalanced trees (n=5)
        # TODO: Write a reconstruct() function.
        assert(self._iscompressed is False)
        _leaf_counter=0
        _node_counter=0

        #this if check looks unnecessary
        if self.root is None: isroot=True

        if tensor is None:
            raise NotFoundError("No tensor is given. Please check if you provided correct input(s)")
        
        dims=list(tensor.shape)

        # initial split for the tensor
        left,right=split_dimensions(dims)
        
        self.root=TuckerCore(dims=dims)

        self.root.left=TuckerCore(parent=self.root, dims=left)
        self.root.right=TuckerCore(parent=self.root, dims=right)
        # Reshape initial tensor into a matrix for the first splt
        tensor=tensor.reshape(np.prod(left),np.prod(right), order='F')
        
        self.root.core, [self.root.left.core, self.root.right.core] = hosvd(tensor)
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
            node.core = np.einsum('ijk,lk->ijl',node.core,lsv3,optimize=True)
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
                node.right=TuckerLeaf(matrix=lsv2,parent=node, dims=right, idx=_leaf_counter)
                self.leaves[_leaf_counter]=node.right
                _leaf_counter+=1
            else:
                node.right=TuckerCore(core=lsv2, parent=node, dims=right)
                self.nodes2Expand.append(node.right)
        
        self._iscompressed=True
        return None
    
    def compress_leaf2root(self,tensor=None,dimension_tree=None):
        assert(self._iscompressed is False)
        if tensor is None:
            raise NotFoundError("No tensor is given. Please check if you provided correct input(s)")
        if (self._dimension_tree is None) and (dimension_tree is None):
            warn("No dimension tree is given, creating one with binary splitting now...")
            self._dimension_tree = createDimensionTree(tensor,2,1)
        else:
            self._dimension_tree = dimension_tree
        
        ## Start with initial HOSVD to get the initial set of leaves
        existing_leaves=[]
        existing_nodes=[]
        node_idx = self._dimension_tree._nodeCount-1

        _ , leafs = hosvd(tensor)
        for li in self._dimension_tree._level_items[-1]:
            if li._isleaf:
                leaf_idx=li._dimension_index[0]
                self.leaves[leaf_idx]=ht.TuckerLeaf(matrix=leafs[leaf_idx],dims=li.val[0],idx=leaf_idx)
                li._ranks[0]=self.leaves[leaf_idx].shape[-1]
                li.real_node = self.leaves[leaf_idx]
                li.parent._ranks[li.parent._ranks.index(None)]=li.real_node.rank
        for lf in self.leaves:
            if (lf is not None) and (lf not in existing_leaves):
                existing_leaves.append(lf)
                tensor = mode_n_product(tensor,lf.core,[lf.leaf_idx,0])


        ## Niye last layer'i kaydettigini hatirla
        ## Sanirim mevcut layerin bir onceki layerla baglantisini yapmak icin var burada last layer
    
        last_layer=self._dimension_tree._level_items[-1]
        
        for layer in self._dimension_tree._level_items[1:-1][::-1]:
            new_shape=[]
            # burada ilk pass hangi dimensionun hangi dimensionla birlesecegini anlamak icin var
            for item in layer:
                print(item._isleaf,item._ranks,item.val,item._dimension_index)
                if item._isleaf:
                    new_shape.append(item.val[0])
                else:
                    for chld in item.children:
                        item.real_children.append(chld.real_node)
                    new_shape.append(np.prod(list(filter(lambda item: item is not None, item._ranks))))
            # dimension contractionun nasil olacagini planladiktan sonra reshape et ve HOSVD hesapla
            # if len(new_shape)>1:
            tensor=tensor.reshape(new_shape,order="F")
            _ , leafs = hosvd(tensor)
            # else:
                # We are at the first level 
                # leafs = [tensor]
            # HOSVD hesaplandiktan sonra left singular vectorleri ilgili dimensionlarla contract etmek gerekiyor
            for item_idx, item in enumerate(layer):
                # LSV'leri ilgili dimensionlarla contract et
                # if len(layer)>1:
                tensor = mode_n_product(tensor,leafs[item_idx],[item_idx,0])
                # else:
                #     pass
                if item._isleaf:
                    # The current item is a leaf.
                    leaf_idx=item._dimension_index[0]
                    lf=ht.TuckerLeaf(matrix=leafs[item_idx],dims=item.val[0],idx=leaf_idx)
                    self.leaves[leaf_idx]=lf
                    item.real_node=lf
                    item._ranks[0] = lf.rank
                    item.parent._ranks[item.parent._ranks.index(None)] = lf.rank
                    pass
                else:
                    # The current item is a transfer node.
                    # Create tucker core and insert the transfer tensor.
                    item._ranks[item._ranks.index(None)]=leafs[item_idx].shape[-1]
                    item.parent._ranks[item.parent._ranks.index(None)]=leafs[item_idx].shape[-1]
                    # new_shape = 
                    node = ht.TuckerCore(
                        core=leafs[item_idx].reshape(item._ranks,order="F"),
                        dims=item.val.copy(),
                        idx=item._dimension_index.copy()
                        )
                    self.transfer_nodes[node_idx]=node
                    node._isexpanded = True
                    node_idx -=1
                    if len(layer) !=1: node._isroot = False
                    # Create 
                    # item.parent.real_children.append(node)
                    item.real_node = node
                    for chld in item.real_children:
                        node.children.append(chld)
                        chld.parent = node
                    for chld in item.children:
                        chld.real_parent = node
                    # self.transfer_nodes
                    node.get_ranks()
                    # np.prod(list(filter(lambda item: item is not None, sk._ranks)))
                    # learn the children and connect it to the current node. 
                    pass

            last_layer=layer
        layer = self._dimension_tree._level_items[0]
        item = layer[0]
        for chld in item.children:
            item.real_children.append(chld.real_node)
        node = ht.TuckerCore(
            core = tensor,
            dims = item.val.copy(),
            idx = item._dimension_index.copy()
        )
        self.root = node
        node._isexpanded = True
        item.real_node = node
        for chld in item.real_children:
            node.children.append(chld)
            chld.parent = node
        for chld in item.children:
            chld.real_parent = node
        node.get_ranks()
        self._iscompressed=True
        return None

    def reconstruct(self):
        # The strategy is to start from the last core and work the way up to the root.
        assert(self._iscompressed)
        _transfer_nodes=self.transfer_nodes.copy()
        if self._dimension_tree is None:
            while _transfer_nodes:
                node=_transfer_nodes.pop(-1)
                node.contract_children()
            self.root.contract_children()
        else:
            while _transfer_nodes:
                node=_transfer_nodes.pop(-1)
                node.contract_children_dimension_tree()
            self.root.contract_children_dimension_tree()

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
        core_l = np.einsum('ijk,lk->ijl', core_l, lsv_l[-1],optimize=True)
        core_r = np.einsum('ijk,li->ljk', core_r, lsv_r[0],optimize=True)
        # print("\n",core_l)
        # print("\n",core_r)
        top = np.diag(s)
        return (lsv_l[0], lsv_l[1], lsv_r[1], lsv_r[2], core_l, core_r, top)

    @property
    def compression_ratio(self):
        num_entries=np.prod(self.root.shape)
        for tf in self.transfer_nodes:
            num_entries+=np.prod(tf.shape)
        for lf in self.leaves:
            num_entries+=np.prod(lf.shape)
        return np.prod(self.original_shape)/num_entries

        
        
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
        return np.diag(s), [u, v.T]

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

def mode_n_product(tensor:np.ndarray, matrix:np.ndarray, modes:list or tuple):
    dims=[idx for idx in range(len(tensor.shape)+len(matrix.shape)-2)]
    tensor_ax, matrix_ax = modes
    dims.pop(tensor_ax)
    dims.append(tensor_ax)
    tensor=np.tensordot(tensor,matrix,axes=modes)
    tensor=tensor.transpose(np.argsort(dims).tolist())
    return tensor

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


class Node:
    def __init__(self, val, children=None, parent=None) -> None:
        self.children = children or []
        self.val = val
        self.parent = parent
        self.real_children = []
        self.real_parent = []
        self.real_node = None
        self._ranks = []
        self._propagated = False
        self._isleaf = False
        self._level = None
        self._dimension_index = None

    def __str__(self) -> str:
        return self.children
    
    def adjust_ranks(self):
        if self.parent is None:
            #This is the root node
            if len(self._ranks)<len(self.children):
                diff =len(self.children)-len(self._ranks)
                self._ranks += [None]*diff
        else:
            # This is any node (incl. leaves)
            if len(self._ranks)<len(self.children)+1:
                diff =len(self.children)-len(self._ranks)+1
                self._ranks += [None]*diff


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

        

        
