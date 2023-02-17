import unittest
import numpy as np

import htucker as ht

seed = 2
np.random.seed(seed)
        
class TestCase(unittest.TestCase):

    def setUp(self):

        num_dim = 4
        self.size = [3, 4, 7, 5]

        leaf_ranks = [3, 2, 5, 4]
        
        leafs = [np.random.randn(r, n) for r,n in zip(leaf_ranks, self.size)]

        transfer_ranks = [3, 6]

        transfer_tensors = [
            np.random.randn(leaf_ranks[0], leaf_ranks[1], transfer_ranks[0]),
            np.random.randn(leaf_ranks[2], leaf_ranks[3], transfer_ranks[1])
        ]

        root = np.random.randn(transfer_ranks[0], transfer_ranks[1])
        
        eval_left = np.einsum('ij,kl,ikr->jlr', leafs[0], leafs[1], transfer_tensors[0])
        eval_right = np.einsum('ij,kl,ikr->jlr', leafs[2], leafs[3], transfer_tensors[1])

        self.tensor = np.einsum('ijk,lmn,kn->ijlm',eval_left, eval_right, root)
        self.koldaTensor=np.array(
            [
                [
                    [1,13],
                    [4,16],
                    [7,19],
                    [10,22]
                    ],
                [
                    [2,14],
                    [5,17],
                    [8,20],
                    [11,23]
                    ],
                [
                    [3,15],
                    [6,18],
                    [9,21],
                    [12,24]
                    ]
                ]
                
        )
        
    @unittest.skip("add_edge")
    def test_add_edge(self):

        # print("\n", self.tensor)
        self.assertEqual(self.size[0], self.tensor.shape[0])
        self.assertEqual(self.size[1], self.tensor.shape[1])
        self.assertEqual(self.size[2], self.tensor.shape[2])
        self.assertEqual(self.size[3], self.tensor.shape[3])

    # @unittest.skip("mode-n unfolding")        
    def test_kolda_unfolding(self):
        mode0=ht.mode_n_unfolding(self.koldaTensor,0)
        mode1=ht.mode_n_unfolding(self.koldaTensor,1)
        mode2=ht.mode_n_unfolding(self.koldaTensor,2)

        # TODO: Write assert statements for dimensions!

        mode0kolda=np.array([
            [1,4,7,10,13,16,19,22],
            [2,5,8,11,14,17,20,23],
            [3,6,9,12,15,18,21,24]
            ])
        mode1kolda=np.array([
            [1,2,3,13,14,15],
            [4,5,6,16,17,18],
            [7,8,9,19,20,21],
            [10,11,12,22,23,24]
        ])
        mode2kolda=np.array([
            [1,2,3,4,5,6,7,8,9,10,11,12],
            [13,14,15,16,17,18,19,20,21,22,23,24]
        ])
        self.assertTrue(np.allclose((mode0-mode0kolda),np.zeros_like(mode0)))
        self.assertTrue(np.allclose((mode1-mode1kolda),np.zeros_like(mode1)))
        self.assertTrue(np.allclose((mode2-mode2kolda),np.zeros_like(mode2)))
    
    def test_hosvd(self):
        core,matrices=ht.hosvd(self.tensor)
        reconstruction=np.einsum('ij,kl,mn,op,jlnp->ikmo',matrices[0],matrices[1],matrices[2],matrices[3],core)
        self.assertTrue(np.allclose((reconstruction-self.tensor),np.zeros_like(reconstruction)))

    def test_htucker_sanity_check_4d(self):
        tens=ht.HTucker()
        (leaf1, leaf2, leaf3, leaf4, nodel, noder, top) = tens.compress_sanity_check(self.tensor)
        # print('\n')
        # print(leaf1.shape,leaf2.shape,leaf3.shape,leaf4.shape)
        # print(nodel.shape,noder.shape)
        # print(top.shape)
        eval_left = np.einsum('ji,lk,ikr->jlr', leaf1, leaf2, nodel)
        eval_right = np.einsum('ij,kl,rjl->rik', leaf3, leaf4, noder)
        # print(eval_left.shape,eval_right.shape)
        tensor = np.einsum('ijk,lmn,kl->ijmn',eval_left, eval_right, top)
        # print(tensor-self.tensor)
        self.assertTrue(np.allclose((tensor-self.tensor),np.zeros_like(tensor)))

    def test_htucker_4d(self):
        tens=ht.HTucker()
        (leaf1, leaf2, leaf3, leaf4, nodel, noder, top) = tens.compress_sanity_check(self.tensor)
        tens.compress(self.tensor)

        self.assertEqual(self.size[0], tens.leaves[0].core.shape[0])
        self.assertEqual(self.size[1], tens.leaves[1].core.shape[0])
        self.assertEqual(self.size[2], tens.leaves[2].core.shape[0])
        self.assertEqual(self.size[3], tens.leaves[3].core.shape[0])

        # Check rank consistency between left leaves and left core
        self.assertEqual(tens.leaves[0].core.shape[1], tens.transfer_nodes[0].core.shape[0])
        self.assertEqual(tens.leaves[1].core.shape[1], tens.transfer_nodes[0].core.shape[1])

        # Check rank consistency between right leaves and right core
        self.assertEqual(tens.leaves[2].core.shape[1], tens.transfer_nodes[1].core.shape[0])
        self.assertEqual(tens.leaves[3].core.shape[1], tens.transfer_nodes[1].core.shape[1])

        # Check if the leaves are same for 4d case
        self.assertTrue(np.allclose((leaf1-tens.leaves[0].core), np.zeros_like(leaf1)))
        self.assertTrue(np.allclose((leaf2-tens.leaves[1].core), np.zeros_like(leaf2)))
        self.assertTrue(np.allclose((leaf3-tens.leaves[2].core), np.zeros_like(leaf3)))
        self.assertTrue(np.allclose((leaf4-tens.leaves[3].core), np.zeros_like(leaf4)))

        # Check if the transfer cores are same for 4d case
        # Note that we need to swap axes for the hardcoded version since we always
        # keep the tucker rank at the last index
        self.assertTrue(np.allclose((nodel-tens.transfer_nodes[0].core), np.zeros_like(nodel)))
        self.assertTrue(np.allclose((noder.transpose(1,2,0)-tens.transfer_nodes[1].core), np.zeros_like(noder.transpose(1,2,0))))

        # self.assertEqual(leaf3.shape[1], noder.shape[1])
        # self.assertEqual(leaf4.shape[1], noder.shape[2])        
        
        eval_left = np.einsum('ji,lk,ikr->jlr', tens.leaves[0].core, tens.leaves[1].core, tens.transfer_nodes[0].core)
        eval_right = np.einsum('ij,kl,jlm->ikm',tens.leaves[2].core, tens.leaves[3].core, tens.transfer_nodes[1].core)


        # print("eval_left.shape = ", eval_left.shape)
        # print("eval_right.shape = ", eval_right.shape)
        # print("top shape = ", top.shape)
        
        tensor = np.einsum('ijk,lmn,kn->ijlm',eval_left, eval_right, tens.root.core)
        
        # Check if we get the same shape as the original tensor
        self.assertEqual(self.size[0], tensor.shape[0])
        self.assertEqual(self.size[1], tensor.shape[1])
        self.assertEqual(self.size[2], tensor.shape[2])
        self.assertEqual(self.size[3], tensor.shape[3])
        
        # Check if we get the same tensor as the original tensor
        self.assertTrue(np.allclose((tensor-self.tensor),np.zeros_like(tensor)))
    
    def test_reconstruct_4d(self):
        tens=ht.HTucker()
        tens.compress(self.tensor)
        tens.reconstruct()
        np.allclose((tens.root.core-self.tensor),np.zeros_like(self.tensor))

    # TODO: Write test for n-mode unfolding -> Done
    # TODO: Write test for n-dimensional tucker  
        
if __name__ == '__main__':
    unittest.main()
