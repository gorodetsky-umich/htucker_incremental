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

        # print("\n",self.koldaTensor.shape)
        # # This is the way that one should compute the mode-n reshaping in the sense of Kolda!
        # print(self.koldaTensor.reshape(3,8,order='F'))
        # print(self.koldaTensor.transpose(1,0,2).reshape(4,6,order='F'))
        # print(self.koldaTensor.transpose(2,0,1).reshape(2,12,order='F'))
        
        # assert all false
        # self.assertTrue(True)
    
    def test_hosvd(self):
        core,matrices=ht.algs.hosvd(self.tensor)
        reconstruction=np.einsum('ij,kl,mn,op,jlnp->ikmo',matrices[0],matrices[1],matrices[2],matrices[3],core)
        self.assertTrue(np.allclose((reconstruction-self.tensor),np.zeros_like(reconstruction)))

    def test_htucker(self):
        tens=ht.HTucker()
        (leaf1, leaf2, leaf3, leaf4, nodel, noder, top) = tens.compress_sanity_check(self.tensor)
        tens.compress(self.tensor)

        self.assertEqual(self.size[0], tens.leaves[0].matrix.shape[0])
        self.assertEqual(self.size[1], tens.leaves[1].matrix.shape[0])
        self.assertEqual(self.size[2], tens.leaves[2].matrix.shape[0])
        self.assertEqual(self.size[3], tens.leaves[3].matrix.shape[0])

        # Check rank consistency between left leaves and left core
        self.assertEqual(tens.leaves[0].matrix.shape[1], tens.transfer_nodes[0].core.shape[0])
        self.assertEqual(tens.leaves[1].matrix.shape[1], tens.transfer_nodes[0].core.shape[1])

        # Check rank consistency between right leaves and right core
        self.assertEqual(tens.leaves[2].matrix.shape[1], tens.transfer_nodes[1].core.shape[0])
        self.assertEqual(tens.leaves[3].matrix.shape[1], tens.transfer_nodes[1].core.shape[1])


        # print("nodel.shape = ", nodel.shape)
        # print("noder.shape = ", noder.shape)
        # print("leaf3.shape = ", leaf3.shape)
        # print("leaf4.shape = ", leaf4.shape)
        
        
        # self.assertEqual(leaf3.shape[1], noder.shape[1])
        # self.assertEqual(leaf4.shape[1], noder.shape[2])        
        
        eval_left = np.einsum('ji,lk,ikr->jlr', tens.leaves[0].matrix, tens.leaves[1].matrix, tens.transfer_nodes[0].core)
        eval_right = np.einsum('ij,kl,jlm->ikm',tens.leaves[2].matrix, tens.leaves[3].matrix, tens.transfer_nodes[1].core)

        # eval_left = np.einsum('ji,lk,ikr->jlr', leaf1, leaf2, nodel)
        # eval_right = np.einsum('ij,kl,rjl->rik', leaf3, leaf4, noder)

        # print("eval_left.shape = ", eval_left.shape)
        # print("eval_right.shape = ", eval_right.shape)
        # print("top shape = ", top.shape)
        
        tensor = np.einsum('ijk,lmn,kn->ijlm',eval_left, eval_right, tens.root.core)
        
        self.assertEqual(self.size[0], tensor.shape[0])
        self.assertEqual(self.size[1], tensor.shape[1])
        self.assertEqual(self.size[2], tensor.shape[2])
        self.assertEqual(self.size[3], tensor.shape[3])
        self.assertTrue(np.allclose((leaf1-tens.leaves[0].matrix), np.zeros_like(leaf1)))
        self.assertTrue(np.allclose((leaf2-tens.leaves[1].matrix), np.zeros_like(leaf2)))
        self.assertTrue(np.allclose((leaf3-tens.leaves[2].matrix), np.zeros_like(leaf3)))
        self.assertTrue(np.allclose((leaf4-tens.leaves[3].matrix), np.zeros_like(leaf4)))
        
        self.assertTrue(np.allclose((tensor-self.tensor),np.zeros_like(tensor)))
        
    # TODO: Write test for n-dimensional tucker  
    # TODO: Write test for n-mode unfolding
        
if __name__ == '__main__':
    unittest.main()
