# Tested with Python 3.7
# Run tests on command line with:
# python -m unittest test/unittests.py -v

import unittest
import numpy as np
import tempfile
from dependency_embeddings import gen_embeddings


class TestMe(unittest.TestCase):
    def setUp(self):
        # Parsed corpus for testing (in byte format)
        corpus = b"""nsubj(eat, dog)
        dobj(eat, kibble)
        nsubj(fell, tree)
        det(tree, the)
        root(ROOT, fell)
        root(ROOT, eat)"""

        # Putting it in a temporary file to pass to read_standford
        tp = tempfile.NamedTemporaryFile()
        tp.write(corpus)
        tp.flush()

        # The module
        self.deps = gen_embeddings.read_standford(tp.name)
        self.ppmi = gen_embeddings.make_pmi_dict(self.deps, positive=True)
        self.retr_cues = gen_embeddings.lex_spec_feats([('eat', 'nsubj')],
                                                       self.deps, self.ppmi)
        self.sim = gen_embeddings.calc_similarity(['dog', 'tree'],
                                                  ['eat-nsubj'],
                                                  self.ppmi, self.retr_cues)

        # By hand:
        # Co-occurrence matrix:
        # Columns: det, nsubj, obj, root
        cooc = np.array([[0, 1, 0, 0],  # dog
                         [0, 0, 0, 1],  # eat
                         [0, 0, 0 ,1],  # fell
                         [0, 0, 1, 0],  # kibble
                         [1, 0, 0, 0],  # the
                         [0, 1, 0, 0]])  # tree

        # Unigram word probabilities
        pw = cooc.sum(axis=1) / cooc.sum()
        # Unigram context probabilities
        pc = cooc.sum(axis=0)**0.75 / (cooc.sum(axis=0)**0.75).sum()
        # Joing probs.
        jp = cooc / cooc.sum()
        # ppmi
        ppmi = np.zeros(cooc.shape)
        for i in range(ppmi.shape[0]):
            for j in range(ppmi.shape[1]):
                with np.errstate(divide='ignore'):
                    ppmi[i,j] = np.maximum(0,
                                           np.log2(jp[i,j] / (pw[i] * pc[j])))
        self.ppmi_manual = ppmi


    def test_ppmi(self):
        self.assertTrue(np.allclose(self.ppmi, self.ppmi_manual))#, 'True and calculated PPMI matrices not equal'


    def test_retrieval_cues(self):
        # Retrieval cues of nsubj-eat is just ppmi vec. for dog
        self.assertTrue(np.array_equal(self.ppmi_manual[0,:],
                                       self.retr_cues.to_numpy().squeeze()))


    def test_similarity(self):
        simdog = (self.ppmi_manual[0,:].dot(self.ppmi_manual[0,:])
                  / (np.linalg.norm(self.ppmi_manual[0,:])
                     * np.linalg.norm(self.ppmi_manual[0,:])))
        simtree = (self.ppmi_manual[-1,:].dot(self.ppmi_manual[0,:])
                   / (np.linalg.norm(self.ppmi_manual[-1,:])
                      * np.linalg.norm(self.ppmi_manual[0,:])))
        self.assertEqual(simdog, self.sim.loc['dog'].to_numpy())
        self.assertEqual(simtree, self.sim.loc['tree'].to_numpy())


if __name__ == '__main__':
    unittest.main()
