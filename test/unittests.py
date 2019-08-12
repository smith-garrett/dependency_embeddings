# Tested with Python 3.7

# Plan for unit tests:
# 1. Provide super small corpus of dependency triples
# 2. Calculate the whole process "by hand"
# 3. Check those true values against the values output by the implemented algs.

import unittest
import numpy as np
import tempfile
from dependency_embeddings import gen_embeddings

corpus = b"""nsubj(eat, dog)
dobj(eat, kibble)
nsubj(fell, tree)
det(tree, the)
root(ROOT, fell)
root(ROOT, eat)"""

tp = tempfile.NamedTemporaryFile()
tp.write(corpus)
tp.flush()
deps = gen_embeddings.read_standford(tp.name)

totest = gen_embeddings.make_pmi_dict(deps, positive=True)

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
        ppmi[i,j] = np.maximum(0, np.log2(jp[i,j] / (pw[i] * pc[j])))

class TestMe(unittest.TestCase):
    def testme(self):
        assert np.allclose(totest, ppmi), 'True and calculated PPMI matrices not equal'

if __name__ == '__main__':
    unittest.main()
