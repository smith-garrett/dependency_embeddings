# Command line script for taking Stanford parses and outputting a text file
# containing PPMI word embeddings. These are "head" feature representations in
# the sense of Smith & Tabor (2018; ICCM), i.e., the features used for a word
# when it attaches as the dependent of another word. In addition to the word
# embeddings, the script also returns the average word embedding of the words
# that appear as the dependent of another words, i.e., features that a word is
# "looking for" in its dependent(s).

# Tested w/ Python 3.7


import re
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from warnings import warn


def read_standford(file):
    with open(file, 'r') as f:
        deps = []
        for line in f:
            line = line.lower()
            line = re.sub('[0-9]+', '', line)
            curr = re.findall("[\w]+", line)
            if len(curr) == 3:
                deps.append(curr)
    return deps


def calc_pmi(wfreq, dfreq, jfreq, size):
    """This implementation gets around DivideByZero errors and sets
    -inf to zero.
    """
    with np.errstate(divide='ignore'):
        pmi = np.log2((jfreq * size) / (wfreq * dfreq))
        if pmi != -np.inf:
            return pmi
        else:
            return 0.0


def calc_ppmi(wfreq, dfreq, jfreq, size):
    """Only keeps the positive PMI.
    """
    return np.maximum(0, np.log2((jfreq * size) / (wfreq * dfreq)))


def make_pmi_dict(deps, positive=False):
    """Does the bulk of the work. Counts words, dependency types, and their
    joint occurences, and returns a dictionary of (P)PMI for each pair.
    The positive argument is set to False by default, following rei2014looking.
    """
    wfreqs = Counter([item[-1] for item in deps])
    dfreqs = Counter([item[0] for item in deps])
    jfreqs = Counter([(item[0], item[-1]) for item in deps])
    size = len(deps)
    pmidict = Counter()
    for d, w in jfreqs.keys():
        if positive:
            pmidict[w, d] = calc_ppmi(wfreqs[w], dfreqs[d], jfreqs[d, w], size)
        else:
            pmidict[w, d] = calc_pmi(wfreqs[w], dfreqs[d], jfreqs[d, w], size)
    return pmidict


def to_sparse_df(pmidict):
    """Converts from dict to sparse pandas DataFrame.
    """
    return pd.Series(pmidict).unstack(fill_value=0.0).to_sparse()


def svd_reduce(spdf, k=None):
    """Performs dimensionality reduction using singular value decomposition of
    the sparse DataFrame -> U*Sigma*V. Returns U*Sigma as the rank-k word
    representations (levy2015improving).
    """
    U, S, _ = svds(spdf, k=k)
    mat = U.dot(np.diag(S))
    mat = pd.SparseDataFrame(mat)
    mat.index = spdf.index.values
    #mat.columns = spdf.columns.values
    return mat


def calc_dep_feats(deps, df):
    """For each word, need a feature vector for each of its dependents. The
    feature vector should be the average of the head vectors of all the words
    that appeared as that dependent, weighted by how often each word was that
    dependent.
    Actually, I don't know if weighting makes sense, or if it's already in the
    (P)PMI's...
    IDEA: pass in a list of words and the desired dependents you want to
    consider. For the Cunnings & Sturt data, this will be more than fine, and
    much faster.
    """
    return


def feats_by_dep(deps, df):
    """Creates retrieval cues/dependent features by dependency type. Thes are
    *not* specific to lexical items. Instead, we just get the average over all
    head features of words that appear as a given dependency type.
    """
    dtypes = list(set([d[0] for d in deps]))
    if isinstance(df.columns[0], str):
        colnames = df.columns
    else:
        #colnames = ['f'+ str(i) for i in range(len(df.columns))]
        colnames = list(range(len(df.columns)))
    feats = pd.DataFrame(0, index=dtypes, columns=colnames)
    for i, dep in enumerate(dtypes):
        if i % (len(dtypes) // 10) == 0:
            print('{}%\r'.format(np.round(i/len(dtypes) * 100)), end='')
        words = list(set([w[-1] for w in deps if w[0] == dep]))
        for word in words:
            feats.loc[dep] += df.loc[word].values / len(words)
    print('', end='')
    print('Done.')
    return feats


def calc_similarity(words, attch):
    """Calculate the cosine similarity between words and attachment sites.
    """
    #if all(words.columns == attch.columns):
    #    return cosine_similarity(words.append(attch))
    #else:
    #    warn('Attachment site column names don\'t match word column names. Assuming word column names are the correct ones.')
    #    attch.columns = words.columns
    #return cosine_similarity(words.append(attch
    nattch = attch.shape[0]
    nwords = words.shape[0]
    sims = cosine_similarity(words.append(attch))[0:nwords, -nattch:]
    return pd.DataFrame(sims, columns=attch.index.values, index=words.index)


if __name__ == '__main__':
    # Gives NaN error in cosine_similarity
    #file = '/Users/garrettsmith/Google Drive/UniPotsdam/Research/Features/naturalstories/parses/stanford/all-parses-aligned.txt.stanford'
    # Doesn't:
    #file = '/Users/garrettsmith/Google Drive/UniPotsdam/Research/Features/GenEmbeddings/ParsedBrownCorpus/parsedbrown10.txt'
    file = '/Users/garrettsmith/Google Drive/UniPotsdam/Research/Features/GenEmbeddings/ParsedBrownCorpus/parsedbrown0.txt'
    deps = read_standford(file)
    pmi_dict = make_pmi_dict(deps, positive=False)
    sparsedf = to_sparse_df(pmi_dict)
    red = svd_reduce(sparsedf, 20)
    dfeats = feats_by_dep(deps, sparsedf)
    dfeatsred = feats_by_dep(deps, red)
    print(calc_similarity(red.loc[['he', 'she', 'cat', 'dog',]],
                          dfeatsred.loc[['nsubj', 'obj']]))
    print(calc_similarity(sparsedf.loc[['he', 'she', 'cat', 'dog']],
                          dfeats.loc[['nsubj', 'obj']]))
