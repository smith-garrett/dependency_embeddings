"""Command line Python module for taking Stanford parses and outputting a text
file containing (P)PMI word embeddings. These are "head" feature representations in the sense of Smith & Tabor (2018; ICCM), i.e., the features used for a word when it attaches as the dependent of another word. In addition to the word embeddings, the script also returns the average word embedding of the words that appear as the dependent of another words, i.e., features that a word is looking for" in its dependent(s).
"""

# Tested w/ Python 3.7


import re
from collections import Counter, OrderedDict
from itertools import product
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import os
import fileinput
import gzip


def read_standford(files, outpath=None):
    """Takes a (list of) file(s) and returns a nicely formatted set of
    dependent-governor-dependent triples.
    """
    with fileinput.input(files=files) as f:
        deps = []
        for line in f:
            line = line.lower()
            line = re.sub('[0-9]+', '', line)
            curr = re.findall("[\w]+", line)
            # Sometimes the parser was returning dependency n-tuples, but we
            # only want triples for simplicity
            if len(curr) == 3:
                deps.append(curr)
    return deps


def calc_pmi(wfreq, dfreq, jfreq, size, sized):
    """This implementation gets around DivideByZero errors and sets
    -inf to zero. Implements smoothing recommendation of Levy et al. (2015).
    """
    alpha = 0.75
    with np.errstate(divide='ignore'):
        #pmi = np.log2((jfreq * size) / (wfreq * dfreq))
        pj = jfreq / size
        pw = wfreq / size
        pd = dfreq**alpha / sized**alpha
        pmi = np.log2(pj / (pw * pd))
        if pmi != -np.inf:
            return pmi
        else:
            return 0.0


def calc_ppmi(wfreq, dfreq, jfreq, size, sized):
    """Only keeps the positive PMI. Implements smoothing recommendation of
    Levy et al. (2015).
    """
    alpha=0.75
    pj = jfreq / size
    pw = wfreq / size
    pd = dfreq**alpha / sized**alpha
    #return np.maximum(0, np.log2((jfreq * size) / (wfreq * dfreq)))
    return np.maximum(0, np.log2(pj / (pw * pd)))


def make_pmi_dict(deps, positive=True, outpath=None):
    """Based on Jurafsky & Martin (2019), Ch. 6. Includes context smoothing.
    """
    alpha = 0.75  # context smoothing exponent
    df = pd.DataFrame(deps, columns=['DepType', 'Head', 'Dep'])
    ctmat = pd.crosstab(df.Dep, df.DepType)
    pjoint = ctmat / np.sum(ctmat.values)
    pword = ctmat.sum(axis=1) / np.sum(ctmat.values)
    pdeptype = ctmat.sum(axis=0)**alpha / np.sum(ctmat.values**alpha)
    pmimat = ctmat.copy()
    for w in pmimat.index:
        for d in pmimat.columns:
            with np.errstate(divide='ignore'):
                pmi = np.log2(pjoint.loc[w, d] / (pword.loc[w] *
                                                  pdeptype.loc[d]))
            if positive:
                pmimat.loc[w, d] = np.maximum(0, pmi)
            else:
                if pmi != -np.inf:
                    pmimat.loc[w, d] = pmi
                else:
                    pmimat.loc[w, d] = 0.0
    if outpath:
        pmimat.to_csv(os.path.join(outpath, 'lexical_features.csv.gz'),
               compression='gzip')
    return(pmimat.to_sparse())


#def make_pmi_dict(deps, positive=True):
#    """Does the bulk of the work. Counts words, dependency types, and their
#    joint occurences, and returns a dictionary of (P)PMI for each pair.
#    The positive argument is set to False by default, following rei2014looking.
#    """
#    wfreqs = Counter([item[-1] for item in deps])
#    dfreqs = Counter([item[0] for item in deps])
#    jfreqs = Counter([(item[0], item[-1]) for item in deps])
#    size = len(deps)
#    sized = sum(dfreqs.values())
#    pmidict = Counter()
#    for d, w in jfreqs.keys():
#        if positive:
#            pmidict[w, d] = calc_ppmi(wfreqs[w], dfreqs[d], jfreqs[d, w], size, sized)
#        else:
#            pmidict[w, d] = calc_pmi(wfreqs[w], dfreqs[d], jfreqs[d, w], size, sized)
    #fullpairs = product([w for _, w in jfreqs.keys()], [d for d, _ in jfreqs.keys()])
    #unusedpairs = set(fullpairs) -
    #for pair in fullpairs:
#        if pair not in pmidict.keys():
#            pmidict[pair[0], pair[1]] = 0
#    return pmidict


#def to_sparse_df(pmidict, outpath=None):
#    """Converts from dict to sparse pandas DataFrame. If an output file path
#    is provided, it saves the sparse data frame as a gzip-compressed .csv
#    file.
#    """
#    # Filling unknowns with 0s here is not right!!
#    #sdf = pd.Series(pmidict).unstack(fill_value=0.0).to_sparse()
#    sdf = pd.Series(pmidict).unstack().to_sparse()
#    if outpath:
#        sdf.to_csv(os.path.join(outpath, 'lexical_features.csv.gz'),
#                   compression='gzip')
#    return sdf


def svd_reduce(spdf, k=None, sym=False):
    """Performs dimensionality reduction using singular value decomposition of
    the sparse DataFrame -> U*Sigma*V. Returns U*Sigma^0.5 as the rank-k word
    representations (levy2015improving).
    """
    U, S, _ = svds(spdf, k=k)
    if sym:
        mat = U.dot(np.diag(S)**0.5)
    else:
        mat = U.dot(np.diag(S))
    mat = pd.SparseDataFrame(mat)
    mat.index = spdf.index.values
    #mat.columns = spdf.columns.values
    return mat


def lex_spec_feats(pairs, deps, df, outpath=None):
    """For each governor-dependency tuple, calculate the average feature vector across all words that appeared as that type of dependent of the
    governor. Takes a list of tuples of pairs, the dependencies from the
    corpus, and a dataframe containing the word feature vectors/head features.
    """
    print('Calculating lexically specific dependent features...')
    dtypes = df.columns
    lexspec = {}
    for g, dep in pairs:
        print('Working on pair {}-{}.'.format(g, dep))
        words = list(set([w[-1] for w in deps
                          if w[0] == dep and w[1] == g]))
        nwords = len(words)
        if nwords == 0:
            print('{}-{} did not appear in the corpus'.format(g, dep))
            vec = np.full(len(dtypes), np.NaN)
        else:
            vec = np.zeros(len(dtypes))
            for word in words:
                vec += df.loc[word].values / nwords
        lexspec['-'.join([g, dep])] = vec
    retcues = pd.DataFrame.from_dict(lexspec, orient='index', columns=dtypes)
    if outpath:
        print('Saving to file.')
        retcues.to_csv(os.path.join(outpath, 'retrieval_cues.csv.gz'),
                   compression='gzip')
    print('Done.')
    return retcues


def feats_by_dep(deps, df):
    """Creates retrieval cues/dependent features by dependency type. Thes are *not* specific to lexical items. Instead, we just get the average over all head features of words that appear as a given dependency type.
    """
    dtypes = list(set([d[0] for d in deps]))
    feats = pd.DataFrame(0, index=dtypes, columns=df.columns)
    for i, dep in enumerate(dtypes):
        if i % (len(dtypes) // 10) == 0:
            print('{}%\r'.format(np.round(i/len(dtypes) * 100)), end='')
        words = list(set([w[-1] for w in deps if w[0] == dep]))
        nwords = len(words)
        for word in words:
            feats.loc[dep] += df.loc[word].values / nwords
    print('', end='')
    print('Done.')
    return feats


def calc_similarity(words, attch, wdf, adf):
    """Calculate the cosine similarity between words and attachment sites. Takes a list of words, a list of dependent types, a words data frame, and an attachment feature data frame subsetted with the desired dependency type(s).
    """
    assert isinstance(words, list), 'Words should be in a list.'
    assert isinstance(attch, list), 'Dependent types should be in a list.'
    nattch = len(attch)
    nwords = len(words)
    assert all([wdf.index.contains(w) for w in words]), \
            'Not all words in word dataframe.'
    assert all([adf.index.contains(a) for a in attch]), \
            'Not all words in word dataframe.'
    wvecs = wdf.loc[words]
    avecs = adf.loc[attch]
    sims = cosine_similarity(wvecs.append(avecs))[0:nwords, -nattch:]
    return pd.DataFrame(sims, columns=attch, index=words)


if __name__ == '__main__':
    file = '/Users/garrettsmith/Google Drive/UniPotsdam/Research/Features/dependency_embeddings/data/ParsedBrownCorpus/parsedbrown0.txt'
    files = sorted([os.path.abspath(os.path.join(dirp, f)) for dirp, _, fn in
             os.walk('/Users/garrettsmith/Google Drive/UniPotsdam/Research/Features/dependency_embeddings/data/ParsedBrownCorpus/') for f in fn if f.endswith('.txt')])

    #deps = read_standford(files)
    deps = read_standford(file)

    # PPMI embeddings really do make more sense...
    pmi_dict = make_pmi_dict(deps, positive=True)
    #print(pmi_dict.head())
    #sparsedf = to_sparse_df(pmi_dict)#, outpath='/Users/garrettsmith/Google Drive/UniPotsdam/Research/Features/dependency_embeddings/data/')
    #red = svd_reduce(sparsedf, 15)
    #dfeats = feats_by_dep(deps, pmi_dict)
#    dfeatsred = feats_by_dep(deps, red)
#    print(calc_similarity(['he', 'she', 'dog',], ['nsubj', 'obj'],
#                          red, dfeatsred))
#    print(calc_similarity(['he', 'she', 'cat', 'dog',], ['nsubj', 'obj'],
#                          pmi_dict, dfeats))
    pairs = [('read', 'nsubj'), ('read', 'obj')]
    lsfeats = lex_spec_feats(pairs, deps, pmi_dict)#, outpath='/Users/garrettsmith/Desktop/')
    #lsfeats = lex_spec_feats(pairs, deps, red)
    #print(lsfeats)
    print(calc_similarity(['he', 'book'], ['read-nsubj', 'read-obj'], pmi_dict, lsfeats))
    #print(calc_similarity(['he', 'book'], ['read-nsubj', 'read-obj'], red, lsfeats))
