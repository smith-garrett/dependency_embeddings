# Command line script for taking Stanford parses from Natural Stories Corpus
# (Futrell et al., 2018) and outputting a text file containing PPMI word
# embeddings. These are "head" feature representations in the sense of Smith &
# Tabor (2018; ICCM), i.e., the features used for a word when it attaches as the
# dependent of another word. In addition to the word embeddings, the script also
# returns the average word embedding of the words that appear as the dependent
# of another words, i.e., features that a word is "looking for" in its
# dependent(s).

# Tested w/ Python 3.7


import re
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds


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


def get_word_freqs(deps):
    words = [item[-1] for item in deps]
    nwords = len(words)
    cts = Counter(words)
    for w in cts:
        cts[w] /= nwords
    return cts


def get_dep_freqs(deps, smooth=None):
    """ Smoothing recommended in levy2015improving is 0.75.
    """
    dtypes = [item[0] for item in deps]
    ntypes = len(dtypes)
    cts = Counter(dtypes)
    for dt in cts:
        if smooth:
            cts[dt] = cts[dt]**smooth / sum([ct**smooth for ct in cts.values()])
        else:
            cts[dt] /= ntypes
    return cts


def get_joint_freqs(deps):
    pairs = [(item[0], item[-1]) for item in deps]
    npairs = len(pairs)
    #cts = Counter()
    cts = Counter(pairs)
    #for pair in pairs:
    for pair in cts:
        #cts[pair[1], pair[0]] += 1./npairs
        cts[pair] /= npairs
    return cts


def calc_pmi(wfreq, dfreq, jfreq):
    """Often throws divide-by-zero error, so this gets around that and sets
    -inf to zero.
    """
    with np.errstate(divide='ignore'):
        pmi = np.log2(jfreq / (wfreq * dfreq))
        if pmi != -np.inf:
            return pmi
        else:
            return 0.0


def calc_ppmi(wfreq, dfreq, jfreq):
    return np.maximum(0, np.log2(jfreq / (wfreq * dfreq)))


def make_pmi_dict(deps, positive=True, smooth=None):
    #words = sorted([item[-1] for item in deps])
    #dtypes = sorted([item[0] for item in deps])
    wfreqs = get_word_freqs(deps)
    dfreqs = get_dep_freqs(deps, smooth)
    jfreqs = get_joint_freqs(deps)
    #pmidict = OrderedDict()
    pmidict = Counter()
    for d, w in jfreqs.keys():
        if positive:
            pmidict[w, d] = calc_ppmi(wfreqs[w], dfreqs[d], jfreqs[d, w])
        else:
            pmidict[w, d] = calc_pmi(wfreqs[w], dfreqs[d], jfreqs[d, w])
    return pmidict


def to_sparse_df(pmidict):
    return pd.Series(pmidict).unstack(fill_value=0.0).to_sparse()


def svd_reduce(spdf, k=None):
    """Returns U*Sigma as the rank-k word representations (levy2015improving).
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
    """
    return


def feats_by_dep(deps, df):
    dtypes = list(set([d[0] for d in deps]))
    if isinstance(df.columns[0], str):
        colnames = df.columns
    else:
        colnames = ['f'+ str(i) for i in range(len(df.columns))]
    feats = pd.DataFrame(0, index=dtypes, columns=colnames)
    for i, dep in enumerate(dtypes):
        #if np.round(i//len(dtypes)) / 10 == 0:
        #    print('{}%\r'.format(np.round(i/len(dtypes) * 100, 3)), end='')
        words = list(set([w[-1] for w in deps if w[0] == dep]))
        for word in words:
            feats.loc[dep] += df.loc[word].values / len(words)
    #print('Done.\t')
    return feats


if __name__ == '__main__':
    file = '/Users/garrettsmith/Google Drive/UniPotsdam/Research/Features/naturalstories/parses/stanford/all-parses-aligned.txt.stanford'
    deps = read_standford(file)
    #print(deps[0:10])
    #wfreqs = get_word_freqs(deps)
    #print(*wfreqs.most_common(10), sep='\n')
    #dfreqs = get_dep_freqs(deps)
    #print(*dfreqs.most_common(10), sep='\n')
    #jfreqs = get_joint_freqs(deps)
    #print(*jfreqs.most_common(10), sep='\n')
    #print(calc_pmi(wfreqs['king'], dfreqs['vmod'], jfreqs['vmod', 'king']))
    #print(calc_pmi(wfreqs['the'], dfreqs['det'], jfreqs['det', 'the']))
    #print(calc_pmi(wfreqs['were'], dfreqs['nsubj'], jfreqs['nsubj', 'were']))
    #print(calc_pmi(wfreqs['was'], dfreqs['root'], jfreqs['root', 'was']))
    pmi_dict = make_pmi_dict(deps, positive=False, smooth=0.75)
    #print(*pmi_dict.most_common(10), sep='\n')
    #print(*pmi_dict.most_common()[-10:-1], sep='\n')
    #print(np.quantile(np.array(list(pmi_dict.values())),
    #                  [0.0, 0.25, 0.5, 0.75, 1.0]))
    #print(pmi_dict['the', 'det'], pmi_dict['you', 'nsubj'])
    #print(pd.Series(pmi_dict).unstack(fill_value=0.0).head(10))
    sparsedf = to_sparse_df(pmi_dict)
    print(sparsedf.head(10))
    #print(cosine_similarity(sparsedf.loc[['he', 'she', 'was']]))
    #red = svd_reduce(sparsedf, 10)
    #print(cosine_similarity(red.loc[['he', 'she', 'was']]))
    #dfeats = feats_by_dep(deps, red)
    #dfeats = feats_by_dep(deps, sparsedf)
    #print(dfeats.columns)
    #print(dfeats)
    #print(cosine_similarity([red.loc['he'], red.loc['to'],
    #                        dfeats.loc['nsubj']]))
