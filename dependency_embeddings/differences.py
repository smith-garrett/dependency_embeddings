# Tested w/ Python 3.7

"""Functions for calculating the (differences in) differences between cue-feature match for words in sentence processing experiments.

Expects the feature vectors to be stored in gzip-compressed .csv files.

Idea: the difference in
"""


import numpy as np
import pandas as pd
from itertools import filterfalse
from gen_embeddings import *


#def read_cs(filename):
    #df = pd.read_csv(filename)
    #verbs = list(set(df['verb'].values))
    #nouns = (list(set(df['target'].values)) +
    #         list(set(df['distractor'].values)))
    #return verbs, nouns


def get_missing(df, memberlist):
    """Returns a list of words that are missing from the provided dataframe.
    """
    missing = []
    for m in memberlist:
        if not df.index.contains(m):
            print('"{}" is not in the dataframe'.format(m))
            missing.append(m)
    return missing


def get_similarity(words, attch, featdf, cuedf, missing):
    #featsub = featdf[featdf.index.isin(tuple(missing))]
    #cuesub = cuedf[cuedf.index.isin(tuple(missing))]
    wordssub = list(set([w for w in words if w not in missing]))
    #wordssub = list(filterfalse(lambda x: x not in missing, words))
    attchsub = list(set(filterfalse(lambda x: any(x.startswith(m) for m in
                                              missing), attch)))
    return calc_similarity(wordssub, attchsub, featdf, cuedf)


def sim_to_df(simdf, matdf):
    """Take a dataframe with pairwise cue/feature similarities and a a data
    frame with the materials, and return a data frame with the similarities,
    diff_1x and diff_1 values inserted as additional columns
    """
    fulldf = matdf.copy()
    fulldf['cue_targ_sim'] = 0.0
    fulldf['cue_distr_sim'] = 0.0
    fulldf['within_item_diff'] = 0.0
    for verb in set(fulldf['verb']):
        # WARNING: Hard-coded attch. type!!
        vattch = verb + '-obj'
        for targ in set(fulldf.loc[fulldf['verb'] == verb, 'target']):
            try:
                fulldf.loc[(fulldf['verb'] == verb) & (fulldf['target'] ==
                           targ), 'cue_targ_sim'] = sim.loc[targ, vattch]
            except KeyError:
                fulldf.loc[(fulldf['verb'] == verb) & (fulldf['target'] ==
                           targ), 'cue_targ_sim'] = np.nan
        for distr in set(fulldf.loc[fulldf['verb'] == verb, 'distractor']):
            try:
                fulldf.loc[(fulldf['verb'] == verb) & (fulldf['distractor'] ==
                           distr), 'cue_distr_sim'] = sim.loc[distr, vattch]
            except KeyError:
                fulldf.loc[(fulldf['verb'] == verb) & (fulldf['distractor'] ==
                           distr), 'cue_distr_sim'] = np.nan
    fulldf['within_item_diff'] = (fulldf['cue_targ_sim'] -
                                  fulldf['cue_distr_sim'])
    return fulldf


if __name__ == '__main__':
    corpfiles = sorted([os.path.abspath(os.path.join(dirp, f)) for dirp, _, fn in os.walk('/Users/garrettsmith/Google Drive/UniPotsdam/Research/Features/dependency_embeddings/data/ParsedBrownCorpus/') for f in fn if f.endswith('.txt')])
    csfile = '/Users/garrettsmith/Google Drive/UniPotsdam/Research/Features/CunningsSturtMaterials.csv'

    # Setting up basic features
    deps = read_standford(corpfiles)
    ppmi = make_pmi_dict(deps, positive=True)
    sdf = to_sparse_df(ppmi)

    # Getting C&S's materials
    csmat = pd.read_csv(csfile)
    #verbs, nouns = read_cs(csfile)

    # Calculating retrieval cues for each verb
    pairs = [(i, 'obj') for i in set(csmat.verb.values)]
    vfeats = lex_spec_feats(pairs, deps, sdf)

    # Getting any missing words
    missing = get_missing(sdf, set(csmat.distractor.values))
    missing += get_missing(sdf, set(csmat.target.values))

    # Getting similarities
    sim = get_similarity(list(set(csmat.target)) +
                         list(set(csmat.distractor)), vfeats.index.values,
                         sdf, vfeats, missing)
    #print(sim.head())
    #print(sim.loc['letter', 'shattered-obj'], type(sim.loc['letter', 'shattered-obj']))

    # Putting similarities into a the data frame
    fulldf = sim_to_df(sim, csmat)
    pd.set_option('display.max_columns', None)
    print(fulldf.head(10))
    print(fulldf.loc[fulldf['tplaus'] == 'implaus'].head())
    fulldf.to_csv('/Users/garrettsmith/Google Drive/UniPotsdam/Research/Features/CunningsSturtFeatMatch.csv', na_rep='NA')
