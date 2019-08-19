# Tested w/ Python 3.7

"""Functions for calculating the (differences in) differences between cue-feature match for words in sentence processing experiments.

Expects the feature vectors to be stored in gzip-compressed .csv files.
"""

import numpy as np
import pandas as pd
from itertools import filterfalse
from gen_embeddings import *
import argparse


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
    """Removes words missing from corpus and calculates similarities using
    the remaining subset of words.
    """
    wordssub = list(set([w for w in words if w not in missing]))
    attchsub = list(set(filterfalse(lambda x: any(x.startswith(m) for m in
                                              missing), attch)))
    return calc_similarity(wordssub, attchsub, featdf, cuedf)


def sim_to_df(simdf, matdf):
    """Take a dataframe with pairwise cue/feature similarities and a a data
    frame with the materials, and return a data frame with the similarities and similarity difference values inserted as additional columns
    """
    fulldf = matdf.copy()
    fulldf['cue_targ_sim'] = np.nan
    fulldf['cue_distr_sim'] = np.nan
    fulldf['within_item_diff'] = np.nan
    for verb in set(fulldf['verb']):
        # WARNING: Hard-coded attch. type!!
        vattch = verb + '-obj'
        for targ in set(fulldf.loc[fulldf['verb'] == verb, 'target']):
            try:
                fulldf.loc[(fulldf['verb'] == verb) & (fulldf['target'] ==
                           targ), 'cue_targ_sim'] = simdf.loc[targ, vattch]
            except KeyError:
                fulldf.loc[(fulldf['verb'] == verb) & (fulldf['target'] ==
                           targ), 'cue_targ_sim'] = np.nan
        for distr in set(fulldf.loc[fulldf['verb'] == verb, 'distractor']):
            try:
                fulldf.loc[(fulldf['verb'] == verb) & (fulldf['distractor'] ==
                           distr), 'cue_distr_sim'] = simdf.loc[distr, vattch]
            except KeyError:
                fulldf.loc[(fulldf['verb'] == verb) & (fulldf['distractor'] ==
                           distr), 'cue_distr_sim'] = np.nan
    # Calculated as the relative advantage of the distractor over the target
    fulldf['within_item_diff'] = (fulldf['cue_distr_sim'] -
                                  fulldf['cue_targ_sim'])
    return fulldf


def main(args):
    pd.set_option('display.max_columns', None)
    corpfiles = sorted([os.path.abspath(os.path.join(dirp, f)) for dirp, _, fn in os.walk(args.corpus_dir) for f in fn if f.endswith('.txt')])
    csfile = args.materials

    # Setting up basic features
    print('Reading dependency files...')
    deps = read_standford(corpfiles)
    print('Making PPMI matrix...')
    ppmi = make_pmi_dict(deps, positive=args.not_positive, outpath=args.output_dir)

    # SVD to reduce
    if args.svd_k:
        ppmi = svd_reduce(ppmi, k=args.svd_k, sym=args.svd_not_sym)

    # Getting C&S's materials
    print('Loading materials from Cunnings & Sturt...')
    csmat = pd.read_csv(csfile)

    # Calculating retrieval cues for each verb
    print('Calculating retrieval cues...')
    pairs = [(i, 'obj') for i in set(csmat.verb.values)]
    #vfeats = lex_spec_feats(pairs, deps, ppmi, outpath='~/Desktop')
    #vfeats = lex_spec_feats(pairs, deps, red)
    #print(vfeats)
    vfeats = lex_spec_feats(pairs, deps, ppmi, outpath=args.output_dir)

    # Getting any missing words
    missing = get_missing(ppmi, set(csmat.distractor.values))
    missing += get_missing(ppmi, set(csmat.target.values))
    missing += get_missing(ppmi, set(csmat.verb.values))
    #print(missing)

    # Getting similarities
    print('Getting similarities...')
    sim = get_similarity(list(set(csmat.target)) +
                         list(set(csmat.distractor)), vfeats.index.values,
                         ppmi, vfeats, missing)
    #print(sim.head())
    #print(sim.loc['letter', 'shattered-obj'], type(sim.loc['letter', 'shattered-obj']))

    # Putting similarities into a the data frame
    fulldf = sim_to_df(sim, csmat)
    #print(fulldf.head(10))
    #print(fulldf.loc[fulldf['tplaus'] == 'implaus'].head())
    print('Saving to file...')
    if args.svd_k:
        fulldf.to_csv(args.output_dir + 'similaritySVD{}.csv'.format(args.svd_k), na_rep='NA')
    else:
        fulldf.to_csv(args.output_dir + 'similarity.csv', na_rep='NA')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main pipeline for computing dependency-based embedding similarities.')
    reqnamed = parser.add_argument_group('required arguments')
    reqnamed.add_argument('--corpus_dir', help='Directory containing parsed corpus file(s)', required=True)
    reqnamed.add_argument('--output_dir', help='Directory for outputs', required=True)
    reqnamed.add_argument('--materials', help='File containing materials from experiment', required=True)
    parser.add_argument('--not_positive', help='Do PMI instead of PPMI.', action='store_false')
    parser.add_argument('--svd_k', help='Provide number of dimensions to keep when doing SVD. Default is no SVD.', type=int, default=0)
    parser.add_argument('--svd_not_sym', help='Do not make SVD-reduced (P)PMI matrix symmetric', action='store_false')
    args = parser.parse_args()
    main(args)
