# Tested with Python 3.7

import conllu
import glob
import os

path = '/Users/garrettsmith/Google Drive/UniPotsdam/Resources/Corpora/UniversalDependencyTreebanks/English/UD_English-EWT/'

os.chdir(path)

#with open('en_ewt-ud-test.conllu') as f:
    #sents = conllu.parse(f.read())
    #sents = conllu.parse_incr(f.read())
f = open('en_ewt-ud-test.conllu')

#for i, sent in enumerate(sents):
for i, sent in enumerate(conllu.parse_incr(f)):
    for tk in sent.tokens:
        if i % 25 == 0:
            if tk['head'] == 0:
                triple = tk['deprel'] + '(' + 'ROOT' + ', ' + tk['form'] + ')'
            else:
                head = [x['form'] for x in sent.tokens
                        if x['id'] == tk['head']][0]
                triple = tk['deprel']+ '(' + head + ', ' + tk['form'] + ')'
            print(triple)

f.close()

# Now need to use glob.glob(..., recursive=True) to get all of the CONLL-U
# files, then read them with the conllu package, then output the triples
# to a/multiple file/s.
