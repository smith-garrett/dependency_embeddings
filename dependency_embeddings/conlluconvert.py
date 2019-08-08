# Tested with Python 3.7

import conllu
import glob
import os

path = '/Users/garrettsmith/Google Drive/UniPotsdam/Resources/Corpora/UniversalDependencyTreebanks/English/'

os.chdir(path)

files = glob.glob('./**/*.conllu', recursive=True)

#with open('en_ewt-ud-test.conllu') as f:
    #sents = conllu.parse(f.read())
    #sents = conllu.parse_incr(f.read())
#f = open('en_ewt-ud-test.conllu')

outfile = open('/Users/garrettsmith/Desktop/UDtriples.txt', 'w')
for fn, file in enumerate(files):
    print('Working on file {} of {}'.format(fn, len(files)))
    #f = open('en_ewt-ud-test.conllu')
    f = open(file)
    for i, sent in enumerate(conllu.parse_incr(f)):
        for tk in sent.tokens:
            if tk['head'] == 0:
                triple = tk['deprel'] + '(' + 'ROOT' + ', ' + tk['form'] + ')'
            elif tk['head'] == None:
                triple = None
                continue
            else:
                head = [x['form'] for x in sent.tokens
                        if x['id'] == tk['head']][0]
                triple = tk['deprel']+ '(' + head + ', ' + tk['form'] + ')'
            #print(triple)
            if triple:
                outfile.write(triple + '\n')
    f.close()
outfile.close()

# Now need to use glob.glob(..., recursive=True) to get all of the CONLL-U
# files, then read them with the conllu package, then output the triples
# to a/multiple file/s.
