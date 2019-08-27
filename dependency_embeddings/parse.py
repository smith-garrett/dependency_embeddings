# Using the Stanford CoreNLP parser to parse a corpus

# Tested w/ Python 3.7 and stanfordnlp 0.2.0

# Took approx. 1h45min. on 2013 MacBook Pro, 2.4GHz Core i5, 8GB memory


import stanfordnlp
import os
import glob
from itertools import islice
import time
from string import punctuation


# Don't need to tokenize, ID multi-word expressions, or lemmatize
config = {'processors': 'tokenize,pos,depparse', 'tokenize_pretokenized': 'True', 'depparse_batch_size': 1000}

# Initializing
nlp = stanfordnlp.Pipeline(**config)

#readdir = '/Users/garrettsmith/Google Drive/UniPotsdam/Research/Features/dependency_embeddings/data/BrownCorpus/'
readdir = '/Users/garrettsmith/Google Drive/UniPotsdam/Resources/Corpora/BritNatCorp/Texts/A/'
#writedir = '/Users/garrettsmith/Google Drive/UniPotsdam/Research/Features/dependency_embeddings/data/ParsedBrownCorpus/'
#writedir = '/Users/garrettsmith/Google Drive/UniPotsdam/Research/Features/dependency_embeddings/data/ParsedBrownCorpusLemmas/'
writedir = '/Users/garrettsmith/Google Drive/UniPotsdam/Research/Features/dependency_embeddings/data/BritNatCorp/'

start_time = time.time()
print('Beginning parsing...')
for filename in glob.glob(os.path.join(readdir, 'x*.txt')):
    file_time = time.time()
    print('Parsing {}'.format(os.path.basename(filename)))
    with open(filename, 'r') as f:
        # Run through pipeline
        #chunk = islice(f, 100)  # For testing
        #proc = nlp('\n'.join(list(chunk)))
        proc = nlp('\n'.join(list(f)))  # Full thing

    print('Writing parses to file...')
    with open(writedir + 'parsed' + os.path.basename(filename), 'w') as f:
        for sent in proc.sentences:
            for d in sent.dependencies:
                dtype = d[1]
                # Removing punctuation#, adding POS
                head = d[0].text.strip(punctuation)# + '-' + d[0].upos
                dep = d[2].text.strip(punctuation)# + '-' + d[2].upos
                #if d[0].lemma != None:
                #    head = d[0].lemma
                #else:
                #    head = d[0].text
                #if d[2].lemma != None:
                #    dep = d[2].lemma
                #else:
                #    dep = d[2].text
                f.write(dtype + '(' + head + ', ' + dep + ')\n')
    print('Finished file in {} minutes'.format((time.time() - file_time) / 60))

print('Done. Time elapsed: {} minutes'.format((time.time() - start_time) / 60))
