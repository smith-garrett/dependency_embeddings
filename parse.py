# Using the Stanford CoreNLP parser to parse a corpus

# Tested w/ Python 3.7 and stanfordnlp 0.2.0

# Took approx. 1h45min. on 2013 MacBook Pro, 2.4GHz Core i5, 8GB memory


import stanfordnlp
import os
import glob
from itertools import islice
import time


nlp = stanfordnlp.Pipeline()

readdir = '/Users/garrettsmith/Google Drive/UniPotsdam/Research/Features/GenEmbeddings/BrownCorpus/'
writedir = '/Users/garrettsmith/Google Drive/UniPotsdam/Research/Features/GenEmbeddings/ParsedBrownCorpus/'


start_time = time.time()
print('Beginning parsing...')
for filename in glob.glob(os.path.join(readdir, '*.txt')):
    file_time = time.time()
    print('Parsing {}'.format(os.path.basename(filename)))
    with open(filename, 'r') as f:
        #chunk = islice(f, 10)
        proc = nlp(' '.join(list(f)))
        #proc = nlp(' '.join(list(chunk)))

    print('Writing parses to file...')
    with open(writedir + 'parsed' + os.path.basename(filename), 'w') as f:
        for sent in proc.sentences:
            for d in sent.dependencies:
                dtype = d[1]
                head = d[0].text
                dep = d[2].text
                f.write(dtype + '(' + head + ', ' + dep + ')\n')
    print('Finished file in {} minutes'.format((time.time() - file_time) / 60))

print('Done. Time elapsed: {} minutes'.format((time.time() - start_time) / 60))
