# File for getting the Brown corpus into just stings of text, one sentence
# per line, via NLTK
# Brown corpus = 1161192 words in 57340 sentences
# Outputs separate files with ~5000 sentences each

# Tested w/ Python 3.7 and NLTK 3.7.3

from nltk.corpus import brown
import os
import time

filebase = '/Users/garrettsmith/Google Drive/UniPotsdam/Research/Features/GenEmbeddings/BrownCorpus/'

start_time = time.time()
fileno = 0
for i, sent in enumerate(brown.sents()):
    if (i > 0) and (i % 5000 == 0):
        fileno += 1
        print('{} sentences processed\r'.format(i), end='')
    file = filebase + 'brown' + str(fileno) + '.txt'
    # Open file to append to if it exists
    if os.path.exists(filebase):
        mode = 'a'
    else:
        mode = 'w'
    with open(file, mode) as f:
        sent = ' '.join(sent) + '\n'
        f.write(sent)


print('Elapsed time: {} seconds'.format(time.time() - start_time))
