# Tested w/ Python 3.7, SciPy 1.2.1

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

data = pd.read_csv('~/Desktop/lexical_features.csv', index_col=0)

#words = 'the a this those an any some all cat dog man woman plate spoon freedom love watch see saw laughed laugh eat fall fell disappear apparently truly clearly very cold warm blue red orange frail atomic research incredibly amazing structural'.split()
words = list(set('There was a lot to do at the company saw the email laptop that the manager with the report office very hastily wrote earlier in a real hurry There were going to be some changes at work Sometimes things happen by mistake noticed the glass shirt that the girl by the food accidentally cracked earlier near the kitchen sink It was not really anybody fault'.lower().split()))
selection = data.loc[words]
#selection = data.sample(n=50, random_state=42)

print('Calculating linkage')
lkg = linkage(selection.to_numpy(), 'ward')
#lkg = linkage(data.to_numpy(), 'ward')

print('Making dendrogram')
#np.random.seed(42)
#subs = np.random.choice(range(len(data.index)), size=75)
plt.figure(figsize=(10, 8))
dendrogram(lkg, leaf_font_size=8, labels=selection.index, orientation='right')
#dendrogram(lkg[subs,], leaf_font_size=8, labels=data.iloc[subs,].index,
#           orientation='right')
plt.show()
