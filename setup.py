from setuptools import setup

setup(name='dependency_embeddings',
      version='0.1',
      description='A package for parsing a corpus, extracting dependency triples, and calculating PPMI word feature embeddings and retrieval cues for use in computational models of human sentence processing.',
      url='https://github.com/garrett-m-smith/dependency_embeddings',
      author='Garrett Smith',
      author_email='gasmith@uni-potsdam.de',
      license='GPL-3.0-or-later',
      packages=['dependency_embeddings'],
      zip_safe=False,
      install_requires=['nltk', 'numpy', 'pandas', 'sklearn', 'scipy', 'stanfordnlp'])
