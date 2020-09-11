# Old, unsuccessful method for developing corpus-based lexical features and retrieval cues

**Better method reported in Smith and Vasishth (2020, *Cognitive Science*), but we still include this method for completeness.**

Cunnings & Sturt (2018) reported illusion of plausibility effects. They argue in that this type of effect requires lexically specific retrieval cues, e.g., *shattered* is looking for direct objects that are shatterable and not ones that are fluffy, like what *pet (the bunny, e.g.)* is looking for. They don't provide a principled way of deriving those retrieval cues, though.

Using this method we obtained the following results using Bayesian mixed models of the log-transformed total reading times from Cunnings & Sturt's two experiments: Plausible targets induced faster reading times at the verb: -60ms, [-78, -42]. The interaction with experiment showed this effect was consistent in both experiments: Experiment 1, -66ms, [-104, -30]; Experiment 2, -53ms, [-92, -15]. The nested effect of distractor advantage in plausible-target sentences was near zero -8ms, [-38, 23]. Again, the interactions with experiment showed consistent effects close to zero: Experiment 1, -11ms, [-58, 37]; Experiment 2, -5ms, [-56, 46]. In implausible sentences, the effect of distractor advantage was more clearly negative than in plausible sentences (-17ms, [-46, 12]); the more plausible the distractor was compared to the target, the faster reading times were. The interaction with experiment shows that this effect was driven mainly by Experiment 1: Experiment 1: -38ms, [-86, 10]; Experiment 2: 5ms, [-45, 52].

## Old README

This Python package allows us to develop such features and cues in a principled way. It requires a parsed corpus in `dependency(gov, dep)` format. We used [Stanford NLP's pipeline](https://stanfordnlp.github.io/stanfordnlp/) because it is reasonably fast---it parsed the BNC in a few days---and it provides state-of-the-art dependency parsing (Qi et al., 2018). Once we have the parses, the processing pipeline in `pipeline.py` goes as follows:

1. With the dependency triples, create word embeddings (using functions in gen_embeddings.py) using the positive point-wise mutual information (PPMI) between a word and each possible dependency context it could appear in. These are saved in a zipped file.

2. To calculate the retrieval cues, we simply take a word for which we want the cues, the type of dependent we're interested in (e.g., direct object), and finally take the average of all of the lexical features of words that appear as that dependent of the given word. These are also saved. We now have what we need.

3. The last step is to apply these vectors to the a scientific question. Here, we focus on cases where there is a grammatical retrieval target and an ungrammatical distractor. Cunnings and Sturt (2018) varied the plausibility of the target and distractor (validated by a norming study). We hypothesized that the difference in plausibility, i.e., the cosine similarity between an embedding and the retrieval cue, between the target and distractor should predict the reading time at the retrieval site. Calculating these differences is done in the script differences.py.

Further help can be found by typing `python pipeline.py -h`.
