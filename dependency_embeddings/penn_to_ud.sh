#!/bin/bash

# Change to location of Stanford parser
cd ~/stanford-parser-full-2018-10-17
counter=0

# Change path to location of PTB files
for f in ~/Google\ Drive/UniPotsdam/Resources/Corpora/ptb-gold/all/*; do
    echo "Processing file $counter"

    # Change to save path
    currfile=~/Google\ Drive/UniPotsdam/Research/Features/dependency_embeddings/data/PennDep/penn_$counter.txt
    touch "$currfile"

    java -cp stanford-parser.jar edu.stanford.nlp.trees.EnglishGrammaticalStructure -treeFile "$f" -retainNPTmpSubcategories -basic > "$currfile"

    ((counter++))
done
