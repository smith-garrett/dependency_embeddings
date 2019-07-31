#!/bin/bash

# Change to location of Stanford parser
cd ~/stanford-parser-full-2018-10-17
counter=0

# Change path to location of MASC-OANC files
for f in ~/Google\ Drive/UniPotsdam/Resources/Corpora/MASC_OANC/all/*; do
    echo "Processing file $counter"

    # Change to save path
    currfile=~/Google\ Drive/UniPotsdam/Research/Features/dependency_embeddings/data/MASC_OANC/masc$counter.txt
    touch "$currfile"

    java -cp stanford-parser.jar edu.stanford.nlp.trees.EnglishGrammaticalStructure -treeFile "$f" -retainNPTmpSubcategories -basic > "$currfile"

    ((counter++))
done
