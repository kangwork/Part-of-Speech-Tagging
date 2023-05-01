# Part-of-Speech-Tagging
This project implements a Hidden Markov Model (HMM) for Part-Of-Speech (POS) tagging in Natural Language Processing (NLP). The goal is to predict the syntactic role (POS tag) of each word in a given sentence.

## Task Description
The task involves the following steps:

1. Training probability tables (initial, transition, and emission) for the HMM from training files containing text-tag pairs.
2. Performing inference with the trained HMM to predict appropriate POS tags for untagged text.

## Running the Code
To run the POS tagger, execute the following command in the terminal:

`python3 tagger.py -d <training files> -t <test file> -o <output file>`

The parameters are as follows:

- <training files>: One or more training file names, separated by spaces.
- <test file>: A single test file name.
- <output file>: A single output file name.

For example:

`python3 tagger.py -d data/training1.txt data/training2.txt -t data/test1.txt -o data/output1.txt`

## Implementation

The training and test procedures are split within the tagger.py file.

### 1. Training Phase
The hidden variables in the HMM represent the POS tags, while the evidence variables represent the words in the sentences.

During the training phase, the program learns three probability tables:
1. Initial probabilities: Likelihood of each POS tag appearing at the beginning of a sentence.
2. Transition probabilities: Probability of transitioning from one POS tag to another.
3. Emission probabilities: Probability of a POS tag emitting an observed word.

Learning these probabilities involves counting. For example, to determine the initial probability for the POS tag NP0, the program divides the number of times NP0 appears at the beginning of a sentence by the total number of sentences.

The program stores these probability tables for later use.

### 2. Test Phase

During the test phase, the program utilizes the probability tables and inference algorithms like the Viterbi algorithm to calculate the most likely tag for each word in a sentence.

- Independence between sentences is assumed, allowing the splitting of text files into individual sentences during training and testing without losing information.


## Test Results
When testing the program with different configurations, including the ratio of trained sets used for training and the level of direct overlap between the training and test cases, the average correctness percentages are as follows:

- Easy cases (3 training sets : 1 testing set, with overlap): 96% correctness.
- Medium cases (3 training sets : 1 testing set, no overlap): 88.8% correctness.
- Hard cases (1 training set : 1 testing set, no overlap): 88.9% correctness.

These correctness percentages indicate the accuracy of the program in predicting the correct POS tags for the words in the test sentences.

## Note
- Please note that these results may vary depending on the specific dataset and the performance of the implemented HMM model.
- This project was for my university course CSC384.
