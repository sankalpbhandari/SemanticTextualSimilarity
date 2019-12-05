Read me:


Installation and setup:

1 - Make sure you are using Python 3

2 - Install nltk
>> pip install nltk

3 - Download corpus   
>>import nltk
>>nltk.download()	

4 - Install gensim
>>pip install -U gensim

5 - Install pandas
>>pip install pandas

6 - Install sklearn
>>pip install numpy scipy scikit-learn 

7 - Install spacy
>>pip install spacy

8 - Download and extract Google pretrained word2vec model at root location of project from https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz


Running

Feature Extraction - stores the extracted features in output file
python3 STSModel.py <input file> <output file>
eg. >> python3 STSModel.py data/train-set.txt train.csv

Model training and predicting
input file is the file containing features
output file is the result file
python3 Model.py <'predict'/'train'> <input file> [<output file>]

For Training
eg >>python3 Model.py train train.csv

For Predicting
eg >>python3 Model.py predict test.csv test-predicted-answers.txt

Evaluation
python3 evaluation.py <gold labels file> <predicted labels file>
>>python3 evaluation.py test-gold-answers.txt test-predicted-answers.txt










