import pandas as pd
from gensim import corpora
from gensim.similarities import Similarity

from PreprocessData import PreProcessData


class STSModel:

    def __init__(self, input_file):
        self.data = []
        self.res = dict()
        self.read_data(input_file)
        self.preprocessdata_o = PreProcessData(self.data)
        id = []
        for data in self.data:
            id.append(data[0])
        self.feature = pd.DataFrame(id, columns=["id"])

    def read_data(self, input_file):
        with open(input_file, 'r', encoding='utf8') as data:
            lines = data.read().splitlines()
        for line in lines[1:]:
            self.data.append(line.split("\t"))

    def cosine_similarity(self):
        cos_sim = []

        for data in self.data:
            sent1 = [word[0] for word in data[1]]
            sent2 = [word[0] for word in data[2]]
            text = [sent1] + [sent2]
            dict = corpora.Dictionary(text)
            corpus = [dict.doc2bow(t) for t in text]
            sim = Similarity('-Similarity-index', corpus, num_features= len(dict))
            test_corpus_1 = dict.doc2bow(sent1)
            cos_sim_each = sim[test_corpus_1][1]
            cos_sim.append(cos_sim_each)
        self.feature['cos_sim'] = cos_sim


    def jaccard_similarity(self):
        jaccard_sim = []
        for data in self.data:
            intersection = len(set(data[1]).intersection(set(data[2])))
            union = len(set(data[1]).union(set(data[2])))
            jaccard_sim.append(intersection / union)
        self.feature['jaccard_sim'] = jaccard_sim

    def model_init(self):
        self.data = self.preprocessdata_o.preprocess_data()
        self.cosine_similarity()
        self.jaccard_similarity()
        print(self.feature.head)
        self.compare_sentence()

    def sentence_length(self, sentence):
        return len(self.preprocessdata_o.tokenise(sentence))

    def compare_sentence(self):
        for data in self.data:
            if len(data[1]) == len(data[2]) and data[1] == data[2]:
                self.res.update({data[0]: 5})


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Please provide the input file only")
    #     exit(0)
    input_file = "data/dev-set.txt"  # sys.argv[1]
    reader = STSModel(input_file)
    # Task 1
    reader.model_init()
    print(reader.res)
