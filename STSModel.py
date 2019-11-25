from time import time

import pandas as pd
from gensim import corpora
from gensim.similarities import Similarity
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from PreprocessData import PreProcessData


class STSModel:

    def __init__(self, input_file):
        self.data = []
        self.res = dict()
        self.read_data(input_file)
        self.preprocessdata_o = PreProcessData(self.data)
        id = []
        label = []
        for data in self.data:
            id.append(data[0])
            label.append(data[3])
        self.feature = pd.DataFrame(id, columns=["id"])
        self.feature["label"] = label

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
            sent3, sent4 = [], []
            for word in sent1:
                if self.preprocessdata_o.synsets.get(word):
                    sent3 += list(self.preprocessdata_o.synsets.get(word))
            sent1 += sent3
            for word in sent2:
                if self.preprocessdata_o.synsets.get(word):
                    sent4 += list(self.preprocessdata_o.synsets.get(word))
            sent2 += sent4
            text = [sent1] + [sent2]
            dict = corpora.Dictionary(text)
            corpus = [dict.doc2bow(t) for t in text]
            sim = Similarity('-Similarity-index', corpus, num_features=len(dict))
            test_corpus_1 = dict.doc2bow(sent1)
            cos_sim_each = sim[test_corpus_1][1]
            cos_sim.append(cos_sim_each)
        self.feature['cos_sim'] = cos_sim

    def jaccard_similarity(self):
        jaccard_sim = []
        for data in self.data:
            sent3, sent4 = [], []
            for word in set(data[1]):
                if self.preprocessdata_o.synsets.get(word[0]):
                    sent3 += list(self.preprocessdata_o.synsets.get(word[0]))
            sent3 += data[1]
            for word in set(data[2]):
                if self.preprocessdata_o.synsets.get(word[0]):
                    sent4 += list(self.preprocessdata_o.synsets.get(word[0]))
            sent4 += data[2]
            intersection = len(set(sent3).intersection(set(sent4)))
            union = len(set(sent3).union(set(sent4)))
            jaccard_sim.append(intersection / union)
        self.feature['jaccard_sim'] = jaccard_sim

    def model_init(self):
        self.data = self.preprocessdata_o.preprocess_data()
        self.cosine_similarity()
        self.jaccard_similarity()
        self.model()
        self.compare_sentence()
        self.relative_length()

    def compare_sentence(self):
        for data in self.data:
            if len(data[1]) == len(data[2]) and data[1] == data[2]:
                self.res.update({data[0]: 5})

    def relative_length(self):
        length = []
        for data in self.data:
            length.append(min(len(data[1]), len(data[2])) / max(len(data[1]), len(data[2])))
        self.feature['relative_length'] = length

    def model(self):
        fig, ax = plt.subplots()
        ax.scatter(self.feature['jaccard_sim'], self.feature['label'])
        # for i, txt in enumerate(self.feature['label']):
        #     ax.annotate(txt, (self.feature['jaccard_sim'][i], self.feature['cos_sim'][i]))
        pdf = PdfPages(str(time) + '_abc_new.pdf')
        pdf.savefig(fig)
        pdf.close()


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Please provide the input file only")
    #     exit(0)
    input_file = "data/dev-set.txt"  # sys.argv[1]
    reader = STSModel(input_file)
    # Task 1
    reader.model_init()
    print(reader.feature)
