from time import time

import pandas as pd
from gensim import corpora
from gensim.similarities import Similarity
from joblib import dump, load
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nltk.corpus import wordnet
from sklearn.svm import SVR

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
        print("Reading Data from file")
        with open(input_file, 'r', encoding='utf8') as data:
            lines = data.read().splitlines()
        for line in lines[1:]:
            self.data.append(line.split("\t"))

    def cosine_similarity_with_syn(self):
        print("Cosine Similarity with lemmas and synsets")
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
        self.feature['cos_sim_with_syn'] = cos_sim

    def cosine_similarity_only_syn(self):
        print("Cosine Similarity with only synsets")
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
            text = [sent3] + [sent4]
            dict = corpora.Dictionary(text)
            corpus = [dict.doc2bow(t) for t in text]
            sim = Similarity('-Similarity-index', corpus, num_features=len(dict))
            test_corpus_1 = dict.doc2bow(sent1)
            cos_sim_each = sim[test_corpus_1][1]
            cos_sim.append(cos_sim_each)
        self.feature['cos_sim_only_syn'] = cos_sim

    def cosine_similarity_no_syn(self):
        print("Cosine Similarity without synsets")
        cos_sim = []
        for data in self.data:
            sent1 = [word[0] for word in data[1]]
            sent2 = [word[0] for word in data[2]]
            text = [sent1] + [sent2]
            dict = corpora.Dictionary(text)
            corpus = [dict.doc2bow(t) for t in text]
            sim = Similarity('-Similarity-index', corpus, num_features=len(dict))
            test_corpus_1 = dict.doc2bow(sent1)
            cos_sim_each = sim[test_corpus_1][1]
            cos_sim.append(cos_sim_each)
        self.feature['cos_sim_no_syn'] = cos_sim

    def jaccard_similarity_with_synset(self):
        print("Cosine Similarity with lemmas and synsets")
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
        self.feature['jaccard_sim_with_syn'] = jaccard_sim

    def jaccard_similarity_only_synset(self):
        print("Jaccard Similarity with only synsets")
        jaccard_sim = []
        for data in self.data:
            sent3, sent4 = [], []
            for word in set(data[1]):
                if self.preprocessdata_o.synsets.get(word[0]):
                    sent3 += list(self.preprocessdata_o.synsets.get(word[0]))
            for word in set(data[2]):
                if self.preprocessdata_o.synsets.get(word[0]):
                    sent4 += list(self.preprocessdata_o.synsets.get(word[0]))
            intersection = len(set(sent3).intersection(set(sent4)))
            union = len(set(sent3).union(set(sent4)))
            jaccard_sim.append(intersection / union)
        self.feature['jaccard_sim_only_syn'] = jaccard_sim

    def jaccard_similarity_no_syn(self):
        print("Jaccard Similarity without synsets")
        jaccard_sim = []
        for data in self.data:
            intersection = len(set(data[1]).intersection(set(data[2])))
            union = len(set(data[1]).union(set(data[2])))
            jaccard_sim.append(intersection / union)
        self.feature['jaccard_sim_no_syn'] = jaccard_sim

    def penn_to_wn(self, tag):
        if tag.startswith('N'):
            return 'n'
        if tag.startswith('V'):
            return 'v'
        if tag.startswith('J'):
            return 'a'
        if tag.startswith('R'):
            return 'r'
        return None

    def tagged_to_synset(self, word, tag):
        wn_tag = self.penn_to_wn(tag)
        if wn_tag is None:
            return None

        try:
            return wordnet.synsets(word, wn_tag)[0]
        except:
            return None

    def sentence_similarity(self):
        print("Word Sense Disambiguation")
        sent_sim = []
        for data in self.data:
            synsets1 = [self.tagged_to_synset(*tagged_word) for tagged_word in data[1]]
            synsets2 = [self.tagged_to_synset(*tagged_word) for tagged_word in data[2]]

            # Filter out the Nones
            synsets1 = [ss for ss in synsets1 if ss]
            synsets2 = [ss for ss in synsets2 if ss]

            score, count = 0.0, 0

            for synset in synsets1:
                temp = []
                for ss in synsets2:
                    if synset.path_similarity(ss):
                        temp.append(synset.wup_similarity(ss))
                best_score = max(temp) if temp else 0

                if best_score:
                    score += best_score
                    count += 1

            score = score / count if count else 0
            sent_sim.append(score)
        self.feature["sent_sim"] = sent_sim

    def model_init(self):
        self.data = self.preprocessdata_o.preprocess_data()
        self.cosine_similarity_with_syn()
        self.cosine_similarity_no_syn()
        self.cosine_similarity_only_syn()
        self.jaccard_similarity_with_synset()
        self.jaccard_similarity_only_synset()
        self.jaccard_similarity_no_syn()
        self.sentence_similarity()
        self.compare_sentence()
        self.relative_length()
        self.pos_relative_length()
        self.model()

    def compare_sentence(self):
        print("Compare two sentences")
        for data in self.data:
            if len(data[1]) == len(data[2]) and data[1] == data[2]:
                self.res.update({data[0]: 5})

    def relative_length(self):
        print("Finding relative length of sentence")
        length = []
        for data in self.data:
            length.append(min(len(data[1]), len(data[2])) / max(len(data[1]), len(data[2])))
        self.feature['relative_length'] = length

    def pos_relative_length(self):
        print("Calculating relative length of POS Tags")
        s, adj, adv, verb, noun = [], [], [], [], []
        for data in self.data:
            s1 = pos1 = data[1]
            s2 = pos2 = data[2]
            t1 = abs(len(s1) - len(s2)) / float(len(s1) + len(s2))
            cnt1 = len([1 for item in pos1 if item[1].startswith('J')])
            cnt2 = len([1 for item in pos2 if item[1].startswith('J')])
            if cnt1 == 0 and cnt2 == 0:
                t2 = 0
            else:
                t2 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
            # all adverbs
            cnt1 = len([1 for item in pos1 if item[1].startswith('R')])
            cnt2 = len([1 for item in pos2 if item[1].startswith('R')])
            if cnt1 == 0 and cnt2 == 0:
                t3 = 0
            else:
                t3 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
            # all nouns
            cnt1 = len([1 for item in pos1 if item[1].startswith('N')])
            cnt2 = len([1 for item in pos2 if item[1].startswith('N')])
            if cnt1 == 0 and cnt2 == 0:
                t4 = 0
            else:
                t4 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
            # all verbs
            cnt1 = len([1 for item in pos1 if item[1].startswith('V')])
            cnt2 = len([1 for item in pos2 if item[1].startswith('V')])
            if cnt1 == 0 and cnt2 == 0:
                t5 = 0
            else:
                t5 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
            s.append(t1)
            adj.append(t2)
            adv.append(t3)
            noun.append(t4)
            verb.append(t5)

        self.feature['sent_pos_score'] = s
        self.feature['adj_pos_score'] = adj
        self.feature['adv_pos_score'] = adv
        self.feature['noun_pos_score'] = noun
        self.feature['verb_pos_score'] = verb

    def model(self):
        clf = SVR(kernel='linear')
        labels = self.feature['label']
        X_features = self.feature.drop(["id", "label"], axis=1)
        clf = load('filename.joblib')
        res = clf.predict(X_features)
        print(res)
        with open('abc.csv', 'w') as f:
            for item in res:
                f.write("%s\n" % item)
        # clf.fit(X_features, labels)
        # dump(clf, 'filename.joblib')
        fig, ax = plt.subplots()
        ax.scatter(self.feature['sent_sim'], self.feature['label'])
        # for i, txt in enumerate(self.feature['label']):
        #     ax.annotate(txt, (self.feature['jaccard_sim'][i], self.feature['cos_sim'][i]))
        pdf = PdfPages(str(time()) + '_abc_new.pdf')
        pdf.savefig(fig)
        pdf.close()


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Please provide the input file only")
    #     exit(0)
    input_file = "data/train-set.txt"  # sys.argv[1]
    reader = STSModel(input_file)
    # Task 1
    reader.model_init()
    print(reader.feature)
