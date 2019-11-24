import string
from collections import defaultdict

import spacy
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class PreProcessData:
    def __init__(self, data):
        self.synsets = defaultdict(set)
        self.hypernyms = defaultdict(set)
        self.hyponyms = defaultdict(set)
        self.meronyms = defaultdict(set)
        self.holonyms = defaultdict(set)
        self.stopwords = set(stopwords.words('english'))
        self.data = data

    def preprocess_data(self):
        for index in range(len(self.data)):
            data = self.data[index]
            sent1_nopunch = data[1].translate(str.maketrans('', '', string.punctuation)).lower()
            sent2_nopunch = data[2].translate(str.maketrans('', '', string.punctuation)).lower()

            filtered_token1, filtered_token2 = [], []
            sentence1_tokens = self.tokenise(sent1_nopunch)
            sentence2_tokens = self.tokenise(sent2_nopunch)

            # Lemmatizing, POS tagging and removing stop words
            for token in sentence1_tokens:
                if token not in self.stopwords:
                    token = pos_tag(self.lematize(token, False))[0]
                    filtered_token1.append(token)
            for token in sentence2_tokens:
                if token not in self.stopwords:
                    token = pos_tag(self.lematize(token, False))[0]
                    filtered_token2.append(token)
            self.data[index][1] = filtered_token1
            self.data[index][2] = filtered_token2
        return self.data

    def tokenise(self, sentence):
        return word_tokenize(sentence)

    def lematize(self, sentence, is_sentence=True):
        if is_sentence:
            tokenised_words = self.tokenise(sentence)
        else:
            tokenised_words = [sentence]
        lemmatizer = WordNetLemmatizer()
        res = []
        tagged_word = pos_tag(tokenised_words)
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        for i in range(len(tokenised_words)):
            res.append(
                lemmatizer.lemmatize(tagged_word[i][0], tag_dict.get(tagged_word[i][1][0].upper(), wordnet.NOUN)))
        return res

    def pos_tagging(self, sentence, is_sentence=False):
        if is_sentence:
            return pos_tag(self.tokenise(sentence))
        else:
            return pos_tag(sentence)

    def wordnet_features(self, sentence):
        tokenized_words = self.tokenise(sentence)
        for word in tokenized_words:
            temp_synset = wordnet.synsets(word)
            for temp_word in temp_synset:
                self.synsets[word].add(temp_word)
            for each_elem in temp_synset:
                temp_hypernyms = each_elem.hypernyms()
                for hypernym in temp_hypernyms:
                    self.hypernyms[each_elem].add(hypernym)

                temp_hyponyms = each_elem.hyponyms()
                for hyponyms in temp_hyponyms:
                    self.hyponyms[each_elem].add(hyponyms)

                temp_meronyms = each_elem.part_meronyms()
                for meronyms in temp_meronyms:
                    self.meronyms[each_elem].add(meronyms)
                temp_meronyms = each_elem.substance_meronyms()
                for meronyms in temp_meronyms:
                    self.meronyms[each_elem].add(meronyms)

                temp_holonyms = each_elem.part_holonyms()
                for holonyms in temp_holonyms:
                    self.holonyms[each_elem].add(holonyms)
                temp_holonyms = each_elem.substance_holonyms()
                for holonyms in temp_holonyms:
                    self.holonyms[each_elem].add(holonyms)

        print("\nSynsets\t", "--" * 10)
        for key in self.synsets:
            print(key, self.synsets[key])

        print("\n\nHypernyms\t", "--" * 10)
        for key in self.hypernyms:
            print(key, self.hypernyms[key])

        print("\n\nHyponyms\t", "--" * 10)
        for key in self.meronyms:
            print(key, self.meronyms[key])

        print("\n\nHolonyms\t", "--" * 10)
        for key in self.holonyms:
            print(key, self.holonyms[key])

    def parse_tree(self, sentence):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentence)
        spacy.displacy.serve(doc, style="dep")


if __name__ == "__main__":
    preProcessData = PreProcessData()
    while True:
        print("Choose from the following options:")
        print("1. Tokenize sentence")
        print("2. Lemmatize sentence")
        print("3. Find POS tag")
        print("4. Get Parse tree")
        print("5. Get wordnet features")
        option = int(input("Option? "))
        if option in range(0, 7):
            sentence = input("Please enter the sentence ").lower()
            if option == 1:
                print(preProcessData.tokenise(sentence))
            elif option == 2:
                print(preProcessData.lematize(sentence))
            elif option == 3:
                print(preProcessData.pos_tagging(sentence, True))
            elif option == 4:
                print(preProcessData.parse_tree(sentence))
            elif option == 5:
                print(preProcessData.wordnet_features(sentence))
        else:
            print("Please select the correct option")
