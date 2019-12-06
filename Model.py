import sys

import pandas as pd
import sklearn
from joblib import load, dump
from sklearn.ensemble import RandomForestClassifier


class Model:
    def __init__(self):
        self.features = ""

    def model(self, file, output, predict):
        self.feature = pd.read_csv(file)
        random_forest = RandomForestClassifier(300)
        labels = []
        try:
            labels = [int(i) for i in self.feature['label'].tolist()]
        except Exception as e:
            pass
        ids = self.feature["id"]
        X_features = self.feature.drop(["id", "header"], axis=1)
        try:
            X_features = self.feature.drop(["id", "header", "label"], axis=1)
        except Exception as e:
            pass
        if predict:
            random_forest = load('STSModel.joblib')
            scaler = sklearn.preprocessing.StandardScaler()
            scaler.fit(X_features)
            X_features_x = scaler.transform(X_features)
            res = random_forest.predict(X_features_x)
            with open(output, 'w') as f:
                for i in range(len(res)):
                    f.write("{}\t{}\n".format(ids[i], int(round(res[i]))))
            if labels:
                with open('gold.csv', 'w') as f:
                    for i in range(len(res)):
                        f.write("{}\t{}\n".format(ids[i], labels[i]))
        else:
            scaler = sklearn.preprocessing.StandardScaler()
            scaler.fit(X_features)
            X_features_x = scaler.transform(X_features)
            random_forest.fit(X_features_x, labels)
            dump(random_forest, 'STSModel.joblib')

if __name__ == '__main__':
    predict = sys.argv[1].strip().lower() == 'predict'
    output_file = ""
    if predict:
        if len(sys.argv) != 4:
            print("Please provide 'train' or 'predict', the Feature file, and result output file")
            exit(0)
        else:
            input_file = sys.argv[2]
            output_file = sys.argv[3]
    else:
        if len(sys.argv) != 3:
            print("Please provide 'train' or 'predict' and feature file the Feature file")
            exit(0)
        else:
            input_file = sys.argv[2]
    model_o = Model()
    model_o.model(input_file, output_file, predict)