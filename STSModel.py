import sys


class CorpusReader:

    def __init__(self):
        self.data = []

    def read_data(self, input_file):
        with open(input_file, 'r', encoding='utf8') as data:
            lines = data.read().splitlines()
        for line in lines:
            self.data.append(line.split("\t"))


if __name__== "__main__":
    if len(sys.argv) != 2:
        print("Please provide the input file only")
        exit(0)
    input_file = sys.argv[1]
    reader = CorpusReader()
    reader.read_data(input_file)
    print(reader.data)

