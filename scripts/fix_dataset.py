#########################################################################
##
##  Data loader source code for TuSimple dataset
##
#########################################################################
import json
import random
from configs.PINet.dataset_tusimple import DatasetParameters as Parameters


#########################################################################
## Data loader class
#########################################################################
class Generator(object):
    ################################################################################
    ## initialize (load data set from url)
    ################################################################################
    def __init__(self):
        self.p = Parameters()

        # load annotation data (training set)
        self.train_data = []
        self.test_data = []

        with open(self.p.train_root_url + 'label_data_0313.json') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                jsonString = json.loads(line)
                self.train_data.append(jsonString)

        random.shuffle(self.train_data)

        with open(self.p.train_root_url + 'label_data_0531.json') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                jsonString = json.loads(line)
                self.train_data.append(jsonString)

        random.shuffle(self.train_data)

        with open(self.p.train_root_url + 'label_data_0601.json') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                jsonString = json.loads(line)
                self.train_data.append(jsonString)

        random.shuffle(self.train_data)

        self.size_train = len(self.train_data)
        print(self.size_train)

        # load annotation data (test set)
        # with open(self.p.test_root_url+'test_tasks_0627.json') as f:
        with open(self.p.test_root_url + 'test_label.json') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                jsonString = json.loads(line)
                self.test_data.append(jsonString)

        # random.shuffle(self.test_data)

        self.size_test = len(self.test_data)
        print(self.size_test)

    def split(self):
        one = 0
        two = 0
        three = 0
        four = 0
        five = 0
        six = 0
        for i in range(self.size_train):
            if len(self.train_data[i]['lanes']) == 6:
                with open("tmp/six.json", 'a') as make_file:
                    six += 1
                    json.dump(self.train_data[i], make_file, separators=(',', ': '))
                    make_file.write("\n")
            if len(self.train_data[i]['lanes']) == 5:
                with open("tmp/five.json", 'a') as make_file:
                    five += 1
                    json.dump(self.train_data[i], make_file, separators=(',', ': '))
                    make_file.write("\n")
            if len(self.train_data[i]['lanes']) == 4:
                with open("tmp/four.json", 'a') as make_file:
                    four += 1
                    json.dump(self.train_data[i], make_file, separators=(',', ': '))
                    make_file.write("\n")
            if len(self.train_data[i]['lanes']) == 3:
                with open("tmp/three.json", 'a') as make_file:
                    three += 1
                    json.dump(self.train_data[i], make_file, separators=(',', ': '))
                    make_file.write("\n")
            if len(self.train_data[i]['lanes']) == 2:
                with open("tmp/two.json", 'a') as make_file:
                    two += 1
                    json.dump(self.train_data[i], make_file, separators=(',', ': '))
                    make_file.write("\n")
            if len(self.train_data[i]['lanes']) == 1:
                with open("tmp/one.json", 'a') as make_file:
                    one += 1
                    json.dump(self.train_data[i], make_file, separators=(',', ': '))
                    make_file.write("\n")
        print("six = " + str(six))
        print("five = " + str(five))
        print("four = " + str(four))
        print("three = " + str(three))
        print("two = " + str(two))
        print("one = " + str(one))
        print("total = " + str(one + two + three + four + five + six))


if __name__ == '__main__':
    G = Generator()
    G.split()