#########################################################################
#  2020
#  Author: Zhiliang Zhou
#########################################################################
import json
from typing import List, Dict
import cv2


class TuSimpleViewer:
    def __init__(self,
                 train_root_url="/media/zzhou/data-tusimple/lane_detection/train_set/",
                 test_root_url="/media/zzhou/data-tusimple/lane_detection/test_set/"):
        # load annotation data (training set)
        self.train_data: List[Dict] = []  # Dict keys "lanes", "h_samples", "raw_file"
        self.test_data: List[Dict] = []
        self.train_root_url = train_root_url
        self.test_root_url = test_root_url

        with open(self.train_root_url + 'label_data_0313.json') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                json_string = json.loads(line)
                self.train_data.append(json_string)

        with open(self.train_root_url + 'label_data_0531.json') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                json_string = json.loads(line)
                self.train_data.append(json_string)

        with open(self.train_root_url + 'label_data_0601.json') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                json_string = json.loads(line)
                self.train_data.append(json_string)

        self.size_train = len(self.train_data)
        print(self.size_train)

        # load annotation data (test set)
        # with open(self.p.test_root_url+'test_tasks_0627.json') as f:
        with open(self.test_root_url + 'test_label.json') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                json_string = json.loads(line)
                self.test_data.append(json_string)

        # random.shuffle(self.test_data)

        self.size_test = len(self.test_data)
        print(self.size_test)

    def browse_image(self):
        for label in self.train_data:
            img = cv2.imread(self.train_root_url + label["raw_file"])
            rows, cols = img.shape[:2]

            y_list = label["h_samples"]
            for x_list in label["lanes"]:
                for i in range(len(x_list)):
                    if x_list[i] > 0:
                        cv2.circle(img, (x_list[i], y_list[i]), 3, (255, 0, 0), -1)
                    cv2.line(img, (0, y_list[i]), (cols, y_list[i]), (0, 0, 255), 1)

            img_small = cv2.resize(img, (512, 256))

            x_step = 512 // 64
            for x in range(64):
                cv2.line(img_small, (x * x_step, 0), (x * x_step, rows), (0, 0, 255), 1)
            y_step = 256 // 32
            for y in range(32):
                cv2.line(img_small, (0, y * y_step), (cols, y * y_step), (0, 0, 255), 1)

            cv2.imshow("img", img)
            cv2.imshow("small", img_small)
            if cv2.waitKey() == 27:
                break


if __name__ == "__main__":
    viewer = TuSimpleViewer()
    viewer.browse_image()
