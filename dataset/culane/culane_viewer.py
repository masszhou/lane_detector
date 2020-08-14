#########################################################################
#  2020
#  Author: Zhiliang Zhou
#########################################################################
from typing import List
import cv2
import numpy as np
import csv
from pathlib import Path
import fire
from dataset.types import Sample
from tqdm import tqdm


class CulaneViewer:
    def __init__(self,
                 root_url="/media/zzhou/data-culane/",
                 index_file="list/train.txt"):
        self.root_url = Path(root_url)
        self.index_file = Path(index_file)

        # mask for car hood
        carhood_mask = self.get_car_hood_mask()  # key=y, value=(xmin, xmax)

        # read index file
        self.train_files_path = []
        with open(self.root_url / self.index_file) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.train_files_path.append(line[1:-1])  # remove "/" at beginning, "\n" at end

        # read labels
        self.sample_list: List[Sample] = []
        pbar = tqdm(total=len(self.train_files_path))
        for each in self.train_files_path:
            image_path = self.root_url / each
            label_path = image_path.with_suffix(".lines.txt")
            lines = list(self._read_lines(label_path, mask=carhood_mask))
            self.sample_list.append({"image_path": image_path,
                                     "label_path": label_path,
                                     "lane_list": lines})
            pbar.update()
        pbar.close()

    def browse_image(self):
        for sample in self.sample_list:
            img = cv2.imread(sample["image_path"].absolute().as_posix())  # (590, 1640, 3)
            rows, cols = img.shape[:2]

            for lane in sample["lane_list"]:
                for pt in lane:
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)

            img_small = cv2.resize(img, (512, 256))

            for y in range(290, 600, 10):
                cv2.line(img, (0, y), (cols, y), (0, 0, 255), 1)

            x_step = 512 // 64
            for x in range(64):
                cv2.line(img_small, (x * x_step, 0), (x * x_step, rows), (0, 0, 255), 1)
            y_step = 256 // 32
            for y in range(32):
                cv2.line(img_small, (0, y * y_step), (cols, y * y_step), (0, 0, 255), 1)

            cv2.imshow("img", img)
            cv2.imshow("img_small", img_small)
            if cv2.waitKey() == 27:
                break

    @staticmethod
    def get_car_hood_mask():
        mask_y = list(range(430, 600, 10))
        mask_x_min = [718,  660,  610,  567,  531,  503,  478,  453,  434,  412,  386,  361,  342,  321,  295,  272,  249]
        mask_x_max = [1076, 1134, 1175, 1220, 1258, 1307, 1343, 1387, 1425, 1473, 1512, 1555, 1594, 1639, 1639, 1639, 1639]
        carhood_mask = {}  # key=y, value=(xmin, xmax)
        for i in range(len(mask_y)):
            x_min = mask_x_min[i] - 5  # add 5 pixels for margin
            x_max = mask_x_max[i] + 5 if mask_x_max[i] + 5 < 1639 else 1639  # add 5 pixels for margin
            carhood_mask[mask_y[i]] = (x_min, x_max)
        return carhood_mask

    @staticmethod
    def _read_lines(file_path, mask=None):
        with open(file_path) as f:
            reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
            for row in reader:
                row = row[:-1]  # remove a ghost space char
                if mask is None:
                    yield np.array([float(i) for i in row], dtype=float).reshape([-1, 2])
                else:
                    x_list = row[::2]
                    y_list = row[1::2]
                    filtered_list = []
                    for i in range(len(x_list)):
                        y = float(y_list[i])
                        x = float(x_list[i])
                        if y >= 430:
                            if mask[y][0] < x < mask[y][1]:
                                continue
                        filtered_list.append((x, y))
                    yield np.array(filtered_list, dtype=float)


if __name__ == "__main__":
    fire.Fire(CulaneViewer)