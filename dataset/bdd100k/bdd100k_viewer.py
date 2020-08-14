#########################################################################
#  2020
#  Author: Zhiliang Zhou
#########################################################################
import copy
from tqdm import tqdm
import numpy as np
import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.image as mpimg
from matplotlib.path import Path  # efficient way to draw Bezier curve


class BDD100KViewer:
    def __init__(self, bdd_path="/media/local-datasets/data-BDD100K/bdd100k/"):
        self.bdd_path = bdd_path
        self.seg_label_path = bdd_path + "seg/color_labels/train/"
        self.img_path = bdd_path + "images/100k/train/"

        train_file = open(bdd_path+ "labels/bdd100k_labels_images_train.json", "r")
        self.train_data = json.load(train_file)
        self.n_imgs = len(self.train_data)
        self.sample_idx = 0

        dpi = 80
        w = 16
        h = 9
        self.fig = plt.figure(figsize=(w, h), dpi=dpi, frameon=False)
        self.ax = plt.gca()
        self.ax.axis('off')
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        # setup figure for saving image without margin
        # https://stackoverflow.com/questions/11837979/removing-white-space-around-a-saved-image-in-matplotlib/27227718
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        self.with_lane = False
        self.with_lanemask = False
        self.with_drivable = False
        self.with_objects = False
        self.with_image = True
        self.with_v_grid = False
        print("press n/p to browse images")
        print("press 1 to toggle lane line")
        print("press 2 to toggle lane line")
        print("press 3 to toggle drivable area")
        print("press 4 to toggle objects")
        print("press 5 to toggle image")
        print("press 6 to toggle vertical grid")
        print("press s to save")
        print("press q to quit")

        self.lane_colors = np.array([[0,   0,  0,   255],
                                    [255, 0,  0,   255],
                                    [0,   0,  255, 255]]) / 255

        self.lanemask_colors = np.array([[250, 170,  30, 255],          # solid
                                         [0,   255,   0, 255],          # dashed
                                         [0,     0, 255, 255],          # road curb
                                         [255,   0,   0, 255],          # vertical
                                         [50,    0, 100, 255]]) / 255   # other

        self.drivable_colors = np.array([[0, 0, 0, 255],
                                         [217, 83, 79, 255],
                                         [91, 192, 222, 255]]) / 255

        # color definition refers to https://github.com/ucbdrive/bdd-data/blob/master/bdd_data/label.py
        self.box_color_index = {"traffic light": np.array([250, 170,  30, 255])/255,
                                "traffic sign":  np.array([220, 220,   0, 255])/255,
                                "bus":           np.array([  0,  60, 100, 255])/255,
                                "car":           np.array([  0,   0, 142, 255])/255,
                                "person":        np.array([220,  20,  60, 255])/255,
                                "bike":          np.array([119,  11,  32, 255])/255,
                                "truck":         np.array([  0,   0,  70, 255])/255,
                                "motor":         np.array([  0,   0, 230, 255])/255,
                                "train":         np.array([  0,  80, 100, 255])/255,
                                "rider":         np.array([255,   0,   0, 255])/255}

        self.mean_rgb = np.array([0.27869505, 0.29261783, 0.2899581], dtype=np.float32)
        self.counter = 0.0

    def view(self):
        plt.connect('key_release_event', self.next_image)
        self.show_image()
        plt.show()

    def export_lanemask(self):
        out_path = self.bdd_path + "seg/lanemask_label/train/"

        sample = self.train_data[0]

        labels = sample["labels"]
        img_name = sample["name"]
        image_path = self.img_path + img_name
        img = mpimg.imread(image_path)
        im = np.array(img, dtype=np.uint8)

        im_mask = np.zeros_like(im)

        pbar = tqdm(total=len(self.train_data))
        for sample in self.train_data:
            labels = sample["labels"]
            img_name = sample["name"]
            im_mask.fill(0)
            plt.cla()
            self.ax.imshow(im_mask, interpolation='nearest', aspect='auto')
            self.draw_lanemask(labels, alpha=1.0)
            self.fig.savefig(out_path+img_name[:-4:]+".png", pad_inches=0, format="png")
            pbar.update()
        pbar.close()

    def calculate_mean(self):
        self.mean_rgb = np.zeros([3], dtype=np.float32)
        pbar = tqdm(total=len(self.train_data))
        self.counter = 0.0
        for sample in self.train_data:
            self.counter += 1
            img_name = sample["name"]
            image_path = self.img_path + img_name
            img = mpimg.imread(image_path)
            # incremental mean
            self.mean_rgb[0] = self.mean_rgb[0] + (np.mean(img[:, :, 0]) - self.mean_rgb[0]) / self.counter
            self.mean_rgb[1] = self.mean_rgb[1] + (np.mean(img[:, :, 1]) - self.mean_rgb[1]) / self.counter
            self.mean_rgb[2] = self.mean_rgb[2] + (np.mean(img[:, :, 2]) - self.mean_rgb[2]) / self.counter
            pbar.update()
        print(self.mean_rgb)
        pbar.close()

    def next_image(self, event):
        if event.key == 'n':
            self.sample_idx += 1
        elif event.key == 'p':
            self.sample_idx -= 1
        elif event.key == "1":
            self.with_lane = not self.with_lane
        elif event.key == "2":
            self.with_lanemask = not self.with_lanemask
        elif event.key == "3":
            self.with_drivable = not self.with_drivable
        elif event.key == "4":
            self.with_objects = not self.with_objects
        elif event.key == "5":
            self.with_image = not self.with_image
        elif event.key == "6":
            self.with_v_grid = not self.with_v_grid
        else:
            return

        self.sample_idx = max(min(self.sample_idx, len(self.train_data) - 1), 0)

        if self.show_image():
            plt.draw()
        else:
            self.next_image(event)

    def show_image(self):
        plt.cla()
        sample = self.train_data[self.sample_idx]
        labels = sample["labels"]
        img_name = sample["name"]
        print('-------------------------------  ID; {}, Image: {}'.format(self.sample_idx, img_name))
        self.fig.canvas.set_window_title(img_name)

        image_path = self.img_path + img_name
        img = mpimg.imread(image_path)
        im = np.array(img, dtype=np.uint8)  # (720, 1280, 3)
        if self.with_image:
            self.ax.imshow(im, interpolation='nearest', aspect='auto')
        else:
            im_mask = np.zeros_like(im)
            self.ax.imshow(im_mask, interpolation='nearest', aspect='auto')

        if self.with_lane:
            self.draw_lanes(labels)
        if self.with_drivable:
            self.draw_drivable(labels)
        if self.with_objects:
            self.draw_boxes(labels)
        if self.with_lanemask:
            self.draw_lanemask(labels)
        if self.with_v_grid:
            self.draw_v_gird(im.shape)
        return True

    @staticmethod
    def random_color():
        return np.random.rand(3)

    @staticmethod
    def get_lanes(label_list):
        return [label for label in label_list if 'poly2d' in label and label['category'][:4] == 'lane']

    @staticmethod
    def get_areas(label_list):
        return [label for label in label_list if 'poly2d' in label and label['category'] == 'drivable area']

    @staticmethod
    def get_boxes(label_list):
        return [label for label in label_list if 'box2d' in label and label['box2d'] is not None]

    def poly2patch(self, poly2d, closed=False, alpha=1., color=None, scale=1.0):
        """
        Efficient way to draw Bezier curve
        """
        moves = {'L': Path.LINETO,
                 'C': Path.CURVE4}
        verts = [pt for pt in poly2d['vertices']]
        codes = [moves[type] for type in poly2d["types"]]
        codes[0] = Path.MOVETO

        if closed:
            verts.append(verts[0])
            codes.append(Path.CLOSEPOLY)

        if color is None:
            color = self.random_color()

        return mpatches.PathPatch(Path(verts, codes),
                                  facecolor=color if closed else 'none',
                                  edgecolor=color,  # if not closed else 'none',
                                  lw=1 if closed else 2 * scale, alpha=alpha,
                                  antialiased=False, snap=True)

    def draw_lanes(self, labels, alpha=0.4):
        lanes = self.get_lanes(labels)
        print(f"===== draw_lanes {len(lanes)}")
        for lane in lanes:
            print(lane)
            if lane['attributes']['laneDirection'] == 'parallel':
                color = self.lane_colors[1]  # Red
            else:
                color = self.lane_colors[2]  # Blue

            # lane['poly2d'] is len=1 list
            # lane['poly2d'][0] is dict dict_keys(['vertices', 'types', 'closed'])
            patch = self.poly2patch(lane['poly2d'][0], closed=False, alpha=alpha, color=color, scale=1)
            self.ax.add_patch(patch)

    def draw_v_gird(self, hwc):
        rows, cols, _ = hwc
        for y in range(rows//3, rows, 10):
            line = mlines.Line2D([0, cols], [y, y])
            self.ax.add_line(line)

    def draw_lanemask(self, labels, alpha=0.9):
        lanes = self.get_lanes(labels)
        filtered_lanes = []
        duplicated_lanes = []

        for i in range(len(lanes)):
            lane_a = lanes[i]
            if lane_a in duplicated_lanes:
                continue
            # if lane is used, then skip

            # else check if this lane has duplicate one
            lane_merge = None
            for j in range(i+1, len(lanes)):
                lane_b = lanes[j]
                if lane_a['attributes']['laneStyle'] != lane_b['attributes']['laneStyle']:
                    continue
                elif lane_a['attributes']['laneType'] != lane_b['attributes']['laneType']:
                    continue
                elif lane_a['attributes']['laneType'] == "double yellow":
                    continue
                elif lane_a['attributes']['laneType'] == "double white":
                    continue
                elif lane_a['attributes']['laneDirection'] == 'vertical':
                    continue
                elif len(lane_a['poly2d'][0]['vertices']) != len(lane_b['poly2d'][0]['vertices']):
                    continue
                if lane_merge is not None:
                    continue

                bottom_x, bottom_y, top_x, top_y = self.get_dist_of_lanes(lane_a, lane_b)
                if bottom_x < 112 and bottom_y < 30:
                    # merge lanes when found the same lane
                    lane_merge = self.merge_lanes(lane_a, lane_b)
                    duplicated_lanes.append(lane_b)  # mark b as used

            if lane_merge is None:
                filtered_lanes.append(lane_a)
            else:
                filtered_lanes.append(lane_merge)

        print(f"===== draw_lanemask, filtered_lanes {len(filtered_lanes)}")
        for lane in filtered_lanes:
            print(lane)
            if lane['attributes']['laneDirection'] == 'parallel':
                if lane['attributes']['laneType'] == 'road curb':
                    color = self.lanemask_colors[2]
                    scale = 4.0
                elif lane['attributes']['laneStyle'] == 'dashed':
                    color = self.lanemask_colors[1]
                    scale = 5.0
                elif lane['attributes']['laneStyle'] == 'solid':
                    color = self.lanemask_colors[0]
                    scale = 5.0
                else:
                    color = self.lanemask_colors[4]
                    scale = 2.0
            else:
                color = self.lanemask_colors[3]
                scale = 3.0

            patch = self.poly2patch(lane['poly2d'][0], closed=False, alpha=alpha, color=color, scale=1.0)
            self.ax.add_patch(patch)

    @staticmethod
    def get_dist_of_lanes(lane_a, lane_b):
        pt1 = lane_a['poly2d'][0]['vertices'][0]  # begin pt
        pt2 = lane_a['poly2d'][0]['vertices'][-1]  # end pt
        if pt1[1] > pt2[1]:
            a_bottom_x, a_top_x = pt1[0], pt2[0]
            a_bottom_y, a_top_y = pt1[1], pt2[1]
        else:
            a_bottom_x, a_top_x = pt2[0], pt1[0]
            a_bottom_y, a_top_y = pt2[1], pt1[1]

        pt3 = lane_b['poly2d'][0]['vertices'][0]  # begin pt
        pt4 = lane_b['poly2d'][0]['vertices'][-1]  # end pt
        if pt3[1] > pt4[1]:
            b_bottom_x, b_top_x = pt3[0], pt4[0]
            b_bottom_y, b_top_y = pt3[1], pt4[1]
        else:
            b_bottom_x, b_top_x = pt4[0], pt3[0]
            b_bottom_y, b_top_y = pt4[1], pt3[1]

        return abs(a_bottom_x-b_bottom_x), abs(a_bottom_y-b_bottom_y), abs(a_top_x-b_top_x), abs(a_top_y-b_top_y)

    @staticmethod
    def merge_lanes(lane_a, lane_b):
        """
        only merge two lane with same number of key points
        :param lane_a:
        :param lane_b:
        :return:
        """
        pts_a = lane_a['poly2d'][0]['vertices']  # list of points
        pts_b = lane_b['poly2d'][0]['vertices']

        # align lane_a and lane_b
        dist_1 = (pts_a[0][0] - pts_b[0][0])**2 + (pts_a[0][1] - pts_b[0][1])**2
        dist_2 = (pts_a[0][0] - pts_b[-1][0]) ** 2 + (pts_a[0][1] - pts_b[-1][1]) ** 2
        if dist_1 > dist_2:
            pts_b = pts_b[::-1]

        pts_merge = [[(pt_a[0]+pt_b[0])/2, (pt_a[1]+pt_b[1])/2] for pt_a, pt_b in zip(pts_a, pts_b)]
        lane_c = copy.deepcopy(lane_a)  # must deep copy
        lane_c['poly2d'][0]['vertices'] = pts_merge
        return lane_c

    def draw_drivable(self, labels, alpha=0.5):
        objects = self.get_areas(labels)

        for obj in objects:
            if obj['attributes']['areaType'] == 'direct':
                color = self.drivable_colors[1]
            else:
                # areaType == alternative
                color = self.drivable_colors[2]

            self.ax.add_patch(self.poly2patch(obj['poly2d'][0], closed=True, alpha=alpha, color=color))

    def draw_boxes(self, labels):
        boxes = self.get_boxes(labels)
        for box in boxes:
            self.ax.add_patch(self.box2rect(box['category'], box['box2d']))

    def box2rect(self, label_name, box2d, scale=1.0):
        """generate individual bounding box from label"""
        x1 = box2d['x1']
        y1 = box2d['y1']
        x2 = box2d['x2']
        y2 = box2d['y2']

        box_color = self.box_color_index[label_name]

        # Draw and add one box to the figure
        return mpatches.Rectangle((x1, y1),
                                  x2 - x1,
                                  y2 - y1,
                                  linewidth=3 * scale,
                                  edgecolor=box_color,
                                  facecolor='none',
                                  fill=False, alpha=0.75)


if __name__ == "__main__":
    print("usage: python -m [module_name] [function_name] [parameters]")
    print("example: python -m data.datasets.BDD100k view --bdd_path=/media/zzhou/data-BDD100K/bdd100k/")
    import fire
    fire.Fire(BDD100KViewer)