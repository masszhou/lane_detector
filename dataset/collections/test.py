#########################################################################
#  2020
#  Author: Zhiliang Zhou
#########################################################################
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import fire

from dataset.augmentation import Flip, Translate, Rotate, AddGaussianNoise, ChangeIntensity
from dataset.transform_op import Resize, TransposeNumpyArray, NormalizeInstensity
from dataset.collections import DatasetCollections
from dataset.utils import DrawLanemarks


class TestDatasetCULane:
    def __init__(self,):
        print("usage examples:")
        print("python -m dataset.culane.test sample")
        print("python -m dataset.culane.test batch")
        print("python -m dataset.culane.test batch shuffle=False")
        flip = Flip(1.0)
        translate = Translate(1.0)
        rotate = Rotate(1.0)
        add_noise = AddGaussianNoise(1.0)
        change_intensity = ChangeIntensity(1.0)
        resize = Resize(rows=256, cols=512)
        hwc_to_chw = TransposeNumpyArray((2, 0, 1))
        norm_to_1 = NormalizeInstensity()

        self.train_dataset = DatasetCollections(transform=transforms.Compose([flip,
                                                                              translate,
                                                                              rotate,
                                                                              add_noise,
                                                                              change_intensity,
                                                                              resize,
                                                                              norm_to_1,
                                                                              hwc_to_chw]), )

    def sample(self):
        render = DrawLanemarks(draw_line=True)
        chw_to_hwc = TransposeNumpyArray((1, 2, 0))
        for i in range(len(self.train_dataset)):
            each_sample = self.train_dataset[i]
            print(i, each_sample["set_id"], each_sample["image_path"])
            img_src = cv2.imread(each_sample["image_path"])
            each_sample = chw_to_hwc(each_sample)
            img = render(**each_sample)
            cv2.imshow("sample", img[:, :, ::-1])
            cv2.imshow("original image", img_src)
            if cv2.waitKey(0) == 27:
                break

    def batch(self, shuffle=True):
        train_generator = DataLoader(self.train_dataset,
                                     batch_size=8,
                                     shuffle=shuffle,
                                     collate_fn=self.train_dataset.collate_fn)
        for each_batch in train_generator:
            n_batch, chs, cols, rows = each_batch["image"].shape
            img_0 = each_batch["image"][0].transpose((1, 2, 0))
            detection_gt_0 = each_batch["detection_gt"][0]
            print(each_batch["set_id"])
            print(each_batch["image_id"])
            # fig, ax = plt.subplots(nrows=1, ncols=3)
            # ax[0].imshow(detection_gt_0[0])
            # ax[1].imshow(detection_gt_0[1])
            # ax[2].imshow(detection_gt_0[2])
            # plt.show()

            cv2.imshow("img_0", img_0[:, :, ::-1])
            cv2.imshow("confidence", detection_gt_0[0])
            if cv2.waitKey() == 27:
                break


if __name__ == "__main__":
    fire.Fire(TestDatasetCULane)