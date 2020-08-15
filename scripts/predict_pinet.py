import time
import numpy as np
import cv2
import fire
from torchvision import transforms

from instances import LaneDetector
from dataset.transform_op import Resize, TransposeNumpyArray, NormalizeInstensity
from configs.PINet.network import NetworkParameters
from instances.postprocessing_pinet import draw_points


class PredictPINet:
    def __init__(self, model_path=None):
        print("example: python -m scripts.predict_pinet --model_path=./tmp/dummy.pth on_dataset")
        net_params = NetworkParameters()
        if model_path is None:
            self.model_path = "./data/pinet_collection_epoch-11.pth"
        else:
            self.model_path = model_path

        self.detector = LaneDetector(model_path=self.model_path, parameter=net_params)

    def on_tusimple(self):
        from dataset.tusimple import DatasetTusimple
        from configs.PINet import ParamsTuSimple
        dataset_param = ParamsTuSimple()
        resize = Resize(rows=256, cols=512)
        hwc_to_chw = TransposeNumpyArray((2, 0, 1))
        norm_to_1 = NormalizeInstensity()
        dataset = DatasetTusimple(root_path=dataset_param.train_root_url,
                                  json_files=dataset_param.train_json_file,
                                  transform=transforms.Compose([resize, norm_to_1, hwc_to_chw]),)

        for i in range(len(dataset)):
            sample = dataset[i]
            img_src = cv2.imread(sample["image_path"])
            out_x, out_y = self.detector.test_on_image(np.array([sample["image"]]))
            vis_image = draw_points(out_x[0], out_y[0], img_src)
            cv2.imshow("sample", vis_image)
            cv2.imshow("original image", img_src)
            if cv2.waitKey(0) == 27:
                break

    def on_culane(self, save=False):
        from dataset.culane import DatasetCULane
        from configs.PINet import ParamsCuLane
        dataset_param = ParamsCuLane()
        resize = Resize(rows=256, cols=512)
        hwc_to_chw = TransposeNumpyArray((2, 0, 1))
        norm_to_1 = NormalizeInstensity()
        dataset = DatasetCULane(root_path=dataset_param.train_root_url,
                                index_file=dataset_param.train_json_file,
                                transform=transforms.Compose([resize, norm_to_1, hwc_to_chw]),)

        if save:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter("./output.avi", fourcc, 20.0, (1640, 590))
        else:
            out = None

        for i in range(len(dataset)):
            sample = dataset[i]
            img_src = cv2.imread(sample["image_path"])
            out_x, out_y = self.detector.test_on_image(np.array([sample["image"]]))
            vis_image = draw_points(out_x[0], out_y[0], img_src, scale_x=1640/512, scale_y=590/256)
            if save:
                out.write(vis_image)
            cv2.imshow("sample", vis_image)
            cv2.imshow("original image", img_src)
            if cv2.waitKey(1) == 27:
                break
        if save:
            out.release()

    def on_bdd100k(self):
        from dataset.bdd100k import DatasetBDD100K
        from configs.PINet import ParamsBDD100K
        dataset_param = ParamsBDD100K()
        resize = Resize(rows=256, cols=512)
        hwc_to_chw = TransposeNumpyArray((2, 0, 1))
        norm_to_1 = NormalizeInstensity()
        dataset = DatasetBDD100K(root_path=dataset_param.train_root_url,
                                 json_files=dataset_param.train_json_file,
                                 transform=transforms.Compose([resize, norm_to_1, hwc_to_chw]),)

        for i in range(len(dataset)):
            sample = dataset[i]
            img_src = cv2.imread(sample["image_path"])
            out_x, out_y = self.detector.test_on_image(np.array([sample["image"]]))
            vis_image = draw_points(out_x[0], out_y[0], img_src)

            cv2.imshow("sample", vis_image)
            cv2.imshow("original image", img_src)
            if cv2.waitKey(0) == 27:
                break

    def on_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while (cap.isOpened()):
            ret, frame = cap.read()
            rows, cols, _ = frame.shape
            prevTime = time.time()
            frame_small = cv2.resize(frame, (512, 256)) / 255.0
            frame_small = np.rollaxis(frame_small, axis=2, start=0)
            out_x, out_y = self.detector.test_on_image(np.array([frame_small]))
            vis_image = draw_points(out_x[0], out_y[0], frame, scale_x=cols / 512, scale_y=rows / 256)
            curTime = time.time()
            sec = curTime - prevTime
            fps = 1 / (sec)
            s = "FPS : " + str(fps)
            cv2.putText(vis_image, s, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            cv2.imshow('frame', vis_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def on_images(self, *images):
        print("model path: ", self.model_path)
        print("image path: ", images)
        for image_path in images:
            test_image = cv2.imread(image_path)
            rows, cols, _ = test_image.shape
            small_image = cv2.resize(test_image, (512, 256)) / 255.0
            small_image = np.rollaxis(small_image, axis=2, start=0)
            out_x, out_y = self.detector.test_on_image(np.array([small_image]))
            vis_image = draw_points(out_x[0], out_y[0], test_image, scale_x=cols / 512, scale_y=rows / 256)
            cv2.imshow("res", vis_image)
            cv2.imshow("img", test_image)
            cv2.waitKey(0)

    def on_folder(self, path, save=False, recursive=False):
        import os
        if recursive:
            filenames = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path)
                         for f in filenames if os.path.splitext(f)[1] == '.jpg' or os.path.splitext(f)[1] == ".png"]
        else:
            filenames = [os.path.join(path, f) for f in os.listdir(path) if f.endswith((".png", ".jpg"))]

        filenames.sort()

        out_stream = None

        for each in filenames:
            test_image = cv2.imread(each)
            rows, cols, _ = test_image.shape
            if save and (out_stream is None):
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                out_stream = cv2.VideoWriter("./output.avi", fourcc, 10.0, (cols, rows))

            small_image = cv2.resize(test_image, (512, 256)) / 255.0
            small_image = np.rollaxis(small_image, axis=2, start=0)
            out_x, out_y = self.detector.test_on_image(np.array([small_image]))
            vis_image = draw_points(out_x[0], out_y[0], test_image, scale_x=cols/512, scale_y=rows/256)  # nbatch = 1
            if save:
                out_stream.write(vis_image)
            cv2.imshow("img", vis_image)
            if cv2.waitKey(1) == 27:
                break
        if save:
            out_stream.release()


if __name__ == "__main__":
    fire.Fire(PredictPINet)