# use with vars(Parameters()) if need dict
class DatasetParameters:
    def __init__(self):
        # data loader setup
        self.dataset_name = "tusimple"
        self.train_root_url = "/media/zzhou/data-tusimple/lane_detection/train_set/"
        self.train_json_file = ['label_data_0313.json', 'label_data_0531.json', 'label_data_0601.json']
        self.val_root_url = "/media/zzhou/data-tusimple/lane_detection/test_set/"
        self.val_json_file = ["test_tasks_0627.json"]
        self.test_root_url = "/media/zzhou/data-tusimple/lane_detection/test_set/"
        self.test_json_file = ["test_tasks_0627.json"]

        # augmentation
        self.flip_ratio = 0.4
        self.translation_ratio = 0.6
        self.rotate_ratio = 0.6
        self.noise_ratio = 0.4
        self.intensity_ratio = 0.4
        self.shadow_ratio = 0.6
        self.scaling_ratio = 0.2
