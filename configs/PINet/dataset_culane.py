# use with vars(Parameters()) if need dict
class DatasetParameters:
    def __init__(self):
        # data loader setup
        self.dataset_name = "culane"
        self.train_root_url = "/media/zzhou/data-culane/"
        self.train_json_file = "list/train.txt"
        self.val_root_url = None
        self.val_json_file = None
        self.test_root_url = None
        self.test_json_file = None

        # augmentation
        self.flip_ratio = 0.4
        self.translation_ratio = 0.6
        self.rotate_ratio = 0.6
        self.noise_ratio = 0.4
        self.intensity_ratio = 0.4
        self.shadow_ratio = 0.6
        self.scaling_ratio = 0.2
