# use with vars(Parameters()) if need dict
class Parameters:
    def __init__(self):
        # train setup
        self.batch_size = 8
        self.num_epochs = 700
        self.learning_rate = 0.0001
        self.weight_decay = 0
        self.l_rate = 0.0001
        self.weight_decay = 0
        self.save_path = "tmp/"
        self.model_path = "tmp/"
        self.validate_epochs = 10

        # data loader setup
        self.dataset_name = "tusimple"
        self.flip_ratio = 0.4
        self.translation_ratio = 0.6
        self.rotate_ratio = 0.6
        self.noise_ratio = 0.4
        self.intensity_ratio = 0.4
        self.shadow_ratio = 0.6
        self.scaling_ratio = 0.2
        self.train_root_url = "/media/zzhou/data-tusimple/lane_detection/train_set/"
        self.train_json_file = ['label_data_0313.json', 'label_data_0531.json', 'label_data_0601.json']
        self.val_root_url = "/media/zzhou/data-tusimple/lane_detection/test_set/"
        self.val_json_file = ["test_tasks_0627.json"]
        self.test_root_url = "/media/zzhou/data-tusimple/lane_detection/test_set/"
        self.test_json_file = ["test_tasks_0627.json"]

        # anchor setup
        self.x_size = 512
        self.y_size = 256
        self.resize_ratio = 8
        self.grid_x = self.x_size // self.resize_ratio  # 64
        self.grid_y = self.y_size // self.resize_ratio  # 32
        self.feature_size = 4  # feature size in similarity matrix in instance layer

        # loss function
        self.K1 = 1.0
        self.K2 = 2.0
        self.constant_offset = 1.0
        self.constant_exist = 1.0  # 2
        self.constant_nonexist = 1.0  # 1.5 last 200epoch
        self.constant_angle = 1.0
        self.constant_similarity = 1.0
        self.constant_alpha = 1.0  # in SGPN paper, they increase this factor by 2 every 5 epochs
        self.constant_beta = 1.0
        self.constant_gamma = 1.0
        self.constant_back = 1.0
        self.constant_l = 1.0
        self.constant_lane_loss = 1.0  # 1.5 last 200epoch
        self.constant_instance_loss = 1.0

        # post processsing
        self.threshold_confidence = 0.81
        self.threshold_instance = 0.22
        # self.grid_location = np.zeros((self.grid_y, self.grid_x, 2))  # anchor template
        # for y in range(self.grid_y):
        #     for x in range(self.grid_x):
        #         self.grid_location[y][x][0] = x
        #         self.grid_location[y][x][1] = y

        # misc
        self.last_model_path = "tmp/"
        self.color = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
                      (255, 255, 255), (100, 255, 0), (100, 0, 255), (255, 100, 0), (0, 100, 255), (255, 0, 100),
                      (0, 255, 100)]
