# use with vars(Parameters()) if need dict
class NetworkParameters:
    def __init__(self):
        # anchor setup
        self.x_size = 512
        self.y_size = 256
        self.resize_ratio = 8
        self.grid_x = self.x_size // self.resize_ratio  # 64
        self.grid_y = self.y_size // self.resize_ratio  # 32
        self.feature_size = 4  # feature size in similarity matrix in instance layer

        # post processsing
        self.threshold_confidence = 0.81
        self.threshold_instance = 0.22
        # self.grid_location = np.zeros((self.grid_y, self.grid_x, 2))  # anchor template
        # for y in range(self.grid_y):
        #     for x in range(self.grid_x):
        #         self.grid_location[y][x][0] = x
        #         self.grid_location[y][x][1] = y
