# use with vars(Parameters()) if need dict
class TrainingParameters:
    def __init__(self):
        # train setup
        self.batch_size = 8
        self.num_epochs = 100
        self.save_path = "tmp/"
        self.model_path = "tmp/"
        self.validate_epochs = 10

        # optimizer
        self.optimizer_name = "adam"
        self.l_rate = 0.0001
        self.weight_decay = 0
        self.scheduler_gamma = 0.9
        self.scheduler_milestones = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]  # MultiStepLR scheduler

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
