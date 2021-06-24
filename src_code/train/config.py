from easydict import EasyDict as edict

cfg = edict()
__C = cfg

__C.train_data_path   = "/home/ubuntu/accident_detection/data/train_data/itb100Hz/train_X.pkl"
__C.test_data_path    = "/home/ubuntu/accident_detection/data/test_data/itb100Hz/test_X.pkl"
__C.train_label       = "/home/ubuntu/accident_detection/data/train_data/itb100Hz/train_y.pkl"
__C.test_label        = "/home/ubuntu/accident_detection/data/test_data/itb100Hz/test_y.pkl"

__C.batch_size        = 100
__C.aspect_ratio      = 1
__C.n_class           = 1
__C.num_epochs        = 2
__C.learn_rate        = 0.001
__C.weight_decay      = 0.01
__C.channel           = 3
__C.resnet_layer      = [1, 1, 1, 1]

__C.training_model_path  = "/home/ubuntu/accident_detection/model/training_model/"
__C.infer_model_path  = "/home/ubuntu/accident_detection/model/infer_model/"
