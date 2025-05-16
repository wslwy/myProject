# data/helper.py
import yaml
import random

from torch.utils.data import DataLoader

# ================== 路径配置 ==================
# 获取项目根目录（myproject）
# project_root = Path(__file__).resolve().parents[1]
# sys.path.append(str(project_root))
# print(str(sys.path))

import sys

# from pathlib import Path

# # 获取项目根目录（async_simulator）
# import sys
# # sys.path.append('/data/wyliang/async_simulator/')
# # sys.path.append('/data/wyliang/async_simulator/utils/')
# sys.path.append('./')
# sys.path.append('../')

# print(sys.path)
# 导入工具模块
# from utils.load_data import ImageDataset, Ucf101Dataset, get_cifar_100_dataset
from utils.load_data import ImageDataset, Ucf101Dataset, get_cifar_100_dataset



# core/data/helper.py 示例
class DataHelper:
    def __init__(self, config_path="configs/data_config.yaml"):
        self.config_path = config_path
        self.load_config(config_path)

    def load_config(self, config_path):
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

            self.server = config["server"]
            # print(config)
            if self.server == 407:
                self.cifar_datasets_root = config["datasets"][407]["cifar_datasets_root"]
                self.imagenet_1k_datasets_root = config["datasets"][407]["cifar_datasets_root"]
                self.imagenet_100_datasets_train_root = config["datasets"][407]["cifar_datasets_root"]
                self.imagenet_100_datasets_test_root = config["datasets"][407]["cifar_datasets_root"]
                self.ucf101_datasets_root = config["datasets"][407]["ucf101_datasets_root"]
            elif self.server in [402, 405]:
                self.ucf101_datasets_root = config["datasets"][405]["ucf101_datasets_root"]   

            self.default_image_size = config["default_image_size"]     
            
    def load_data(self, dataset='ucf101', img_dir_list_file=None, train_batch_size=64, test_batch_size=256, mode="test", num_per_class=300, num_class=100, step=20):
        # 设置随机数种子
        seed_value = 2024
        random.seed(seed_value)
        
        if dataset == "cifar-100":
            train_dataset, test_dataset = get_cifar_100_dataset()
        elif dataset == 'imagenet-1k':
            if mode == "train":
                train_dataset = ImageDataset(image_dir=self.imagenet_1k_datasets_root, image_size=self.default_image_size, mode=mode)
            else:
                test_dataset =ImageDataset(image_dir=self.imagenet_1k_datasets_root, image_size=self.default_image_size, mode=mode)
        elif dataset == 'imagenet-100':
            if mode == "train":
                train_dataset = ImageDataset(image_dir=self.imagenet_100_datasets_train_root, image_size=self.default_image_size, mode="test", num_per_class=num_per_class, num_class=num_class)
            else:
                test_dataset = ImageDataset(image_dir=self.imagenet_100_datasets_test_root, image_size=self.default_image_size, mode=mode, num_per_class=num_per_class, num_class=num_class)
        elif dataset == 'ucf101':
            if mode == "train":
                train_dataset = Ucf101Dataset(image_dir_list_file=img_dir_list_file, img_dir_root=self.ucf101_datasets_root, image_size=self.default_image_size, mode=mode, shuffle=True, num_per_class=num_per_class, num_class=num_class, step=step)
            else:
                test_dataset = Ucf101Dataset(image_dir_list_file=img_dir_list_file, img_dir_root=self.ucf101_datasets_root, image_size=self.default_image_size, mode=mode, shuffle=True, num_per_class=num_per_class, num_class=num_class, step=step)
        else:
            print("error, no dataset matched")

        # 创建训练集和测试集的数据加载器 (num_workers 看看是否需要设为1),根据需要定义合适的dataloader, shuffle是否为True
        if mode == "train":
            train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
            data_loader = train_loader
        else:
            test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
            data_loader = test_loader
        return data_loader
    

    
    def split_data(self, data, num_clients: int, strategy: str="iid"):
        """划分数据到多个客户端"""
        # 调用 partition.py 中的策略
    
    def distribute(self, clients):
        """分发数据分片到客户端"""


if __name__ == "__main__":
    import os

     # 构建配置文件的绝对路径
    config_path = "/data/wyliang/async_simulator/configs/data_config.yaml"

    dataHelper = DataHelper(config_path)
    if dataHelper.server == 407:
        train_dir_list_file = os.path.join("/data0/wyliang/datasets/ucf101/ucfTrainTestlist", "trainlist01.txt")
        test_dir_list_file = os.path.join("/data0/wyliang/datasets/ucf101/ucfTrainTestlist", "testlist01.txt")
    elif dataHelper.server in [402, 405]:
        train_dir_list_file = os.path.join("/data/wyliang/datasets/ucf101/ucfTrainTestlist", "trainlist01.txt")
        test_dir_list_file = os.path.join("/data/wyliang/datasets/ucf101/ucfTrainTestlist", "testlist01.txt")

    class_num = 50  # 101
    num_per_class = 10   #  10000   # 足够大
    batch_size = 60

    step = 5
    train_loader = dataHelper.load_data("ucf101", train_dir_list_file, 64, 256, "train", 15, class_num, 5)
    test_loader = dataHelper.load_data("ucf101", test_dir_list_file, 64, batch_size, "test", num_per_class, class_num, step)
    print(len(train_loader))
    print(len(train_loader.dataset))
    print(len(test_loader))
    print(len(test_loader.dataset))