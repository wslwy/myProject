from queue import PriorityQueue 
import os
import copy

from .server import Server
from .client import Client
from .utils.logger import sim_logger

from ..data.dataHelper import DataHelper
from ..model.modelHelper import ModelHelper
from ..cache.cacheHelper import CacheHelper

class Simulator:
    def __init__(self, cNum: int):
        self.clientNum = cNum

        self.logger = sim_logger.getChild("Simulator")
        self.logger.info(f"Initializing {cNum} clients")
        return
    
    def init_client(self, idx):
        model_config_path = "/data0/wyliang/async_simulator/configs/model_config.yml"
        model_helper = ModelHelper(model_config_path, "cpu", "resnet101", "ucf101")

        class_num = 50  # 101
        num_per_class = 10   #  10000   # 足够大
        batch_size = 60
        step = 5
        data_config_path = "/data0/wyliang/async_simulator/configs/data_config.yml"
        test_dir_list_file = os.path.join("/data0/wyliang/datasets/ucf101/ucfTrainTestlist", "testlist01.txt")
        data_helper = DataHelper(data_config_path)
        data_helper.load_data("ucf101", test_dir_list_file, 64, batch_size, "test", num_per_class, class_num, step)
        
        client_config_path = "/data0/wyliang/async_simulator/configs/cache_config.yml"
        cache_helper = CacheHelper(client_config_path, "local", "resnet101", "ucf101", 50, 0.01, 60)

        return Client(idx, model_helper, data_helper, cache_helper)
    
    def init_server(self):
        server = Server()

        client_config_path = "/data0/wyliang/async_simulator/configs/cache_config.yml"
        cache_helper = CacheHelper(client_config_path, "global", "resnet101", "ucf101", 101, 0.008, 60)
        server.cache_helper = cache_helper

        return server
    
    def simulate(self):
        self.server = self.init_server()
        self.clientList = [self.init_client(idx) for idx in range(self.clientNum)]

        self.executeQueue = PriorityQueue()
        for idx, client in enumerate(self.clientList):
            self.executeQueue.put((client.timeStamp, idx))


        # 变化 不同缓存层 与准确率，平均推理时间的验证
        sign_id_lists = [
            [],
            [9],
            [17],
            [25],
            [33],
            [17, 33],
            [25, 33],
            [17, 25],
            [9, 25],
            [17, 25, 33],
            [ 9, 17, 33],
            [ 9, 17, 25, 33],
            [ 5,  9, 13, 17, 21, 25, 29, 33],
            [ 3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33],
            list(range(34))
        ]

        # 测试对比两个
        # sign_id_lists = [ sign_id_lists[0], sign_id_lists[8], sign_id_lists[11], sign_id_lists[12] ]
        sign_id_lists = [ sign_id_lists[11] ]

        sign_lists = []
        base_sign_idx_list = []
        for idx, x in enumerate(self.server.cache_helper.cache.cache_sign_list):
            if x == 1:
                base_sign_idx_list.append(idx)
        # print(base_sign_idx_list)

        for sign_id_list in sign_id_lists:
            sign_list = [0] * len(self.server.cache_helper.cache.cache_sign_list)
            for idx in sign_id_list:
                sign_list[base_sign_idx_list[idx]] = idx + 1
            sign_lists.append(sign_list)    
        
        for idx, sign_list in enumerate(sign_lists):
            self.clientList[0].cache_helper.cache.freq_table = copy.deepcopy(self.server.cache_helper.cache.freq_table)
            self.clientList[0].cache_helper.cache.cache_sign_list = sign_list

        while not self.executeQueue.empty():
            time, taskClientIdx = self.executeQueue.get()
            taskClient = self.clientList[taskClientIdx]

            # 分配缓存
            self.server.cache_helper.allocate_cache(taskClient.cache_helper, 50)

            # 本地推理
            t, epc_acc, sample_num = taskClient.step_cache_infer()
            
            # 打印信息
            # print(taskClient.ID, taskClient.step, t, "ms")
            self.logger.info(f"{taskClient.step}, acc:{epc_acc}, step_time:{taskClient.timeStamp/sample_num:>10.3f}, all_time:{taskClient.all_time/sample_num:>10.3f} ms")

            if taskClient.step > 0:
                self.executeQueue.put((taskClient.timeStamp, taskClientIdx))


if __name__ == "__main__":
    simulator = Simulator(1)
    simulator.simulate()

