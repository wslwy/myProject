import random
from .utils.logger import sim_logger
import time

import logging
import os

class Client:
    def __init__(self, idx: int, model_helper, data_helper, cache_helper):
        self.ID = idx
        self.timeStamp = 0.0
        

        self.step = len(data_helper.data_loader)

        self.logger = sim_logger.getChild(f"Client[{idx}]")  # 子Logger
        self.logger.propagate = False  # 阻止传播到父Logger
        # 创建专属文件处理器
        parent_handlers = [h for h in sim_logger.handlers if isinstance(h, logging.FileHandler)]
        if parent_handlers:
            # 假设父Logger只有一个FileHandler
            parent_log_path = os.path.dirname(parent_handlers[0].baseFilename)
            # 构建子Logger路径
            file_handler = logging.FileHandler(os.path.join(parent_log_path, f"client_{idx}.log"))
        else:
            # 父Logger未配置文件处理器时的默认处理
            file_handler = logging.FileHandler(f"client_{idx}.log")
            
        file_handler.setFormatter(sim_logger.handlers[0].formatter)  # 沿用父Logger格式
        self.logger.addHandler(file_handler)

        self.logger.info(f"Client {idx} initialized")

        

        self.model_helper = model_helper
        self.data_helper = data_helper
        self.cache_helper = cache_helper

        self.cache_update = True
        self.hit_count = 0
        self.correct = 0
        self.sample_num = 0
        self.epc_acc_list = []
    

        return
    
    def step_cache_infer(self):
        # self.logger.debug(f"Client {self.ID} Executing step_forward")
        num = random.random()
        self.step -= 1
        self.timeStamp += num

        start_time = time.perf_counter()
        self.correct, acc, self.sample_num, step_hit_count  = self.model_helper.cache_step_infer(self.cache_helper, self.data_helper, self.cache_update, self.correct, self.sample_num)
        end_time = time.perf_counter()

        step_time = end_time - start_time

        step_time *= 1000

        self.timeStamp += step_time
        self.hit_count += step_hit_count

        self.logger.info(f"epc:{self.data_helper.epc}, {self.timeStamp:>10.3f} ms, avg_time: {self.timeStamp/self.sample_num:>10.3f} ms, acc:{self.correct * 1.0 / self.sample_num}, hit ratio:{self.hit_count * 1.0 / self.sample_num}, hit count/sample_num:{self.hit_count:>5}/{self.sample_num}")

        return self.timeStamp, acc, self.sample_num