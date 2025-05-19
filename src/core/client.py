import random
from .utils.logger import sim_logger
import time

class Client:
    def __init__(self, idx: int, model_helper, data_helper, cache_helper):
        self.ID = idx
        self.timeStamp = 0.0
        self.all_time = 0.0
        self.step = len(data_helper.data_loader)

        self.logger = sim_logger.getChild(f"Client[{idx}]")  # 子Logger
        self.logger.info(f"Client {idx} initialized")

        self.model_helper = model_helper
        self.data_helper = data_helper
        self.cache_helper = cache_helper

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
        self.correct, step_time, epc_acc, self.sample_num  = self.model_helper.cache_step_infer(self.cache_helper, self.data_helper, False, self.correct, self.sample_num)
        end_time = time.perf_counter()

        sample_time = end_time - start_time

        step_time *= 1000
        sample_time *= 1000

        self.timeStamp += step_time
        self.all_time += sample_time

        self.logger.info(f"epc:{self.data_helper.epc}, {self.timeStamp:>10.3f} ms, {self.all_time:>10.3f}, acc:{epc_acc} ms")

        self.step -= 1
        return self.timeStamp, epc_acc, self.sample_num