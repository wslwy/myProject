import random
from .utils.logger import sim_logger

class Client:
    def __init__(self, idx: int):
        self.ID = idx
        self.timeStamp = 0.0
        self.step = 10

        self.logger = sim_logger.getChild(f"Client[{idx}]")  # 子Logger
        self.logger.info(f"Client {idx} initialized")
        return
    
    def step_forward(self):
        # self.logger.debug(f"Client {self.ID} Executing step_forward")
        num = random.random()
        self.step -= 1
        self.timeStamp += num