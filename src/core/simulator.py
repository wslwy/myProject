from queue import PriorityQueue 

from .server import Server
from .client import Client
from .utils.logger import sim_logger

class Simulator:
    def __init__(self, cNum: int):
        self.clientNum = cNum

        self.logger = sim_logger.getChild("Simulator")
        self.logger.info(f"Initializing {cNum} clients")
        return
    
    def simulate(self):
        self.server = Server()
        self.clientList = [Client(idx) for idx in range(self.clientNum)]

        self.executeQueue = PriorityQueue()
        for idx, client in enumerate(self.clientList):
            self.executeQueue.put((client.timeStamp, idx))

        while not self.executeQueue.empty():
            time, taskClientIdx = self.executeQueue.get()
            taskClient = self.clientList[taskClientIdx]
            taskClient.step_forward()
            
            # 打印信息
            print(taskClient.ID, taskClient.step, time, "ms")
            self.logger.info(f"{taskClient.ID}, {taskClient.step}, {time:>6.3f} ms")

            if taskClient.step > 0:
                self.executeQueue.put((taskClient.timeStamp, taskClientIdx))


if __name__ == "__main__":
    simulator = Simulator(4)
    simulator.simulate()

