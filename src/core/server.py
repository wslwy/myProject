from .utils.logger import sim_logger

class Server:
    def __init__(self):
        self.logger = sim_logger.getChild("Server")
        self.logger.info("Server started")
        return
    
    