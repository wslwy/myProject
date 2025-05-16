
class ModelHelper:
    def __init__(self, config_path="configs/model_config.yaml"):
        self.config = self._load_config(config_path)
    
    def load_model(self, model_name: str):
        """加载预训练模型"""
        # 实现细节
    
    def split_model(self, model, strategy: str):
        """按策略切分模型（如分割客户端/服务端部分）"""
        # 调用 splitter.py 中的算法
    
    def inference(self, model_part, data):
        """执行模型推理"""