# # core/utils/logger.py
# import logging
# from logging.handlers import RotatingFileHandler

# def init_logger(log_file="simulation.log"):
#     # 创建全局Logger
#     logger = logging.getLogger("AsyncSimulator")
#     logger.setLevel(logging.DEBUG)

#     # 文件处理器（自动轮转）
#     file_handler = RotatingFileHandler(
#         log_file, maxBytes=1e6, backupCount=3, encoding="utf-8"
#     )
#     file_formatter = logging.Formatter(
#         "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
#     )
#     file_handler.setFormatter(file_formatter)
    
#     # 控制台处理器
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.INFO)
#     console_formatter = logging.Formatter("%(levelname)s - %(message)s")
#     console_handler.setFormatter(console_formatter)

#     # 添加处理器
#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)
    
#     return logger

# # 初始化全局Logger实例
# # sim_logger = init_logger()

import os
import logging
from logging.handlers import RotatingFileHandler

import yaml

# 获取项目根目录（与src同级）
def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件路径（utils/logger.py）
    return os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))      # 上三级到项目根目录

def get_log_file_path():
    # 1. 构建日志目录路径
    project_root = get_project_root()
    logs_dir = os.path.join(project_root, "logs")

    # 2. 自动创建日志目录（网页4）
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)

    config_file = os.path.join(project_root, "configs", "sim_config.yml")
    with open(config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)

    log_file = os.path.join(logs_dir, f'm-{config["model"]}_d-{config["dataset"]}.log')
    return log_file

# 配置全局日志
def init_logger():    
    # 配置日志处理器
    logger = logging.getLogger("Sim")
    logger.setLevel(logging.DEBUG)
    
    # 文件处理器（轮转日志）
    log_file = get_log_file_path()
    file_handler = RotatingFileHandler(
        log_file, maxBytes=1e6, backupCount=3, encoding="utf-8"
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 统一格式
    # formatter = logging.Formatter(
    #     "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    # ) # 太长了
    formatter = logging.Formatter(
        "%(name)s | %(levelname)s | %(message)s"
    )
    file_handler.setFormatter(formatter)
    # console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.handlers.clear()  # 清除所有已有处理器
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)
    return logger

# 构建全局 logger
sim_logger = init_logger()