import torch
import torch.nn as nn
import torchvision.models as models

import os
import yaml
import numpy as np
import time
from collections import defaultdict

from .utils.load_model import load_model, split_model
from ..data.dataHelper import DataHelper
from ..cache.cacheHelper import CacheHelper

img_size = 224
filter_time = 0.1

class ModelHelper:
    def __init__(self, config_file, device: str, model_type: str, dataset_type: str):
        self.load_config(config_file)

        self.device = device
        self.model_type = model_type
        self.dataset_type = dataset_type

        self.model = self._load_model()
        self.model_list = self._split_model()

        # 初始化
        for sub_model in self.model_list:
            sub_model.eval()
            sub_model.to(device)

        # 定义提取中间向量语义表示的 GAP 层
        self.gap_layer = nn.AdaptiveAvgPool2d(1).to(device)

        return
    
    def load_config(self, config_file):
        # 读取配置文件
        with open(config_file, 'r') as config_file:
            config = yaml.safe_load(config_file)
            self.model_weights_dir = config["model_weights_dir"]

    def _load_model(self):
        """加载预训练模型"""
        return load_model(self.model_weights_dir, self.device, self.model_type, self.dataset_type)
    
    def _split_model(self):
        """按策略切分模型（如分割客户端/服务端部分）"""
        # 调用 splitter.py 中的算法
        return split_model(self.model, self.model_type)
    
    def cache_forward(self, cache_helper, x, cache_size, cache_update):
        """
        根据切分好的模型进行带缓存的推理
        """
        # print(f"length: {len(cache.cache_table[0])}")
        hit = 0
        layer_idx = 0
        scores = np.zeros(cache_size, dtype=float)
        up_data = dict()

        for idx, sub_model in enumerate(self.model_list):
            if idx == len(self.model_list) - 1:
                x = torch.flatten(x, 1)
            x = sub_model(x)

            if cache_helper.cache.cache_sign_list[idx]:
                vec = self.gap_layer(x)
                vec = vec.squeeze()
                weight = 1 << layer_idx
                if cache_update:
                    tmp = vec.detach().numpy()
                    tmp = tmp / np.linalg.norm(tmp) 
                    up_data[idx] = tmp

                hit, pred_id = cache_helper.match_layer(vec, cache_helper.cache.cache_table[idx], scores, weight)
                if hit:
                    x = pred_id
                    break
                layer_idx += 1

        return idx, hit, x, up_data
    
    def warm_up(self, cache_helper):
        # warm up
        warm_up_epoch = 100
        total_time = 0.0

        print('warm up ...')
        dummy_input = torch.rand(1, 3, img_size, img_size).to(self.device)
        for i in range(warm_up_epoch):
            start_time = time.perf_counter()
            _ = self.cache_forward(cache_helper.cache, dummy_input, cache_helper.cache.cache_size, False)
            end_time = time.perf_counter()

            total_time += end_time - start_time
        avg_time = total_time / warm_up_epoch
        print(f"warm up avg time: {avg_time * 1000:.3f} ms")

    def cache_step_infer(self, cache_helper, data_helper, cache_update, correct, total_num):
        # 正式推理
        total_time = 0.0
        layers_hits = defaultdict(int)
        layers_correct = defaultdict(int)
        layers_sum_time = defaultdict(float)
        filtered_num = 0
        
        
        data_block = data_helper.next_iter()
        if data_block is None:
            return None
        else:
            (data, labels) = data_block

        for x, y in zip(data, labels):
            # print(f"label: {y}")
            for idx in range(len(cache_helper.cache.ts_table)):
                cache_helper.cache.ts_table[idx] += 1
                cache_helper.cache.ts_table[y] = 0

            x = x.unsqueeze(0)
            start_time = time.perf_counter()
            hit_idx, hit, res, up_data = self.cache_forward(cache_helper, x, cache_helper.cache.cache_size, cache_update)
            end_time = time.perf_counter()

            sample_time = end_time - start_time

            if hit:
                pred = cache_helper.cache.id2label[res]
            else:
                # # loss 部分是否需要保留
                # loss = criterion(res, y.unsqueeze(0))
                # test_loss = test_loss + loss.item()

                pred = torch.max(res, 1)[1]

            test_correct = (pred == y).sum().item()

            if sample_time < filter_time:
                total_time += sample_time
                correct = correct + test_correct

                # 添加额外详细记录信息
                if hit:
                    layers_hits[hit_idx] += 1
                    layers_correct[hit_idx] += test_correct
                    layers_sum_time[hit_idx] += sample_time * 1000

                add_str = ""
            else:
                filtered_num += 1
                add_str = "### filtered"
            
            # 缓存更新部分
            if not hit and cache_update:
                for idx, sign in enumerate(cache_helper.cache.cache_sign_list):
                    if sign:    # 将缓存更新暂存
                        cache_helper.cache.up_cache_table[idx][pred] += up_data[idx]
                        cache_helper.cache.up_freq_table[idx][pred] += 1

        # # 将暂存的缓存写入全局缓存
        # for idx, sign in enumerate(local_cache.cache_sign_list):
        #     if sign:
        #         for label in range(global_cache.cache_size):
        #             if local_cache.up_freq_table[idx][pred]:
        #                 global_cache.cache_table[idx][label] = update_equation(global_cache.cache_table[idx][label], global_cache.up_freq_table[idx][label], local_cache.up_cache_table[idx][label], local_cache.up_freq_table[idx][label])
        #                 global_cache.up_freq_table[idx][label] += local_cache.up_freq_table[idx][label]

        # local_cache.update_table_clear()

        # print(f"ts_table: {local_cache.ts_table}")


        # epc_acc = float(correct) / (cache_helper.W * (data_helper.epc + 1) - filtered_num)
        epc_acc = float(correct) / (total_num + cache_helper.W - filtered_num)

        return correct, total_time, epc_acc, (total_num + cache_helper.W - filtered_num)
    
    


if __name__ == "__main__":
    # 构建配置文件的绝对路径
    config_path = "/data0/wyliang/async_simulator/configs/model_config.yml"

    modelHelper = ModelHelper(config_path, "cpu", "resnet101", "ucf101")

    print(modelHelper.model_type)
    print(modelHelper.dataset_type)
    print(modelHelper.model)
    print(len(modelHelper.model_list))