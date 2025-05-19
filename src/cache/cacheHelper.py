import yaml
from .cache import Cache
import os

import numpy as np

class CacheHelper:
    def __init__(self, config_file, cache_state, model_type, dataset_type, cache_size, threshold, W):
        self.cache_state = cache_state
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.cache_size = cache_size
        self.th = threshold
        self.W = W

        self.load_config(config_file)
        self.load_cache()

        return

    def load_config(self, config_file):
        # 读取配置文件
        with open(config_file, 'r') as config_file:
            config = yaml.safe_load(config_file)
            cache_files_dir = config["cache_files_dir"]

            self.cache_file = os.path.join(cache_files_dir, config[self.dataset_type][self.model_type])
    
    def load_cache(self):
        if self.dataset_type == "ucf101":
            init_cache_size = 101
        elif self.dataset_type == "imagenet100":
            init_cache_size = 100
        
        if self.cache_state == "global":
            self.cache_size = init_cache_size
            self.cache = Cache(state="global", model_type=self.model_type, data_set=self.dataset_type, cache_size=init_cache_size)
            self.cache.load(self.cache_file)
        else:
            self.cache = Cache(state="local", model_type=self.model_type, data_set=self.dataset_type, cache_size=init_cache_size)
    
    def match_layer(self, vec, cache_array, scores, weight):
        # 余弦相似度统计表
        similarity_table = np.zeros(len(cache_array), dtype=float)

        # 标准化参考向量vec
        vec = vec.detach().numpy()
        vec = vec / np.linalg.norm(vec)
        
        # 以下两个步骤可以合并
        for idx, row in enumerate(cache_array):
            # 标准化cache中的向量(标准化步骤不一定需要)
            if np.linalg.norm(row) != 0:
                row = row / np.linalg.norm(row)
            else:
                # 在这里处理向量范数为零的情况，可以选择将向量保持为零向量或者采取其他操作
                pass

            similarity = np.dot(vec, row)
            similarity_table[idx] = similarity

        for idx in range(len(scores)):
            scores[idx] += weight * similarity_table[idx]

        # 找到数组中元素的排序索引
        sorted_indices = np.argsort(scores)

        # 最大值的索引是排序后的最后一个元素
        max_index = sorted_indices[-1]

        # 第二大值的索引是排序后的倒数第二个元素
        second_max_index = sorted_indices[-2]

        # 找到最大值和第二大值
        max_score = scores[max_index]
        second_score = scores[second_max_index]

        sep = (max_score - second_score) / second_score

        # cache 匹配信息
        # print(f"{id2label[max_index]}, {id2label[second_max_index]}, {max_score:.5f}, {second_score:.5f}, {sep:.5f}")
        if sep > self.th:
            # print("amazing, score is more than Threhold, cache hit")
            # print(f"{id2label[max_index]}, {id2label[second_max_index]}, {max_score:.5f}, {second_score:.5f}, {sep:.5f}")
            hit = 1
        else:
            hit = 0

        return hit, max_index

    def allocate_cache(self, local_cache_helper, cache_size):
        global_cache = self.cache
        local_cache = local_cache_helper.cache

        scores = np.zeros(global_cache.cache_size, dtype=float)

        for idx in range(global_cache.cache_size):
            scores[idx] = local_cache.freq_table[idx] * (0.25) ** np.floor(local_cache.ts_table[idx] / self.W)
        
        local_cache.id2label = np.argsort(scores)[::-1][:cache_size]

        # print(cache_size, len(local_cache.id2label))
        for layer in range(global_cache.cache_layer_num):
            for idx, label in enumerate(local_cache.id2label):
                local_cache.cache_table[layer][idx] = global_cache.cache_table[layer][label]

        # 检查缓存分配结果
        # print(local_cache.id2label, local_cache.cache_table)
        return 


    def display(self):
        self.cache.display_info()

if __name__ == "__main__":
     # 构建配置文件的绝对路径
    config_path = "/data0/wyliang/async_simulator/configs/cache_config.yml"

    cacheHelper = CacheHelper(config_path, "local", "resnet101", "ucf101", 101, 0.008, 60)

    cacheHelper.display()
