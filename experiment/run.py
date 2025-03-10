import os
import random
import numpy as np
import pandas as pd
import sys
from scipy.stats import norm
current_dir = os.path.dirname(__file__)  # 获取当前脚本所在目录（如果是文件中使用）
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))  # 父目录路径
sys.path.append(parent_dir) 

#print(parent_dir)

import argparse
import helper.io_util as io_util

def data_collect(path: str):
    traces = io_util.load(f'{path}/traces.pkl')
    return traces

parser = argparse.ArgumentParser()
parser.add_argument('--dataDir', type=str, default='../data')
parser.add_argument('--dataSet', type=str, default='A')
parser.add_argument('--saveDir', type=str, default='../../output')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--sampleRate', type=float, default=0.1)
args = parser.parse_args()


from collections import Counter
operation_counter = Counter()

def generate_random_value(key, results, sample_size=1):
    if key not in results:
        raise ValueError(f"Key '{key}' 未在 results 中找到！")
    mu = results[key]["mean"]
    sigma = results[key]["std"]
    random_values = norm.rvs(loc=mu, scale=sigma, size=sample_size)
    return random_values.tolist()

def test_generation_duration():

    
if __name__ == "__main__":
    os.makedirs(args.saveDir, exist_ok=True)

    traces = data_collect(f'{args.dataDir}/{args.dataSet}')
    
    #instance_operation_counter = {}
    duration_dict = {}
    for trace in traces:
        for span in trace.getSpans():
            instance = span.instance
            operation = span.operation
            #print(instance,operation,span.service)
            key = f"{instance}:{operation}"
            value = span.duration 
            #instance_operation_counter[key] = instance_operation_counter.get(key, 0) + 1
            if key not in duration_dict:
                duration_dict[key] = []
            duration_dict[key].append(value)    
    
    #sorted_counter = sorted(instance_operation_counter.items(), key=lambda x: -x[1])
                
    results = {}
    for key, durations in duration_dict.items():
        # 转换为 numpy 数组
        data = np.array(durations)
        # 拟合正态分布参数（均值和标准差）
        mu, sigma = norm.fit(data)
        # 存储结果
        results[key] = {
            "mean": mu,
            "std": sigma,
            "sample_size": len(durations),
            "data": data  # 可选：保留原始数据用于后续验证
        }

    for key, res in results.items():
        print(f"Key: {key}")
        print(f"  均值 : {res['mean']:.2f}")
        print(f"  标准差 : {res['std']:.2f}")
        print(f"  样本数量: {res['sample_size']}")
        print("-----------------------")
    
    test_generation_duration()
