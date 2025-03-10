import os
import random
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
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

def generate_duration_normalDistribution(key, results, sample_size=1):
    if key not in results:
        raise ValueError(f"Key '{key}' 未在 results 中找到！")
    mu = results[key]["mean"]
    sigma = results[key]["std"]
    random_values = norm.rvs(loc=mu, scale=sigma, size=sample_size)
    random_values = np.maximum(random_values, 0)
    return random_values.tolist()

def Distribution(distName,span,results):
    instance = span.instance
    operation = span.operation
    key = f"{instance}:{operation}"
    if distName=="normal":
        return generate_duration_normalDistribution(key,results,1)[0]

def duration_difference(dur1,dur2):
    if max(dur1,dur2) == 0:
        return 1
    else:
        #print(max(dur1,dur2))
        return abs((dur1-dur2))/max(dur1,dur2)
    
def test_Distribution(distName,traces,results):
    sum_diff=0.0
    sum_span=0
    for trace in traces:
        sum_span+=trace.getSpanNum()
        for span in trace.getSpans():
            sum_diff+=duration_difference(Distribution(distName,span,results),span.duration)
    print(f"{distName} distribution similarity:{1-sum_diff/sum_span}")    

def build_Distribution(distName,duration_dict):
    results={}
    if distName=="normal":
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
    return results

def show_distribution(duration_dict):
     for key, durations in duration_dict.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        #直方图 
        sns.histplot(
            durations,
            kde=True,
            ax=ax1,
            color="skyblue",
            edgecolor="black"
        )
        ax1.set_title(f"Duration Distribution for {key}")
        ax1.set_xlabel("Duration (ms)")
        ax1.set_ylabel("Frequency")
        plt.tight_layout()
        plt.show()

def make_durationDict(traces):
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
    return duration_dict

if __name__ == "__main__":
    os.makedirs(args.saveDir, exist_ok=True)

    traces = data_collect(f'{args.dataDir}/{args.dataSet}')
    
    #instance_operation_counter = {}
    duration_dict=make_durationDict(traces)

    #真实分布
    #show_distribution(duration_dict)  
    dists = [
        "normal",
        "expon",
        "gamma",
        "weibull_min",
        "lognorm"
    ]

    for dist in dists:
        results = build_Distribution(dist,duration_dict)
        test_Distribution(dist,traces,results)
