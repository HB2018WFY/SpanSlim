import os
import random
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, expon, gamma, weibull_min, lognorm, truncnorm
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

def isError(span):
    if args.dataSet in ['hipster-more', 'hipster-less']:
        if span.statusCode not in [0, 200, 302, np.nan, 1]:
            return True
    elif args.dataSet in ['media', 'socialNetwork']:
        if span.statusCode not in [200, 302, np.nan]:
            return True
    return False

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

def generate_random_value(span, results, l=0.0, r=float('inf')):
    instance = span.instance
    operation = span.operation
    key = f"{instance}:{operation}"
    dist_info = results[key]
    dist_type = dist_info["distribution"]
    params = dist_info["params"]
    
    # 参数校验
    if l > r:
        raise ValueError("参数错误：l 必须小于等于 r")
    
    if dist_type == "normal":
        mu = params["mu"]
        sigma = params["sigma"]
        # 截断正态分布
        if sigma == 0:
            sigma = 1e-9  # 设置极小值
        a = (l - mu) / sigma
        b = (r - mu) / sigma
        return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=1)[0]
    elif dist_type == "expon":
        lambd = params["lambda"]
        scale = 1 / lambd
        # 指数分布的上限 r 必须 >= 0，且自然范围是 [0, ∞)
        if l < 0:
            raise ValueError("指数分布的 l 必须 >= 0")
        # 截断到 [l, r]
        # 计算截断后的概率归一化因子
        Z = expon.cdf(r, scale=scale) - expon.cdf(l, scale=scale)
        if Z <= 0:
            raise ValueError(f"区间 [{l}, {r}] 与指数分布无重叠")
        # 生成均匀分布的随机数，反向采样
        u = np.random.uniform(expon.cdf(l, scale=scale), 
                             expon.cdf(r, scale=scale))
        return expon.ppf(u, scale=scale)
    elif dist_type == "gamma":
        shape = params["shape"]
        scale = params["scale"]
        if scale <= 0 or np.isnan(scale):
            if l <= 0 <= r:
                return 0.0
            else:
                raise ValueError("参数无效且 0 不在区间内")
        # Gamma 分布的自然范围是 [0, ∞)，截断到 [l, r]
        # 使用 scipy 的截断方法或自定义实现
        # 这里用 rejection sampling（效率可能较低，但保证正确性）
        while True:
            value = gamma.rvs(shape, scale=scale, size=1)[0]
            if l <= value <= r:
                break
        return value
    elif dist_type == "weibull_min":
        shape = params["shape"]
        scale = params["scale"]
        # Weibull 分布的自然范围是 [0, ∞)，截断到 [l, r]
        # 同样用 rejection sampling
        while True:
            value = weibull_min.rvs(shape, scale=scale, size=1)[0]
            if l <= value <= r:
                break
        return value
    elif dist_type == "lognorm":
        mu_log = params["mu_log"]
        sigma_log = params["sigma_log"]
        if mu_log <= 0 or sigma_log <= 0:
            if l <= 0 <= r:
                return 0.0
            else:
                raise ValueError("参数无效且 0 不在区间内")
        scale = np.exp(mu_log)
        # 对数正态分布的自然范围是 [0, ∞)，截断到 [l, r]
        # 使用 rejection sampling
        while True:
            value = lognorm.rvs(s=sigma_log, scale=scale, size=1)[0]
            if l <= value <= r:
                break
        return value
    else:
        raise ValueError(f"未知的分布类型: {dist_type}")

def duration_difference(dur1,dur2):
    if max(dur1,dur2) == 0:
        return 0
    else:
        #print(max(dur1,dur2))
        return abs((dur1-dur2))/max(dur1,dur2)
    
def test_Distribution(distName,traces,results):
    sum_diff=0.0
    sum_span=0
    sum_error_span=0
    for trace in traces:
        sum_span+=trace.getSpanNum()
        for span in trace.getSpans():
            if not isError(span):
                generated_duration = max(0,generate_random_value(span,results))
                sum_diff+=duration_difference(generated_duration,span.duration)
            else:
                sum_error_span+=1
    print(f"sum_span:{sum_span} sum_error_span:{sum_error_span} sum_normal_span:{sum_span-sum_error_span}")
    print(f"{distName} distribution similarity:{1-sum_diff/(sum_span-sum_error_span)}")    

def build_Distribution(distName,duration_dict):
    results = {}

    for key, durations in duration_dict.items():
        data = np.array(durations)
        #print(distName)
        if distName == "normal":
            # 正态分布：均值（mu）和标准差（sigma）
            mu, sigma = norm.fit(data)
            results[key] = {
                "distribution": "normal",
                "params": {
                    "mu": mu,
                    "sigma": sigma
                },
                "sample_size": len(durations),
                "data": data
            }
            #print(key,results[key])
        elif distName == "expon":
            # 指数分布：速率参数（lambda）
            # expon.fit 返回 (loc, scale)，其中 scale = 1/lambda，强制 loc=0
            loc, scale = expon.fit(data, floc=0)

            epsilon = 1e-9  # 定义一个极小值（如 0.000000001）
            scale = max(scale, epsilon)  # 确保 scale 至少为 epsilon
            lambd = 1 / scale

            results[key] = {
                "distribution": "expon",
                "params": {
                    "lambda": lambd
                },
                "sample_size": len(durations),
                "data": data
            }
            
        elif distName == "gamma":
            # 伽马分布：形状参数（shape）和尺度参数（scale）
            # gamma.fit 返回 (shape, loc, scale)，强制 loc=0
            clean_data = [x for x in data if x > 0]
            # 设置初始猜测值（例如 shape=1, scale=mean(clean_data)）
            initial_guess = (1, 0, np.mean(clean_data))
            shape, loc, scale = gamma.fit(clean_data, floc=0)

            
            results[key] = {
                "distribution": "gamma",
                "params": {
                    "shape": shape,
                    "scale": scale
                },
                "sample_size": len(durations),
                "data": data
            }
            
        elif distName == "weibull_min":
            # 威布尔分布：形状参数（shape）和尺度参数（scale）
            # weibull_min.fit 返回 (shape, loc, scale)，强制 loc=0
            shape, loc, scale = weibull_min.fit(data, floc=0)
            results[key] = {
                "distribution": "weibull_min",
                "params": {
                    "shape": shape,
                    "scale": scale
                },
                "sample_size": len(durations),
                "data": data
            }
            
        elif distName == "lognorm":
            # 对数正态分布：对数均值（mu_log）和对数标准差（sigma_log）
            clean_data = np.array([x for x in data if x > 0])

            log_data = np.log(clean_data)
            mu_log = np.mean(log_data)
            sigma_log = np.std(log_data)

            results[key] = {
                "distribution": "lognorm",
                "params": {
                    "mu_log": mu_log,
                    "sigma_log": sigma_log
                },
                "sample_size": len(durations),
                "data": data
            }
        else:
            raise ValueError(f"未知的分布类型: {distName}")
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

def get_key(span):
    instance = span.instance
    operation = span.operation
    #print(instance,operation,span.service)
    key = f"{instance}:{operation}"
    return key

def make_durationDict(traces):
    duration_dict = {}
    for trace in traces:
        #print(f"trace.isError={trace.isError}")
        for span in trace.getSpans():
            #print(span.statusCode)
            span.key=get_key(span)
            key=span.key
            value = span.duration 
            #instance_operation_counter[key] = instance_operation_counter.get(key, 0) + 1
            if not isError(span):
                if span.key not in duration_dict:
                    duration_dict[key] = []
                duration_dict[key].append(value)    
    
    #print(duration_dict)
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
        #print(results)
        test_Distribution(dist,traces,results)
