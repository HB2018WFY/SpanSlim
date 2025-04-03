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
        #print(f"span.statusCode:{span.statusCode}")
        if span.statusCode not in [0, 200, 302, np.nan, 1]:
            return True
    elif args.dataSet in ['media', 'socialNetwork']:
        if span.statusCode not in [200, 302, np.nan]:
            return True
    return False

from collections import Counter
operation_counter = Counter()
'''
def generate_duration_normalDistribution(key, results, sample_size=1):
    if key not in results:
        raise ValueError(f"Key '{key}' 未在 results 中找到！")
    mu = results[key]["mean"]
    sigma = results[key]["std"]
    random_values = norm.rvs(loc=mu, scale=sigma, size=sample_size)
    random_values = np.maximum(random_values, 0)
    return random_values.tolist()
'''
def generate_random_value(span, results, l=0.0, r=float('inf')):
    instance = span.instance
    operation = span.operation
    key = f"{instance}:{operation}"
    dist_info = results[key]
    dist_type = dist_info["distribution"]
    params = dist_info["params"]
    
    if l > r:
        return 0.0
    
    # 参数校验
    if l < 0 and dist_type in ["expon", "gamma", "weibull_min", "lognorm"]:
        raise ValueError(f"{dist_type}分布的l必须>=0")
    
    if dist_type == "normal":
        mu = params["mu"]
        sigma = params["sigma"]
        if sigma == 0:
            sigma = 1e-9  # 设置极小值
        a = (l - mu) / sigma
        b = (r - mu) / sigma
        return truncnorm(a, b, loc=mu, scale=sigma).rvs()
    
    elif dist_type == "expon":
        lambd = params["lambda"]
        scale = 1 / lambd
        # 截断到 [l, r]
        cdf_l = expon.cdf(l, scale=scale)
        cdf_r = expon.cdf(r, scale=scale)
        if cdf_r - cdf_l < 1e-9:
            return 0
        u = np.random.uniform(cdf_l, cdf_r)
        return expon.ppf(u, scale=scale)
    
    elif dist_type == "gamma":
        shape = params["shape"]
        scale = params["scale"]
        if scale <= 0 or np.isnan(scale):
            if l <= 0 <= r:
                return 0
            else:
                raise 0
        # 使用截断伽马分布
        a = l / scale  # 截断参数转换（Gamma 分布的 loc=0）
        b = r / scale
        return gamma.ppf(np.random.uniform(gamma.cdf(a, shape),
                                           gamma.cdf(b, shape)),
                         shape) * scale
    
    elif dist_type == "weibull_min":
        shape = params["shape"]
        scale = params["scale"]
        # Weibull 分布的自然范围是 [0, ∞)，截断到 [l, r]
        cdf_l = weibull_min.cdf(l, shape, scale=scale)
        cdf_r = weibull_min.cdf(r, shape, scale=scale)
        if cdf_r - cdf_l < 1e-9:
            return 0
        u = np.random.uniform(cdf_l, cdf_r)
        return weibull_min.ppf(u, shape, scale=scale)
    
    elif dist_type == "lognorm":
        mu_log = params["mu_log"]
        sigma_log = params["sigma_log"]
        if sigma_log <= 0:
            sigma_log = 1e-9
        # 对数正态分布的截断：转换到正态分布空间
        lower = np.log(l) if l > 0 else -np.inf
        upper = np.log(r)
        a = (lower - mu_log) / sigma_log
        b = (upper - mu_log) / sigma_log
        log_val = truncnorm(a, b, loc=mu_log, scale=sigma_log).rvs()
        return np.exp(log_val)
    
    else:
        raise ValueError(f"未知的分布类型: {dist_type}")

def difference(dur1,dur2):
    if max(dur1,dur2) == 0:
        return 0
    else:
        #print(max(dur1,dur2))
        return abs((dur1-dur2))/max(dur1,dur2)

def dfs_rebuild_duration(node,childs,results,l=0.0, r=float('inf')):
    rebuild_duration=0.0
    sum_duration_diff=0.0
    son_duration=0.0  #son limit
    if node.getParentId()=='-1'or isError(node):
        rebuild_duration=node.duration
    else:
        for son in childs[node.getSpanId()]:
            if(isError(son)):
                son_duration+=son.duration
        rebuild_duration=max(0,generate_random_value(node,results,max(l,son_duration),r))     

    sum_duration_diff+=difference(rebuild_duration,node.duration)
    #print(f"duration:{node.duration},rebuild:{rebuild_duration}")
    free_duration=rebuild_duration-son_duration
    for son in childs[node.getSpanId()]:
        if(isError(son)):
            son_duration_diff,_=dfs_rebuild_duration(son,childs,results)
            sum_duration_diff+=son_duration_diff
        else:
            son_duration_diff,son_rebuild_duration=dfs_rebuild_duration(son,childs,results,0,free_duration)
            free_duration-=son_rebuild_duration
            sum_duration_diff+=son_duration_diff

    #print(sum_duration_diff)
    return sum_duration_diff,rebuild_duration

def dfs_rebuild_latency(node,childs,results,latency_dict,l=0.0, r=float('inf')):
    rebuild_latency=0.0
    sum_latency_diff=0.0
    son_duration_and_latency=0.0  #son limit
    if node.getParentId()=='-1'or isError(node):
        rebuild_latency=latency_dict[node.getSpanId()]
    else:
        for son in childs[node.getSpanId()]:
            if(isError(son)):
                son_duration_and_latency+=son.duration+latency_dict[son.getSpanId()]
        rebuild_latency=max(0,generate_random_value(node,results,max(l,son_duration_and_latency),r))     

    sum_latency_diff+=difference(rebuild_latency,latency_dict[node.getSpanId()])
    #print(f"latency:{latency_dict[node.getSpanId()]},rebuild:{rebuild_latency}")

    free_latency=node.duration-son_duration_and_latency
    for son in childs[node.getSpanId()]:
        if(isError(son)):
            son_latency_diff,_=dfs_rebuild_latency(son,childs,results,latency_dict)
            sum_latency_diff+=son_latency_diff
        else:
            son_latency_diff,son_rebuild_latency=dfs_rebuild_latency(son,childs,results,latency_dict,0,free_latency)
            free_latency-=son_rebuild_latency
            sum_latency_diff+=son_latency_diff
    return sum_latency_diff,rebuild_latency
                                          

def build_childs(trace):
    childs = {}
    for span in trace.getSpans():
        if span.spanId not in childs:
            childs[span.spanId] = []
        parent_id = span.getParentId()
        if parent_id != '-1':  # 非根节点
            if parent_id not in childs:
                childs[parent_id] = []
            childs[parent_id].append(span)
    
    # 对每个父节点的子节点列表按 startTime 排序
    for parent_id in childs:
        # 使用 lambda 表达式按 startTime 排序
        childs[parent_id].sort(key=lambda s: s.startTime)
    
    return childs


def test_Distribution(distName,traces,duration_results,latency_results):
    sum_duration_diff=0.0
    sum_latency_diff=0.0
    sum_span=0
    sum_error_span=0
    for trace in traces:
        sum_span+=trace.getSpanNum()
        root=trace.getRoot()
        childs=build_childs(trace)
        for span in trace.getSpans():
            if isError(span):
                sum_error_span+=1
        duration_diff,_=dfs_rebuild_duration(root,childs,duration_results)
        latency_diff,_=dfs_rebuild_latency(root,childs,latency_results,trace.latency_dict) #Todo
        sum_duration_diff+=duration_diff
        sum_latency_diff+=latency_diff

    print(f"sum_span:{sum_span} sum_error_span:{sum_error_span} sum_normal_span:{sum_span-sum_error_span}")
    print(f"{distName} distribution duration similarity:{1-sum_duration_diff/(sum_span-sum_error_span)}")    
    print(f"{distName} distribution latency similarity:{1-sum_latency_diff/(sum_span-sum_error_span)}")    

def build_Distribution(distName, value_dict):
    results = {}

    for key, values in value_dict.items():
        data = np.array(values)
        if distName == "normal":
            # 正态分布：均值（mu）和标准差（sigma）
            mu, sigma = norm.fit(data)
            results[key] = {
                "distribution": "normal",
                "params": {
                    "mu": mu,
                    "sigma": sigma
                },
                "sample_size": len(values),
                "data": data
            }
        elif distName == "expon":
            # 指数分布：速率参数（lambda）
            loc, scale = expon.fit(data, floc=0)
            epsilon = 1e-9  # 定义一个极小值（如 0.000000001）
            scale = max(scale, epsilon)  # 确保 scale 至少为 epsilon
            lambd = 1 / scale

            results[key] = {
                "distribution": "expon",
                "params": {
                    "lambda": lambd
                },
                "sample_size": len(values),
                "data": data
            }
        elif distName == "gamma":
            # 伽马分布：形状参数（shape）和尺度参数（scale）
            clean_data = [x for x in data if x > 0]
            initial_guess = (1, 0, np.mean(clean_data))
            shape, loc, scale = gamma.fit(clean_data, floc=0)
            
            results[key] = {
                "distribution": "gamma",
                "params": {
                    "shape": shape,
                    "scale": scale
                },
                "sample_size": len(values),
                "data": data
            }
        elif distName == "weibull_min":
            # 威布尔分布：形状参数（shape）和尺度参数（scale）
            shape, loc, scale = weibull_min.fit(data, floc=0)
            results[key] = {
                "distribution": "weibull_min",
                "params": {
                    "shape": shape,
                    "scale": scale
                },
                "sample_size": len(values),
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
                "sample_size": len(values),
                "data": data
            }
        else:
            raise ValueError(f"未知的分布类型: {distName}")
    return results
'''
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
'''
def get_key(span):
    instance = span.instance
    operation = span.operation
    #print(instance,operation,span.service)
    key = f"{instance}:{operation}"
    return key

def make_metric_dict(traces, metric_type):
    """
    根据传入的 metric_type 生成对应的字典：
    - 'duration': 收集 span 的 duration 值
    - 'latency': 收集 span 的 latency 值
    """
    metric_dict = {}
    for trace in traces:
        latency_dict=trace.latency_dict
        #print(len(latency_dict))
        for span in trace.getSpans():
            #print(span.getParentId(),span.latency)
            if isError(span):
                continue  # 跳过错误 span
            
            key = get_key(span)
            if metric_type == 'duration':
                value = span.duration
            elif metric_type == 'latency':
             #   print(span.parentSpanId)
                value = latency_dict[span.getSpanId()]  # 假设 span 有该属性
               # print(value)
              #  print(span.parentSpanId,value)
            else:
                raise ValueError(f"Unsupported metric_type: {metric_type}")
            
            if key not in metric_dict:
                metric_dict[key] = []
            metric_dict[key].append(value)
    
    return metric_dict

def build_latency(traces):
    latency_dict = {}  # 存储 spanId 到 latency 的映射
    for trace in traces:
        childs=build_childs(trace)
        for span in trace.getSpans():
            #print(span.getParentId())
            if span.getParentId() == '-1':
                #print(span.getSpanId(),"wa")
                latency_dict[span.getSpanId()]=0
            #print(span.getParentId())
            previous_end = span.startTime
            for child in childs[span.getSpanId()]:
                #print(child)
                latency = child.startTime - previous_end
                latency_dict[child.getSpanId()] = latency  # 存入字典
                previous_end = child.startTime + child.duration
                
        trace.latency_dict=latency_dict  # 返回字典供后续使用


if __name__ == "__main__":
    os.makedirs(args.saveDir, exist_ok=True)

    traces = data_collect(f'{args.dataDir}/{args.dataSet}')
    
    #真实分布
    #show_distribution(duration_dict)  
    dists = [
        "normal",
        "expon",
        "gamma",
        "weibull_min",
        "lognorm"
    ]

    build_latency(traces)
    duration_dict=make_metric_dict(traces,'duration')
    latency_dict=make_metric_dict(traces,"latency")

    for dist in dists:
        results_duration = build_Distribution(dist,duration_dict)
        results_latency = build_Distribution(dist,latency_dict)
        
        #print(results)
        test_Distribution(dist,traces,results_duration,results_latency)
