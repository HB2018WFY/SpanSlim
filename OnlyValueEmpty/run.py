import os
import random
import numpy as np
import pandas as pd
import sys

current_dir = os.path.dirname(__file__)  # 获取当前脚本所在目录（如果是文件中使用）
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))  # 父目录路径
sys.path.append(parent_dir) 

#print(parent_dir)

import argparse
import helper.io_util as io_util
import time

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

if __name__ == "__main__":
    os.makedirs(args.saveDir, exist_ok=True)

    traces = data_collect(f'{args.dataDir}/{args.dataSet}')
    
    instance_operation_counter = {}
    for trace in traces:
        for span in trace.getSpans():
            instance = span.instance
            operation = span.operation
            #print(instance,operation)
            key = f"{instance}:{operation}"
            instance_operation_counter[key] = instance_operation_counter.get(key, 0) + 1

    sorted_counter = sorted(instance_operation_counter.items(), key=lambda x: -x[1])
    for key,cnt in sorted_counter:
        print(key,cnt)
