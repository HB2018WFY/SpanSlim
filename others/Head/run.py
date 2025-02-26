import os
import random
import numpy as np
import pandas as pd
import sys
sys.path.insert(1, 'D:\\Repo\\TraceSample')
# sys.path.insert(1, 'C:\\Users\\theor\\Desktop\\TraceSample')
import argparse
import helper.io_util as io_util
import time

def data_collect(path: str):
    traces = io_util.load(f'{path}/traces.pkl')
    return traces

def setSeed(seed):
    np.random.seed(seed)
    random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataDir', type=str, default='../../data')
parser.add_argument('--dataSet', type=str, default='A')
parser.add_argument('--saveDir', type=str, default='../../output')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--sampleRate', type=float, default=0.1)
args = parser.parse_args()

if __name__ == "__main__":
    os.makedirs(args.saveDir, exist_ok=True)
    setSeed(args.seed)

    traces = data_collect(f'{args.dataDir}/{args.dataSet}')

    ids = []
    decisions = []
    exec_ts = []
    for trace in traces:
        ids.append(trace.traceID)
        st = time.time()
        # sample with a fixed rate
        r = random.random()
        if r < args.sampleRate:
            decision = True
        else:
            decision = False
        decisions.append(decision)
        et = time.time()
        exec_ts.append(et-st)

    res = pd.DataFrame(data={
        'traceId': ids, 
        'decision': decisions,
    })
    cost_res = pd.DataFrame(data={
        'encode_t': [0],
        'sample_t': [sum(exec_ts)],
        'other_t': [0],
        'total_t': [sum(exec_ts)]
    })
    res.to_csv(f'{args.saveDir}/{args.dataSet}-Head-sample.csv', index=False)
    cost_res.to_csv(f'{args.saveDir}/{args.dataSet}-Head-cost.csv', index=False)