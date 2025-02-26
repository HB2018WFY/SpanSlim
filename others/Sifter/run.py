import argparse
import os
import random
import sys
import numpy as np
import pandas as pd
sys.path.insert(1, 'D:\\Repo\\TraceSample')
# sys.path.insert(1, 'C:\\Users\\theor\\Desktop\\TraceSample')
from Sifter import Sifer
from helper import io_util
import time


parser = argparse.ArgumentParser()
parser.add_argument('--dataDir', type=str, default='data')
parser.add_argument('--dataSet', type=str, default='A')
parser.add_argument('--saveDir', type=str, default='output')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--path_length', type=int, default=5)
parser.add_argument('--embedding_dim', type=int, default=10)
parser.add_argument('--memory_size', type=int, default=50)
parser.add_argument('--sampleRate', type=float, default=0.1)
args = parser.parse_args()


def setSeed(seed):
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    os.makedirs(args.saveDir, exist_ok=True)
    setSeed(args.seed)

    traces = io_util.load(f'{args.dataDir}/{args.dataSet}/traces.pkl')
    spanLabels = io_util.load(f'{args.dataDir}/{args.dataSet}/spanLabels.pkl')

    if args.dataSet == 'D':
        args.path_length=3

    # init vocabulary
    vocab = set(spanLabels)
    sampler = Sifer(
        list(vocab),
        N=args.path_length,
        P=args.embedding_dim,
        k=args.memory_size,
        alpha=args.sampleRate
    )

    encode_ts, optim_ts, sample_ts = [], [], []
    ids = []
    decisions = []
    for trace in traces:
        ids.append(trace.traceID)
        decision, P, encode_t, optim_t, sample_t = sampler.sample(trace)
        encode_ts.append(encode_t)
        optim_ts.append(optim_t)
        sample_ts.append(sample_t)
        decisions.append(decision)

    encode_cost = sum(encode_ts)
    sample_cost = sum(sample_ts)
    optim_cost = sum(optim_ts)
    res = pd.DataFrame(data={
        'traceId': ids, 
        'decision': decisions
    })
    cost_res = pd.DataFrame(data={
        'encode_t': [encode_cost],
        'sample_t': [sample_cost],
        'other_t': [optim_cost],
        'total_t': [encode_cost+sample_cost+optim_cost]
    })
    res.to_csv(f'{args.saveDir}/{args.dataSet}-Sifter-sample.csv', index=False)
    cost_res.to_csv(f'{args.saveDir}/{args.dataSet}-Sifter-cost.csv', index=False)
