import argparse
import os
import random
import sys
sys.path.insert(1, 'D:\\Repo\\TraceSample')
# sys.path.insert(1, 'C:\\Users\\theor\\Desktop\\TraceSample')
import pandas as pd
from sieve import *
import helper.io_util as io_util
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataDir', type=str, default='../../data')
parser.add_argument('--dataSet', type=str, default='D')
parser.add_argument('--saveDir', type=str, default='output')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--sampleRate', type=float, default=0.1)
args = parser.parse_args()

def setSeed(seed):
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    os.makedirs(args.saveDir, exist_ok=True)
    setSeed(args.seed)

    traces = io_util.load(f'{args.dataDir}/{args.dataSet}/traces.pkl')

    sieve = Sieve(tree_num=50, tree_size=128, k=50, threshold=0.3)
    ids = []
    decisions = []
    count = 0
    encode_ts, sample_ts, compact_ts = [], [], []

    for trace in traces:
        ids.append(trace.traceID)
        decision, encode_t, sample_t = sieve.isSample(trace)
        encode_ts.append(encode_t)
        sample_ts.append(sample_t)
        st=time.time()
        if decision:
            decisions.append(True)
            # 每处理128条trace降一次维
            if count % 128 == 0:
                print('before compact: %d' % sieve.getEncodeLength(), end=', ')
                sieve.compact()
                print('after compact: %d' % sieve.getEncodeLength())
        else:
            decisions.append(False)
        ed=time.time()
        compact_ts.append(ed-st)
    
    encode_cost = sum(encode_ts)
    sample_cost = sum(sample_ts)
    compact_cost = sum(compact_ts)
    res = pd.DataFrame(data={
        'traceId': ids, 
        'decision': decisions
    })
    cost_res = pd.DataFrame(data={
        'encode_t': [encode_cost],
        'sample_t': [sample_cost],
        'other_t': [compact_cost],
        'total_t': [encode_cost+sample_cost+compact_cost]
    })
    res.to_csv(f'{args.saveDir}/{args.dataSet}-Sieve-sample.csv', index=False)
    cost_res.to_csv(f'{args.saveDir}/{args.dataSet}-Sieve-cost.csv', index=False)
