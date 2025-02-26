import random
import numpy as np
import sys
import os
import argparse
import pandas as pd
sys.path.insert(1, 'D:\\Repo\\TraceSample')
# sys.path.insert(1, 'C:\\Users\\theor\\Desktop\\TraceSample')
from PNode import PNode
import helper.io_util as io_util
import time

class WTSampling:
    def __init__(self, nodes, max_leaves, exact_dist_thres=10):
        self.nodes = nodes
        self.max_leaves = max_leaves
        self.count = 0
        self.root = PNode(exact_dist_thres=exact_dist_thres)

    def insert(self, trace):
        # print(trace)
        if self.count < self.max_leaves:
            self.root = self.root.insert(trace)
            self.count += 1
        else:
            self.delete_unlikely_node()
            self.root = self.root.insert(trace)

    def delete_unlikely_node(self):
        node = self.root

        while not node.is_leaf():
            node = max(node.children, key=lambda x: x.point_counter)

        sibling = node.siblings()[0]
        parent = node.parent
        parent.children = []
        parent.pts = sibling.pts
        parent.point_counter = sibling.point_counter
        sibling.deleted = True
        node.deleted = True

        parent._update_params_recursively()

    def sampling(self, num, seed=None):
        np.random.seed(seed)
        samples = set()
        count = 0
        while count < num:
            node = self.root
            while not node.is_leaf():
                node = np.random.choice(node.children, 1)[0]
            if node not in samples:
                samples.add(node)
                count += 1
        return samples

    def traceEncoding(self, trace):
        spanCount = [0] * len(self.nodes)
        for span in trace.spans:
            label = span.service
            spanCount[self.nodes.index(label)] += 1
        code = (np.array(spanCount), trace.traceID)
        return code

def setSeed(seed):
    np.random.seed(seed)
    random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataDir', type=str, default='data')
parser.add_argument('--dataSet', type=str, default='A')
parser.add_argument('--saveDir', type=str, default='output')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--sampleRate', type=float, default=0.1)
args = parser.parse_args()

if __name__ == "__main__":
    os.makedirs(args.saveDir, exist_ok=True)
    
    setSeed(args.seed)
    traces = io_util.load(f'{args.dataDir}/{args.dataSet}/traces.pkl')
    nodes = io_util.load(f'{args.dataDir}/{args.dataSet}/nodes.pkl')

    sampler = WTSampling(nodes=nodes, max_leaves=3000, exact_dist_thres=10)
    ids = []

    encode_ts, other_ts=[], []
    for trace in traces:
        st = time.time()
        code = sampler.traceEncoding(trace)
        et = time.time()
        encode_ts.append(et-st)

        st = et
        sampler.insert(code)
        et = time.time()
        other_ts.append(et-st)
        ids.append(code[1])

    st = time.time()
    samples = sampler.sampling(num=int(args.sampleRate*len(traces)))
    et = time.time()
    sample_t = et-st
    sampleIds = [sample.pts[0][1] for sample in samples]

    decisions = []
    for id in ids:
        if id in sampleIds:
            decisions.append(True)
        else:
            decisions.append(False)

    encode_cost=sum(encode_ts)
    other_cost=sum(other_ts)
    res = pd.DataFrame(data={
        'traceId': ids, 
        'decision': decisions
    })
    cost_res = pd.DataFrame(data={
        'encode_t': [encode_cost],
        'sample_t': [sample_t],
        'other_t': [other_cost],
        'total_t': [encode_cost+other_cost+sample_t]
    })
    res.to_csv(f'{args.saveDir}/{args.dataSet}-PERCH-sample.csv', index=False)
    cost_res.to_csv(f'{args.saveDir}/{args.dataSet}-PERCH-cost.csv', index=False)
