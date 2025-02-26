import random
from collections import deque
import numpy as np
import time
import pandas as pd
import networkx as nx
import torch
from torch import nn

from entity.Trace import Trace
from model import CBOW
from helper import graph_util



class Sifer:
    def __init__(self, vocab, N=5, P=10, k=50, alpha=0.01):
        """
        Args:
            N: the length of each path
            P: the dimension of embedding
            k: the number of recently seen traces
            alpha: initial sampling rate
        """
        if N % 2 == 0:
            raise Exception("N must be odd")

        self.vocab = vocab
        self.N = N
        self.P = P
        self.k = k
        self.alpha = alpha

        self.lastSeenLosses = deque(maxlen=k+1)
        self.model = CBOW(len(vocab), self.P, N-1)
        self.loss_fn = nn.NLLLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        self.word_to_ix = {word: ix for ix, word in enumerate(vocab)}
        self.ix_to_word = {ix: word for ix, word in enumerate(vocab)}


    def sample(self, trace):
        st = time.time()
        paths = self.extract_paths(trace)
        data = self.build_inputs_labels(paths)
        et = time.time()

        encode_t = et-st

        optim_t, sample_t = 0,0
        total_loss = 0
        st = et
        if len(data) != 0:
            for context, target in data:
                idxs = [self.word_to_ix[w] for w in context]
                context_vector = torch.tensor(idxs, dtype=torch.long)
                log_probs = self.model(context_vector)
                total_loss += self.loss_fn(log_probs, torch.tensor([self.word_to_ix[target]]))
            
            mean_loss = (total_loss.detach().item()) / len(data)

            self.lastSeenLosses.append(mean_loss)
            # update weights
            weights=[]
            min_loss=min(self.lastSeenLosses)
            for loss in self.lastSeenLosses:
                weights.append(loss-min_loss)

            equal_all = all(w == weights[0] for w in weights)

            if equal_all:
                p = self.alpha
            else:
                p = (weights[-1] / (sum(weights[1:])+1e-9)) * (self.k + 1) * self.alpha

            et = time.time()
            sample_t=(et-st)

            op_st=time.time()
            # optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            op_et=time.time()
            optim_t=(op_et-op_st)

            print(f"loss: {mean_loss}, 'prob: {p:.3f}")

        else:
            et = time.time()
            sample_t=(et-st)
            p = self.alpha

        r = random.random()
        if r < p:
            decision = True
        else:
            decision = False
        
        return decision, p, encode_t, optim_t, sample_t

    def extract_paths(self, trace: Trace):
        # init a DAG
        dg = graph_util.build_dg_with_trace(trace, 'operation')
        root = graph_util.get_root(dg)
        leaves = graph_util.get_leaves(dg)
        # filter N-length paths
        paths = []
        pths = nx.all_simple_paths(G=dg, source=root, target=leaves)
        for pth in pths:
            if len(pth) == self.N:
                paths.append(pth)
            elif len(pth) > self.N:
                for i in range(0, len(pth)-self.N+1):
                    paths.append(pth[i:self.N+i])
        return paths

    def build_inputs_labels(self, paths):
        data = []
        context_size = int(self.N / 2)
        for path in paths:
            context = []
            context.extend(path[:context_size])
            context.extend(path[-context_size:])
            target = path[context_size]
            data.append((context, target))
        return data