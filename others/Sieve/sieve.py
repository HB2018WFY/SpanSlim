import math
import pdb

from networkx import DiGraph
import others.Sieve.myrrcf as myrrcf
import numpy as np
from queue import Queue
import time
'''
    调用链是一棵树，树上每个节点的类型是TNode，表示一个span
'''
class TNode:
    def __init__(self, span, parent):
        self.nid = span.getSpanId()                  # spanId作为node的id
        self.parent = parent                         # 父节点
        self.label = span.service                   # span的标签 [TODO]
        # self.label = span.getSpanLabel()             # span的标签
        self.children = []                           # 孩子节点
        self.elapsedTime = span.getElapsedTime()     # span的持续时间
#         self.callType = span.getCallType()           # span的调用类型
        
#     def getCallType(self):
#         return self.callType
        
    def getNid(self):
        return self.nid

    def getParent(self):
        return self.parent
    
    def setParent(self, parent):
        self.parent = parent
    
    def addChild(self, tnode):
        self.children.append(tnode)
        
    def removeChild(self, tnode):
        self.children.remove(tnode)
        
    def getChildren(self):
        return self.children
    
    def setChildren(self, children):
        self.children = children
    
    def getLabel(self):
        return self.label
    
    def getElapsedTime(self):
        return self.elapsedTime
    
    # 返回从根节点到该节点的路径
    def getPath(self):
        # 第一次调用该函数会给node添加一个path属性，记录从根节点到该节点的路径
        if not hasattr(self, 'path'):
            if self.parent == None:
                # 根节点直接返回label
                self.path = self.label
            else:
                self.path = self.parent.getPath() + '/' + self.label
            
        return self.path

'''
    正常情况下调用链应该是一棵树。
    可能出现断链的情况，就会有不止一棵树，用TForest来对调用链建模。
'''
class TForest:
    def __init__(self, trace):
        self.spanNum = trace.getSpanNum()
        tNodes = {}                                  
        for span in trace.getSpans():
            tn = TNode(span, span.getParentId())
            tNodes[span.getSpanId()] = tn
        self.roots = []                               # 可能会有多个根
        for span in trace.getSpans():
            tn = tNodes[span.getSpanId()]
            if span.getParentId() == None or span.getParentId() == 'None' or span.getParentId() not in tNodes:
                tn.parent = None
                self.roots.append(tn)
            else:
                tnp = tNodes[span.getParentId()]
                tnp.addChild(tn)
                tn.parent = tnp
                
    def isEmpty(self):
        return len(self.roots) == 0
    
    # 先序遍历，对遍历的每个node执行func
    def traversal(self, func):
        def preOrder(node, func):
            func(node)
            children = node.children
            if children == []:
                return
            else:
                for child in children:
                    preOrder(child, func)
        
        for root in self.roots:
            preOrder(root, func)
    
    # 获取调用链中存在的所有路径
    def getPaths(self):
        paths = []
        def record(tNode):
            paths.append(tNode.getPath())
        self.traversal(record)
        return paths
    
    # 获取无重复的路径
    def getUniquePaths(self):
        paths = self.getPaths()
        return set(paths)
    
    # 可视化调用链
    def draw(self, filename):
        dot = DiGraph(format='svg')
        q = Queue()
        for root in self.roots:
            q.put(root)
            while (not q.empty()):
                parent = q.get()
                dot.node(parent.getNid(), label=('%s:\n%d' % (parent.getLabel(), parent.getElapsedTime())))
                for child in parent.getChildren():
                    dot.edge(parent.getNid(), child.getNid())
                    q.put(child)
        dot.render(filename)
    
    '''
        去掉callType为'Local'的节点
        针对西藏移动的数据集所做的优化，对其他的数据集不适用
    '''
    def compress(self):
        count = 0
        q = Queue()
        oldRoots = []
        oldRoots.extend(self.roots)
        newRoots = []
        for root in oldRoots:
            q.put(root)
            while not q.empty():
                tn = q.get()
                children = tn.getChildren()

                if tn.getCallType() == 'LOCAL':
                    count += 1
                    if tn in self.roots:
                        self.roots.remove(tn)
                    if tn in newRoots:
                        newRoots.remove(tn)
                    
                    parent =tn.getParent()
                    if parent != None:
                        parent.removeChild(tn)
                        for child in children:
                            parent.addChild(child)
                            child.setParent(parent)
                    else:
                        for child in children:
                            child.setParent(None)
                            newRoots.append(child)
                for child in children:
                    q.put(child)
        self.roots.extend(newRoots)
#         print('remove %d LOCAL nodes' % count)
    
    # 将连续的label相同的节点合并成一个节点
    def compress2(self):
        def func(tnode):
            childrenToBeChecked = tnode.getChildren()
            childrenChecked = []
            
            while len(childrenToBeChecked):
                tmp = []    # 保存label相同的子节点的孩子
                for child in childrenToBeChecked:
                    if child.getLabel() == tnode.getLabel():
                        tmp.extend(child.getChildren())
                    else:
                        child.setParent(tnode)
                        childrenChecked.append(child)
                childrenToBeChecked = tmp
            tnode.setChildren(childrenChecked)
            
        self.traversal(func)
    
    # 路径向量编码，不考虑弱维度。mapping为path到index的映射，可传入空映射({})，在遍历TForest时会建立映射关系。
    def encodeWithMapping(self, mapping):
        def wrapper(e):
            def func(tnode):
                path = tnode.getPath()
                if path not in mapping: # 发现新维度
                    mapping[path] = len(mapping)
                    e.append(-1)
                e[mapping[path]] = max(e[mapping[path]], tnode.getElapsedTime())
            return func
        
        e = [-1] * len(mapping) # 所有维度的值初始化为-1，包括无效维度
        self.traversal(wrapper(e))
        return e
    
    '''
        路径向量编码，考虑弱维度。
        参数：
            path2dim: path到index的映射
            dim2path: index到path的映射
            weakDim: 弱维度所对应的path
        返回值:
            e: 强维度编码
            weakPaths: 弱维度下的值
            newPaths: 新维度下的值
    '''
    def encode(self, path2dim, dim2path, weakDim):
        def wrapper(e, weakPaths, newPaths):
            def func(tnode):
                path = tnode.getPath()     
                if path in weakDim:    # path对应弱维度
                    if path in weakPaths:
                        weakPaths[path] = max(weakPaths[path], tnode.getElapsedTime())
                    else:
                        weakPaths[path] = tnode.getElapsedTime()
                elif path not in path2dim:    # path对应新维度
                    if path in newPaths:
                        newPaths[path] = max(newPaths[path], tnode.getElapsedTime())
                    else:
                        newPaths[path] = tnode.getElapsedTime()
                else:    # path对应强维度
                    e[path2dim[path]] = max(e[path2dim[path]], tnode.getElapsedTime())
            return func
        
        weakPaths = {}
        newPaths = {}
        e = [-1] * len(path2dim)    # 所有强维度的值初始化为-1
        self.traversal(wrapper(e, weakPaths, newPaths))
        return e, weakPaths, newPaths

'''
    point是路径向量的抽象
'''
class Point:
    def __init__(self, x):
        self.x = x

'''
    采样算法，使用路径向量构建rrcf。
    参数:
        tree_num: rrct的棵数
        tree_size: rrct的叶子数量
        k: 历史数据的窗口，表示与过去多少个点的得分做比较
        threshold: 阈值，超过它使用sigmoid函数计算采样率，低于它使用线性函数计算采样率
        var_limit: 方差低于这个值的维度视为弱维度
        seed: 随机数种子
'''
class Sieve:
    def __init__(self, tree_num=20, tree_size=128, k=50, threshold=0.3, var_limit=0.1, seed=None):
        self.tree_num = tree_num        # rrct的棵数
        self.tree_size = tree_size      # rrct的叶子数量
        self.k = k                      # 历史数据的窗口，表示与过去多少个点的注意力得分做比较
        self.var_limit = var_limit      # 差低于这个值的维度视为弱维度
        self.kScores = []               # 过去k条调用链的注意力得分
        self.threshold = threshold      # 阈值，超过它使用sigmoid函数计算采样率，低于它使用线性函数计算采样率
        self.points = {}                # rrct上的路径向量
        self.dim2q = None               # 这是一个list，与路径向量等长，某个维度i被选为cutting dimension时，list[i]记录对应的Branch，否则为None
        self.dim2path = []              # 这是一个list，与路径向量等长。list[i] = p：表示维度i对应于路径p
        self.path2dim = {}              # 路径向量中path到index的映射
        self.weakDim = {}               # 弱维度对应的path到值的映射
        self.nextIdx = 1                # 下一个数据点的编号，在rrct上的每个数据点都有一个编号
        self.rrcf = []                  # 多棵rrct
        for _ in range(tree_num):
            rrct = myrrcf.RCTree(random_state=seed)
            self.rrcf.append(rrct)
    
    # 路径向量编码
    def encode(self, trace):
#         self.checkDim2q()
        tf = TForest(trace)
#         tf.compress()    # 去掉callType为'Local'的节点。针对西藏移动数据集的优化，其他的数据集不适用
        tf.compress2()   # 将连续的label相同的节点合并成一个节点
        
        # 对强维度进行编码，新维度和弱维度单独挑出来
        e, weakPaths, newPaths = tf.encode(self.path2dim, self.dim2path, self.weakDim.keys())
#         assert len(e) == len(self.dim2path)
        
        
        values = []                                     # 新维度的向量，每一维都是-1
        for path in newPaths:
            values.append(-1)
            self.path2dim[path] = len(self.dim2path)    # 建立新维度对应的path到index的映射
            self.dim2path.append(path)                  # 建立新维度的index到path的映射
            e.append(newPaths[path])                    # 新维度的值合并到路径向量编码中                            

        weakDimPaths = list(self.weakDim.keys())        # 弱维度所对应的path

        # 追踪弱维度的取值，如果弱维度取值的方差超过var_limit，将弱维度提升为强维度
        for path in weakDimPaths:
            '''
                由于rrct的叶子数是固定的，新数据点的插入伴随着旧数据点的移除，所以弱维度的取值是一种streaming data。
                为了降低计算弱维度方差的开销，使用https://nestedsoftware.com/2018/03/27/calculating-standard-deviation-on-streaming-data-253l.23919.html方法来计算方差
            '''
            x_n = weakPaths.get(path, -1)
            x_0 = self.weakDim[path]['data'][0]
            avg1 = self.weakDim[path]['avg']
            avg2 = avg1 + (x_n - x_0) / self.tree_size
            self.weakDim[path]['avg'] = avg2    # 更新均值
            
            dSquare = self.weakDim[path]['var'] * (self.tree_size - 1)
            dSquare = dSquare + (x_n - x_0) * (x_n + x_0 - avg1 - avg2)
            var = dSquare / (self.tree_size - 1)
            self.weakDim[path]['var'] = var    # 更新方差
            self.weakDim[path]['data'].append(x_n)    # 更新滑动窗口
            self.weakDim[path]['data'].pop(0)
            
            if var > self.var_limit and path in weakPaths:    # 提升弱维度
#                 values.append(avg2)
                values.append(-1)                             # 弱维度也是新维度
                self.path2dim[path] = len(self.dim2path)
                self.dim2path.append(path)
                e.append(weakPaths[path])
                self.weakDim.pop(path)
        
        '''
        assert len(e) == len(self.dim2path)
        for rrct in self.rrcf:
            assert rrct.ndim == None and len(values) == len(self.dim2path) or rrct.ndim + len(values) == len(self.dim2path)
        '''
        
        # 叶子扩展新维度
        if len(values):
            for idx in self.points:
                # 对rrct上的所有point拼接新维度的向量
                self.points[idx].x = np.append(self.points[idx].x, values)
            
            if self.dim2q is not None:
                for q in self.dim2q:
                    if q is not None:
                        for branch in q:
                            branch.b = np.append(branch.b, [values, values], axis=1)    # 更新bounding box
#                             assert branch.b.shape[1] == len(self.dim2path)
                        
                # dim2q扩展新维度
                self.dim2q.extend([None] * len(values))
#                 assert len(self.dim2q) == len(self.dim2path)
#         self.checkDim2q()
        if len(e) != len(self.dim2path):
            pdb.set_trace()
        return Point(e)        
    
    # 计算trace的注意力得分
    def score(self, trace):
        st=time.time()
        newPoint = self.encode(trace)
        et=time.time()
        encode_t = et-st
        scores = []
        
        st=et
        # rrct的叶子数已满
        if len(self.rrcf[0].leaves) == self.tree_size:
#             duplicate_add_set = set()
#             duplicate_rm_set = set()
            for rrct in self.rrcf:
                rrct.ndim = len(self.dim2path)    # 更新维数
            
                # 移除最老叶子
                leaf_rm, old_branch = rrct.forget_point(self.nextIdx - self.tree_size)
                # 如果被移除的叶子在树上有其他副本，duplicate_rm为True，old_branch为None
                duplicate_rm = True if leaf_rm == None else False
#                 duplicate_rm_set.add(duplicate_rm)
#                 assert len(duplicate_rm_set) == 1
                if old_branch is not None:
                    self.dim2q[old_branch.q].remove(old_branch)
                    if len(self.dim2q[old_branch.q]) == 0:
                        self.dim2q[old_branch.q] = None
                        
                # 插入新数据点
                leaf_add, new_branch = rrct.insert_point(newPoint, index=self.nextIdx)
                # 如果新数据点在树上有其他副本，duplicate_add为True，new_branch为None
                duplicate_add = True if leaf_add is None else False
#                 duplicate_add_set.add(duplicate_add)
#                 assert len(duplicate_add_set) == 1
                if new_branch is not None:
                    if self.dim2q is None:
                        self.dim2q = [None] * len(self.dim2path)
                    if self.dim2q[new_branch.q] == None:
                        self.dim2q[new_branch.q] = []
                    self.dim2q[new_branch.q].append(new_branch)
                
                # 计算得分
                scores.append(rrct.codisp(self.nextIdx))
            if not duplicate_rm:    # 移除的不是重复节点
                self.points.pop(leaf_rm.i)
            if not duplicate_add:    # 插入的不是重复节点
                self.points[self.nextIdx] = newPoint
        else:
            # rrct叶子数未满
#             duplicate_add_set = set()
            for rrct in self.rrcf:
                rrct.ndim = len(self.dim2path)    # 更新维数
            
                # 插入新数据点
                leaf_add, new_branch = rrct.insert_point(newPoint, index=self.nextIdx)
                duplicate_add = True if leaf_add is None else False
#                 duplicate_add_set.add(duplicate_add)
#                 assert len(duplicate_add_set) == 1
                if new_branch is not None:
                    if self.dim2q is None:
                        self.dim2q = [None] * len(self.dim2path)
                    if self.dim2q[new_branch.q] == None:
                        self.dim2q[new_branch.q] = []
                    self.dim2q[new_branch.q].append(new_branch)
                scores.append(rrct.codisp(self.nextIdx))
            if not duplicate_add:
                self.points[self.nextIdx] = newPoint
        self.nextIdx += 1
        
        et = time.time()
        score_t = et-st
        # 取平均分为最终得分
        return np.average(scores), encode_t, score_t
    
    # 找出弱维度，返回的弱维度列表为逆序
    def findWeakDimensions(self):
        rrct = self.rrcf[0]
        data = np.array([leaf.x.x for leaf in rrct.leaves.values()])
        weak_dimensions = []
        
        # rrct的叶子数未满，路径向量的维度不会很高，不需要找出弱维度
        if data.shape[0] < self.tree_size:
            return weak_dimensions
        
        for i in range(data.shape[1] - 1, -1, -1):
            # 方差小于var_limit并且没有被选为分割维度的维度是弱维度
            if (data[:, i].var()) < self.var_limit and self.dim2q[i] is None:
                weak_dimensions.append(i)

        # weak_dimensions的顺序为从大到小
        return weak_dimensions
    
    def compact(self):
       # 去除无效维度
        weakPaths = list(self.weakDim.keys())
        for weakPath in weakPaths:
            # 弱维度会越积越多，包含大量-1的弱维度直接去除
            if self.weakDim[weakPath]['avg'] < -0.9 or (np.array(self.weakDim[weakPath]['data']) == -1).all():
                self.weakDim.pop(weakPath)
        
        # 找出弱维度
        weak_dimensions = self.findWeakDimensions()
        if len(weak_dimensions) == 0:
            return
        
        data = np.array([leaf.x.x for leaf in self.rrcf[0].leaves.values()])
        for wd in weak_dimensions:
            # 追踪新的弱维度
            if not (data[:,wd] == -1).all():
                wp = self.dim2path[wd]
                self.weakDim[wp] = {'data': data[:, wd].tolist(), 'avg': data[:, wd].mean(), 'var': data[:, wd].var()}

        # 去除叶子弱维度
        for idx in self.points:
            self.points[idx].x = np.delete(self.points[idx].x, weak_dimensions)
            
        # 去除内部点弱维度
        for q in self.dim2q:
            if q is not None:
                for branch in q:
                    branch.b  = np.delete(branch.b, weak_dimensions, axis=1)
        
        # 更新与维度有关的变量
        for d in weak_dimensions:   
            # path2dim中把弱维度移除
            path = self.dim2path[d]
            self.path2dim.pop(path)
            # 由于dim2path是list，所以pop(d)会使index大于d的路径自动调整index
            self.dim2path.pop(d)
            # dim2q与上面同理
            self.dim2q.pop(d)
        
        # 校正叶子维数
        for rrct in self.rrcf:    
            rrct.ndim = len(self.dim2path)
        
        '''
            weak_dimensions中的元素是逆序排列的，最小的弱维度在最后的位置。
            把弱维度去除之后，原先index大于最小弱维度的path需要重新调整index
        '''
        for d in range(weak_dimensions[-1], len(self.dim2path)):    
            path = self.dim2path[d]
            self.path2dim[path] = d
            
            # 分割维度大于最小弱维度的也需要校正
            if self.dim2q[d] is not None:
                for branch in self.dim2q[d]:
                    branch.q = d
    
    '''
        计算trace的采样概率。
        可以传一个trace，score为None，这种情况下对trace编码，计算注意力得分，计算采样率。
        也可以传一个注意力得分score，trace为None，这种情况下直接计算采样率。
    '''
    def getProbability(self, trace=None, score=None):
        assert (score is None and trace is not None) or (score is not None and trace is None)
        encode_t, score_t = 0, 0
        if score == None:
            score, encode_t, score_t = self.score(trace)
        self.kScores.append(score)
        
        st = time.time()
        # 历史数据少于两个时采样概率为0
        if len(self.kScores) <= 2:
            return 0, encode_t, score_t
        
        if len(self.kScores) > self.k:
            self.kScores = self.kScores[1:]
        
        diff = np.array(self.kScores).var(ddof=1) - np.array(self.kScores[:-1]).var(ddof=1)
        if diff > self.threshold:
#         mean = np.mean(self.kScores)
#         std = np.std(self.kScores)
#         if score > mean + 3 * std or score < mean - 3 * std:
            # sigmoid函数
            p = 1 / (1 + 2 * math.exp(np.average(self.kScores) - self.kScores[-1]))
        else:
            p = self.kScores[-1] / sum(self.kScores)
        if len(self.kScores) > self.k:
            self.kScores = self.kScores[1:]
        
        et = time.time()
        sample_t = et-st+score_t
        return p, encode_t, sample_t
    
    def isSample(self, trace):
        p, encode_t, sample_t = self.getProbability(trace)
        if np.random.uniform() < p:
            return True, encode_t, sample_t 
        return False, encode_t, sample_t 
    
    def getEncodeLength(self):
        return self.rrcf[0].ndim
    
    def getWeakDimensions(self):
        return list(self.weakDim.keys())
    
    def checkDim2q(self):
        if self.dim2q:
            for d in self.dim2q:
                if d:
                    for branch in d:
                        span = branch.b[-1,:] - branch.b[0,:]
                        assert span[branch.q] != 0
                            
    def getInvalidDimensionCount(self):
        rrct = self.rrcf[0]
        data = np.array([leaf.x.x for leaf in rrct.leaves.values()])
        count = 0
        for i in range(0, data.shape[1]):
            if (data[:,i] == -1).all():
                count += 1
        return count
    
    def getWeakDimensionCount(self):
        rrct = self.rrcf[0]
        data = np.array([leaf.x.x for leaf in rrct.leaves.values()])
        count = 0
        for i in range(0, data.shape[1]):
            if (data[:,i] != -1).any() and data[:, i].var() < self.var_limit:
                count += 1
        return count
