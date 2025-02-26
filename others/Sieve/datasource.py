import os
import time
from myconcurrent import *
from trace import *
from others.Sieve.sieve_utils import load_multithread, read_raw

class DataSource:
    def __init__(self, **kwargs):
        if 'path' in kwargs.keys() and 'spec' in kwargs.keys(): 
            self.raw = read_raw(kwargs['path'])
            self.spec = kwargs['spec']
            self.groups = self.raw.groupby('traceId')
            self.groupNames = list(self.groups.groups.keys())
            self.numGroup = len(self.groupNames)
        
    def getTraceNum(self):
        return self.numGroup
    
    def getTraces(self):
        if hasattr(self, 'traces'):
            return self.traces
        print('load trace first!!!')
        
    def parse_and_save_trace(self, path, numPerFile):
        startTime = time.time()
        count_n = 0
        count_a = 0
        suffix_n = 0
        suffix_a = 0
        fp_n = open(os.path.join(path, 'traces_normal.txt_' + str(suffix_n)), 'w')
        fp_a = open(os.path.join(path, 'traces_abnormal.txt_' + str(suffix_a)), 'w')
        
        for gn in self.groupNames:
            group = self.groups.get_group(gn)
            if self.spec.loc[gn, 'abnormal']:
                trace = Trace.buildTrace(group, abnormal=True)
                fp_a.write(Trace.serialize(trace) + '\n')
                count_a += 1
                if count_a == numPerFile:
                    print("write %d abnormal traces" % numPerFile)
                    suffix_a += 1
                    fp_a.close()
                    fp_a = open(os.path.join(path, 'traces_abnormal.txt_' + str(suffix_a)), 'w')
                    count_a = 0       
            else:
                trace = Trace.buildTrace(group, abnormal=False)
                fp_n.write(Trace.serialize(trace) + '\n')
                count_n += 1
                if count_n == numPerFile:
                    print("write %d normal traces" % numPerFile)
                    suffix_n += 1
                    fp_n.close()
                    fp_n = open(os.path.join(path, 'traces_abnormal.txt_' + str(suffix_n)), 'w')
                    count_n = 0
        fp_n.close()
        fp_a.close()
        endTime = time.time()
        # print('task %d runs %0.2f seconds.' % (taskid, (endTime - startTime)))
            
    def parse_and_save_trace_mp(self, path, numPerEpoch):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.startswith('traces_'):
                    # 清空文件内容
                    with open(os.path.join(root, file), 'w'):
                        pass
        
        epochs = self.numGroup // numPerEpoch
        
        for i in range(epochs):
            print("**************************************************************************************************************")
            print("                                                epoch %d" % i)
            print("                                       process group from %d - %d" % (i*numPerEpoch, (i+1)*numPerEpoch))
            print("**************************************************************************************************************")
            pstTask = MPTask(6)
            pstTask.run(task_parse_and_save_traces, 
                        self.groupNames[i*numPerEpoch:(i+1)*numPerEpoch], 
                        numPerEpoch, 
                        self.spec, 
                        self.groups, 
                        path)

        remain = self.numGroup % numPerEpoch
        if remain != 0:
            pstTask = MPTask(3)
            pstTask.run(task_parse_and_save_traces, 
                        self.groupNames[self.numGroup-remain:], 
                        remain, 
                        self.spec, 
                        self.groups, 
                        path)

    def load(self, path, withSpan=False, num=None):
        traces_n = load_multithread(path, 'normal', withSpan, num)
        traces_a = load_multithread(path, 'abnormal', withSpan, num)
        self.traces = {'normal': traces_n, 'abnormal': traces_a}

    def getElapsedTimes(self):
        if hasattr(self, 'elapsedTimes'):
            return self.elapsedTimes
        if not hasattr(self, "traces"):
            print("load trace first!!!")
            return None
        et_n = [t.elapsedTime for t in self.traces['normal']]
        et_a = [t.elapsedTime for t in self.traces['abnormal']]
        self.elapsedTimes = {'normal': et_n, 'abnormal': et_a}
        return self.elapsedTimes
    
    def getSpanNums(self):
        if hasattr(self, 'spanNums'):
            return self.spanNums
        if not hasattr(self, 'traces'):
            print('load trace first!!!')
            return None
        sn_n = [t.spanNum for t in self.traces['normal']]
        sn_a = [t.spanNum for t in self.traces['abnormal']]
        self.spanNums = {'normal': sn_n, 'abnormal': sn_a}
        return self.spanNums
        
    def getTraceDataFlow(self):
        if not hasattr(self, 'traces'):
            print('load trace first!!!')
            return
        traces = self.traces['normal'] + self.traces['abnormal']
        self.traceDataFlow = sorted(traces, key=lambda x:x.startTime)  
        return self.traceDataFlow