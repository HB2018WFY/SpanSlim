from multiprocessing import Pool
import threading
import os
import time
from trace import *

# 多线程任务执行
class TaskThread(threading.Thread):
    def __init__(self, func, args=()):
        super(TaskThread, self).__init__()
        self.func = func
        self.args = args
    
    def run(self):
        print('TaskThread %s starts...' % threading.currentThread().getName())
        startTime = time.time()
        self.result = self.func(*self.args)
        endTime = time.time()
        print('TaskThread %s run %.2f seconds' % (threading.currentThread().getName(), endTime - startTime))
    
    def getResult(self):
        threading.Thread.join(self)
        try:
            return self.result
        except Exception as err:
            print(err)
            
            
# 多进程任务执行
class MPTask:
    def __init__(self, numProcess):
        self.numProcess = numProcess
    
    def run(self, task, data, size, *args):
        num = size // self.numProcess
        p = Pool(self.numProcess)
        for i in range(self.numProcess):
            p.apply_async(task, args=(i, i*num, num, data, *args))
        
        print('wait for all processes done...')
        p.close()
        p.join()
        print('all processes done...')
    

def task_parse_and_save_traces(taskid, startIdx, num, groupNames, spec, groups, path):
    print('========================= Run task %d (%d) ===========================' % (taskid, os.getpid()))
    print("taskid:%d, startIdx:%d, num:%d, groupNames:%d, spec:%d, path%s" % (taskid, startIdx, num, len(groupNames), len(spec), path))
    startTime = time.time()
    i = startIdx
    
    fp_n = open(os.path.join(path, "traces_normal.txt_" + str(taskid)), 'a')
    fp_a = open(os.path.join(path, "traces_abnormal.txt_" + str(taskid)), 'a')
    for j in range(num):
        group = groups.get_group(groupNames[i])
        if spec.loc[groupNames[i], 'abnormal']:
            trace = Trace.buildTrace(group, abnormal=True)
            fp_a.write(Trace.serialize(trace) + '\n')
        else:
            trace = Trace.buildTrace(group, abnormal=False)
            fp_n.write(Trace.serialize(trace) + '\n')
        i += 1
    
    fp_n.close()
    fp_a.close()
    endTime = time.time()
    print('task %d runs %0.2f seconds.' % (taskid, (endTime - startTime)))
    print("process trace(%d, %d)" % (startIdx, startIdx + num))
