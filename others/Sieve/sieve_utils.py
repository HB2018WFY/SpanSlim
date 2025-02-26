import pandas as pd
import os
import time
from functools import reduce
from trace import *
from myconcurrent import *

def getSpec(esbPath, osbPath):
    esb = pd.read_csv(esbPath)
    esb['startTime'] = esb['startTime'].apply(lambda x: x//60000)
    esb_agg = esb.groupby('startTime').agg({'num': 'sum', 'succee_num': 'sum'})
    esb_agg['fail_rate'] = (esb_agg['num'] - esb_agg['succee_num']) / esb_agg['num']

    osb = pd.read_csv(osbPath)
    osb['startTime'] = osb['startTime'].apply(lambda x: x//60000)
    osbGroupbyST = osb.groupby('startTime')
    groupNames = osbGroupbyST.groups.keys()

    spec = pd.DataFrame(columns=['abnormal'], index=osb['traceId']).fillna(False)
    for gn in groupNames:
        osb_g = osbGroupbyST.get_group(gn)
        failExpected = int(osb_g.shape[0] * esb_agg.loc[gn, 'fail_rate'])
        osb_f = osb[osb['success'] == False]
        osb_t = osb[osb['success'] == True]
        spec.loc[osb_f['traceId'], 'abnormal'] = True

        if osb_f.shape[0] < failExpected:
            diff = failExpected - osb_f.shape[0]
            osb_t.sort_values(by=['elapsedTime'], ascending=False, inplace=True)
            spec.loc[osb_t[0:diff]['traceId'], 'abnormal'] = True
    return spec
    
def read_raw(path):
    raw_csf = pd.read_csv(os.path.join(path, 'trace_csf.csv'))
    raw_fly_remote = pd.read_csv(os.path.join(path, 'trace_fly_remote.csv'))
    raw_jdbc = pd.read_csv(os.path.join(path,'trace_jdbc.csv'))
    raw_local = pd.read_csv(os.path.join(path, 'trace_local.csv'))
    raw_osb = pd.read_csv(os.path.join(path, 'trace_osb.csv'))
    raw_remote_process = pd.read_csv(os.path.join(path, 'trace_remote_process.csv'))
    raw_all = pd.concat([raw_csf, raw_fly_remote, raw_jdbc, raw_local, raw_osb, raw_remote_process]).fillna('')
    return raw_all

def load_multithread(path, traceType, withSpan=False, num=None):
    taskThreads = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith('traces_' + traceType):
                if withSpan:
                    taskThreads.append(TaskThread(loadTraceWithSpan, (os.path.join(root, file), num)))
                else:
                    taskThreads.append(TaskThread(loadTraceWithoutSpan, (os.path.join(root, file), num)))
    for tt in taskThreads:
        tt.start()
    return reduce(lambda x,y: x+y, [t.getResult() for t in taskThreads])


def loadTraceWithSpan(filename, num=None):
    trace = []
    if num:
        with open(filename, 'r') as fp:
            traceStr = fp.readline()
            while traceStr != '' and num:
                trace.append(Trace.deserialize(traceStr))
                traceStr = fp.readline()
                num -= 1
    elif num == None:
        with open(filename, 'r') as fp:
            traceStr = fp.readline()
            while traceStr != '':
                trace.append(Trace.deserialize(traceStr))
                traceStr = fp.readline()
    return trace

def loadTraceWithoutSpan(filename, num=None):
    trace = []
    if num:
        with open(filename, 'r') as fp:
            traceStr = fp.readline()
            while traceStr != '' and num:
                t = Trace.deserialize(traceStr)
                # 去掉span
                t.spans = None      
                trace.append(t)
                traceStr = fp.readline()
                num -= 1
    elif num == None:
        with open(filename, 'r') as fp:
            traceStr = fp.readline()
            while traceStr != '':
                t = Trace.deserialize(traceStr)
                # 去掉span
                t.spans = None
                trace.append(t)
                traceStr = fp.readline()
    return trace

# timestamp to datetime
def ts2dt(ts):
        time_local = time.localtime(ts)
        return time.strftime("%Y-%m-%d %H:%M:%S", time_local)
