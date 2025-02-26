# Sieve采样算法实现

## 主要文件
- sieve.py
	- TNode类
	- TForest类: 路径向量编码实现
	- Sieve类: 采样实现
- myrrcf.py: 随机砍伐森林模型实现
- interface.py
	- TraceBase类: Trace基类，接入的trace数据需要实现这个接口
	- SpanBase类: Span基类，接入的trace数据需要实现这个接口

## trace数据接入
- 定义Trace类，实现TraceBase中的方法
- 定义Span类，实现SpanBase的方法

### 示例
```python
from interface import *

class Span(SpanBase):
    def __init__(self, traceID, spanID, parentID, startTime, elapsedTime, serviceName, callType):
        self.traceID = traceID
        self.spanID = spanID
        self.parentID = parentID
        self.startTime = startTime
        self.elapsedTime = elapsedTime
        self.serviceName = serviceName
        self.callType = callType
        
    def getTraceId(self):
        return self.traceID
    
    def getSpanId(self):
        return self.spanID
    
    def getParentId(self):
        return self.parentID
    
    def getElapsedTime(self):
        return self.elapsedTime
    
    def getSpanLabel(self):
        return self.serviceName + ":" + self.callType

class Trace(TraceBase):
    def __init__(self, traceID, spans, abnormal=False):
        self.traceID = traceID
        self.spanNum = len(spans)
        self.spans = spans
        self.abnormal = abnormal
        
    def getTraceId(self):
        return self.traceID
    
    def getSpanNum(self):
        return self.spanNum
    
    def getSpans(self):
        return self.spans
```

## 用法

```python
from sieve import *

sieve = Sieve(tree_num=50, tree_size=128, k=50, threshold=0.3)
samples = []
count = 0
for trace in traces:
    count += 1
    if sieve.isSample(trace):
        samples.append(trace)
    # 每处理128条trace降一次维
    if count % 128 == 0:
        print('before compact: %d' % sieve.getEncodeLength(), end=', ')
        sieve.compact()
        print('after compact: %d' % sieve.getEncodeLength())
```
