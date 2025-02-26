import copy
import json

class Span:
    def __init__(self, traceID, spanID, parentID, startTime, elapsedTime, serviceName, success, thrown, callType):
        self.traceID = traceID
        self.spanID = spanID
        self.parentID = parentID
        self.startTime = startTime
        self.elapsedTime = elapsedTime
        self.serviceName = serviceName
        self.success = success
        self.thrown = thrown
        self.callType = callType
        
    def getTraceID(self):
        return self.traceID
    
    def getSpanID(self):
        return self.spanID
    
    def getParentID(self):
        return self.parentID
    
    def getElapsedTime(self):
        return self.elapsedTime
    
    def getCallType(self):
        return self.callType
    
    def getSpanLabel(self):
        return self.serviceName + ":" + self.callType
        
    def buildSpan(record):
        traceID = record['traceId']
        spanID = record['id']
        parentID = record['pid']
        startTime = record['startTime']
        elapsedTime = record['elapsedTime']
        serviceName = record['serviceName']
        success = record['success']
        callType = record['callType']
        if ('thrown' in record):
            thrown = True
        else:
            thrown = False
        
        return Span(traceID, spanID, parentID, startTime, elapsedTime, serviceName, success, thrown, callType)
        
    def serialize(span):
        return json.dumps(span.__dict__)
    
    def deserialize(spanStr):
        spanDict = json.loads(spanStr)
        return Span(spanDict['traceID'],
                    spanDict['spanID'], 
                    spanDict['parentID'], 
                    spanDict['startTime'], 
                    spanDict['elapsedTime'],
                    spanDict['serviceName'],
                    spanDict['success'],
                    spanDict['thrown'],
                    spanDict['callType'])
    
class Trace:
    def __init__(self, traceID, spans, abnormal=False):
        self.traceID = traceID
        self.spanNum = len(spans)
        self.spans = spans
        self.abnormal = abnormal
        
    def getTraceID(self):
        return self.traceID
    
    def getSpanNum(self):
        return self.spanNum
    
    def getSpans(self):
        return self.spans
        
    def serialize(trace):
        traceDict = copy.deepcopy(trace.__dict__)
        spansDict = []
        for sp in trace.spans:
            spansDict.append(Span.serialize(sp))
        traceDict['spans'] = spansDict
        return json.dumps(traceDict)
    
    def deserialize(traceStr):
        traceDict = json.loads(traceStr)
        spans = [Span.deserialize(sp) for sp in traceDict['spans']]
        return Trace(traceDict['traceID'],
                     spans, traceDict.get('abnormal', False))