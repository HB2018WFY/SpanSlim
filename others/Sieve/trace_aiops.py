import json
import copy

class Span:
    def __init__(self, callType, startTime, elapsedTime, success, traceId, spanId, parentSpanId, cmdbId, serviceName, dsName):
        self.callType = callType
        self.startTime = int(startTime)
        self.elapsedTime = int(elapsedTime)
        self.success = success
        self.traceId = traceId
        self.spanId = spanId
        self.parentSpanId = parentSpanId
        self.cmdbId = cmdbId
        self.serviceName = serviceName
        self.dsName = dsName
    
    def getCallType(self):
        return self.callType
    
    def getElapsedTime(self):
        return self.elapsedTime
    
    def getTraceID(self):
        return self.traceId
    
    def getSpanID(self):
        return self.spanId
    
    def getParentID(self):
        return self.parentSpanId
    
    # cmdbId、serviceName、dsName拼接作为span的label，无后缀版本
    def getSpanLabel(self):
        cmdbId = '_'.join(self.cmdbId.split('_')[:-1]) if self.cmdbId != '' else '_'
        serviceName = '_'.join(self.serviceName.split('_')[:-1]) if self.serviceName != '' else '_'
        dsName = '_'.join(self.dsName.split('_')[:-1]) if self.dsName != '' else '_'
        return cmdbId + ':' + serviceName + ':' + dsName 
    
    def buildSpan(record):
        callType = record['callType']
        startTime = record['startTime']
        elapsedTime = record['elapsedTime']
        success = record['success']
        traceId = record['traceId']
        spanId = record['id']
        parentSpanId = record['pid']
        serviceName = record['serviceName']
        cmdbId = record['cmdb_id']
        dsName = record['dsName']
        return Span(callType, startTime, elapsedTime, success, traceId, spanId, parentSpanId, cmdbId, serviceName, dsName)
    
    def serialize(span):
        return json.dumps(span.__dict__)
    
    def deserialize(spanStr):
        spanDict = json.loads(spanStr)
        return Span(spanDict['callType'],
                    spanDict['startTime'], 
                    spanDict['elapsedTime'], 
                    spanDict['success'], 
                    spanDict['traceId'], 
                    spanDict['spanId'], 
                    spanDict['parentSpanId'],
                    spanDict['cmdbId'],
                    spanDict['serviceName'],
                    spanDict['dsName'])
    
    def toString(self):
        return ("callType: " + str(self.callType) + ",\n" +
                "startTime: " + str(self.startTime) + ",\n" +
                "elapsedTime: " + str(self.elapsedTime) + ",\n" +
                "success: " + str(self.success) + ",\n" +
                "traceId: " + str(self.traceId) + ",\n" +
                "spanId: " + str(self.spanId) + ",\n" +
                "parentSpanId: " + str(self.parentSpanId) + ",\n" +
                "cmdbId: " + str(self.cmdbId) + ",\n" +
                "serviceName: " + str(self.serviceName) + ",\n" +
                "dsName: " + str(self.dsName))
        
class Trace:
    def __init__(self, traceId, abnormal, elapsedTime, spanNum, spans, pattern, startTime):
        self.traceId = traceId
        self.abnormal = abnormal
        self.elapsedTime = int(elapsedTime)
        self.spanNum = spanNum
        self.spans = spans
        self.pattern = pattern
        self.startTime = startTime
        
    def get(self, name):
        return self.__dict__[name]
    
    def getTraceID(self):
        return self.traceId
    
    def getSpanNum(self):
        return self.spanNum
    
    def getSpans(self):
        return self.spans
        
    def buildTrace(group, abnormal):
        traceId = group['traceId'].iat[0]
        elapsedTime = group[group['callType'] == 'OSB']['elapsedTime'].iat[0]
        spanNum = group.shape[0]
        spans = group.apply(Span.buildSpan, axis=1).to_list()
        pattern = ''#.join(group['serviceName'].str.cat(group['dsName']).to_list())
        startTime = min([span.startTime for span in spans])
        return Trace(traceId, abnormal, elapsedTime, spanNum, spans, pattern, startTime)

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
        return Trace(traceDict['traceId'],
                     traceDict['abnormal'],
                     traceDict['elapsedTime'],
                     traceDict['spanNum'],
                     spans,
                     traceDict['pattern'],
                     traceDict['startTime'])
    
    def toString(self):
        return ("traceId: " + str(self.traceId) + ",\n" +
                "abnormal: " + str(self.abnormal) + ",\n" +
                "elapsedTime: " + str(self.elapsedTime) + ",\n" +
                "spanNum: " + str(self.spanNum) + ",\n" +
                "spans: [\n" +
                ''.join(["\t{" + sp.toString().replace('\n', '') + "},\n" for sp in self.spans]) +
                "],\n" +
                "startTime: " + str(self.startTime) + ",\n" +
                "pattern: " + str(self.pattern))
