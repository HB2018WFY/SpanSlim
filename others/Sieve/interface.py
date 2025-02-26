from abc import ABCMeta, abstractmethod

class TraceBase(metaclass=ABCMeta):
    @abstractmethod
    def getSpanNum(self):
        pass
    
    @abstractmethod
    def getSpans(self):
        pass
    
class SpanBase(metaclass=ABCMeta):
    @abstractmethod
    def getSpanId(self):
        pass
    
    @abstractmethod
    def getParentId(self):
        pass
    
    @abstractmethod
    def getSpanLabel(self):
        pass
    
    @abstractmethod
    def getElapsedTime(self):
        pass
