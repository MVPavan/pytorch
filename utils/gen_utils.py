import json
import numpy as np
from json import JSONEncoder

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.__dict__
        return JSONEncoder.default(self, obj)


class _JsonEncoder:
    def toJSON(self):
        # return json.dumps(self, default=lambda o: o.__dict__, cls=NumpyEncoder)
        return json.dumps(self, cls=NumpyEncoder)
    
    def toDict(self):
        return json.loads(self.toJSON())


class ClassEncoder:
    def toDict(self):
        return self.__dict__
    
    def addDict(self,**kwargs):
        for key, value in kwargs.items():
            self.addVar(key=key, value=value)
    
    def addVar(self,key,value):
        setattr(self, key, value)
    
    def getVar(self,key):
        return getattr(self,key,None)