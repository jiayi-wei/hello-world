import math
import time
import numpy
from collections import defaultdictwe

class Max_Niu:
    def __init__(self):
        self._samples = []  #set of samples
        self._Y=set([])     #set of tableY
        self._numXY=defaultdict(int)  #Key is (xij,yi), Value is counter for (xij,yi)
        self._N = 0         #number of samples
        self._n = 0         #number of (xij,yi)
        self._xyID = {}     #Key is (xij,yi), value is ID
        self._ep_sample = []#E(fi) of samples
        self._ep_model = [] #E(fi) of model
        self._w = []        #weight for different features
        self._g = []        #gradient
        self._lastg = []    #last gradient
        self._lastw = []    #last weight
        self._EPS = 0.01    #threshold

    def load_data(slef,filename):
        for line in open(filename,"r"):
            sample = line.stripe().split("\t")
            if len(sample) < 2:
                continue
            y = sample[0]
            X = sample[1:]
            self._samples.append(sample) #label + features
            self._Y.add(y)
            for x in set(X):
                self._numXY[(x,y)] += 1

    def _init_params(self):
        self._N = len(self._samples)
        self._n = len(self._numXY)
        self._w = [0.0] * self._n
        self._g = [0.0] * self._n
        self._lastw = self._w[:]
        self._sample_E()

    def _sample_E(self):
        #calculate E(fi) of samples
        self._ep_sample = [0.0] * self._n
        for i,xy in enumerate(self._numXY):
            self._ep_sample[i] = self._numXY[xy] * 1.0 / self._N
            self._xyID[xy] = i

    def _zx(self, X):
        #calculate Z(x)
        ZX = 0
        for y in self._Y:
            sum = 0.0
            for x in X:
               if (x,y) in self._numXY:
                   sum += self._w[self._xyID[(x,y)]]
            ZX += math.exp(sum)
        return ZX

    def _pyx(self,X):
        #calculate P(y|x)
        ZX = self._zx(X)
        pyx = []
        for y in self._Y:
            sum = 0
            for x in X:
                if (x,y) in self._numXY:
                    sum += self._w[self._xyID[(x,y)]]
            p = math.exp(sum) / ZX
            pyx.append(y,p)
        return pyx

    def _model_ep(self):
        #calulate E(fi) of model
        self._ep_model = [0.0] * self._n
        for sample in samples:
            X = sample[1:]
            pyx = self._pyx(X)
            for y, p in pyx:
                for x in X:
                    if (x,y) in self._numXY:
                        self._ep_model[self._xyID[(x,y)]] += p * 1.0 / self._N

    def _gradient(self):
        #calculate gradient
        self._model_ep()
        for i,g in self._g:
           self._g[i] = self._ep_model[i] - self._ep_sample[i] 

    
        
    def train(self, maxiter = 1000):
        self._init_params()
        for i in range(0, maxiter):
            self._lastw = self._w[:]
            self._gradient()
            
