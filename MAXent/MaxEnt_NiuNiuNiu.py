import math
import time
from numpy import *
import collections

class Max_Niu:
    def __init__(self):
        self._samples = []  #set of samples
        self._Y=set([])     #set of tableY
        self._X = set([])   #set of X
        self._numXY=collections.defaultdict(int)  #Key is (xij,yi), Value is counter for (xij,yi)
        self._N = 0         #number of samples
        self._n = 0         #number of (xij,yi)
        self._xyID = {}     #Key is (xij,yi), value is ID
        self._ep_sample = []#E(fi) of samples
        self._ep_model = [] #E(fi) of model
        self._w = mat(empty([1, self._n]))        #weight for different features
        self._g = mat(empty_like(self._w))        #gradient
        self._lastg = mat(empty_like(self._w))    #last gradient
        self._lastw = mat(empty_like(self._w))    #last weight
        self._p = mat(empty_like(self._w))        #p=-g*B
        self._y = mat(empty_like(self._w))        #y=g - lastg
        self._delta = mat(empty_like(self._w))    #delta=w - lastw
        self._B = mat(empty([self._n, self._n]))  #matrix that replace H-1
        self._lamda = 0.0                    #one dimension to search
        self._EPS = 0.01    #threshold

    def load_data(self,filename):
        for line in open(filename,"r"):
            sample = line.strip().split("\t")
            if len(sample) < 2:
                continue
            y = sample[0]
            X = sample[1:]
            self._samples.append(sample) #label + features
            self._Y.add(y)
            for x in set(X):
                self._X.add(x)
                self._numXY[(x,y)] += 1

    def _init_params(self):
        self._N = len(self._samples)
        self._n = len(self._numXY)
        self._w = mat(zeros(self._n))
        self._g = mat(zeros(self._n))
        self._lastw = mat(zeros(self._n))
        self._lastg = mat(zeros(self._n))
        self._y = self._g - self._lastg
        self._delta = self._w - self._lastw
        self._B = mat(eye(self._n))
        self._sample_E()

    def _sample_E(self):
        #calculate E(fi) of samples
        self._ep_sample = [0.0] * self._n
        for i,xy in enumerate(self._numXY):
            self._ep_sample[i] = self._numXY[xy] * 1.0 / self._N
            self._xyID[xy] = i

    def _zx(self, X):
        #calculate Z(x)
        ZX = 0.0
        for y in self._Y:
            sum = 0.0
            for x in X:
               if (x,y) in self._numXY:
                   sum += self._w[0, self._xyID[(x,y)]]
            ZX += math.exp(sum)
        return ZX

    def _zx_lamda(self, X, lamda):
        #calculate ZX for find lamda
        ZX = 0.0
        for y in self._Y:
            sum = 0.0
            for x in X:
                if (x,y) in self._numXY:
                    sum += self._w[0, self._xyID[(x,y)]] + lamda * self._p[0, self._xyID[(x,y)]]
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
                    sum += self._w[0, self._xyID[(x,y)]]
            p = math.exp(sum) / ZX
            pyx.append((y, p))
        return pyx

    def _model_ep(self):
        #calulate E(fi) of model
        self._ep_model = [0.0] * self._n
        for sample in self._samples:
            X = sample[1:]
            pyx = self._pyx(X)
            for y, p in pyx:
                for x in X:
                    if (x,y) in self._numXY:
                        self._ep_model[self._xyID[(x,y)]] += p * 1.0 / self._N

    def _gradient(self):
        #calculate gradient
        self._model_ep()
        for i in range(0, len(self._ep_model)):
           self._g[0, i] = self._ep_model[i] - self._ep_sample[i] 


    def _threshold(self):
        #judge for threshold
        sum = 0.0
        temp_matric = dot(self._g, self._g.T)
        sum = temp_matric[0, 0]
        result = sum**(1/2)
        if result >= self._EPS:
            return False
        return True
        

    def _get_lamda(self):
        min_flag = 100.0
        for lamda_ in range(0, 100):
            lamda = lamda_ / 100.0
            sum_front = 0.0
            ZX = self._zx_lamda(X, lamda)
            for x in X:
                sum_front += 1.0 / self._N * math.log(ZX) 
            sum_back = 0.0
            for i, xy in enumerate(self._numXY):
                sum_back += self._numXY[xy] * 1.0 / self._N * (self._w[0, self._xyID[xy]] + lamda * self._p[0, self._xyID[xy]])
            diff = sum_front - sum_back
            if diff < min_flag:
                self._lamda = lamda
                min_falg = diff

    def _updata_params(self):
        self._lastw = self._w
        self._w = self._w + self._lamda * self._p
        self._delta = self._w - self._lastw
        self._lastg = self._g
        self._gradient()
        self._y = self._g - self._lastg
        front_down = dot(self._delta, self._delta.T)
        front_up = dot(self._y.T, self._y)
        back_up = dot(dot(self._B, self._delta.T), dot(self._delta, self._B))
        back_down = dot(dot(self._delta, self._B), self._delta.T)
        self._B += (1.0 / front_down[0,0]) * front_up + (1.0 / back_down[0,0]) * back_up
        
    def train(self, maxiter = 10):
        self._init_params()
        self._gradient()
        for i in range(0, maxiter):
            self._p = -1.0 * dot(self._g, self._B)
            self._get_lamda()
            self._updata_params()
            if self._threshold():
                break

    def predict(self ,input):
         X = input.strip().split("\t")
         prob = self._pyx(X)
         return prob
            
if __name__ == "__main__":
    maxent = Max_Niu()
    maxent.load_data('data.txt')
    maxent.train()
    print(maxent.predict("sunny\thot\thigh\tFALSE"))
    print(maxent.predict("overcast\thot\thigh\tFALSE"))
    print(maxent.predict("sunny\tcool\thigh\tTRUE"))
    print(maxent.predict("sunny\tcool\thigh\tFALSE"))
