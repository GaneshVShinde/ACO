import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
eps=1e-5
np.seterr(all='print')
class ants(object):
    def __init__(self):
        self.history=[]
    
    def ant_explore(self,pherClose):
        self.history.append(np.where(np.max(np.delete(pherClose,self.history))==pherClose)[0][0])
        return(self.history[-1])
    
    def ant_exploit(self,pherClose):
        print(pherClose)
        r=np.random.uniform()
        df=pd.DataFrame(pherClose,index=range(len(pherClose)))
        pherClose=df.drop(df.index[self.history])
        pherClose=pherClose.cumsum()/np.array(pherClose)[-1]
        self.history.append(pherClose.index[pherClose[0]>r][0])
        return(self.history[-1])

    
    def distance(self,distMat):
        dist=0
        for i in range(len(self.history)-1):
            dist+=distMat[self.history[i],self.history[i+1]]
        return dist

class colony(object):
    def __init__(self,nants,dist_mat):
        self.distance_mat=dist_mat
        self.dist_close=1/dist_mat
        self.pheremone_mat =    np.zeros(self.distance_mat.shape)+0.8
        np.fill_diagonal(self.pheremone_mat,0)
        self.nants=nants
        self.ants = [ants() for i in range(nants)]
        self.q = 0.7
        self.local_update_rate = 0.99
        self.dist_val_perAnt=[]
        self.pher_close = self.pheremone_mat * self.dist_close
        np.fill_diagonal(self.pher_close,0)
        self.found_path =False
        self.randomWalk_dist()
        self.oldBest =0
    
    def randomWalk_dist(self):
        paths=[np.random.choice(self.distance_mat.shape[0], self.distance_mat.shape[0], 
                    replace=False) for _ in range(self.distance_mat.shape[0]) ]
        dist=[]
        for path in paths:
            dst=0
            for i in range(len(path)-1):
                dst+=  self.distance_mat[path[i],path[i+1]]
            dist.append(dst)
        max_=np.max(dist)
        min_=np.min(dist)
        self.m = (30-2)/(min_-max_)
        self.c = 30-self.m*min_

    def travel(self):
        self.pher_close = self.pheremone_mat * self.dist_close
        np.fill_diagonal(self.pher_close,0)
        for ant in self.ants:
            start=np.random.choice(range(self.dist_close.shape[0]))
            ant.history.append(start)
            for i in range(self.distance_mat.shape[0]-1):
                r= np.random.uniform()
                if r<self.q :
                    start=ant.ant_exploit(self.pher_close[start])
                else:
                    start=ant.ant_explore(self.pher_close[start])
            ant.history.append(ant.history[0])
        #local update
        self.pheremone_mat *= self.local_update_rate
        self.global_update()

    
    def calculate_dist(self):
        self.dist_val_perAnt=[]
        for ant in self.ants:
            self.dist_val_perAnt.append(ant.distance(self.distance_mat))

    def run(self):
        while (self.found_path==False): 
            self.ants = [ants() for i in range(self.nants)]
            self.travel()
            self.best_path_by_pheremones()
    
    def update_pheremone_matrix(self,pher_per,history):
        for i in range(len(history)-1):
            self.pheremone_mat[history[i],history[i+1]]+=self.pheremone_mat[history[i],history[i+1]]*pher_per/100
            self.pheremone_mat[history[i+1],history[i]]+=self.pheremone_mat[history[i],history[i+1]]*pher_per/100

    def global_update(self):
        self.calculate_dist()
        max_arg = np.argmax(self.dist_val_perAnt)
        if type(max_arg) =="list":
            max_arg = max_arg[0]
        pher_per = self.m *self.dist_val_perAnt[max_arg] +self.c
        self.update_pheremone_matrix(pher_per,self.ants[max_arg].history)
        if (np.abs(self.oldBest-self.dist_val_perAnt[max_arg])<eps):
            print("************yippee found optimum path***************")
            self.found_path=True
        self.oldBest = self.dist_val_perAnt[max_arg]
        # if min_ != max_:
        #     m = (30-2)/(min_-max_)
        #     c = 30-m*min_
        #     y = m *np.array(self.dist_val_perAnt) +c
        #     for i in range(len(y)):
        #         self.update_pheremone_matrix(y[i],self.ants[i].history)
        #     self.found_parh =False
        # else:
        #     print("************yippee found optimum path***************")
        #     self.found_path=True
    
    def best_path_by_pheremones(self):
        self.calculate_dist()
        minAnt=np.argmin(self.dist_val_perAnt)
        print("optimumPath:",self.ants[minAnt].history)
        print("distance:",self.dist_val_perAnt[minAnt])

        

if __name__=="__main__":
    b = np.random.randint(0,10,size=(10,10))
    b=(b+b.T)/2
    np.fill_diagonal(b,0)
    ant_colony=colony(10,b)
    ant_colony.run()
    
