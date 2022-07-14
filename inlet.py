import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
import json
from math import cos,sin,tan,pi
from scipy.interpolate import griddata

def carToPolar(x,y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x)
    return [r,theta]

def plotPolarFunction(func,maxrad:float,resolution:int = 200):
    xvalues = np.linspace(-maxrad,maxrad,resolution)
    yvalues = np.linspace(-maxrad,maxrad,resolution)
    xg,yg = np.meshgrid(xvalues,yvalues)
    r,theta = carToPolar(xg,yg)

    def tempfunc(r,theta):
        if r > maxrad:
            return 0
        else: 
            return func(r,theta)
    newfunc = np.vectorize(tempfunc)
    zg = newfunc(r,theta)
    fig,ax = plt.subplots(figsize=(6,6))
    ax.contourf(xg,yg,zg)

class Rake:
    def __init__(self):
        self.od = 0
    def read_rake_json(self,filepath:str):
        '''
        Reads a json file defining a rake array
        '''
        with open(filepath,'r') as fh:
            tempdir = json.load(fh)
            self.od = tempdir["OD"]
            self.rakedf= pd.DataFrame.from_dict(tempdir['rakepositions'])
            self.rakedf['x'] = self.rakedf.apply(lambda row: self.od*row['r']*cos(row['theta'] * pi/180),axis =1)
            self.rakedf['y'] = self.rakedf.apply(lambda row: self.od*row['r']*sin(row['theta'] * pi/180),axis =1)

        return None

    def create_figure(self,rakepts:bool = False):
        self.fig,self.ax = plt.subplots(figsize=(8,8))
        self.ax.set_xlim([-self.od,self.od])
        self.ax.set_ylim([-self.od,self.od])
        circle = Circle((0,0),self.od,fill=False,ec='black')
        self.ax.add_patch(circle)

        if rakepts is True:
            for rakept in self.rakedf.iterrows():
                data = rakept[1]
                self.ax.plot(data['x'],data['y'],marker = '.',color = 'grey')
        return None

class Result:
    def __init__(self):
        self.resultsdir = {}
    def set_rake(self,rake:object):
        self.rake = rake
    def blankdf(self):
        return self.Rake.rakedf[['name','r','theta']].set_index('name')
    def createInterpolator(self,func,dx:float=200):
        '''
        Returns a function that utilizes scipy.griddata to return values based on grid
        of known values
        '''
        results = []
        for measure in self.rake.rakedf.iterrows():
            temp = measure[1]
            r,theta = carToPolar(temp['x'],temp['y'])
            results.append(func(r,theta))
        results = np.array(results)
        
        def returnfunc(x,y):
            return griddata(self.rake.rakedf[['x','y']]).to_numpy(), results,[x,y]

        return returnfunc

    def createResults(self,name,interp_func,dx:float=200):

        #Create the meshgrid
        results =[]
        for measure in self.rake.rakedf.iterrows():
            temp = measure[1]
            r,theta = carToPolar(temp['x'],temp['y'])
            r = np.sqrt(temp['x']**2 + temp['y']**2)
            theta = np.arctan2(temp['y'],temp['x'])
            results.append(func(r,theta))
        results = np.array(results)
