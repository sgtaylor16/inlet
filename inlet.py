import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
import json
from math import cos,sin,tan,pi,atan
from scipy.interpolate import griddata

def arctan2(xy) -> float:
    x = float(xy[0])
    y = float(xy[1])
    if (x==0.0) and (y > 0):
        return pi/2
    elif (x ==0.0) and (y < 0):
        return 1.5 * pi
    elif (x==0.0) and (y==0.0):
        return 0.0
    #Quadrant 1
    elif (x >= 0) and (y >= 0):
        return atan(y/x)
    #Quadrant 2
    elif (x < 0) and (y > 0):
        return atan(y/x) + pi
    #Quadrant 3
    elif (x < 0) and (y <= 0):
        return atan(y/x) + pi
    #Quadrant 4
    elif (x >= 0) and (y < 0):
        return atan(y/x) + 2 * pi
    else:
        raise ValueError

def sumsquare(xy) -> float:
    return (xy[0]**2 + xy[1]**2)**.5

def carToPolar(xy) -> list:
    return [sumsquare(xy),arctan2(xy)]
    
def polarToCar(r,theta):
    return [r * np.cos(theta * pi/180), r * np.sin(theta * pi/180)]

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
    def read_rake_json(self,filepath:str) -> None:
        '''
        Reads a json file defining a rake array
        '''
        with open(filepath,'r') as fh:
            tempdir = json.load(fh)
            self.od = tempdir["OD"]
            self.rakedf= pd.DataFrame.from_dict(tempdir['rakepositions'])
            self.rakedf['x'] = self.rakedf.apply(lambda row: self.od*row['r']*cos(row['theta'] * pi/180),axis =1)
            self.rakedf['y'] = self.rakedf.apply(lambda row: self.od*row['r']*sin(row['theta'] * pi/180),axis =1)
            self.rakedf.set_index('name')

        return None

    def create_figure(self,rakepts:bool = False) -> None:
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
        self.resultdir = {}
    def set_rake(self,rake:object):
        self.rake = rake
    def blankdf(self):
        return self.rake.rakedf[['name']] 
    def polarFunctionResult(self,name:str,function):
        '''
        Method to take an analytical function and map it's values onto the 
        rake locations.
        '''
        resultdf = self.blankdf().copy()
        for probe in resultdf.index:
            r = self.rake.rakedf.loc[probe,'r']
            theta = self.rake.rakedf.loc[probe,'theta']
            resultdf.loc[probe,'measurement'] = function(r,theta)
        self.resultdir[name] = resultdf

        return resultdf

    def createInterpolator(self,resultname,dx:float=200):
        '''
        Returns a function that utilizes scipy.griddata to return values based on grid
        of known values
        '''
        # Evalue the function at the meshpoints in cartesian coordinates
        xy = None
        def newFunc(x,y):
            #Convert x and y into r and theta
            r,theta = carToPolar(x,y)



        

