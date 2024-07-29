import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from matplotlib.tri import Triangulation
import json
from math import cos,sin,tan,pi,atan
from scipy.interpolate import griddata
from typing import Iterable,Callable,List,Annotated, Literal
from inlettypes import RakePositions, OneProbe, DataPoint

def arctan2(xy:Iterable) -> float:
    x = xy[0]
    y = xy[1]
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

def sumsquare(xy:Iterable[float]) -> float:
    return (xy[0]**2 + xy[1]**2)**.5

def carToPolar(xy:Iterable[float]) -> list[float]:
    return [sumsquare(xy),arctan2(xy)]

def carToPolar2(x:float,y:float) -> list:
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)
    return [r,theta]
    
def loopCarToPolar(xylist:list)-> list:
    intlist = []
    for pair in xylist:
        intlist.append(carToPolar(pair))
    return intlist

def polarToCar(rtheta:Iterable[float]) -> list[float]:
    r = rtheta[0]
    theta = rtheta[1]
    return [r * cos(theta), r * sin(theta)]

def loopPolarToCar(rthetalist:Iterable) -> list:
    intlist = []
    for pair in rthetalist:
        intlist.append(polarToCar(pair))
    return intlist

def plotPolarFunction(func:Callable,maxrad:float,resolution:int = 200):
    xvalues = np.linspace(-maxrad,maxrad,resolution)
    yvalues = np.linspace(-maxrad,maxrad,resolution)
    xg,yg = np.meshgrid(xvalues,yvalues)
    r,theta = carToPolar2(xg,yg)

    def tempfunc(r,theta):
        if r > maxrad:
            return 0
        else: 
            return func(r,theta)
    newfunc = np.vectorize(tempfunc)
    zg = newfunc(r,theta)
    fig,ax = plt.subplots(figsize=(6,6))
    ax.contourf(xg,yg,zg)

class Array:
    """
    Class to define a Array Class. Stores the base locations of all of the rakes in an array.
    """
    def __init__(self,od:float=1):
        self.od = od
    def read_array_json(self,filepath:str) -> None:
        '''
        Reads a json file defining a  array
        '''
        with open(filepath,'r') as fh:
            tempdir = json.load(fh)
            self.od = tempdir["OD"]
            self.arraydf= pd.DataFrame.from_dict(tempdir['rakepositions'])
            self.arraydf['x'] = self.arraydf.apply(lambda row: self.od*row['r']*cos(row['theta'] * pi/180),axis =1)
            self.ararydf['y'] = self.arraydf.apply(lambda row: self.od*row['r']*sin(row['theta'] * pi/180),axis =1)
            self.arraydf.set_index('name')

        return None

    def create_figure(self,rakepts:bool = False) -> None:
        '''
        Method to create a figure with the rake locations plotted
        '''
        self.fig,self.ax = plt.subplots(figsize=(8,8))
        self.ax.set_xlim([-self.od,self.od])
        self.ax.set_ylim([-self.od,self.od])
        circle = Circle((0,0),self.od,fill=False,ec='black')
        self.ax.add_patch(circle)

        if rakepts is True:
            for rakept in self.arraydf.iterrows():
                data = rakept[1]
                self.ax.plot(data['x'],data['y'],marker = '.',color = 'grey')
        return None

class Measurement:
    def __init__(self,array:object,condition:str):
        self.rake = array
        self.measurements = []
   
    def record(self,offset:float,meas:DataPoint) -> None:
        tempdf = self.rake.arraydf
        r = tempdf.loc[tempdf.name == meas.name,'r'].values[0]
        theta = tempdf.loc[tempdf.name == meas.name,'theta'].values[0] + offset
        self.measurements.append([r,theta,meas.yaw,meas.pitch,meas.pt])

    def recordout(self,filepath:str) -> None:
        outlist = []
        for onemeas in self.measurements:
            outlist.append({'r':float(onemeas[0]),'theta':float(onemeas[1]),'yaw':float(onemeas[2]),'pitch':float(onemeas[3]),'pt':float(onemeas[4])})
        with open(filepath,'w') as fh:
            json.dump(outlist,fh)

    def simMeasurement(self,func:Callable,offsets=[0]) -> None:
        '''
        Method to simulate a measurement based on a function
        '''
        for offset in offsets:
            for index,probehead in self.rake.arraydf.iterrows():
                name = probehead['name']
                r = probehead['r']
                theta = probehead['theta'] * pi/180 + offset * pi/180
                measurement = func(r,theta)
                self.record(offset,DataPoint(name=name,yaw=measurement,pitch=measurement,pt=measurement))

    def plotMeasurement(self,mesurement:Literal['yaw','pitch','pt']) -> None:
        temparray = np.array(self.measurements)
        radii = temparray[:,0]
        angles =  temparray[:,1] * pi/180
        
        x = (radii * np.cos(angles)).flatten()
        y = (radii * np.sin(angles)).flatten()
        if mesurement == 'yaw':
            z = temparray[:,2]
        elif mesurement == 'pitch':
            z = temparray[:,3]
        elif mesurement == 'pt':
            z = temparray[:,4]
        else:
            raise ValueError
        triang = Triangulation(x, y)
        #triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1), y[triang.triangles].mean(axis=1)) <  self.rake.od)
        
        fig,ax = plt.subplots(figsize=(8,8))
        ax.set_aspect('equal')
        
        tcf = ax.tricontourf(triang,z)
        fig.colorbar(tcf)



        

