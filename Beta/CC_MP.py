#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:47:39 2020

@author: k1899676
"""

import numpy as np

import multiprocessing as mp

from ase.io import read

import time

from scipy.stats import norm

def distance(a, b):
    
    """ Robert
    
    A simple distance function which takes arguments of
    
    a, b
        These are expected to be arrays containing three elements
        (x, y, z)
        Being the respective euclidean coordinates for atoms a and b
        at a given point in time.
        
    Reurns a single float being the euclidean distance between the atoms.
    
    """
    
    dx = abs(a[0] - b[0])
     
    dy = abs(a[1] - b[1])
     
    dz = abs(a[2] - b[2])
 
    return np.sqrt(dx**2 + dy**2 + dz**2)

def Euc_Dist2(i, positions):
    Distances = [distance(positions[i],positions[j]) for j in range(i+1,len(positions))]

    return Distances
    
pool = mp.Pool(mp.cpu_count())

results = []

def collect_result(result):
    global results
    results.append(result)
    
def gauss2(i, Data,Band, Space,  mon=False):
    A= [(norm.pdf(Space, Data[i],Band))]
    return A
    
File = read("../../20/March/CMD/Au1103Pt309/Melt/100/Sim-1345/movie.xyz", index=0)

pos = File.get_positions()

tick = time.time()

for i in range(len(pos)-1):
    pool.apply_async(Euc_Dist2, args = (i, pos), callback=collect_result)

pool.close()
pool.join() 
print(time.time()-tick)

import DistFuncs

tick = time.time()
res = DistFuncs.Euc_Dist(pos)
print(time.time()-tick)

import functools
import operator

tick=time.time()
Data = functools.reduce(operator.iconcat, results, [])

Space = np.linspace(2, 8, 100); Data=np.asarray(Data)
Data = [elem for elem in Data if 1 < elem < 9]

results = []

def collect_results(result):
    global results
    results.append(result)

pool = mp.Pool(mp.cpu_count())

results = pool.starmap_async(gauss2, [(i, Data, 0.05, Space) for i in range(len(Data))]).get()


pool.close()

#A = functools.reduce(operator.iconcat, results, [])
A = results
Density = np.asarray(np.sum(A, axis=0))
Density = Density/np.trapz(Density, Space) #For normalisation purposes
Density[np.where(Density < 0.01)] = 0
Min = (np.diff(np.sign(np.diff(Density[0]))) > 0).nonzero()[0] + 1 # local min
R_Cut = Space[Min][np.where(Space[Min]>3)][0]

print(time.time()-tick)

from Kernels import Kernels
tick = time.time()
a,b,c = Kernels.Gauss(Data,0.05)
print(time.time()-tick)
    