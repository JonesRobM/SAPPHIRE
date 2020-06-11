import numpy as np
from numpy import *
from scipy.stats import norm
import time
import matplotlib.pyplot as plt

from ase.io import read

from DistFuncs import *
from Kernels import Kernels
"""

        for i in Space:
            P=0
            for j in range(len(Data)):
                X = (Data[j]-i)/Band
                Gauss = (1/np.sqrt(2*np.pi))*np.exp(-(X**2)/2)
                P+=Gauss
            Density.append(P/(len(Data)*Band))
            
"""


filename = '../../../Feb20/Structures/Ico309.xyz'
Data = read(filename, index = 0)

Pos = Data.get_positions()

Distances = Euc_Dist(0, Pos)

Space = linspace(2.5, 3.5, 200); A=[]
for i in range(len(Distances)):
     A.append(norm.pdf(Space, Distances[i],0.05))
Density = np.asarray(np.sum(A, axis=0))
Density = Density/np.trapz(Density, Space) #For normalisation purposes
Density[where(Density < 0.01)] = 0
Min = (diff(sign(diff(Density))) > 0).nonzero()[0] + 1 # local min
R_Cut = Space[Min][where(Space[Min]>3)][0]


"""
tick = time.time()
X,Y = Kernels.Gauss(Distances, 0.075)
Y=np.asarray(Y)
MinNew = (diff(sign(diff(Y))) > 0).nonzero()[0] + 1 # local min
R_CutNew = X[MinNew[1]]
print('This took %.3f seconds to sort out.' %(time.time()-tick))

tick = time.time()
a,b,c = Kernels.Uniform(Distances, 0.25)
print('This took %.3f seconds to sort out.' %(time.time()-tick))

plt.plot(Space, Density); plt.plot(Space[Min], Density[Min], 'o')
plt.show()

plt.plot(X,Y);plt.plot(X[MinNew],Y[MinNew], 'o')
plt.show()


plt.plot(a,b);a = list(a);plt.plot(a[a.index(c)], b[a.index(c)], 'o')
"""