import pickle
import matplotlib.pyplot as plt
import numpy as np
Base_Dir = '../../../Feb20/AuPd/1127/Melting75/'


with open(Base_Dir+'Meta2.csv', "rb+") as f:
    Data1 = pickle.load(f)
    
with open(Base_Dir+'Meta3.csv', "rb+") as f:
    Data2 = pickle.load(f)

with open(Base_Dir+'Meta4.csv', "rb+") as f:
    Data3 = pickle.load(f)

with open(Base_Dir+'Meta5.csv', "rb+") as f:
    Data4 = pickle.load(f)    
    
