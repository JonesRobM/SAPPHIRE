import ProcessV2
import pickle
import time
import multiprocessing as mp

import TrajAGCN
import Pressure

def collect_result(result):
    global results
    results.append(result)

Supported=[
        'rdf', 'cna', 'adj', 'pdf', 'pdfhomo', 'agcn', 'nn',
        'SimTime', 'EPot', 'ETot', 'EKin', 'EDelta', 'MeanETot', 'Temp'
           ]

System = {
        'base_dir' : '../../20/March/CMD/Au/1103/NVT/700/Sim-1345/',
        'movie_file_name' : 'movie.xyz',
        'energy_file_name' : 'energy.out',
        'Path_to_Pot' : '../../20/March/CMD/Au/Au_Pt.pot',
        'New_agcn_movie' : True,
        
        #'Homo' : ['Au', 'Pd'], 'HomoQuants' : [ 'HoPDF', 'HoRDF', 'CoM', 'HoAdj', 'CoMDist', 'MidCoMDist'], 
        #'Hetero' : True, 'HeteroQuants' : [ 'HePDF', 'HeRDF', 'HeAdj' ],
        
        'Start' : 0, 'End' : 200, 'Step' : 10, 'Skip' : 10, 'UniformPDF' : False, 'Band' : 0.05,
        
        'HCStats' : True,
        
        'SimTime': True, 'EPot': True, 'ETot' : True, 
        'EKin' : True, 'EDelta' : True, 'MeanETot' : True, 'Temp' : True
        }

Quantities = {
        'euc' : None, 'rdf' : None, 'pos' : None,  'CoMDist' : None,
        'adj' : None, 'pdf' : None, 'agcn' : None, 'nn' : None, 'CoM' : None,
        'SimTime': None, 'EPot': None, 'ETot' : None, 
        'EKin' : None, 'EDelta' : None, 'MeanETot' : None, 'Temp' : None
        }

Analysis = {
    "JSD" : ["rdf", "pdf", "cna"],
    "Kullback" : ["rdf", "pdf", "cna"],
    "PStat" : ["rdf", "pdf", "cna"]
    }


Data = ProcessV2.Process(System, Quantities)
Data.Initialising()
Data.run_pdf()
Data.clean_pdf()
Data.run_core()
Data.clean_core()

Data.New_File()

"""
Press = Pressure.pressure(System, Data.metadata)
movie_file = Press.readMovieFileXYZ()
pot_file = Press.readPotentialFile()
Press.pressureMain(movie_file, pot_file)
"""

"""
New_Agcn = TrajAGCN.Trajagcn(Data.metadata, System)
New_Agcn.process_movie()
New_Agcn.New_File()
"""


Meta = Data.analyse(Analysis)

with open(System['base_dir']+"MetaTrial.csv", "wb") as file:
    pickle.dump(Data.metadata, file)
