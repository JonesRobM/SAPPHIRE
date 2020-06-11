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

        'base_dir' : '../../20/March/CMD/AuPt/Au2057Pt55Melt/25/Sim-4009/',

        'movie_file_name' : 'movie.xyz',

        'energy_file_name' : 'energy.out',

        'Path_to_Pot' : '../../20/March/CMD/AuPt/Au_Pt.pot',

        

        'Homo' : ['Au', 'Pd'], 'HomoQuants' : [ 'HoPDF', 'HoRDF', 'CoM', 'HoAdj' ], 

        'Hetero' : True, 'HeteroQuants' : [ 'HePDF', 'HeRDF', 'HeAdj' ],

        

        'Start' : 0, 'End' : None, 'Step' : 10, 'Skip' : 50, 'UniformPDF' : False, 'Band' : 0.05,

        

        'HCStats' : True,

        

        'SimTime': True, 'EPot': True, 'ETot' : True, 

        'EKin' : True, 'EDelta' : True, 'MeanETot' : True, 'Temp' : True

        }



Quantities = {

        'euc' : None, 'rdf' : None, 'pos' : None, 'cna' : None, 

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



"""

Press = Pressure.pressure(System, Data.metadata)

movie_file = Press.readMovieFileXYZ()

pot_file = Press.readPotentialFile()

Press.pressureMain(movie_file, pot_file)

"""



New_Agcn = TrajAGCN.Trajagcn(Data.metadata, System)

New_Agcn.edit_movie()

New_Agcn.New_File()



Meta = Data.analyse(Analysis)



with open(System['base_dir']+"MetaTrial.csv", "wb") as file:

    pickle.dump(Data.metadata, file)


