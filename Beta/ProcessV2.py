#Load the required modules for post-processing
import Adjacent
import Kernels
from ase.io import read
import DistFuncs
import AGCN
import CNA

import builtins
import numpy as np
import time
import functools
import operator
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
from contextlib import closing
from inspect import getmembers, isfunction

class Process():
    
    def __init__(self, System=None, Quantities=None):
        self.System=System
        self.Quantities=Quantities
        self.filename=System['base_dir']+System['movie_file_name']
        self.Supported=[
                'rdf', 'cna', 'adj', 'pdf', 'pdfhomo', 'agcn', 'nn', 'CoM', 'CoMDist',
                'SimTime', 'EPot', 'ETot', 'EKin', 'EDelta', 'MeanETot', 'Temp'
                ]

        
        self.metadata={}
    
        self.result_cache={}
        self.Spool = ['pdf', 'PDF', 'rdf', 'RDF', 'R_Cut']
        self.T = time.time()

    
    
    def Initialising(self):
        tick = time.time()
    
    

    

        print('\nReading from the %s file.' %(self.filename), "\n")

    
        """
        In the full version, the user will either be called for these arguments or may simply submit a script.
        I'd like for both versions to be effective. But, for now, it shall remain hard-coded to faciliate 
        debugging and support transparency.
     
        """
    
    
        try:
            self.System['Start']
            if type(self.System['Start']) is not int:
                self.Start = 0
                print('Bad value set for initial frame. Start has been set to 0 by default. Please set an integer value in the future', "\n")
            else:
    
                self.Start = self.System['Start']
                print('Initial frame at %s.' %(self.Start), "\n")
            
        except KeyError:
            self.Start = 0
            print('No value set for initial frame. Start has been set to 0 by default.', "\n")
        
        self.metadata['Start'] = self.Start
    
        try:
            self.System['End']
            if type(self.System['End']) is not int:
                self.End  = len(read(self.filename, index= ':'))
                print('Bad value set for final frame. End has been set to %s, the final frame in this trajectory.\n'
                      'Please set an integer value in the future.' %(self.End), "\n")
            elif self.System['End']<self.Start:
                self.End  = len(read(self.filename, index= ':'))
                print('Bad value set for final frame. End has been set to %s, the final frame in this trajectory.\n'
                      'Please set a value greater than your start frame in the future.' %(self.End), "\n")
            
            else: 
                self.End = self.System['End']
                print('Final frame set to %s.' %(self.End), "\n")
            
        except KeyError:
            self.End  = len(read(self.filename, index= ':'))
            print('No value set for final frame. End has been set to %s, the final frame in this trajectory.'%(self.End), "\n")
            
        self.metadata['End'] = self.End
                
        try:
            self.System['Step']
            if type(self.System['Step']) is not int:
                self.Step = 1
                print('Bad value set for Step. This has been set to 1 by default. Please set an integer value in the future', "\n")
            else:
                self.Step = self.System['Step']
                print('Step set to %s.' %(self.Step), "\n")
        except KeyError:
            self.Step = 1
            print('No value set for Step. The default of 1 has been used instead.', "\n")
            
        self.metadata['Step'] = self.Step
        
        try:
            self.System['Skip']
            if type(self.System['Skip']) is not int:
                self.Skip = int(self.End-self.Start)/25.0
                print('Bad value set for Skip. This has been set to %s such that R_Cut will be evaluated roughly every 25 frames.\n'
                      'Be aware that this may slow down your processing considerably.' %(self.Skip), "\n")
            else:
                self.Skip = self.System['Skip']
                print('self.Skip has been set to %s.' %(self.Skip), "\n")
        except KeyError:
            self.Skip = int(self.End-self.Start)/25.0
            print('No value set for Skip. This has been set to %s such that R_Cut will be evaluated roughly every 25 frames.\n'
                      'Be aware that this may slow down your processing considerably.' %(self.Skip), "\n")
            
        self.metadata['Skip'] = self.Skip
    
        self.Time=int((self.End-self.Start)/self.Step)
        
        print("Reading trajectory from frames %s to %s with an increment of %s." %(self.Start, self.End, self.Step), "\n")
        print('The PDF and, by extension, R_Cut will be evaluated every %s frames.' %(self.Skip), "\n")
        
        
        try:
            self.System['UniformPDF']
            if self.System['UniformPDF'] is False:
                self.PDF = Kernels.Kernels.Gauss
                print('The set method for calculating the PDF is with a Gaussian kernel function.\nBe aware that this method'
                      'is slower than using a Uniform kernel. However; the distribution will be smoother.', "\n")
                self.metadata['pdftype'] = 'Gauss'
                try:
                    self.System['Band']
                    if bool(type(self.System['Band']) is float or int):
                        self.Band = self.System['Band']
                        print('Bandwidth for the Kernel Density Estimator set to %s.' %(self.Band), "\n")
                        self.metadata['Band'] = self.Band
                    else:
                        self.Band = 0.05
                        print('Bad value set for the Kernel function bandwidth. \n Defaulting to % for the Gaussian Kernel Density Estimator.' %(self.Band), "\n")
                        metadata['Band'] = self.Band
                except KeyError:
                    self.Band = 0.05
                    print('Default setting for the Gaussian Kernel Density Estimator is set to %s.' %(self.Band), "\n")
                    self.metadata['Band'] = self.Band
                    
            else:
                self.PDF = Kernels.Kernels.Uniform
                print('The selected method for calculating the PDF is with a Uniform kernel function. \n Be aware that this method'
                      'may yield non-smooth distributions for certain structures. However; this is a much faster calculator.', "\n")
                self.metadata['pdftype'] = 'Uniform'
                try:
                    self.System['Band']
                    if bool(type(self.System['Band']) is float or int):
                        self.Band = self.System['Band']
                        print('Bandwidth for the Kernel Density Estimator set to %s.' %(self.Band), "\n")
                        self.metadata['Band'] = self.Band
                    else:
                        self.Band = 0.25
                        print('Bad value set for the Kernel function bandwidth. \n Defaulting to % for the Uniform Kernel Density Estimator.' %(self.Band), "\n")
                        self.metadata['Band'] = self.Band
                except KeyError:
                    self.Band = 0.25
                    print('Default setting for the Uniform Kernel Density Estimator is set to %s.' %(self.Band), "\n")
                    self.metadata['Band'] = self.Band
                    
        except KeyError:
            self.PDF = Kernels.Kernels.Uniform
            print('The default method for calculating the PDF is with a Uniform kernel function. \n Be aware that this method'
                  'may yield non-smooth distributions for certain structures. However; this is a much faster calculator.',"\n")
            self.metadata['pdftype'] = 'Uniform'
            try:
                self.System['Band']
                if bool(type(self.System['Band']) is float or int):
                    self.Band = self.System['Band']
                    print('Bandwidth for the Kernel Density Estimator set to %.' %(self.Band), "\n")
                    self.metadata['Band'] = self.Band
                else:
                    self.Band = 0.25
                    print('Bad value set for the Kernel function bandwidth. \n Defaulting to % for the Uniform Kernel Density Estimator.' %(self.Band), "\n")
                    self.metadata['Band'] = self.Band
            except KeyError:
                self.Band = 0.25
                print('Default setting for the Uniform Kernel Density Estimator is set to %.' %(self.Band), "\n")
                self.metadata['Band'] = self.Band
    
    
        try: 
            self.System['energy_file_name']
    
            self.energy = np.loadtxt(self.System['base_dir']+self.System['energy_file_name'])
            print('Reading from the %s file.' %(self.System['energy_file_name']), "\n")
        except KeyError:
            print("No energy file given, no quantities related to energy will be evaluated.", "\n")
            self.System['SimTime'] = False; self.System['EPot'] = False; self.System['ETot'] = False; self.System['EKin'] = False
            self.System['EDelta'] = False; self.System['MeanETot'] = False; self.System['Temp'] = False
            
        for x in self.Supported:
            try:
                self.Quantities[x]; print("Calculating the %s." %(x), "\n"); globals()[x] = True
                self.Quantities[x] = np.empty((self.Time,), dtype=object)
                if x == 'pdf':
                    self.Quantities[x] = np.empty((int((self.Time)/(self.Skip)),), dtype=object)
                    self.Quantities['R_Cut'] = np.empty((int((self.Time)/(self.Skip)),), dtype=object)
                if x == 'rdf':
                    self.Quantities[x] = np.empty((int((self.Time)/(self.Skip)),), dtype=object)
            except KeyError:
                print("Will not calculate %s in this run." %(x), "\n"); globals()[x] = False
    
    
            """
        
            Robert:
            
                I know that this quantities dictionary is a mess, the general idea is that this will
                be filled up with information from the class implementation 'automatically'. Because
                this has been thrown together in an afternoon with all of the calculators rejigged for
                ease of calling as opposed to efficiency, it looks ugly as sin! Sorry, team.
            
            
                Supported quantities as of this release:
                
                    Euclidean distance: euc - Pairwise distance between all atoms
                
                    RDF: rdf - radial distribution function
                
                    Common Neighbour Analysis: cna - all signatures and the number of observed counts
                
                    Adjacency matrix: adj - Sparse matrix of truth elements regarding whether or not two atoms are neighbours
                
                    Pair distance distribution function: pdf - Kernel densiy estimator (uniform approximation) for the pdf.
                    This function also sets a new R_Cut each time it is called and calculated.
                
                    Atop generalised coordination number: agcn - ask Fra
                
                    Nearest neighbours: nn - Number of nearest neighbours each atom has. 
            
            """
            
        try: 
            self.System['HCStats']
            if bool(self.System['HCStats']) is not False:
                self.Quantities['h'] = np.empty((self.Time,), dtype=object); globals()['h'] = True
                self.Quantities['c'] = np.empty((self.Time,), dtype=object); globals()['c'] = True
                print("Will be calculating and evaluating collectednes and concertednes of cluster rearrangement.", "\n")
            else:
                print("Will not be calculating collectednes or concertednes of cluster rearrangements.", "\n")
        except KeyError:
            print("Will not be calculating collectednes or concertednes of cluster rearrangements.", "\n")
            
        print("Initialising system environment took %.3f seconds." %(time.time()-tick), "\n")
    

        import CNA
        tick = time.time()

        # Load the simulation dataset to be analyzed.
        self.Masterkey = []
    
        
        
        self.Dataset = read(self.filename, index = 0)
        self.all_positions = self.Dataset.get_positions()
        self.all_atoms = self.Dataset.get_chemical_symbols()

        used=set()
        self.Species = [x for x in self.all_atoms if x not in used and (used.add(x) or True)]
    
        self.NAtoms = len(self.all_atoms)

        tick = time.time()
        self.metadata['Elements'] = self.all_atoms
        self.metadata['Species'] = self.Species
        self.metadata['NSpecies']=len(self.Species)
        self.metadata['NFrames'] = self.Time
        self.metadata['NAtoms'] = self.NAtoms
        
        print("Checking user input for calculating homo properties in this run.", "\n")
        try:
            self.System['Homo']
            
            if self.System['Homo'] is None:
                try:
                    self.System['HomoQuants']
                    if self.System['HomoQuants'] is None:
                        print("No bimetallic properties for homo species will be calculated in this run.", "\n")
                    else:
                        self.System['Homo'] = self.metadata['Species']
                        print("No homo atom species requested, but you wish to calculate bimetallic homo properties." 
                              "\n Instead we shall calculate homo properties for %s and hetero properties for the system." %(self.metadata['Species']), "\n")
                   
                        for x in self.System['HomoQuants']:
                            for y in self.System['Homo']:
                                self.Quantities[x+y] = np.empty((self.Time,), dtype=object); globals()[x+y] = True
                                print("Calculating %s as a homo property." %(x+y), "\n")
                                if 'PDF' in x:
                                    self.Quantities[x+y] = np.empty((int((self.Time*self.Step)/(self.Skip)),), dtype=object)
                                elif 'RDF' in x:
                                    self.Quantities[x+y] = np.empty((int((self.Time*self.Step)/(self.Skip)),), dtype=object)
                except KeyError:
                    print("Will not be calculating any homo properties this run." , "\n") 
                    
                    
            elif False in [x not in self.metadata['Species'] for x in self.System['Homo']]:
                print("Specie entered in homo not found in the system. The only observed species are %s and you have requested to observe %s." 
                      "\n Defaulting to atoms found in the system for evaluation." %(self.metadata['Species'], self.System['Homo']), "\n")
                self.System['Homo'] = self.metadata['Species']
                try:
                    self.System['HomoQuants']
                    if self.System['HomoQuants'] is None:
                        print("No homo properties will be calculated in this run.", "\n")
                    else:
                        for x in self.System['HomoQuants']:
                            for y in self.System['Homo']:
                                self.Quantities[x+y] = np.empty((self.Time,), dtype=object); globals()[x+y] = True
                                print("Calculating %s as a homo property." %(x+y), "\n")
                                if 'PDF' in x:
                                    self.Quantities[x+y] = np.empty((int((self.Time*self.Step)/(self.Skip)),), dtype=object)
                                elif 'RDF' in x:
                                    self.Quantities[x+y] = np.empty((int((self.Time*self.Step)/(self.Skip)),), dtype=object)
                except KeyError:
                    print("Will not be calculating any homo properties this run as no qauntities have been given to calculate." , "\n") 
                    
                    
            else:
                print("Homo atom properties will be caluclated for %s in this run." %(self.System['Homo']), "\n")
                try:
                    self.System['HomoQuants']
                    if self.System['HomoQuants'] is None:
                        print("No bimetallic properties will be calculated in this run as none have been requested.", "\n")
                    else:
                        for x in self.System['HomoQuants']:
                            for y in self.System['Homo']:
                                self.Quantities[x+y] = np.empty((self.Time,), dtype=object); globals()[x+y] = True
                                print("Calculating %s as a homo property." %(x+y), "\n")
                                if 'PDF' in x:
                                    self.Quantities[x+y] = np.empty((int((self.Time*self.Step)/(self.Skip)),), dtype=object)
                                elif 'RDF' in x:
                                    self.Quantities[x+y] = np.empty((int((self.Time*self.Step)/(self.Skip)),), dtype=object)
                except KeyError:
                    print("Will not be calculating any homo properties this run." , "\n")
                    
                    
        except KeyError:
            print("No homo atoms have been requested for calculation. Checking if bimetallic properties have been requested.", "\n")
            
            try:
                self.System['HomoQuants']
                if self.System['HomoQuants'] is None:
                    print("No homo properties have been requested, either. Continuing to calculate whole system properties, only.", "\n")
                else:
                    print("You have requested to calculate %s while not asking for any atoms. Defaulting to considering all species identified in the system." %(self.System['HomoQuants']), "\n")
                    self.System['Homo'] = self.metadata['Species']
                    
                    for x in self.System['HomoQuants']:
                        for y in self.System['Homo']:
                            self.Quantities[x+y] = np.empty((self.Time,), dtype = object); globals()[x+y] = True
                            print("Calculating %s as a homo property." %(x+y), "\n")
                            if 'PDF' in x:
                                self.Quantities[x+y] = np.empty((int((self.Time*self.Step)/(self.Skip)),), dtype = object)
                            elif 'RDF' in x:
                                self.Quantities[x+y] = np.empty((int((self.Time*self.Step)/(self.Skip)),), dtype=object)
                            
            except KeyError:
                print("No homo quantities have been requested, either.", "\n")
        
        print("Finished evaluating user input for homo atomic properties." , "\n")
        
        print("Checking user input for hetero atomic species.", "\n")
        
        try:
            self.System['Hetero']
            if self.System['Hetero'] is not True:
                print("Bad input detected for the 'Hetero' argument'. \n Checking if the user has requested hetero quantities to calculate.", "\n")
                try: 
                    self.System['HeteroQuants']
                    if self.System['HeteroQuants'] is None:
                        print("Bad input variable decalred for calculating hetero quantities. Nothing hetero will happen here, today!", "\n")
                    else:
                        print("User has requested hetero quantities without specifying the desire to do so. We shall assume that this is an error and calculate anyway.", "\n")
                        self.System['Hetero'] = True
                        for x in self.System['HeteroQuants']:
                            self.Quantities[x] = np.empty((self.Time,), dtype = object); globals()[x] = True
                            print("Calculating %s as a hetero property." %(x), "\n")
                            if 'PDF' in x:
                                self.Quantities[x] = np.empty((int((self.Time*self.Step)/(self.Skip)),), dtype = object)
                            elif 'RDF' in x:
                                self.Quantities[x] = np.empty((int((self.Time*self.Step)/(self.Skip)),), dtype =object)
                except KeyError:
                    print("No hetero quantities requested and so none shall be calculated.", "\n")
            
            else:
                print("Hetero quantities have been requested by the user.", "\n")
                try:
                    self.System['HeteroQuants']
                    if self.System['HeteroQuants'] is None:
                        print("Bad input variable decalred for calculating hetero quantities. Nothing hetero will happen here, today!", "\n")
                    else:
                        print("User has requested hetero quantities.", "\n")
                        self.System['Hetero'] = True
                        for x in self.System['HeteroQuants']:
                            self.Quantities[x] = np.empty((self.Time,), dtype = object); globals()[x] = True
                            print("Calculating %s as a hetero property." %(x), "\n")
                            if 'PDF' in x:
                                self.Quantities[x] = np.empty((int((self.Time*self.Step)/(self.Skip)),), dtype = object)
                            elif 'RDF' in x:
                                self.Quantities[x] = np.empty((int((self.Time*self.Step)/(self.Skip)),), dtype =object)
                except KeyError:
                    print("No hetero quantities requested and so none shall be calculated.", "\n")
        except KeyError:
            print("No input variable declared for 'Hetero' calculations. Checking if user has requested quantities without specifying the wish to calculate.", "\n")
            try:
                self.System['HeteroQuants']
                if self.System['HeteroQuants'] is None:
                    print("Bad input variable decalred for calculating hetero quantities. Nothing hetero will happen here, today!", "\n")
                else:
                    print("User has requested hetero quantities.", "\n")
                    self.System['Hetero'] = True
                    for x in self.System['HeteroQuants']:
                        self.Quantities[x] = np.empty((self.Time,), dtype = object); globals()[x] = True
                        print("Calculating %s as a hetero property." %(x), "\n")
                        if 'PDF' in x:
                            self.Quantities[x] = np.empty((int((self.Time*self.Step)/(self.Skip)),), dtype =object)
                        elif 'RDF' in x:
                            self.Quantities[x] = np.empty((int((self.Time*self.Step)/(self.Skip)),), dtype =object)
            except KeyError:
                print("No hetero quantities requested and so none shall be calculated.", "\n")
                
                
        print("Finished evaluating input arguments for homo/hetero calculations.", "\n")
                   
        #This block initialises the metadata
        for key in self.Quantities:
            self.metadata[key] = self.Quantities[key]
        print("Initialising Metadata took %.3f seconds." %(time.time() - tick),"\n")
        print("This system contains", self.NAtoms, "atoms","\n",
              "consisting of", self.Species, "as present atomic species.","\n")
        
        
        self.L1 = list(range(self.metadata['Start'], self.metadata['End'], self.metadata['Skip']*self.metadata['Step']))
        self.L2 = list(range(self.metadata['Start'], self.metadata['End'], self.metadata['Step']))
        self.L3 = [x for x in self.L2 if x not in self.L1]
        
        if self.System['New_agcn_movie'] is True:
            self.New_Obj = np.empty((self.Time,), dtype=object)
        
        
        
        
    def calculate(self, i):
        T0=time.time()
    

        temptime=time.time()
        self.All_Atoms = read(self.filename, index = i)
        self.result_cache['pos'] = self.All_Atoms.get_positions()
        self.result_cache['euc'] = DistFuncs.Euc_Dist(self.result_cache['pos'])
          
        
        #All RDF calculations performed in the following block
        if i%(self.Skip*self.Step)==0 and bool(globals()['rdf']) is True: 
            self.result_cache['rdf'] = DistFuncs.RDF(self.result_cache['pos'], 100, 10.0)
            self.metadata['rdf'][int(i/(self.Step*self.Skip))] = self.result_cache['rdf'] 
            try:
                if bool(bool(self.System['Homo'])*bool('HoRDF' in self.System['HomoQuants'])) is True:
                    for x in self.System['Homo']:
                        self.result_cache['homopos'+x] = DistFuncs.get_subspecieslist(x, self.metadata['Elements'], self.result_cache['pos'])
                        self.metadata['HoRDF'+x][int(i/(self.Step*self.Skip))] = DistFuncs.RDF(self.result_cache['homopos'+x])
            except KeyError:
                pass
            try:
                if bool(bool(self.System['Hetero'])*globals()['HeRDF']) is True:
                    self.metadata['HeRDF'][int(i/(self.Step*self.Skip))] = DistFuncs.RDF(self.result_cache['pos'], Res=100, R_Cut=10.0, Hetero = True, 
                                                                          Species = self.metadata['Species'], Elements = self.metadata['Elements'])
            except KeyError:
                pass
    
        #All PDF calculations performed in the following block
        if i%(self.Skip*self.Step)==0 and bool(globals()['pdf']) is True:
            self.result_cache['pdf'] = self.PDF(self.result_cache['euc'], self.Band)
            self.metadata['pdf'][int(i/(self.Step*self.Skip))] = self.result_cache['pdf']
            self.R_Cut = self.result_cache['pdf'][-1]
            self.metadata['R_Cut'][int(i/(self.Step*self.Skip))] = self.R_Cut
            print("R_Cut is now set to %s." %(self.R_Cut), "\n")
            try:
                if bool(bool(self.System['Homo'])*bool('HoPDF' in self.System['HomoQuants'])) is True:
                    for x in self.System['Homo']:
                        self.result_cache['homoed'+x] = DistFuncs.Euc_Dist(positions=self.result_cache['pos'], homo = True, specie = x, elements = self.metadata['Elements'])
                        if self.result_cache['homoed'+x] is not None:
                            self.metadata['HoPDF'+x][int(i/(self.Step*self.Skip))] = self.PDF(self.result_cache['homoed'+x], self.Band, mon=True)
                        else:
                            pass
            except KeyError:
                pass
            try:
                if bool(self.System['Hetero']*globals()['HePDF']) is True:
                    self.result_cache['heteropos'] = DistFuncs.Hetero(self.result_cache['pos'], self.metadata['Species'], self.metadata['Elements'])
                    self.result_cache['heterodist'] = functools.reduce(operator.iconcat, self.result_cache['heteropos'], [])
                    if self.result_cache['heterodist'] is not None:
                        self.metadata['HePDF'][int(i/(self.Step*self.Skip))] = self.PDF(self.result_cache['heterodist'], self.Band, mon=True)
                    else:
                        self.metadata['HePDF'][int(i/(self.Step*self.Skip))] = None
                        print("There was an error with the heterogenous distance array. No PDF calculated for frame %s."%(i), "\n")
            except KeyError:
                pass

    
        #This block evaluates all of the CoM calculations
        if bool(globals()['CoM']) is True:
            self.result_cache['CoM'] = self.All_Atoms.get_center_of_mass()
            self.metadata['CoM'][int(i/self.Step)] = self.result_cache['CoM']
            if bool(globals()['CoMDist']) is True:
                self.metadata['CoMDist'][int(i/self.Step)] = DistFuncs.CoM_Dist(self.result_cache['pos'], CoM = self.result_cache['CoM'], homo = False, specie = None, elements = None)
        try:
            if bool(bool(self.System['Homo'])*bool('CoM' in self.System['HomoQuants'])) is True:
                for x in self.System['Homo']:
                    self.metadata['CoM'+x][int(i/self.Step)] = DistFuncs.get_CoM(DistFuncs.get_subspecieslist( specie = x, elements = self.metadata['Elements'], positions = self.result_cache['pos']) )
                    if 'CoMDist' in System['HomoQuants']:
                        self.metadata['CoMDist'+x][int(i/self.Step)] = DistFuncs.CoM_Dist(positions = self.result_cache['pos'], homo=True, specie = x, elements = self.metadata['Elements'])
                        self.metadata['MidCoMDist'+x][int(i/self.Step)] = DistFuncs.CoM_Dist(positions = DistFuncs.get_subspecieslist( specie = x, elements = self.metadata['Elements'], 
                                                                                                                                  positions = self.result_cache['pos']), CoM = self.result_cache['CoM'] )
        except KeyError:
            pass


 
        #This block calculates the CNA signatures for the whole system, only
        if bool(globals()['cna']) is True:
            self.result_cache['cna'] = CNA.get_cnas(i, self.metadata['R_Cut'][int(i/(self.Step*self.Skip))], self.Masterkey, self.filename)
            self.metadata['cna'][int(i/self.Step)] = self.result_cache['cna']
        
        
        #This block evaluates the adjacency matrices for the whole system, homo pair(s), & hetero atoms 
        if bool(globals()['adj']) is True:
            self.result_cache['adj'] = Adjacent.Adjacency_Matrix(self.result_cache['pos'], self.result_cache['euc'], self.metadata['R_Cut'][int(i/(self.Step*self.Skip))])
            self.metadata['adj'][int(i/(self.Step))] = self.result_cache['adj']
        try:
            if bool(bool(self.System['Homo'])*bool('HoAdj' in self.System['HomoQuants'])) is True:
                for x in self.System['Homo']:
                    self.result_cache['HomoED'+x] = DistFuncs.Euc_Dist(self.result_cache['pos'], homo = True, specie = x, elements = self.metadata['Elements'])
                    
                    self.metadata['HoAdj'+x][int(i/self.Step)] = Adjacent.get_coordination(Adjacent.Adjacency_Matrix(
                                                                                               DistFuncs.get_subspecieslist
                                                                                               (
                                                                                               x, self.metadata['Elements'], self.result_cache['pos']
                                                                                               ),
                                                                                               self.result_cache['HomoED'+x], self.metadata['R_Cut'][int(i/(self.Step*self.Skip))]) )
        except KeyError:
            pass
        try:
            if bool(self.System['Hetero']*globals()['HeAdj']) is True:
                self.result_cache['HeDist'] = DistFuncs.Hetero(self.result_cache['pos'], self.metadata['Species'], self.metadata['Elements'])
                if self.result_cache['heteropos'] is not None:
                    self.metadata['HeAdj'][int(i/self.Step)] = Adjacent.get_coordination_hetero(self.result_cache['HeDist'], self.metadata['R_Cut'][int(i/(self.Step*self.Skip))])
                else:
                    self.metadata['HeAdj'] = None
                    print("There was an error with hetero positions, no respective adjacency matrix calculated for frame %s." %(i), "\n")
        except KeyError:
            pass
        
        
    
        #This block evaluates the atop generalised coordination number for the whole system
        if bool(globals()['agcn']*globals()['nn']*globals()['adj']) is True:
            self.Agcn, self.NN = AGCN.agcn_generator(self.result_cache['adj'], NN = True)
            self.metadata['agcn'][int(i/self.Step)] = self.Agcn; self.metadata['nn'][int(i/self.Step)] = self.NN
        elif bool(globals['agcn']*globals()['adj']) is True:
            self.Agcn = AGCN.agcn_generator(self.result_cache['adj'])[0]
            self.metadata['agcn'][int(i/self.Step)] = self.Agcn
        elif bool(globals['nn']*globals()['adj']) is True:
            _,self.NN = AGCN.agcn_generator(self.result_cache['adj'], NN = True)
            self.metadata['nn'][int(i/self.Step)] = self.NN
    
        """
        ##This is simply a progress updater which informs the user how every 5% is getting along.
        if len(self.metadata['adj'])%int((self.End-self.Start)/20) == 0:
            Per = int(/int((self.End-self.Start)/100))
            print("Currently performed %.3f%% of the calculation." %(Per), "\n")
            print('['+int(Per/5)*'##'+(20-int(Per/5))*'  '+']', "\n")
        """
        

        if self.System['New_agcn_movie'] is True:
            self.Temp_aGCN = (np.column_stack( (self.all_atoms,self.result_cache['pos'],self.Agcn) ))


        self.Masterkey.sort()
        self.metadata['masterkey']=self.Masterkey
    
        """
                    
        #############################################################################################
        
        Robert:
            And now we check to see if the users wishes to evaluate any of the quantities
            from the energy file and add them to the metadata.
        
        """
    
    
        if bool(globals()['SimTime']) is True:
            self.metadata['SimTime'][int(i/self.Step)] = self.energy[:,0][int(i)]
    
        if bool(globals()['EPot']) is True:
            self.metadata['EPot'][int(i/self.Step)] = self.energy[:,1][int(i)]
        
        if bool(globals()['ETot']) is True:
            self.metadata['ETot'][int(i/self.Step)] = self.energy[:,2][int(i)]
        
        if bool(globals()['EKin']) is True:
            self.metadata['EKin'][int(i/self.Step)] = self.energy[:,3][int(i)]
        
        if bool(globals()['EDelta']) is True:
            self.metadata['EDelta'][int(i/self.Step)] = self.energy[:,4][int(i)]
        
        if bool(globals()['MeanETot']) is True:
            self.metadata['MeanETot'][int(i/self.Step)] = self.energy[:,5][int(i)]
        
        if bool(globals()['Temp']) is True:
            self.metadata['Temp'][int(i/self.Step)] = self.energy[:,6][int(i)]
        
        
    
        return self.metadata, self.Temp_aGCN
    
    
    def run_pdf(self, cores = mp.cpu_count()-1):
        
        p = mp.Pool(cores)
        self.result_pdf = np.asanyarray(p.map(self.calculate, (self.L1)))
        p.close()
        p.join()
        self.T0=time.time()
        print('Time for completing RCut calculation is %s.' %(time.strftime("%H:%M:%S", time.gmtime((self.T0-self.T)))), "\n")
        self.aGCN_Data = self.result_pdf[:,1]
        self.result_pdf = self.result_pdf[:,0]
        return self.result_pdf, self.aGCN_Data

    
    def clean_pdf(self):
        self.Keyring = list(self.Quantities.keys())
        
        
    
    
    
        for Key in self.Quantities.keys():
            for i in self.L1:
                for code in self.Spool:
                    if code in Key:
                        self.metadata[Key][self.L1.index(i)] = self.result_pdf[self.L1.index(i)][Key][self.L1.index(i)]
                        try:
                            self.Keyring.remove(Key)
                        except ValueError:
                            continue
                            
                            
        for Key in self.Keyring:
            for i in self.L1:
                self.New_Obj[int(i/self.Step)] = self.aGCN_Data[self.L1.index(i)]
                try:
                    self.metadata[Key][int(i/self.Step)] = self.result_pdf[self.L1.index(i)][Key][int(i/self.Step)]
                except TypeError:
                    continue
                    try:
                        self.Keyring.remove(Key)
                    except ValueError:
                        continue
                except IndexError:
                    continue
                    
        self.metadata['masterkey'] = self.result_pdf[0]['masterkey']
        
        for i in self.L1:
            for item in self.result_pdf[self.L1.index(i)]['masterkey']:
                if item not in self.metadata['masterkey']:
                    self.metadata['masterkey'].append(item)
        self.T1 = time.time()
        print('Time for completing RCut clean is %s.' %(time.strftime("%H:%M:%S", time.gmtime((self.T1-self.T0)))), "\n")
        
                
    def run_core(self, cores = mp.cpu_count()-1):
        
        self.Keyring_core=list(self.Quantities.keys())

        p = mp.Pool(cores)
        self.result_core = np.asanyarray(p.map(self.calculate, (self.L3)))
        p.close()
        p.join()
        
        self.T2 = time.time()
        print('Time for completing core calculation is %s.' %(time.strftime("%H:%M:%S", time.gmtime((self.T2-self.T1)))), "\n")
        self.aGCN_Data = self.result_core[:,1]
        self.result_core = self.result_core[:,0]
        return self.result_core, self.aGCN_Data
    
    def clean_core(self):
        
        
        for obj in self.Spool:
            for Key in self.Keyring_core:
                if obj in Key:
                    self.Keyring_core.remove(Key)
                else:
                    continue
        
                    
        for key in self.Keyring_core:
            for i in self.L3:
                self.New_Obj[int(i/self.Step)] = self.aGCN_Data[self.L3.index(i)]
                try:
                    self.metadata[key][int(i/self.Step)] = self.result_core[self.L3.index(i)][key][int(i/self.Step)]
                except TypeError:
                    continue
                    try:
                        self.Keyring.remove(Key)
                    except ValueError:
                        continue
                except IndexError:
                    continue
                    try:
                        self.Keyring.remove(Key)
                    except ValueError:
                        continue

        for i in self.L3:
            for item in self.result_core[self.L3.index(i)]['masterkey']:
                if item not in self.metadata['masterkey']:
                    self.metadata['masterkey'].append(item)
        self.T3 = time.time()
        print('Time for completing core clean is %s.' %(time.strftime("%H:%M:%S", time.gmtime((self.T3-self.T2)))), "\n")
        print('Time for completion is %s.' %(time.strftime("%H:%M:%S", time.gmtime((self.T3-self.T)))), "\n")
                

    def analyse(self, Stat_Tools):
            
        self.Stat_Tools = Stat_Tools
        self.functions_list = [o for o in getmembers(CNA.Dist_Stats) if isfunction(o[1])]
        for i in range(1, int((self.End - self.Start)/self.Step) ):
        
            #This  block calculates the concertedness and collectivity of atom rearrangements    
            if bool(self.System['HCStats']*i) is not False:
                self.result_cache['r'] = Adjacent.R(self.metadata['adj'][i], self.metadata['adj'][i-1])
                self.metadata['h'][i-1] = Adjacent.Collectivity(self.result_cache['r'])
                if not(i<3):
                    self.metadata['c'][i-2] = Adjacent.Concertedness(self.metadata['h'][i-1], self.metadata['h'][i-3])
            
        #This block reconfigures the CNA signatures to include all observed throughout the trajeectory
        
        try:
            self.Quantities['cna']
            self.cna = self.metadata['cna']
            for j in range(0, int( (self.End - self.Start) / self.Step ) ):
                self.metadata['cna'][j] = CNA.get_heights(self.cna, self.metadata['masterkey'], j)
        except KeyError:
            print("No CNA signatures to be calculated.")
        
        
        """
        This next block creates a dictionary whose keys are the analysis tools to be implemented.
        The first entry is the function to be called.
        All of the subsequent entries are the keys, as they appear in the metadata, to be processed.
        """
        
        self.Stat_Keys = self.Stat_Tools.keys()
        self.Meta_Keys = self.metadata.keys()
        self.Calc_Dict = {}
        for obj in self.Stat_Keys:
            for item in self.functions_list:
                if obj.lower() in item[0].lower():
                    self.Calc_Dict[item[0]] = [item[1]]
                    
        for A_Key in self.Stat_Keys:
            for M_Key in self.Meta_Keys:
                for obj in self.Stat_Tools[A_Key]:
                    if obj.lower() in M_Key.lower():
                        if M_Key.lower() is 'pdftype':
                            pass
                        else:
                            self.Calc_Dict[A_Key].append(M_Key)
            self.Calc_Dict[A_Key].remove('pdftype')
                            
        """
        This next block reads over the previously created dictionary and then doctors the relevant
        metadata entry to be ready for processing.
        
        That is to say, that the heights of the distributions are to be analysed as the x-axis are 
        largely uniform across the sample.
        """
        
        for A_Key in self.Stat_Keys:
            for obj in self.Calc_Dict[A_Key][1:]:
                self.metadata[A_Key+obj] = np.empty((len(self.metadata[obj]),), dtype=object)
                Init = self.metadata[obj][0][1] # This is the initial distribution to which we shall make comparrisons 
                for frame in range( len(self.metadata[obj]) ):
                    try:
                        Temp = self.metadata[obj][frame][1] #This is the y-axis of the distribution under consideration
                        
                        self.metadata[A_Key+obj][frame] = self.Calc_Dict[A_Key][0](Init, Temp)
                    except TypeError:
                        continue
        return self.metadata
                    
    def New_File(self, new_movie='agcn_movie.xyz'):
        with open(self.System['base_dir'] + new_movie, 'w+') as self.movie:
            self.movie.write(str(self.metadata['NAtoms']) +'\n')
            self.movie.write('\t' + "This was made by Jones' post-processing code." + '\n')
            for Frame in self.New_Obj:
                for items in Frame:
                    self.movie.write(' \t'.join(str(item) for item in items) +'\n')
                self.movie.write(str(self.metadata['NAtoms']) + '\n')
                self.movie.write('\n')
        print("This movie has been saved as %s in %s.\n" %(new_movie, self.System['base_dir']))
        print('Time for writing new aGCN trajectroy is %s.' %(time.strftime("%H:%M:%S", time.gmtime((time.time()-self.T0)))), "\n")