# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:08:33 2019

@author: jones
"""

import numpy as np
from itertools import groupby
from collections import namedtuple
from ase.io import read


class pressure():
    
    def __init__(self, system, metadata):
        self.system = system
        self.file_loc = self.system['base_dir']+self.system['movie_file_name']
        self.start = self.system['Start']
        self.end = self.system['End']
        self.step = self.system['Step']
        self.pot = self.system['Path_to_Pot']
        self.metadata = metadata


    def readMovieFileXYZ(self):
        """
        Reads a LoDiS movie.xyz file and fetches the coordinates
        for each atom for each frame.
        
        Input:
            path_to_movie: path to movie.xyz
        
        Returns:
            Named tuple read_movie:
                - read_movie.Frames: list of frames; each is an array of atoms each described by [Atom, x, y, z, Col]
                - read_movie.Headers: list of the movie frames headers    
                - pressure is in TPa 10^12 Pa"""
    
        self.read_movie = []
        self.Frames = range(self.start, self.end, self.step)
        for i in self.Frames:
            temp = read(self.file_loc, index = i)
            c = temp.get_chemical_symbols()
            xyz = temp.get_positions()
            self.read_movie.append( np.column_stack( (c, xyz) ) )

        return self.read_movie

    def readPotentialFile(self):
        """
        Reads the .pot file and extracts all parameters from it.
        Including the potential analytical continuation.
        
        Returns:
            potential (named tuple): contains all parameters extracted:
                'AtomTypes P Q A Qsi Cohesion Radius Mass Cutoff dik0 x3 x4 x5 a3 a4 a5'
        """
        
        def readValuesFromLine(line):
            """
            For input line directly read from pot file, removes white spaces and Fortran d0
            and only extract the numerical values
            """
    
            str_sep = [value.replace('d0', '') for value in [i for i in line.replace('\t', ' ').split(' ')\
                                                             if i not in [' ', '\t', '\n']]]
            str_sep = [value for value in str_sep if value!='']
    
            self.values_lst = []
            for elem in str_sep:
                try:
                    self.values_lst.append(float(elem))
                except ValueError:
                    pass
    
            return self.values_lst
    
    
        self.read_file_chars = []
    
        with open(self.pot, 'r') as file:
            for line in file:
                self.read_file_chars.append(line)
        # Delete line jump
    
        # Read all values
        self.read_file_chars = [line[:-1] for line in self.read_file_chars]
    
        self.atoms_type = [elem for elem in self.read_file_chars[2].split(' ') if elem != '']
    
        self.p_val = readValuesFromLine(self.read_file_chars[5])
        self.q_val = readValuesFromLine(self.read_file_chars[6])
        self.a_val = readValuesFromLine(self.read_file_chars[7])
        self.qsi_val = readValuesFromLine(self.read_file_chars[8])
        self.cohe_val = readValuesFromLine(self.read_file_chars[11])
        self.atom_rad_val = readValuesFromLine(self.read_file_chars[12])
        self.mass_val = readValuesFromLine(self.read_file_chars[13])
        self.cutoff_val = readValuesFromLine(self.read_file_chars[16])
    
        #Determines if system is bimetallic by comparing the p values
    
        if self.p_val[0]==self.p_val[1]:
            self.sys_bim = False
        else:
            self.sys_bim = True
    
        self.arete = [self.atom_rad_val[0]*np.sqrt(8)]
        if self.sys_bim:
            self.arete.append(self.atom_rad_val[1]*np.sqrt(8))
            self.arete.append((self.arete[0]+self.arete[1])/2)
    
        # Unit conversions to arete
        self.nn = self.arete/np.sqrt(2)
        self.dik0 = self.atom_rad_val[0]+self.atom_rad_val[1] # Minimal distance between atoms. Sum of atomic radii
        self.dist = [1/np.sqrt(2)]
        if self.sys_bim:
            self.dist.append(self.nn[1]/self.arete[0])
            self.dist.append(self.nn[2]/self.arete[0])
    
    
        # Converts cutoffs    
        self.cutoff_start = self.cutoff_val[0]#/arete[0]
        self.cutoff_end = self.cutoff_val[1]#/arete[0]
    
    
        # Analytical continuation of potential
    
        self.x3 = [0.0, 0.0, 0.0]
        self.x4 = [0.0, 0.0, 0.0]
        self.x5 = [0.0, 0.0, 0.0]
    
        self.a3 = [0.0, 0.0, 0.0]
        self.a4 = [0.0, 0.0, 0.0]
        self.a5 = [0.0, 0.0, 0.0]
    
    
        for i in range(3):
            # Old dik0, maybe to correct with units    dik0 = dist[min(i, len(dist)-1)]
            #dik0 = dist[min(i, len(dist)-1)]
    
            self.ar = -self.a_val[i] * np.exp( -self.p_val[i] * ( ( self.cutoff_start / self.dik0 )-1 ) ) / ( ( self.cutoff_end-self.cutoff_start)**3)
            self.br = -( self.p_val[i] / self.dik0 ) * self.a_val[i] * np.exp( -self.p_val[i] * ( (self.cutoff_start / self.dik0) - 1 ) ) / ( ( self.cutoff_end - self.cutoff_start )**2 )
            self.cr = - ( ( self.p_val[i] / self.dik0 )**2 ) * self.a_val[i] * np.exp( - self.p_val[i] * ( ( self.cutoff_start / self.dik0 ) - 1 ) ) / ( ( self.cutoff_end - self.cutoff_start ) )
    
            self.ab = - self.qsi_val[i] * np.exp( -self.q_val[i] * ( ( self.cutoff_start / self.dik0 ) - 1 ) ) / ( ( self.cutoff_end - self.cutoff_start )**3 )
            self.bb = -( self.q_val[i] / self.dik0 ) * self.qsi_val[i] * np.exp( - self.q_val[i] * ( ( self.cutoff_start / self.dik0 ) - 1 ) ) / ( ( self.cutoff_end - self.cutoff_start )**2 )
            self.cb = - ( ( ( self.q_val[i] / self.dik0 )**2 ) * self.qsi_val[i] * np.exp( - self.q_val[i] * ( ( self.cutoff_start / self.dik0 ) - 1 ) ) / ( ( self.cutoff_end - self.cutoff_start ) ) )
    
            self.x5[i] = ( 12 * self.ab - 6 * self.bb + self.cb ) / ( 2 * ( ( self.cutoff_end - self.cutoff_start )**2 ) )
            self.x4[i] = ( 15 * self.ab - 7 * self.bb + self.cb ) / ( ( ( self.cutoff_end - self.cutoff_start ) ) )
            self.x3[i] = ( 20 * self.ab - 8 * self.bb + self.cb ) / 2
    
            self.a5[i] = ( 12 * self.ar - 6 * self.br + self.cr ) / ( 2 * ( ( self.cutoff_end - self.cutoff_start )**2 ) )
            self.a4[i] = ( 15 * self.ar - 7 * self.br + self.cr ) / ( ( ( self.cutoff_end - self.cutoff_start ) ) )
            self.a3[i] = ( 20 * self.ar - 8 * self.br + self.cr ) / 2
    
    
        self.Potential = namedtuple('Potential','AtomTypes P Q A Qsi Cohesion Radius Mass CutoffStart CutoffEnd dik0 x3 x4 x5 a3 a4 a5')
        self.potential = self.Potential( self.atoms_type, self.p_val, self.q_val, self.a_val, self.qsi_val ,
                                        self.cohe_val, self.atom_rad_val, self.mass_val, self.cutoff_val[0], 
                                        self.cutoff_val[1], self.dik0, self.x3, self.x4, self.x5, self.a3, self.a4, self.a5)
    
        return self.potential

    def getPressureTwoAtoms(self, atom_i, atom_j, potential):
        """
        For two atoms 1 and 2 -- given as arrays of form [Atom, x, y, z, Colour]--
        returns the bonding and repulsion pressures. The potential params are all
        stored in the potential named tuple.
        """
        
        ###### den_i should be set zero only when I change the sum over atom-i but summing over all j
        
        # 1. Determining interaction type
        if bool(atom_i[0] == self.Pot.AtomTypes[0]) * bool(atom_j[0] == self.Pot.AtomTypes[0]):
            interaction_type = 0 # Monometallic interaction between atoms of type 1 (cf Pot file)
        elif bool(atom_i[0] == self.Pot.AtomTypes[1]) * bool(atom_j[0] == self.Pot.AtomTypes[1]):
            interaction_type = 1 # Monometallic interaction between atoms of type 2 (cf Pot file)
        else:
            interaction_type = 2 # Bimetallic interaction
        # 2. Determining interatomic distance
        
        tempi = np.asarray(atom_i[1:], dtype = np.float128) 
        tempj = np.asarray(atom_j[1:], dtype = np.float128)
        
        dist_ij = np.linalg.norm(tempi-tempj)
        
        # 3. Pressure calculations
        if dist_ij<=self.Pot.CutoffStart and dist_ij>0: #Distances in A
            
        # 3.1 Pressure calculation for NN
            espo = (dist_ij/self.Pot.dik0)-1
    
            pres_repul = self.Pot.P[interaction_type]*self.Pot.A[interaction_type]*\
            (np.exp(-self.Pot.P[interaction_type]*espo))/self.Pot.dik0
    
            pres_bond =-self.Pot.Q[interaction_type]*(self.Pot.Qsi[interaction_type]**2)*\
            (np.exp(-2*self.Pot.Q[interaction_type]*espo))/self.Pot.dik0
    
            #summing over all j-atoms contributing to force on i
            denom_i = -self.Pot.Qsi[interaction_type]**2*np.exp(-2*self.Pot.Q[interaction_type]*espo)
            # Units denom: eV**2
    
    
        # 3.2 Pressure calculation for AN
        elif dist_ij<=self.Pot.CutoffEnd:
            dist_ij_m = dist_ij - self.Pot.CutoffEnd
            
            pres_repul = (5*self.Pot.a5[interaction_type]*(dist_ij_m**4))+\
            (4*self.Pot.a4[interaction_type]*(dist_ij_m**3))+\
            (3*self.Pot.a3[interaction_type]*dist_ij_m**2)
            
            pres_bond = (self.Pot.x5[interaction_type]*(dist_ij_m**5)+\
                         self.Pot.x4[interaction_type]*(dist_ij_m**4)+\
                         self.Pot.x3[interaction_type]*(dist_ij_m**3))*\
            (5*self.Pot.x5[interaction_type]*(dist_ij_m**4)+\
             4*self.Pot.x4[interaction_type]*(dist_ij_m**3)+\
             3*self.Pot.x3[interaction_type]*(dist_ij_m**2))
            
            denom_i = (self.Pot.x5[interaction_type]*(dist_ij_m**5)+\
                       self.Pot.x4[interaction_type]*(dist_ij_m**4)+\
                       self.Pot.x3[interaction_type]*(dist_ij_m**3))**2
            # Units denom: eV**2
            
        # 3.3 No need to calculate pressure for far neighbours
    
        elif dist_ij>self.Pot.CutoffEnd:
            pres_repul = 0
            pres_bond = 0
            denom_i = 0
        
        # 4. Calculate final pressure between atoms i and j, accounting for distance  
        self.Press = np.array([pres_repul*dist_ij, pres_bond*dist_ij, denom_i])
        return self.Press


    ### Loop over all atoms to get pressure for each
    def pressureMain(self, movie_file, pot_file, new_file = 'press_traj.xyz'):
        """
        Reads .pot and movie files and outputs to location
        the same movie with pressures added.
        """
        # 1. Read Files
        self.Pot = pot_file
        self.movie = movie_file
        
        self.NATOM = self.metadata['NAtoms']
        open(self.system['base_dir'] + new_file, 'w').close() #Clear old movie pressure file
        
        # 2. Startin the loop over all frames
        for frame_num, current_frame in enumerate(self.movie):
            print('Analyzing frame: {}/{}'.format(frame_num+1, len(self.movie)))
    
            # Pressure Calculation for that frame
            self.atom_pressures = []
            for i in range(len(current_frame)): # Loop over i
                pressure_repul_i = 0.0
                pressure_bond_i = 0.0
                summed_denom_i = 0.0
                atomic_volume = (4./3.) * np.pi * self.Pot.Radius[0]**3
                # by definition the pressure is divided by the atomic volume --Wales suggested the Wigner-Seitz volume instead
                # the atomic one as written here
                # being all distances in Ang and energy in eV this should be in eV/A^3 ~ 0.1602 TPa
                for j in range(len(current_frame)): # Loop over j of i
                    pressure_repul_i += self.getPressureTwoAtoms(current_frame[i], current_frame[j], self.Pot)[0]
                    pressure_bond_i += self.getPressureTwoAtoms(current_frame[i], current_frame[j], self.Pot)[1]
                    summed_denom_i +=self.getPressureTwoAtoms(current_frame[i], current_frame[j], self.Pot)[2]
                
                pressure_bond_i = atomic_volume*pressure_bond_i/np.sqrt(summed_denom_i)
                pressure_bond_i = 0.1602*pressure_bond_i #in TPa
                
                self.atom_pressures.append(pressure_repul_i+pressure_bond_i)
    
            # Check that we have one pressure value for each atom
            if (len(self.atom_pressures)!= self.NATOM):
                raise ValueError('CAREFUL: Not all atoms have one pressure value')
    
            #return(atom_pressures)
            # 3. Output Pressure to xyz file
            with open(self.system['base_dir'] + new_file, 'a+') as newmovie: # Mode chosen: append
                
                num_lines = sum(1 for line in open(self.system['base_dir']  + new_file))
    
                if (num_lines==0): # No newline for first line -- bugs Ovito if there is newline at beginning
                    newmovie.write(str(self.NATOM)+'\n')
                else:
                    newmovie.write('\n' + str(self.NATOM)+'\n')
                    
                newmovie.write('\t' + "This edited trajectory was created by Jones' post-processing software." + '\n' ) 
                
                for atom_index, atom_info in enumerate(current_frame):
                    atom_info[-1] = self.atom_pressures[atom_index] # Adding pressure to tuple
                    newmovie.write('  \t'.join(str(item) for item in atom_info))
