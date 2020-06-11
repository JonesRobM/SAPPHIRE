import pickle
import numpy as np
from ase.io import read
import time
import multiprocessing as mp

class Trajagcn():


    def __init__(self, metadata, system):
        self.system = system
        self.metadata = metadata
        self.movie_file = self.system['base_dir']+self.system['movie_file_name']
        self.start = self.metadata['Start']
        self.end = self.metadata['End']
        self.step = self.metadata['Step']
        try:
            self.agcn = self.metadata['agcn']
        except KeyError:
            print("It would appear that you have not evaluated the agcn for this simulation, yet. \n")
        self.T = time.time()
        self.Frames = range(self.start, self.end, self.step)
        self.New_Obj = []
    
    def edit_movie_frame(self, Frame):
        temp = read(self.movie_file, index = Frame)
        c = temp.get_chemical_symbols()
        xyz = temp.get_positions()
        ag = self.agcn[self.Frames.index(Frame)]
        return  np.column_stack( (c,xyz,ag) ) 
    
    def process_movie(self):
        p = mp.Pool(6)
        self.object_pool = p.map(self.edit_movie_frame, (self.Frames))
        p.close()
        p.join()
        
        for i in Frames:
            self.New_Obj.append(self.object_pool[i])
        return self.New_File
        self.T0=time.time()
        print('Time for completing editing the movie is %s.' %(time.strftime("%H:%M:%S", time.gmtime((self.T0-self.T)))), "\n")
        

    
    def New_File(self, new_movie='agcn_movie.xyz'):
        with open(self.system['base_dir'] + new_movie, 'w+') as self.movie:
            self.movie.write(str(self.metadata['NAtoms']) +'\n')
            self.movie.write('\t' + "This was made by Jones' post-processing code." + '\n')
            for Frame in self.New_Obj:
                for items in Frame:
                    self.movie.write(' \t'.join(str(item) for item in items) +'\n')
                self.movie.write(str(self.metadata['NAtoms']) + '\n')
                self.movie.write('\n')
        print("This movie has been saved as %s in %s.\n" %(new_movie, self.system['base_dir']))
        print('Time for writing new aGCN trajectroy is %s.' %(time.strftime("%H:%M:%S", time.gmtime((time.time()-self.T0)))), "\n")