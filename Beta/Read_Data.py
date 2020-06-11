import numpy as np
import csv
import pandas as pd

base_dir = '../../../Feb20/AuPd/1127/Melting75/'

reader = csv.DictReader(open(base_dir+'Meta2.csv', 'r'))
for row in reader:
    print(row, "\n")
    

