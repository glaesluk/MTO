

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

import os
import pickle
from random import randint

#--- MAIN ---------------------------------------------------------------------+

x_target = 500
x_routes = 500
elements = 10000
route_length = 6

outfolder = 'C:/Users/GNQBG/Documents/AI4Synthesis/random/500x500x6'
filename = 'sets.pkl'

sets = []
for target in range(x_target):
    target = []
    for route in range(x_routes):
        interms = set()
        for i in range(route_length):
            interms.add("{}".format(randint(0,elements)))
        target.append(interms)
    sets.append(target)

if not os.path.exists(outfolder):
    os.makedirs(outfolder)
file = open(os.path.join(outfolder,filename),'wb')
pickle.dump(sets, file)
