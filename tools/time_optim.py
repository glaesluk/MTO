import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt
from tools.optim_colony import AntColony
from tools.tree_join import score_tree
from random import randint


if __name__ == '__main__':
    route_dir = 'C:/Users/GNQBG/OneDrive - Bayer/Personal Data/AI4Synthesis_service_calculate_routes/routes'
    all_files = list(os.listdir(route_dir))
    all_files.sort()
    problem = []
    # load the problem from the given input directory 
    for file in all_files:
        if file.endswith('pkl'):
            with open(os.path.join(route_dir, file), 'rb') as fp:
                route = pickle.load(fp)
                if len(route) > 0:
                    problem.append(route)

    # evaluate problems of different length n
    solution = {}
    for n in [45]:
        # select random subset of length n
        routes = list(range(len(problem)))
        subset = set()
        for i in range(n):
            subset.add(routes.pop(randint(0,len(routes)-1)))

        small_problem = []
        for route in subset:
            small_problem.append(problem[route])
        
        # run AntColony with different timelimits

        for limit in range(1):
            
            limit = (limit+1)*100
            print(limit)
            aco = AntColony()
            aco.add_sets(small_problem)
            path = aco.run()
            
            # path = [0]*n
            # evaluate solution
            routes = []
            i = 0
            for selected in path:
                routes.append(small_problem[i][selected])
                i += 1
            score = score_tree({'smiles':'START>>TARGETS', 'children': routes}, 'num_unique_interm')
            solution[(n,limit)]=score
        x=[]
        y=[]
        print(solution)
        for i in range(1):
            x.append(i)
            y.append(solution[(n,(i+1)*100)])
        print(x,y)
        plt.plot(x,y,label="{} targets".format(n))
    plt.legend()
    plt.show()
    print(solution)
            

