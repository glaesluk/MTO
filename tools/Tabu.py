

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

import random
from random import randint
import time

#--- MAIN ---------------------------------------------------------------------+

class Tabu():
    '''Simple Tabu search
    
    In this search a neighbor is defined as a path that differ in at most (five?) targets.
    New states not only mustn't be in the tabu-list but also mustn't be similar to them.
    Similarity is defined throu difference in at most one target.
    '''

    def __init__(self, func, problem, startsol, greedy = False,  step_max=100000, timelimit = 500):

        # initialize starting conditions
        self.t_max = timelimit
        self.step_max = step_max
        self.hist = []

        self.cost_func = func
        self.problem = problem
        self.x0 = startsol
        self.current_state = self.x0
        self.current_energy = func(self.x0)
        self.best_state = self.current_state
        self.best_energy = self.current_energy
        self.tabus = []
        self.greedy = greedy

        # begin optimizing
        self.step = 1
        self.start_t = time.time()
        self.t = 0
        while self.step < self.step_max and self.t < self.t_max:
            
            self.tabus.append(self.current_state)
            # get neighbor
            proposed_neighborhood = self.get_neighborhood()
            self.getBest(proposed_neighborhood)

            # check if the current state is best solution so far
            if self.current_energy < self.best_energy:
                self.best_energy = self.current_energy
                self.best_state = self.current_state

            # persist some info for later
            self.hist.append([
                self.step,
                self.t,
                self.current_energy,
                self.best_energy])

            # update some stuff
            self.t = time.time()-self.start_t
            # print(self.t)
            self.step += 1


    def get_neighborhood(self):
        '''
       not finished

       should return a list/set of paths with are in some way 
       neighbors of the current_state, but also not to similar,
       because otherwise they would have the same greedy solution
       and it will take a long time to go out of this (local) 
       greedy-minimum, where we start.
        '''
        # list of 25 neighbors to be returned
        neighborhood =[]
        # we want 25 "neighbors" (HYPERPARAMETER)
        for i in self.current_state.keys():
            neighbor = self.current_state.copy()
            # want a difference of maximum 5? targets (HYPERPARAMETER)
            neighbor.pop(i)
            # chose a new target for the one we popped
            new_target = randint(0,len(self.problem)-1)
            while new_target in neighbor.keys():
                new_target = randint(0,len(self.problem)-1)
            # determine the route for the new target
            if self.greedy:
                best_cost = 1000000
                best_neighbor = 0
                for route_id in range(len(self.problem[new_target])):
                    neighbor[new_target] = route_id
                    cost_now = self.cost_func(neighbor)
                    if cost_now < best_cost:
                        best_cost = cost_now
                        best_neighbor = route_id
                new_route = best_neighbor
            else:
                new_route = randint(0,len(self.problem[new_target])-1)
            neighbor[new_target] = new_route
            neighborhood.append(neighbor.copy())
        return neighborhood

    def getBest(self, neighborhood):
        '''
        gets the best canidate from the given neigborhood-list, 
        which isn't similar to one of the tabu-paths and save it.
        '''
        for path in neighborhood:
            new_cost = self.cost_func(path)
            if path not in self.tabus:    
                if new_cost < self.current_energy:
                # difference = 0
                # for tabu_target in self.tabus:    
                #     for target in path:
                #         try: 
                #             if tabu_target[target] == path[target]:
                #                 pass
                #             else:
                #                 difference += 1
                #         except: 
                #             difference += 1
                # if difference > 2:
                
                    self.current_state = path
                    self.best_energy = new_cost

    def results(self):
        print('+------------------------ RESULTS -------------------------+\n')
        print(f'      max time: {self.t_max}')
        print(f'    final time: {self.t:0.6f}')
        print(f'     max steps: {self.step_max}')
        print(f'    final step: {self.step}\n')

        print(f'  final energy: {self.best_energy:0.6f}\n')
        print('+-------------------------- END ---------------------------+')
#--- END ----------------------------------------------------------------------+