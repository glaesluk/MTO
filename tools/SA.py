

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

import random
from random import randint
from math import exp
from math import log
from time import time
from tqdm import tqdm
from collections import Counter

#--- MAIN ---------------------------------------------------------------------+

class SimAnn():
    '''Simple Simulated Annealing
    '''

    def __init__(self, func, problem, startsol, greedy = False, cooling_schedule='linear', timelimit = 1000, t_min=0, t_max=100, bounds=[], alpha=None, damping=1):

        # checks
       
        assert cooling_schedule in ['linear','exponential','logarithmic', 'quadratic'], 'cooling_schedule must be either "linear", "exponential", "logarithmic", or "quadratic"'


        # initialize starting conditions
        self.t = t_max
        self.t_max = t_max
        self.t_min = t_min
        self.opt_mode = "combinatorial"
        self.hist = []
        self.cooling_schedule = cooling_schedule

        self.cost_func = func
        self.problem = problem
        self.x0 = startsol
        self.bounds = bounds[:]
        self.damping = damping
        self.current_state = self.x0
        self.current_energy = func(self.x0)
        self.best_state = self.current_state
        self.best_energy = self.current_energy
        self.greedy = greedy
        self.timelimit = timelimit

        self.step_max = 100000000 /  (500*5) # (avrg. routes/target * avrg. routelength)
        self.get_neighbor = self.move_combinatorial


        # initialize cooling schedule
        if self.cooling_schedule == 'linear':
            if alpha != None:
                self.update_t = self.cooling_linear_m
                self.cooling_schedule = 'linear multiplicative cooling'
                self.alpha = alpha

            if alpha == None:
                self.update_t = self.cooling_linear_a
                self.cooling_schedule = 'linear additive cooling'

        if self.cooling_schedule == 'quadratic':
            if alpha != None:
                self.update_t = self.cooling_quadratic_m
                self.cooling_schedule = 'quadratic multiplicative cooling'
                self.alpha = alpha

            if alpha == None:
                self.update_t = self.cooling_quadratic_a
                self.cooling_schedule = 'quadratic additive cooling'

        if self.cooling_schedule == 'exponential':
            if alpha == None: self.alpha =  0.8
            else: self.alpha = alpha
            self.update_t = self.cooling_exponential_m

        if self.cooling_schedule == 'logarithmic':
            if alpha == None: self.alpha =  0.8
            else: self.alpha = alpha
            self.update_t = self.cooling_logarithmic_m


        # begin optimizing
        self.step, self.accept = 1, 0
        self.sec = 0
        self.start_time = time()
        pbar = tqdm(desc="Total search steps", total=self.step_max)
        while self.step < self.step_max and self.t >= self.t_min and self.t>0 and self.sec < self.timelimit:

            # get neighbor
            proposed_neighbor = self.get_neighbor()
            # check energy level of neighbor
            E_n = self.cost_func(proposed_neighbor)
            dE = E_n - self.current_energy

            # determine if we should accept the current neighbor
            if random.random() < self.safe_exp(-dE / self.t):
                self.current_energy = E_n
                self.current_state = proposed_neighbor
                self.accept += 1

            # check if the current neighbor is best solution so far
            if E_n < self.best_energy:
                self.best_energy = E_n
                self.best_state = proposed_neighbor

            # persist some info for later
            self.hist.append([
                self.step,
                self.t,
                self.current_energy,
                self.best_energy])

            # update some stuff
            self.t = self.update_t(self.step)
            self.step += 1
            self.sec = time()-self.start_time
            pbar.update(1)
        pbar.close()
        # generate some final stats
        self.acceptance_rate = self.accept / self.step


    def move_combinatorial(self):
        '''
        randomly replaces target with other 
        Or with random route or selects route greedy 
        '''
        neighbor = self.current_state.copy()
        neighbor.pop(random.choice(list(neighbor.keys())))
        new_target = randint(0,len(self.problem)-1)
        while new_target in neighbor.keys():
            new_target = randint(0,len(self.problem)-1)
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
        neighbor.update({new_target:new_route})
        return neighbor

    def move_combinatorial_2(self):
        '''
        randomly replaces target with other 
        Or with random route or selects route greedy 
        '''
        neighbor = self.current_state.copy()
        intermediates = Counter()
        for target,route in neighbor.items():
            intermediates.update(self.problem[target][route])
        weights = {}
        for target, route in neighbor.items():
            weights[target] = sum(1/intermediates[mol] for mol in self.problem[target][route])
        '''
        out = randint(1,self.current_energy)
        for target in dict(sorted(weights.items(),key= lambda x:x[1])):
            out -= weights[target]
            if out <= 0:
                target_out = target
                break
        neighbor.pop(target)
        '''
        # pop out route with biggest participation on costFunc
        neighbor.pop(max(weights, key=weights.get))
        new_target = randint(0,len(self.problem)-1)
        while new_target in neighbor.keys():
            new_target = randint(0,len(self.problem)-1)
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
        neighbor.update({new_target:new_route})
        return neighbor

    def results(self):
        print('+------------------------ RESULTS -------------------------+\n')
        print(f'      opt.mode: {self.opt_mode}')
        print(f'cooling sched.: {self.cooling_schedule}')
        print(f'  accept. rate: {self.acceptance_rate}')
        if self.damping != 1: print(f'       damping: {self.damping}\n')
        else: print('\n')

        print(f'      max time: {self.timelimit}')
        print(f'    final time: {self.sec}')
        print(f'  initial temp: {self.t_max}')
        print(f'    final temp: {self.t:0.6f}')
        print(f'     max steps: {self.step_max}')
        print(f'    final step: {self.step}\n')

        print(f'  final energy: {self.best_energy:0.6f}\n')
        print('+-------------------------- END ---------------------------+')
    # linear multiplicative cooling
    def cooling_linear_m(self, step):
        return self.t_max /  (1 + self.alpha * step)

    # linear additive cooling
    def cooling_linear_a(self, step):
        return self.t_min + (self.t_max - self.t_min) * ((self.step_max - step)/self.step_max)

    # quadratic multiplicative cooling
    def cooling_quadratic_m(self, step):
        return self.t_min / (1 + self.alpha * step**2)

    # quadratic additive cooling
    def cooling_quadratic_a(self, step):
        return self.t_min + (self.t_max - self.t_min) * ((self.step_max - step)/self.step_max)**2

    # exponential multiplicative cooling
    def cooling_exponential_m(self, step):
        return self.t_max * self.alpha**step

    # logarithmical multiplicative cooling
    def cooling_logarithmic_m(self, step):
        return self.t_max / (self.alpha * log(step + 1))


    def safe_exp(self, x):
        try: return exp(x)
        except: return 0

#--- END ----------------------------------------------------------------------+