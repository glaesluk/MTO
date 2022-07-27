#!/usr/bin/python3
from ctypes import Union
import logging
import numpy as np
import os
import sys
import pickle
import pulp
import random
import argparse
from collections import Counter, namedtuple
from multiprocessing import Queue, Pool
from tools.tree_join import find_shorter_paths, score_tree, nodeGenerator, connected_components, removeRouteIsotopes
from tools.report import plot_routes, plot_routes_order, write_multiroute_report
from tqdm import tqdm
from pulp import LpMaximize, LpMinimize, LpProblem, LpStatus, lpSum, LpVariable, LpInteger

def get_buyables(molecule):
    queue = [molecule]
    while len(queue) > 0:
        curr = queue.pop()
        if '>>' not in curr['smiles']:
            yield curr['smiles']
        if len(curr['children']) > 0:
            for child in curr['children']:
                queue.append(child)
        #else:
        #    yield curr['smiles']

def get_intermediates(molecule):
    #returns a list of all intermediates of the molecule
    queue = [molecule]
    while len(queue) > 0:
        curr = queue.pop()
        if len(curr['children']) > 0:
            for child in curr['children']:
                queue.append(child)
            if '>>' not in curr['smiles']:
                yield curr['smiles']
        #else:
        #    yield curr['smiles']

class AntColony:
    def __init__(self):
        self.n = 0
        self.k = 0

    def add_sets(self, problem):
        """ Sets the specific parameters of a problem. In our case, that means
        a list of tree sets.

        Parameters
        ----------
        problem : list(list(dict))
            A list of tree sets. The i-th element of the set contains all
            possible trees for the i-th molecule. Every tree is represented as
            a nested dictionary.
        """
        self.problem = problem
        self.n = len(self.problem)
        self.k = max(map(len, self.problem))

    def run(self, startsol = None, timelimit = 100):
        """ Runs the ACO algorithm.

        Parameters
        ----------
        num_steps : int
            Number of iterations

        startsol: dict{target:route, ...}
            Solution to do the warmstart.


        Returns
        -------
        list
            A list with the ids of the best routes found for each target of the problem
        """
        #import pdb; pdb.set_trace()
        # Set of all individual buyables
        all_buyables = set()
        # I[d][r] is the set of buyables for molecule d and route r
        I = []
        for molecule in self.problem:
            new_row = []
            for idx, route in enumerate(molecule):
                # this line is for optimalization of intermediates
                buyables = set(get_intermediates(route))
                # following line is for optimization of (buyables + intermediates)
                # buyables = set(get_buyables(route))
                all_buyables.update(buyables)
                new_row.append(buyables)
            I.append(new_row)
        all_buyables = list(all_buyables)

        # Create a binary variable that indicates whether a specific buyable is used
        y = LpVariable.dicts("buyable", all_buyables, lowBound=0, upBound=1, cat=LpInteger)
        # Create a binary variable that indicates whether molecule d uses route r
        x = LpVariable.dicts("molecule_uses_route", (list(range(self.n)), list(range(self.k))), lowBound=0, upBound=1, cat=LpInteger)
        # Create the problem
        model = LpProblem(name="Minimize buyables", sense=LpMinimize)
        # Set the objective function: minimize the number of buyables
        model += lpSum(y[buy] for buy in all_buyables)
        # First constraint: each molecule uses one of its routes
        # x[d][r] = molecule d uses route r
        for d in range(self.n):
            model += (lpSum([x[d][r] for r in range(self.k)]) == 1, "Molecule_{}_uses_one_recipe".format(d))
        # Second constraint: forces all buyables to be used in their respective route and molecule
        for d in range(self.n):
            for r in range(self.k):
                try:
                    for buyable in I[d][r]:
                        model += (x[d][r] <= y[buyable], "forces_all_buyables_{}-{}-{}".format(d, r, buyable))
                except IndexError:
                    model += (x[d][r] == 0, "out_of_bounds_buyable_{}-{}".format(d, r))
        
        # Hand over warm-start solution
        for target, route in startsol.items():
            x[target][route].setInitialValue(1)
            for interm in I[target][route]:
                y[interm].setInitialValue(1)
        
        # Solve it
        solution = dict()
        solver = pulp.getSolver('PULP_CBC_CMD', timeLimit=timelimit, msg=True, warmStart=True)
        status = model.solve(solver)
        
        for var in model.variables():
            if var.value() > 0 and var.name[0] == 'm':
                print("{}: {}".format(var.name, var.value()))
                molecule_route = var.name.split('_')
                mol = int(molecule_route[-2])
                route = int(molecule_route[-1])
                solution[mol] = route
        retval = []
        for i in range(self.n):
            if solution[i] < len(self.problem[i]):
                retval.append(solution[i])
            else:
                print("Value {} for molecule {} is out of bounds".format(solution[i], i))
        print(retval)
        return retval
    
    def heuristic(self):
        smaller_problem = self.problem
        path = {}
        Depth = namedtuple('Depth', 'index route_depth')
        DepthNode = namedtuple('CurrNode', 'depth data')
        for id, route_set in enumerate(smaller_problem):
            min_depth = Depth(0, 99)
            for idx, route in enumerate(route_set):
                # Search the shortest route
                queue = [DepthNode(1, route)]
                while len(queue) > 0 and min_depth.route_depth > 1:
                    current, queue = queue[0], queue[1:]
                    if len(current.data['children']) == 0:
                        if current.depth < min_depth.route_depth:
                            min_depth = Depth(idx, current.depth)
                        break
                    else:
                        for c in current.data['children']:
                            queue.append(DepthNode(current.depth+1, c))
            path.update({id:min_depth.index})

    def greedy(self):
        path = {}
        seen_interm = set()
        for id, molecule in enumerate(self.problem):
            Union = namedtuple('union',['route_id','union_total'])
            shortest = Union(None,100000)
            for route_id, route in enumerate(molecule):
                union_now = self.route_to_intermediates(route, keep_reactions=False).union(seen_interm)
                if  len(union_now) < shortest.union_total:
                    shortest = Union(route_id, len(union_now))
            seen_interm.update(union_now)
            path.update({id:shortest.route_id})
        return path

    @staticmethod
    def get_top_N_mol(all_routes_list, ordered_routes, num_mol=25):
        """Gets the top N molecules regarding the synthesis order and returns their
        indices in the full list as list

        Args:
            all_routes_list (list): All molecules
            ordered_routes (list): Synthesis order top (least effort) to bottom (maximum effort)
            num_mol (int, optional): Number of molecules to retrieve. Defaults to 25.

        Returns:
            list: list of indices in the full molecule list
        """
        indices = []
        for oo in range(num_mol):
            for rr in range(len(all_routes_list)):
                if all_routes_list[rr]["smiles"] == next(iter(ordered_routes[oo][0])):
                    indices.append(rr)
        return indices

    @staticmethod
    def route_to_intermediates(route, keep_reactions=True):
        """ Given a route (written as a nested dictionary), returns a set of all
        its intermediates.

        Parameters
        ----------
        route : dict
            Nested dictionary of intermediates representing a route.
        keep_reactions : bool
            If True, reactions will also be returned as intermediates. Otherwise,
            only true intermediates are returned.

        Returns
        -------
        set
            A set of all the intermediates in this route.
        """
        intermediates = set()

        pending = [route]
        while len(pending) > 0:
            current, pending = pending[0], pending[1:]
            if keep_reactions or '>>' not in current['smiles'] and (len(current['children']) > 0):
                intermediates.add(current['smiles'])
            for child in current['children']:
                pending.append(child)
        return intermediates

    @staticmethod
    def route_to_buyables(route):
        """ Given a route (written as a nested dictionary), returns a set of all
        its buyables.

        Parameters
        ----------
        route : dict
            Nested dictionary of buyables representing a route.

        Returns
        -------
        set
            A set of all the buyables in this route.
        """
        buyables = set()

        pending = [route]
        while len(pending) > 0:
            current, pending = pending[0], pending[1:]
            if ('>>' not in current['smiles']) and (len(current['children']) == 0):
                buyables.add(current['smiles'])
            for child in current['children']:
                pending.append(child)
        return buyables

    @staticmethod
    def collect_all_intermediates(routes, keep_reactions=True):
        """ Given a list of routes, it returns a set of all the intermediates
        present in any of them. Targets are counted as intermediates also.

        Parameters
        ----------
        routes : list(dict)
            List of routes.
        keep_reactions : bool
            If True, reactions will also be returned as intermediates. Otherwise,
            only true intermediates are returned.

        Returns
        -------
        set
            A set with any and all intermediates present in any of the input
            routes.
        """
        intermediates = set()
        for route in routes:
            for node in nodeGenerator(route):
                if (keep_reactions or '>>' not in node['smiles']) and len(node['children']) > 0:
                    intermediates.add(node['smiles'])
        return intermediates

    @staticmethod
    def get_right_order(routes, n =25, best_routes = [], selection = []):
        """ Calculates the order in which intermediates are needed to obtain the
        highest amount of molecules with the fewer number of intermediates.

        Parameters
        ----------
        routes : list
            A list of routes, where each route is a nested dictionary.

        n : int
            Number of how many targets should be seen further forward
        
        best_routes : list
            A list with the indices of the best n routes we have found so far. 

        selection : list
            A list of length n, which contains values from {-1,0,1}. (-1) means 
            that this target shouldn't be used, (0) is neutral and (1) means, that
            the target should be used. The index of this list equals the indexing of best_routes.
       

        Returns
        -------
        list
            A list of pairs, where the element (molecules, intermediates) in the
            n-th position represents the molecule(s) that can be created after
            adding the selected intermediates. See `Notes` for some details on
            the optimality of this list.

        Notes
        -----
        The returned list has an extra property: it tries to build the largest
        number of elements from the least number of intermediates. This means
        that, given a big enough list of routes, it should be possible to look
        at the graph of #molecules given #intermediates and decide where it
        would be more economical to stop.
        """

        # Named tuples to make the code easier to read
        Route = namedtuple('Route', 'smiles intermediates buyables generalid')
        # RouteInterms = namedtuple('RouteInterms', 'idx common unique generalid')
        RouteInterms = namedtuple('RouteInterms', 'idx unique_score generalid')
        # List that will be returned
        retval = []

        if len(selection)>0:
            postopt = True
        else:
            postopt = False
        inRoutes = []
        # In this code, a route is a set of intermediates and buyables. This code turns the
        # nested dictionaries from the input into sets.
        pending_molecules = []
        for idx, route in enumerate(routes):
            # looking, if we want to do postopt
            if not postopt:
                route_interms = AntColony.route_to_intermediates(route, keep_reactions=False)
                route_buyables = AntColony.route_to_buyables(route)
                pending_molecules.append(Route(route['smiles'], route_interms, route_buyables, idx))
            else:
                # look if route has special condition from list selection
                try: # gives an Error if route isn't in the best_routes
                    val = selection[best_routes.index(idx)]
                    if val != '-1':
                        if val == '1':
                            inRoutes.append(idx)
                        route_interms = AntColony.route_to_intermediates(route, keep_reactions=False)
                        route_buyables = AntColony.route_to_buyables(route)
                        pending_molecules.append(Route(route['smiles'], route_interms, route_buyables, idx))
                except:
                    route_interms = AntColony.route_to_intermediates(route, keep_reactions=False)
                    route_buyables = AntColony.route_to_buyables(route)
                    pending_molecules.append(Route(route['smiles'], route_interms, route_buyables, idx))
        # Start of the code that chooses at every step which route should be
        # created next. We attach a progress bar to it.
        pbar = tqdm(desc="Calculating synthesis order", total=len(pending_molecules))
        while len(pending_molecules) > 0:
            # Identify all pending intermediates
            all_interms = Counter()
            loop_routes = []
            for r in pending_molecules:
                all_interms.update(r.intermediates)
            # Convert a route to the "shared/unique" format
            for idx, r in enumerate(pending_molecules):
                # Note: this used to be a list. I hope that using a set won't break anything.
                # common = set(filter(lambda x: all_interms[x] > 1, list(r.intermediates)))
                # unique = set(filter(lambda x: all_interms[x] == 1, list(r.intermediates)))
                # The unique score tells the "weight" of an intermediate. (Lighter if many routes use this intermediate)
                unique_score={x:(1/all_interms[x]) for x in r.intermediates}
                #loop_routes.append(RouteInterms(idx, common, unique, r.generalid))
                loop_routes.append(RouteInterms(idx, unique_score, r.generalid))
            # Finds the route with the largest amount of non-shared intermediates.
            # In case of a tie, picks the route with the largest amount of *shared*
            # intermediates.
            loop_routes.sort(key=lambda x: np.sum(list(x.unique_score.values())), reverse=True)
            # molecules of inRoutes should only be chosen in the last n steps
            i = 0
            while loop_routes[i].generalid in inRoutes and len(pending_molecules) > n :
                i += 1
            chosen = loop_routes[i]
            new_smiles = pending_molecules[chosen.idx].smiles
            # new_interms = chosen.common.union(chosen.unique) - seen_intermediates
            new_interms = pending_molecules[chosen.idx].intermediates
            new_buyables = pending_molecules[chosen.idx].buyables
            # Remove the selected route from the "pending" list
            pending_molecules = pending_molecules[:chosen.idx] + pending_molecules[chosen.idx + 1:]
            retval.append(Route(new_smiles, new_interms, new_buyables, None))
            # Update the progress bar
            pbar.update(1)
        pbar.close()

        # Post-processing: if a particular step has no new intermediates that
        # means that the previous step should have added two (or more) molecules.
        # This step unifies that.
        # TODO: the current code doesn't make a separation between a buyable as
        # target and a molecule that can be built because all the intermediates
        # are suddenly there
        i = 0
        while i < len(retval):
            curr_route = retval[i]
            retval[i] = ({curr_route.smiles}, curr_route.intermediates, curr_route.buyables, curr_route.generalid)
            i += 1
            """
            if i == 0 or len(curr_route.intermediates) > 0:
                # We replace the old Route named tuple with a set and a list
                retval[i] = (set([curr_route.smiles]), curr_route.intermediates)
                i += 1
            else:
                # The current node has no new chemicals, so I maybe need to
                # unify it with the previous node.
                retval[i-1][0].add(curr_route.smiles)
                retval = retval[:i] + retval[i+1:]
                # i is not incremented because we removed one element of the
                # list and that suffices to move the algorithm forward.
            """
        # The list has to be read back-to-front. Therefore, we reverse the list
        retval.reverse()
        seen_intermediates = set()
        seen_buyables = set()
        pending_molecules = []
        for idx, route in enumerate(routes):
            route_interms = AntColony.route_to_intermediates(route, keep_reactions=False)
            route_buyables = AntColony.route_to_buyables(route)
            pending_molecules.append(Route(route['smiles'], route_interms, route_buyables, idx))
        ii = 0
        while ii < len(retval):
            for pp in pending_molecules:
                if next(iter(retval[ii][0])) == pp.smiles:
                    curr_route = pp
            new_interms = curr_route.intermediates - seen_intermediates
            new_buyables = curr_route.buyables - seen_buyables
            retval[ii] = ({curr_route.smiles}, new_interms, new_buyables)
            seen_intermediates.update(new_interms)
            seen_buyables.update(new_buyables)
            ii += 1

        return retval


if __name__ == '__main__':
    # route_dir = 'C:/Users/GNQBG/OneDrive - Bayer/Personal Data/AI4Synthesis_service_calculate_routes/routes'
    # route_dir = '/tmp/newroutes/V8/01'
    parser = argparse.ArgumentParser()
    parser.add_argument('routedir', type=str)
    parser.add_argument('outdir', type=str)
    parser.add_argument('number_smallTree', type=int)
    parser.add_argument('timelimit_seconds',type=int)
    args = parser.parse_args()
    route_dir = args.routedir
    outdir = args.outdir
    num_mol_smallTree = args.number_smallTree
    timelimit = args.timelimit_seconds

    all_files = list(os.listdir(route_dir))
    all_files.sort()
    problem = []
    problem_interms = []
    seen_targets = []
    # Collect all intermediates and check their clustering.
    for file in all_files:
        if file.endswith('pkl'):
            with open(os.path.join(route_dir, file), 'rb') as fp:
                routes = pickle.load(fp)
                for idx, route in enumerate(routes):
                    routes[idx] = removeRouteIsotopes(route)
                if len(routes) > 0:
                    if routes[0]["smiles"] not in seen_targets:
                        seen_targets.append(routes[0]["smiles"])
                    else:
                        continue
                    problem.append(routes)
                    # Calculate the set of intermediates
                    all_interms = set()
                    for route in routes:
                        for node in nodeGenerator(route):
                            all_interms.add(node['smiles'])
                    problem_interms.append(all_interms)

    # Calculate how many disconnected components are there
    all_components = connected_components(problem_interms)
    print("{} component(s) found: {}".format(len(all_components), all_components))

    for component_id, component in enumerate(all_components):
        # Every set of components gets its own report
        smaller_problem = list(map(lambda x: problem[x], component))

        # For development this is set to False. For production,
        # this should be set to True
        run_ants = True
        aco = AntColony()
        if run_ants:
            aco.add_sets(smaller_problem)
            path = aco.run(startsol = aco.greedy(), timelimit=timelimit)
        else:
            # path = [0] * len(component)
            # Choose the shallowest route for every target
            path = []
            Depth = namedtuple('Depth', 'index route_depth')
            DepthNode = namedtuple('CurrNode', 'depth data')
            for route_set in smaller_problem:
                min_depth = Depth(0, 99)
                for idx, route in enumerate(route_set):
                    # Search the shortest route
                    queue = [DepthNode(1, route)]
                    while len(queue) > 0 and min_depth.route_depth > 1:
                        current, queue = queue[0], queue[1:]
                        if len(current.data['children']) == 0:
                            if current.depth < min_depth.route_depth:
                                min_depth = Depth(idx, current.depth)
                            break
                        else:
                            for c in current.data['children']:
                                queue.append(DepthNode(current.depth+1, c))
                path.append(min_depth.index)
        final_routes = []
        for idx, winner in enumerate(path):
            final_routes.append(smaller_problem[idx][winner])

        # This step should not be necessary: if you did not remove the isotopes
        # before running the algorithm then the results will be disappointing (and the filenames prob. to long in windows).
        # I apply the isotope removal here because I need it for debugging the
        # report generation, but "real" code shouldn't need it
        for idx, route in enumerate(final_routes):
            final_routes[idx] = removeRouteIsotopes(route)
        # testing if the find_shorter_paths function improves the result
        final_routes_score = score_tree({'smiles':'START>>TARGETS', 'children': final_routes}, 'num_unique_interm')
        final_routes_test = find_shorter_paths(final_routes)
        final_routes_test_score = score_tree({'smiles':'START>>TARGETS', 'children': final_routes_test}, 'num_unique_interm')
        if final_routes_test_score < final_routes_score:
            final_routes = final_routes_test
            final_routes_score = final_routes_test_score
            print("Found shorter routes")

        # New feature: what's the optimal way to ...
        synth_order = AntColony.get_right_order(final_routes)
        final_routes_order = aco.get_top_N_mol(final_routes, synth_order, num_mol=len(final_routes))
        outdir = os.path.join(outdir,"component_{}".format(component_id))
        os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\Graphviz\\bin"
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        write_multiroute_report(final_routes, outdir, synth_order)
        plot_routes(final_routes, os.path.join(outdir, "full_graph"))
        plot_routes_order(synth_order, os.path.join(outdir, "synthesis_order.png"))
        
        # write (smaller_problem, path, final_routes_order) in a pikle file for further selection of Top-Targets
        outdir_pkl = os.path.join(outdir,"pkl")
        if not os.path.exists(outdir_pkl):
                os.makedirs(outdir_pkl)
        outfile = open(os.path.join(outdir_pkl,"problem.pkl"),'wb')
        pickle.dump(smaller_problem, outfile)
        outfile.close()
        outfile = open(os.path.join(outdir_pkl,"selected_routes.pkl"),'wb')
        pickle.dump(path, outfile)
        outfile.close()
        outfile = open(os.path.join(outdir_pkl,"final_routes_top_n_order.pkl"),'wb')
        pickle.dump(final_routes_order, outfile)
        outfile.close()

        # choose top n targets
                
        final_routes_top_n_order = final_routes_order[:num_mol_smallTree]
        # get smaller report with only the top N molecules regarding synthesis order (old routes)
        top_n_routes = [final_routes[x] for x in final_routes_top_n_order]
        # top_n_routes = find_shorter_paths(top_n_routes)

        outdir_smallTree = outdir + "_smallTree"
        if not os.path.exists(outdir_smallTree):
            os.makedirs(outdir_smallTree)
        write_multiroute_report(top_n_routes, outdir_smallTree, synth_order[:num_mol_smallTree])
        plot_routes(top_n_routes, os.path.join(outdir_smallTree, "full_graph"))
        plot_routes_order(synth_order[:num_mol_smallTree], os.path.join(outdir_smallTree, "synthesis_order.png"))
        # write (final_routes, final_routes_top_n_order) in a pikle file for further selection of Top-Targets
        outdir_pkl = os.path.join(outdir_smallTree,"pkl")
        if not os.path.exists(outdir_pkl):
                os.makedirs(outdir_pkl)
        outfile = open(os.path.join(outdir_pkl,"problem.pkl"),'wb')
        pickle.dump(smaller_problem, outfile)
        outfile.close()
        outfile = open(os.path.join(outdir_pkl,"selected_routes.pkl"),'wb')
        pickle.dump(path, outfile)
        outfile.close()
        outfile = open(os.path.join(outdir_pkl,"final_routes_top_n_order.pkl"),'wb')
        pickle.dump(final_routes_top_n_order, outfile)
        outfile.close()

        final_score = score_tree({'smiles':'START>>TARGETS', 'children': top_n_routes}, 'num_unique_interm')
        
        # Optimize routes for top n targets with AntColony another time:
        top_n_problem = []
        for idx in final_routes_top_n_order:
            top_n_problem.append(smaller_problem[idx])

        aco2 = AntColony()
        aco2.add_sets(top_n_problem)
        smaller_path = aco2.run(timelimit)
        
        '''
        # using total_ordering for reoptimization
        from tools.total_ordering import AntColony
        top_n_problem = []
        for idx in final_routes_top_n_order:
            top_n_problem.append(smaller_problem[idx])
        aco2 = AntColony()
        aco2.add_sets(top_n_problem)
        smaller_path, order = aco2.run()
        '''

        # list with best routes of top n targets 
        final_small_routes = []
        for idx, winner in enumerate(smaller_path):
            final_small_routes.append(top_n_problem[idx][winner])

        for idx, route in enumerate(final_small_routes):
            final_small_routes[idx] = removeRouteIsotopes(route)

        # Testing if find_shorter path improves the Routes
        final_small_routes_test = find_shorter_paths(final_small_routes)
        final_score_opt_test = score_tree({'smiles': 'START>>TARGETS', 'children':final_small_routes_test}, 'num_unique_interm')
        final_score_opt = score_tree({'smiles':'START>>TARGETS', 'children':final_small_routes}, 'num_unique_interm')
        
        if final_score_opt_test < final_score_opt:
            final_small_routes = final_small_routes_test
            final_score_opt = final_score_opt_test
            print("Found shorter routes")

        # determine synthesis order another time 
        final_synth_order = AntColony.get_right_order(final_small_routes)
        final_routes_top_n_order_new = aco.get_top_N_mol(final_small_routes, final_synth_order, num_mol=len(final_small_routes))
        print("Tried to optimize routes another time (from score {} to {}).".format(final_score,final_score_opt))
        if final_score_opt < final_score:
            # write report
            # outdir_smallTree = "\\".join(outdir.split("\\")[:-1])+"_smallTreeOpt"
            outdir_smallTreeOpt = outdir+"_smallTreeOpt"
            print("See results in ",outdir_smallTreeOpt)
            if not os.path.exists(outdir_smallTreeOpt):
                os.makedirs(outdir_smallTreeOpt)

            write_multiroute_report(final_small_routes, outdir_smallTreeOpt, final_synth_order)
            plot_routes(final_small_routes, os.path.join(outdir_smallTreeOpt, "full_graph"))
            plot_routes_order(final_synth_order, os.path.join(outdir_smallTreeOpt, "synthesis_order.png"))

            final_routes_top_n_order_comb = []
            # write (final_small_routes, final_routes_top_n_order) in a pikle file for further selection of Top-Targets
            for idx, target_id in enumerate(final_routes_top_n_order):
                path[target_id] = smaller_path[idx]
            for target_id in final_routes_top_n_order_new:
                final_routes_top_n_order_comb.append(final_routes_top_n_order[target_id])
            outdir_pkl = os.path.join(outdir_smallTreeOpt,"pkl")
            if not os.path.exists(outdir_pkl):
                    os.makedirs(outdir_pkl)
            outfile = open(os.path.join(outdir_pkl,"problem.pkl"),'wb')
            pickle.dump(smaller_problem, outfile)
            outfile.close()
            outfile = open(os.path.join(outdir_pkl,"selected_routes.pkl"),'wb')
            pickle.dump(path, outfile)
            outfile.close()
            outfile = open(os.path.join(outdir_pkl,"final_routes_top_n_order.pkl"),'wb')
            pickle.dump(final_routes_top_n_order_comb, outfile)
            outfile.close()
