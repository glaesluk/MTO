#!/usr/bin/python3
import logging
import numpy as np
import os
import sys
import pickle
import pulp
import random
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

    def run(self):
        """ Looks which routes to take in order to get the highest amount of targets with 
        ascending number of intermediates 

        Returns
        -------
        list
            A list of pairs (target_id, route_id), with the best Routes for 
            each target. The order of the list represent the synthesis_order.


        Notes
        -----
        The returned list has an extra property: it tries to build the largest
        number of elements from the least number of intermediates. This means
        that, given a big enough list of routes, it should be possible to look
        at the graph of #molecules given #intermediates and decide where it
        would be more economical to stop.
        """

        # Named tuples to make the code easier to read
        Route = namedtuple('Route', 'target_id route_id intermediates buyables')
        RouteInterms = namedtuple('RouteInterms', 'idx unique_score')
        # List with ids ordered according their common intermediates
        order = []
        # In this code, a route is a set of intermediates and buyables. This code turns the
        # nested dictionaries from the input into sets.
        pending_molecules = []
        for target_id, target in enumerate(self.problem):
            for route_id, route in enumerate(target):
                route_interms = AntColony.route_to_intermediates(route, keep_reactions=False)
                route_buyables = AntColony.route_to_buyables(route)
                pending_molecules.append(Route(target_id, route_id, route_interms, route_buyables))
            
        # Start of the code that chooses at every step which route should be
        # created next. We attach a progress bar to it.
        pbar = tqdm(desc="Calculating compatibility of routes", total=len(pending_molecules))
        # iterate until there are only routes of n targets pending
        pending_targets = len(self.problem)
        while len(pending_molecules) > 0:
            loop_routes = []
            # Identify all pending intermediates
            all_interms = Counter()
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
                loop_routes.append(RouteInterms(idx, unique_score))
            # Finds the route with the largest amount of non-shared intermediates.
            # In case of a tie, picks the route with the largest amount of *shared*
            # intermediates.
            loop_routes.sort(key=lambda x: np.sum(list(x.unique_score.values())), reverse=True)
            chosen = loop_routes[0]
            # Put the ids of the chosen route to the final order
            order.append([pending_molecules[chosen.idx].target_id, pending_molecules[chosen.idx].route_id])
            # Remove the selected route from the "pending" list
            pending_molecules = pending_molecules[:chosen.idx] + pending_molecules[chosen.idx + 1:]
            # Update the progress bar
            pbar.update(1)
        pbar.close()
        # The list has to be read back-to-front. Therefore, we reverse the list
        order.reverse()
        # we only want one route for each target
        final_routes = []
        final_order = []
        for route in order:
            if route[0] not in final_order:
                final_routes.append(route)
                final_order.append(route[0])
        final_routes.sort(key = lambda x: x[0])
        for i in range(len(final_routes)):
            final_routes[i] = final_routes[i][1]
        return final_routes, final_order

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
            if keep_reactions or '>>' not in current['smiles'] and (len(current['children']) >0):
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
            A list with the best n routes we have found so far. 

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
        RouteInterms = namedtuple('RouteInterms', 'idx common unique generalid')
        # We keep of list of which intermediates we have already seen to know
        # which ones are new for a route and which ones are already there.
        seen_intermediates = set()
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
                    val = selection[best_routes.index(route)]
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
                common = set(filter(lambda x: all_interms[x] > 1, list(r.intermediates)))
                unique = set(filter(lambda x: all_interms[x] == 1, list(r.intermediates)))
                loop_routes.append(RouteInterms(idx, common, unique, r.generalid))
            # Finds the route with the largest amount of non-shared intermediates.
            # In case of a tie, picks the route with the largest amount of *shared*
            # intermediates.
            loop_routes.sort(key=lambda x: (len(x.unique), len(x.common)), reverse=True)
            # molecules of inRoutes should only be chosen in the last n steps
            i = 0
            while loop_routes[i].generalid in inRoutes and len(pending_molecules) > n :
                i += 1
            chosen = loop_routes[i]
            new_smiles = pending_molecules[chosen.idx].smiles
            new_interms = chosen.common.union(chosen.unique) - seen_intermediates
            new_buyables = pending_molecules[chosen.idx].buyables.intersection(new_interms)
            # Remove the selected route from the "pending" list
            pending_molecules = pending_molecules[:chosen.idx] + pending_molecules[chosen.idx + 1:]
            # Update the list of intermediates we have already seen
            seen_intermediates.update(new_interms)
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
            new_buyables = curr_route.buyables.intersection(new_interms)
            retval[ii] = ({curr_route.smiles}, new_interms, new_buyables)
            seen_intermediates.update(new_interms)
            ii += 1

        return retval



if __name__ == '__main__':
    route_dir = 'C:/Users/GNQBG/OneDrive - Bayer/Personal Data/AI4Synthesis_service_calculate_routes/routes'
    # route_dir = '/tmp/newroutes/V8/01'
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
            path, order = aco.run()
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
        # before running the algorithm then the results will be disappointing.
        # I apply the isotope removal here because I need it for debugging the
        # report generation, but "real" code shouldn't need it
        for idx, route in enumerate(final_routes):
            final_routes[idx] = removeRouteIsotopes(route)
        # testing if the find_shorter_paths function improves the result
        final_routes_test = find_shorter_paths(final_routes)
        if score_tree({'smiles':'START>>TARGETS', 'children': final_routes_test}, 'num_unique_interm') < \
        score_tree({'smiles':'START>>TARGETS', 'children': final_routes}, 'num_unique_interm'):
            final_routes = final_routes_test
            print("Found shorter routes")

        # New feature: what's the optimal way to ...
        synth_order = AntColony.get_right_order(final_routes)
        final_routes_id = aco.get_top_N_mol(final_routes, synth_order, num_mol=len(final_routes))
        outdir = "C:\\Temp\\total_ordering\\component_{}".format(component_id)
        os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\Graphviz\\bin"
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        write_multiroute_report(final_routes, outdir, synth_order)
        plot_routes(final_routes, os.path.join(outdir, "full_graph"))
        plot_routes_order(synth_order, os.path.join(outdir, "synthesis_order.png"))
        
        # write (final_routes, final_routes_id) in a pikle file for further selection of Top-Targets
        outdir_pkl = os.path.join(outdir,"pkl")
        if not os.path.exists(outdir_pkl):
                os.makedirs(outdir_pkl)
        outfile = open(os.path.join(outdir_pkl,"final_routes.pkl"),'wb')
        pickle.dump(final_routes, outfile)
        outfile.close()
        outfile = open(os.path.join(outdir_pkl,"final_routes_top_n_id.pkl"),'wb')
        pickle.dump(final_routes_id, outfile)
        outfile.close()

          # choose top n targets
                
        num_mol_smallTree = 10
        final_routes_top_n_id = final_routes_id[:num_mol_smallTree]        
        # get smaller report with only the top N molecules regarding synthesis order (old routes)
        top_n_routes = [final_routes[x] for x in final_routes_top_n_id]

        outdir_smallTree = outdir + "_smallTree"
        if not os.path.exists(outdir_smallTree):
            os.makedirs(outdir_smallTree)
        write_multiroute_report(top_n_routes, outdir_smallTree, synth_order[:num_mol_smallTree])
        plot_routes(top_n_routes, os.path.join(outdir_smallTree, "full_graph"))
        plot_routes_order(synth_order[:num_mol_smallTree], os.path.join(outdir_smallTree, "synthesis_order.png"))
        # write (final_routes, final_routes_top_n_id) in a pikle file for further selection of Top-Targets
        outdir_pkl = os.path.join(outdir_smallTree,"pkl")
        if not os.path.exists(outdir_pkl):
                os.makedirs(outdir_pkl)
        outfile = open(os.path.join(outdir_pkl,"final_routes.pkl"),'wb')
        pickle.dump(final_routes, outfile)
        outfile.close()
        outfile = open(os.path.join(outdir_pkl,"final_routes_top_n_id.pkl"),'wb')
        pickle.dump(final_routes_top_n_id, outfile)
        outfile.close()
        final_score = score_tree({'smiles':'START>>TARGETS', 'children': top_n_routes}, 'num_unique_interm')
        '''
        # Optimize routes for top n targets with AntColony another time:
        top_n_problem = []
        for idx in final_routes_top_n_id:
            top_n_problem.append(smaller_problem[idx])
        aco2 = AntColony()
        aco2.add_sets(top_n_problem)
        smaller_path, final_order = aco2.run()

        # list with best routes of top n targets 
        final_small_routes = []
        for idx, winner in enumerate(smaller_path):
            final_small_routes.append(top_n_problem[idx][winner])

        for idx, route in enumerate(final_small_routes):
            final_small_routes[idx] = removeRouteIsotopes(route)

        # Testing if find_shorter path improves the Routes
        final_small_routes_test = find_shorter_paths(final_small_routes)
        final_score_opt_test = score_tree({'smiles': 'START', 'children':final_small_routes_test}, 'num_unique_interm')
        final_score_opt = score_tree({'smiles':'START>>TARGETS', 'children':final_small_routes}, 'num_unique_interm')
        
        if final_score_opt_test < final_score_opt:
            final_small_routes = final_small_routes_test
            final_score_opt = final_score_opt_test
            print("Found shorter routes")

        # multiroute optimalization
        final_synth_order = AntColony.get_right_order(final_small_routes)
        final_routes_id = aco.get_top_N_mol(final_small_routes, final_synth_order, num_mol=len(final_small_routes))
        print("Tried to ptimize routes another time (from score {} to {}).".format(final_score,final_score_opt))
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
                
            # write (final_small_routes, final_routes_top_n_id) in a pikle file for further selection of Top-Targets
            outdir_pkl = os.path.join(outdir_smallTreeOpt,"pkl")
            if not os.path.exists(outdir_pkl):
                    os.makedirs(outdir_pkl)
            outfile = open(os.path.join(outdir_pkl,"final_routes.pkl"),'wb')
            pickle.dump(final_small_routes, outfile)
            outfile.close()
            outfile = open(os.path.join(outdir_pkl,"final_routes_top_n_id.pkl"),'wb')
            pickle.dump(final_routes_top_n_id, outfile)
            outfile.close()
            '''