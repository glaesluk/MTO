import logging
import numpy as np
import os
import sys
import pickle
import random
from collections import Counter, namedtuple
from multiprocessing import Queue, Pool
from tools.tree_join import find_shorter_paths, score_tree, nodeGenerator, connected_components, removeRouteIsotopes
from tools.report import plot_routes, plot_routes_order, write_multiroute_report
from tqdm import tqdm


class AntColony:
    def __init__(self, workers=1, evap_coef=0.134, pheromone_prob=0.074):
        """
        Class for running an Ant Colony Optimization (ACO) algorithm.

        Parameters
        ----------
        workers : int
            Number of sub-processes to use for this algorithm. Defaults to 1.
        evap_coef : float
            Evaporation coeficient. A value of 0 evaporates all traces at every
            step, while a value of 1 has no evaporation at all.
        pheromone_prob : float
            Probability of choosing the pheromone trails over the heuristic.
            A value of 1 will always follow the trail, while a value of 0 will
            only follow the heuristic.

        Notes
        -----
        This algorithm implements one of the many variations of the ACO
        algorithm. The main difference is the use of the `self.alpha` parameter:
        instead of using p1^alpha * p2^beta to decide which node to follow, this
        method uses alpha * p1 + (1-alpha)*p2.
        The default values for `evap_coef` and `pheromone_prob` were found via
        hyperparameter search. Consider them magic.
        """
        self.workers = workers
        self.pheromones = None
        self.heuristic = None
        self.problem = None

        self.evap_coef = evap_coef
        self.alpha = pheromone_prob

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
        routes = len(self.problem)
        max_trees = max(map(len, self.problem))

        # The heuristic is going to be the number of nodes (or intermediates) in a graph before
        # joining graphs. This code collects those heuristics and normalizes
        # them.
        self.heuristic = np.full((routes, max_trees), np.nan)
        for i in range(routes):
            for j in range(len(self.problem[i])):
                # The probability of choosing a target should be larger if the
                # number of children is small. Therefore, we use 1/<num_children>
                # as measure for how good a node is for the heuristic
                self.heuristic[i][j] = 1/score_tree(self.problem[i][j], 'num_unique_interm')

        self.heuristic = self.heuristic / np.nansum(self.heuristic, axis=1).reshape((-1,1))
        self.pheromones = np.full((routes, max_trees), 1/max_trees)
        for i in range(routes):
            r = self.problem[i]
            if len(r) < max_trees:
                # This row has less trees that the maximum number of trees,
                # and therefore needs a bunch of 0s
                self.pheromones[i] = [1/len(r)]*len(r) + [0]*(max_trees-len(r))

    def postopt_add(self, final_routes_top_n_id):
        """
        Sets the parameters for Post-Optimizing a smaller number of Targets reusing the pheromones 
        of the first time ACO

        Parameters
        ----------
        final_routes_top_n_id : list 
        A list with the ids of the targets/trees, that should be optimized another time
        """
        # the problem we want to solve is a part of the original problem
        self.problem = [self.problem[idx] for idx in final_routes_top_n_id]
               
        routes = len(self.problem)
        max_trees = max(map(len, self.problem))

        # The heuristic is the same as above for the smaller set of targets
        '''
        self.heuristic = np.full((routes, max_trees), np.nan)
        for i in range(routes):
            for j in range(len(self.problem[i])):
                self.heuristic[i][j] = 1/score_tree(self.problem[i][j], 'num_children')
        self.heuristic = self.heuristic / np.nansum(self.heuristic, axis=1).reshape((-1, 1))
        print("Heuristic: ", np.nanmax(self.heuristic))
        '''
        # we use the heuristic of the old Problem
        self.heuristic = np.array([[self.heuristic[ii][jj] for jj in range(max_trees)] for ii in final_routes_top_n_id])
        
        # For the Pheromones we use the pheromones we already have
        self.pheromones = np.array([[self.pheromones[ii][jj] for jj in range(max_trees)] for ii in final_routes_top_n_id])

    @staticmethod
    def eval_single_path(shared_problem, cumulative_probs, num_ants=50):
        """

        Parameters
        ----------
        shared_problem : list(list(dict))
            A list of lists of trees, where a tree is expressed as a nested dictionary.
        cumulative_probs : np.array
            A two-dimensional array with the normalized pheromones for each path in the
            graph. See `Notes` in `eval_single_path` for more details.
        queue : Queue
            A queue to send the final results back to the parent process.
        num_ants : int
            How many ants to test in a single run.

        Returns
        -------
        list
            A list of size num_ants where the i-th item is a pair. The first
            element of the pair is the score of a path, and the second element
            is the indices that identify the selected path.
            Note that this value is sent via the queue - the function itself
            has no return value.

        Notes
        -----
        Our problem is defined as a list of lists of trees. Given that an ACO
        problem works on graphs, we model the `self.pheromones` matrix as follows:
          * The first column of this array is equivalent to the pheromones in the
            edge that starts at the ant hill and ends in the first tree.
          * The second column of this array is the pheromones between the first
            tree and the second
          * The last row of this array is the cost of reaching the n-th tree from
            the (n-1)-th tree. There is no cost from the last tree to the target
            because we assume that to be an edge of cost 0.
        Each column of this array is a matrix of dimensions (max_trees, max_trees)
        and represents the attractiveness of going from node i in the first
        layer to node j in the next layer. The attractiveness is expressed as a
        cumulative probability because it makes the sampling of a random number
        faster.

        Regarding sampling: if my prob. dist. is [0.2, 0.3, 0.05, 0.45], one way
        to sample from here is by storing this array as cumulative probabilities
        (in this case, [0.2, 0.5, 0.55, 1]), sample a random uniform number,
        and pick the index of the first element that exceeds that value. For
        instance, if I sample a 0.52, then I would pick the 3rd element of the
        original distribution. This is how we randomly pick based on pheromones:
        they bias our probability towards the best paths, while still allowing
        some exploration.
        """
        random.seed()
        retval = []
        trees = len(shared_problem)
        for i in range(num_ants):
            # A solution to the problem is a forest of size
            # len(shared_problem), that is, one tree from each tree set.
            selected_trees = []
            indices = []
            for j in range(trees):
                # The first index of cumulative_probs has to be understood as
                # "I am looking at the j-th transition in a graph".
                # cumulative_probs[j][1] is the probapility to use the 1st route in the next step.

                # We first need to define which probabilities apply at this step.
                # That means choosing the correct sub-matrix from the pheromone
                # table.
                node_probs = cumulative_probs[j]

                # Now, we sample a random value and substract the value from
                # the cumulative probabilities.
                # As a result, the first non-zero value in each column is the one I
                # want in my sampling.
                rand_val = random.random()
                diff_table = node_probs-rand_val

                # I set all negative values and NANs to a really high value that cannot happen.
                diff_table[np.isnan(diff_table)] = 2
                diff_table[diff_table < 0] = 2

                # Finally, I now get the indices of the smallest indices (that is,
                # the smallest non-zero element from the table above)
                next_node = np.argmin(diff_table)
                selected_trees.append(shared_problem[j][next_node])
                indices.append(next_node)

            # Finally, I calculate the score of the selected trees
            
            # I have deleted the find_shorter_paths here because I wasn't sure if it is effective
            #combined_trees = find_shorter_paths(selected_trees)
            combined_trees = {'smiles':'START>>TARGETS', 'children': selected_trees}
            # The following line is for opimizing the number of (children + intermediates)
            # score = sum(map(lambda t: score_tree(t, 'num_children_noreactions'), combined_trees))
            # The folliwing line is for optimizing the number of intermediates
            # score_old = sum(map(lambda t: score_tree(t, 'num_interm'), selected_trees))
            score = score_tree(combined_trees, 'num_unique_interm')
            # retval.append((score, selected_trees))
            # print((score, indices))
            retval.append((score, indices))
        return retval

    def run(self, num_steps):
        """ Runs the ACO algorithm.

        Parameters
        ----------
        num_steps : int
            Number of iterations

        Returns
        -------
        list
            A list with the ids of the best routes found for each target of the problem
        """
        best_cost = 1e6
        best_path = None
        pbar = tqdm(desc="Ant hill algorithm", total=num_steps)
        for s in range(num_steps):
            best_local_cost = float('inf')
            # We need to calculate two factors: the attractiveness of the
            # heuristic (which is already in the self.heuristic variable) and
            # the attractiveness of the pheromone trail (which changes at every
            # step)

            # Converts the pheromone table (which contains un-normalized values) into
            # a table of probability distributions.
            pheromone_probs = self.pheromones / np.nansum(self.pheromones, axis=1)[:, None]
            # Multiplies both probabilities, and convert them into a cumulative
            # probability table
            final_ant_probs = self.alpha * pheromone_probs + (1-self.alpha) * self.heuristic
            # Because some states are impossible, there are plenty of divisions
            # by 0 that we don't care about. Therefore, we supress warnings
            # here (and only here).
            with np.errstate(divide='ignore'):
                a = np.cumsum(final_ant_probs, axis=1)
                b = np.nansum(final_ant_probs, axis=1)[:, None]
                cumulative_probs = a/b

            logging.debug("Running iteration {}/{}".format(s, num_steps))
            params = [(self.problem, cumulative_probs)]*self.workers
            with Pool(processes=self.workers) as p:
                answer_set = p.starmap(AntColony.eval_single_path, params)
            # Collect the results and update the pheromone trail
            evaporation = self.pheromones * self.evap_coef
            visits = np.zeros(self.pheromones.shape)
            for worker_results in answer_set:
                for cost, trees in worker_results:
                    q = 1
                    if cost < best_local_cost:
                        best_local_cost = cost
                    if cost < best_cost:
                        # This should be a hyperparameter
                        q = 10
                        best_cost = cost
                        best_path = trees
                    # Update pheromone costs
                    for i in range(len(trees)):
                        visits[i, trees[i]] += q * len(self.problem) / cost
            self.pheromones = evaporation + visits
            logging.debug("End of iteration {}. Best cost: {}/{}. Best path so far: {}".format(s, best_local_cost, best_cost, best_path))
            pbar.set_description("Best cost: {}/{}".format(best_local_cost, best_cost))
            pbar.update(1)
        pbar.close()
        return best_path

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
        Route = namedtuple('Route', 'generalid smiles intermediates buyables')
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
                pending_molecules.append(Route(idx, route['smiles'], route_interms, route_buyables))
            else:
                # look if route has special condition from list selection
                try: # gives an Error if route isn't in the best_routes
                    val = selection[best_routes.index(route)]
                    if val != '-1':
                        if val == '1':
                            inRoutes.append(idx)
                        route_interms = AntColony.route_to_intermediates(route, keep_reactions=False)
                        route_buyables = AntColony.route_to_buyables(route)
                        pending_molecules.append(Route(idx, route['smiles'], route_interms, route_buyables))
                except:
                    route_interms = AntColony.route_to_intermediates(route, keep_reactions=False)
                    route_buyables = AntColony.route_to_buyables(route)
                    pending_molecules.append(Route(idx, route['smiles'], route_interms, route_buyables))
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
            retval.append(Route(i, new_smiles, new_interms, new_buyables))
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
            retval[i] = ({curr_route.smiles}, curr_route.intermediates, curr_route.buyables)
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
            pending_molecules.append(Route(idx, route['smiles'], route_interms, route_buyables))
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
        aco = AntColony(workers=30)
        aco.add_sets(smaller_problem)
        if run_ants:
            path = aco.run(num_steps=50)
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
        # testing if the find_shorter_path functoin improves the result
        final_routes_test = find_shorter_paths(final_routes)
        if score_tree({'smiles':'START>>TARGETS', 'children': final_routes_test}, 'num_unique_interm') < \
        score_tree({'smiles':'START>>TARGETS', 'children': final_routes}, 'num_unique_interm'):
            final_routes = final_routes_test
            print("Found shorter routes")

        # New feature: what's the optimal way to ...
        synth_order = AntColony.get_right_order(final_routes)
        final_routes_id = aco.get_top_N_mol(final_routes, synth_order, num_mol=len(final_routes))
        outdir = "C:\\Temp\\ant_colony_copy\\component_{}".format(component_id)
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
        # top_n_routes = find_shorter_paths(top_n_routes)


        outdir_smallTree = outdir + "_smallTree"
        if not os.path.exists(outdir_smallTree):
            os.makedirs(outdir_smallTree)
        write_multiroute_report(top_n_routes, outdir_smallTree, synth_order[:num_mol_smallTree]) # synth_order[:num_mol_smallTree]
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
        # Optimize routes for top n targets with AntColony another time:
        
        top_n_problem = []
        for idx in final_routes_top_n_id:
            top_n_problem.append(smaller_problem[idx])
        '''
        #Here we can use the old Pheromones and so don't need to construct another Colony-Objekt
        aco2 = AntColony(workers=20)
        aco2.add_sets(top_n_problem)
        smaller_path = aco2.run(num_steps=10)
        '''
        aco.postopt_add(final_routes_top_n_id)
        smaller_path = aco.run(num_steps = 200)

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

        # multiroute optimalization
        final_synth_order = AntColony.get_right_order(final_small_routes)
        final_routes_id = aco.get_top_N_mol(final_small_routes, final_synth_order, num_mol=len(final_small_routes))

        if final_score_opt < final_score:
            print("Optimized routes another time (from score {} to {}). See results in ..._smallTreeOpt".format(final_score,final_score_opt))
            # write report
            # outdir_smallTree = "\\".join(outdir.split("\\")[:-1])+"_smallTreeOpt"
            outdir_smallTreeOpt = outdir+"_smallTreeOpt"
            print(outdir_smallTreeOpt)
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