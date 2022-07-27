import os
import pickle
import rdkit.Chem as Chem
import time
from collections import defaultdict, namedtuple

# A record is a tuple containing a node and its score.
Record = namedtuple('Record', 'elem score')


def removeIsotopes(smi):
    """ Removes the isotope label of each atom of a SMILES.
    If the provided smiles string is not canonical it will be after
    the removal.

    Parameters
    ----------
    smi : str
        A SMILES string.
    Returns
    -------
    str
        A SMILES string without isotopes.
    """
    all_smis = smi.split(">>")
    retval = []
    for one_smi in all_smis:
        mol_tmp = Chem.MolFromSmiles(one_smi)
        atom_data = [(atom, atom.GetIsotope()) for atom in mol_tmp.GetAtoms()]
        for atom, isotope in atom_data:
            if isotope:
                atom.SetIsotope(0)
        smiles = Chem.MolToSmiles(mol_tmp, True)
        retval.append(smiles)
    return ">>".join(retval)


def removeRouteIsotopes(route):
    """ Applies the `removeIsotopes` function to every node of this route.

    Parameters
    ----------
    route : dict
        A route, written as a dictionary.

    Returns
    -------
    dict
        The same route as input, but with the `removeIsotopes` function applied
        to every smiles inside.
    """
    queue = [route]
    while len(queue) > 0:
        node, queue = queue[0], queue[1:]
        node['smiles'] = removeIsotopes(node['smiles'])
        for c in node['children']:
            queue.append(c)
    return route


def connected_components(all_sets):
    """ Given a list of sets, returns sets of indices of which sets share at
    least one common element. If we assume these indices to be vertices in
    a common graph, then the result can be understood as the sets of
    disconnected components.

    Parameters
    ----------
    all_sets : list(set)
        A list of all the sets that need to be merged.

    Returns
    -------
    list(set(int))
        A list where each element is a set, and each set contains the indices
        of sets in the input list that share at least one element.

    Notes
    -----
    The algorithm works as follows: we have a list of sets:
        [{1}, {2,3}, {2,4}, {5}]
    We take the first element ({1}) and try to merge it with the next sets
    (which in this case doesn't work, and nothing happens).
    We take the second and try to merge it, which works. Therefore, the sets
    {2,3} and {2,4} are merged into the set {2,3,4}. We now try to merge this
    new set with the remaining elements ({5}), and so on until we are done.
    Once we reach the end of the loop, the algorithm ends.
    Note that the algorithm hangs on the following...

    Theorem: Once we have verified that elements e_0, e_1, ..., e_n cannot be
    merged, we don't need to check them again. This guarantees both that
    the algorithm does what it claims to do and that it will end.
    Informal Proof: let n be the index of all elements that I have already
    tried to merge (and failed) with all elements with index higher than n.
    If I take one element e_i to the left of n and two elements e_j, e_k to the
    right of n, I already know that e_i has no common elements with neither
    e_j nor e_k. Therefore, e_i has no common elements with their union either.
    """
    items_set = all_sets
    indices_set = [{i} for i in range(len(all_sets))]
    idx = 0
    # We will stop once we can no longer merge any two sets with each other.
    while idx < len(items_set):
        curr_item = items_set[idx]
        next_idx = idx + 1
        merged = False
        while next_idx < len(items_set) and not merged:
            candidate = items_set[next_idx]
            if len(curr_item.intersection(candidate)) > 0:
                # These two sets have a common element, so I can merge them.
                items_set[idx] = items_set[idx].union(candidate)
                indices_set[idx] = indices_set[idx].union(indices_set[next_idx])
                # Remove the rightmost element
                del items_set[next_idx]
                del indices_set[next_idx]
                merged = True
            next_idx += 1
        if not merged:
            # I reached the end of the list and couldn't merge anything, so I
            # can move the pivot one element to the right.
            idx += 1
    return indices_set


def score_tree(tree, criteria):
    """ Given a tree, it returns its score as defined by one out of several
    possible criteria. All scores are returned such that smaller is better.

    Parameters
    ----------
    tree : dict
        A tree, defined as a recursive dictionary.
    criteria : str
        Criteria to use to score this tree. Possible values are:
           * depth: returns the depth of this tree
           * depth_noreactions: returns the depth of this tree without counting reactions
           * num_children: returns the number of nodes in this tree
           * num_children_noreactions: returns the number of nodes in this tree
             without counting reactions
           * num_interm: returns the number of nodes in the tree 
             without counting reactions and buyables
           * num_unique_interm: returns the number of different nodes in the tree 
             without counting reactions and buyables
           * weakest_link: returns 1-(the smallest success probability in this tree)
           * prob: returns 1-(the product of all individual probabilities)

    Returns
    -------
    int
        An integer score for the input tree.

    Notes
    -----
    All these algorithms would be more easily defined if we did it recursively,
    but that would be inefficient and we have a *lot* of trees to go through.
    Therefore, all algorithms are implemented using a queue instead.
    """
    assert criteria in {"depth", "depth_noreactions", "num_children", "num_children_noreactions",
                         "num_interm", "num_unique_interm", "weakest_link", "prob"},\
        "Unknown criteria {}".format(criteria)

    # All algorithms use a queue, so I simply define a common one
    retval = 0
    queue = []
    if criteria == "depth":
        # The first value is the depth I have seen so far. The idea being:
        # I will keep track of how deep I go, and return the maximum value
        queue.append((1, tree))
        while len(queue) > 0:
            elem, queue = queue[0], queue[1:]
            retval = max(retval, elem[0])
            for c in elem[1]['children']:
                queue.append((1+elem[0], c))
    elif criteria == "depth_noreactions":
        # The first value is the depth I have seen so far. The idea being:
        # I will keep track of how deep I go, and return the maximum value
        queue.append((1, tree))
        while len(queue) > 0:
            elem, queue = queue[0], queue[1:]
            retval = max(retval, elem[0])
            for c in elem[1]['children']:
                if '>>' in c['smiles']:
                    # Reactions don't increase the counter
                    queue.append((elem[0], c))
                else:
                    queue.append((1+elem[0], c))
    elif criteria == "num_children":
        queue.append(tree)
        while len(queue) > 0:
            elem, queue = queue[0], queue[1:]
            retval += 1
            for c in elem['children']:
                queue.append(c)
    elif criteria == "num_children_noreactions":
        queue.append(tree)
        while len(queue) > 0:
            elem, queue = queue[0], queue[1:]
            if '>>' not in elem['smiles']:
                retval += 1
            for c in elem['children']:
                queue.append(c)
    elif criteria == "num_interm":
        queue.append(tree)
        while len(queue) > 0:
            elem, queue = queue[0], queue[1:]
            if '>>' not in elem['smiles'] and len(elem['children']) != 0:
                retval += 1
            for c in elem['children']:
                queue.append(c)
    elif criteria == "num_unique_interm":
        queue.append(tree)
        seen_intermediates = []
        while len(queue) > 0:
            elem, queue = queue[0], queue[1:]
            if '>>' not in elem['smiles'] and len(elem['children']) != 0 and elem['smiles'] not in seen_intermediates:
                seen_intermediates.append(elem['smiles'])
                retval += 1
            for c in elem['children']:
                queue.append(c)  
    elif criteria == "weakest_link":
        retval = 1
        queue.append(tree)
        while len(queue) > 0:
            elem, queue = queue[0], queue[1:]
            if 'plausibility' in elem.keys():
                retval = min(retval, elem['plausibility'])
            for c in elem['children']:
                queue.append(c)
        if retval != 1:
            retval = 1-retval
    elif criteria == "prob":
        retval = 1
        queue.append(tree)
        while len(queue) > 0:
            elem, queue = queue[0], queue[1:]
            if 'plausibility' in elem.keys():
                retval = retval * elem['plausibility']
            for c in elem['children']:
                queue.append(c)
        if retval != 1:
            retval = 1-retval
    return retval


def nodeGenerator(tree):
    """ Returns a generator for a tree, returning one node at the time.
    The nodes are returned top-to-bottom in Depth-(Breath-??) First Search order.

    Parameters
    ----------
    tree : dict()
        A tree, defined as a recursive dictionary.

    Returns
    -------
    dict
        A sub-node of the tree
    """
    queue = [tree]
    while len(queue) > 0:
        elem, queue = queue[0], queue[1:]
        for c in elem['children']:
            queue.append(c)
        yield elem


def rebuild_tree(smiles, mol_to_best_mol):
    """ Given a starting smiles, builds an entirely new tree.

    Parameters
    ----------
    smiles : str
        SMILES for the target molecule
    mol_to_best_mol : dict
        Dictionary mapping smiles to a Record. Required to rebuild the tree.

    Returns
    -------
    dict
        A tree, defined as a recursive dictionary. The root of this tree is
        the SMILES given as input.
    """
    # TODO: Make sure that I am not having problems with pass-by-copy instead
    # of pass-by-reference
    # 2020/09/21: I actually do have some issues here. Make a copy of everything.
    start_node = mol_to_best_mol[smiles].elem
    queue = [start_node]
    while len(queue) > 0:
        elem, queue = queue[0], queue[1:]
        for i in range(len(elem['children'])):
            elem['children'][i] = mol_to_best_mol[elem['children'][i]['smiles']].elem
            queue.append(elem['children'][i])
    return start_node


def find_shorter_paths(all_trees):
    """

    Parameters
    ----------
    all_trees : list(dict)
        A list of trees, defined as recursive dictionaries.

    Returns
    -------
    list(dict)
        A list of trees where the target molecules are the same, but each tree
        reuses as many sub-steps as possible.

    Notes
    -----
    When dealing with trees, the following is a problem that can happen often:
        Tree1 = A(B_8, C_7)
        Tree2 = F(A_9, B_9)
        Tree3 = A(B_5, C_5)
    Tree2 may replace its B_9 for the B_8 that it saw before. However, Tree3
    brings an even better B_5, but neither Tree1 nor Tree2 know about this.
    Therefore, once I'm aware of B_5 I need to go back to the beginning for
    both Tree1 and Tree2 to update their routes.
    This problem can be also solved with doubly-linked lists, in which an
    update requires going up the tree of everyone who uses the old node and
    letting them know that a better node may now be available. This is more
    efficient but more difficult to program, and therefore it is not
    implemented right now.
    """
    # return(all_trees)
    # This dictionary keeps track of the best way to build a specific molecule,
    # along with the score for that molecule.
    mol_to_best_mol = defaultdict(lambda: Record(None, float('inf')))

    """
    # Debug information
    print("Before the process:\n==================")
    for idx, t in enumerate(all_trees):
        print("Tree {}: {}".format(idx, score_tree(t, 'num_children')))
    """

    stable = False
    while not stable:
        stable = True
        for t in all_trees:
            for node in nodeGenerator(t):
                # Note: for 'num_children' a lower value is better, but for
                # other metrics this is the other way around.
                # If you change the scoring function here, check the next line
                # too!
                # Also, don't use `<=` or your code will probably never stop.
                node_score = score_tree(node, 'num_interm')
                if node_score < mol_to_best_mol[node['smiles']].score:
                    # I found a sub-node that is better than the one I had
                    # before in this route. Therefore, I record this node as my
                    # new best route, and set stable to `False` because other
                    # records may change too.
                    mol_to_best_mol[node['smiles']] = Record(node, node_score)
                    stable = False
    # At this point, I can rebuild the original routes
    new_trees = []
    for t in all_trees:
        new_trees.append(rebuild_tree(t['smiles'], mol_to_best_mol))

    """
    # Debug information
    print("After the process:\n==================")
    for idx, t in enumerate(all_trees):
        print("Tree {}: {}".format(idx, score_tree(t, 'num_children')))
    """
    return new_trees

def total_ordering(problem):
        """ Looks which routes to take in order to get the highest amount of targets with 
        ascending number of intermediates 

        Returns
        -------
        list
            A list with the ids of the best Targets for each route

        order
            A list of all target-ids defining synth_order

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
        for target_id, target in enumerate(problem):
            for route_id, route in enumerate(target):
                route_interms = AntColony.route_to_intermediates(route, keep_reactions=False)
                route_buyables = AntColony.route_to_buyables(route)
                pending_molecules.append(Route(target_id, route_id, route_interms, route_buyables))
            
        # Start of the code that chooses at every step which route should be
        # created next. We attach a progress bar to it.
        pbar = tqdm(desc="Calculating compatibility of routes", total=len(pending_molecules))
        # iterate until there are only routes of n targets pending
        while len(pending_molecules) > 0:
            loop_routes = []
            # Identify all pending intermediates
            all_interms = Counter()
            for r in pending_molecules:
                all_interms.update(r.intermediates)
            # Convert a route to the "shared/unique" format
            for idx, r in enumerate(pending_molecules):
                # Note: this used to be a list. I hope that using a set won't break anything.
                unique_score={x:(1/all_interms[x]) for x in r.intermediates}
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

def print_tree(tree, depth=0):
    leading_space = ''.join([" " for _ in range(depth)])
    print("{}{}".format(leading_space, removeIsotopes(tree['smiles'])))
    for c in tree['children']:
        print_tree(c, depth+1)


if __name__ == '__main__':
    limit = 100
    full_routes = []
    route_set = []
    route_dir = '../trees/T2_LSTM_beck_zuerich_coop/03'
    all_files = list(os.listdir(route_dir))
    all_files.sort()
    # Step 1: collect all intermediates and check their clustering.
    for file in all_files:
        if file.endswith('pkl') and limit > 0:
            with open(os.path.join(route_dir, file), 'rb') as fp:
                routes = pickle.load(fp)
                if len(routes) > 0:
                    all_interms = set()
                    route = routes[0]
                    for node in nodeGenerator(route):
                        all_interms.add(node['smiles'])
                    route_set.append(all_interms)
                    full_routes.append(routes)
                    limit -= 1
    all_components = connected_components(route_set)
    print(all_components)
    # Step 2: for the first cluster (which I know has more than one route), see
    # how good does the route filtering works
    test_cluster = []
    for i in all_components[0]:
        test_cluster.append(full_routes[i][0])

    print_tree(test_cluster[1])
    final_routes = find_shorter_paths(test_cluster)
    print_tree(final_routes[1])