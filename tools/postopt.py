import numpy as np
import os
import sys
import pickle
from tools.ant_colony import AntColony
from tools.report import write_multiroute_report, plot_routes, plot_routes_order
from tools.optimal import optimal
from tools.tree_join import score_tree, removeRouteIsotopes, find_shorter_paths
import argparse

if __name__ == '__main__':       
    # direction where the data is stored
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir', type=str)
    args = parser.parse_args()
    outdir = args.datadir
    # outdir = "C:\\Temp\\heuristic_test\\component_0"
    os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\Graphviz\\bin"
    
    # read problem (list(list(dict))), selected_routes (list(id)) and final_routes_top_n_order (list with ids) from pickle files
    with open(os.path.join(outdir, "pkl", "problem.pkl"), 'rb') as fp:
        problem = pickle.load(fp)
    with open(os.path.join(outdir, "pkl", "final_routes_top_n_order.pkl"), 'rb') as fp:
        final_routes_top_n_order = pickle.load(fp)
    with open(os.path.join(outdir, "pkl", "selected_routes.pkl"), 'rb') as fp:
        selected_routes = pickle.load(fp)
    final_routes = []
    for target, route in enumerate(selected_routes):
        final_routes.append(problem[target][route])
    
    # needs this two parameters from the HTML
    '''
    # parameters have to be givin when executing file
    print(sys.argv)
    selection = sys.argv[1].split(" ") # list [0,0,0,0,1,-1,0] of length of old num_mol_smallTree
    num_mol_smallTree = int(sys.argv[2]) # length of new top_n_routes
    print(selection, num_mol_smallTree)
    '''
    with open(os.path.join(outdir,"parameters.txt")) as params:
        selection = params.readline()[1:-1].split(" ")
        number = params.readline()
        if number == 'all':
            num_mol_smallTree = len(problem)
        else:
            num_mol_smallTree = min(int(number), len(problem))
        print("Caluculating new best {} routes".format(num_mol_smallTree))

    # Calculate new synthesis order
    synth_order_small = AntColony.get_right_order(final_routes, num_mol_smallTree, final_routes_top_n_order, selection)
    final_routes_top_n_order = AntColony.get_top_N_mol(final_routes, synth_order_small, num_mol=num_mol_smallTree)
    top_n_routes = [final_routes[x] for x in final_routes_top_n_order]
    
    # Testing is find_shorter_path improves the routes
    top_n_score = score_tree({'smiles':'START>>TARGETS', 'children':top_n_routes}, 'num_unique_interm')
    top_n_routes_test = top_n_routes.copy()
    top_n_routes_test = find_shorter_paths(top_n_routes_test)
    top_n_score_test = score_tree({'smiles': 'START>>TARGETS', 'children':top_n_routes_test}, 'num_unique_interm')
    
    if top_n_score_test < top_n_score:
        top_n_routes = top_n_routes_test
        print("Found shorter routes")
    
    # This step should not be necessary: if you did not remove the isotopes
    # before running the algorithm then the results will be disappointing 
    # (and the filenames prob. to long for windows).
    # I apply the isotope removal here because I need it for debugging the
    # report generation, but "real" code shouldn't need it
    for idx, route in enumerate(top_n_routes):
        top_n_routes[idx] = removeRouteIsotopes(route)

    # write top_n_routes in pikle file
    filename = os.path.join(outdir, "pkl", "final_routes_top_n_order.pkl")
    outfile = open(filename,'wb')
    pickle.dump(final_routes_top_n_order, outfile)
    outfile.close()

    #write report
    write_multiroute_report(top_n_routes, outdir, synth_order_small)
    plot_routes(top_n_routes, os.path.join(outdir, "full_graph"))
    plot_routes_order(synth_order_small, os.path.join(outdir, "synthesis_order.png"))