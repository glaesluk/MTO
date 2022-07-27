import numpy as np
import os
import sys
import pickle
from tools.optim_colony import AntColony
from tools.report import write_multiroute_report, plot_routes, plot_routes_order
from tools.optimal import optimal
from tools.tree_join import score_tree, removeRouteIsotopes, find_shorter_paths

if __name__ == '__main__':       
    # direction where the data is stored
    outdir = "C:\\Temp\\optim_colony\\component_0_smallTree"
    os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\Graphviz\\bin"
    
    # read problem (list(list(dict))), selected routes and final_routes_top_n_order (list with ids) from pickle files
    with open(os.path.join(outdir, "pkl", "problem.pkl"), 'rb') as fp:
        problem = pickle.load(fp)
    with open(os.path.join(outdir, "pkl", "final_routes_top_n_order.pkl"), 'rb') as fp:
        final_routes_top_n_order = pickle.load(fp)
    with open(os.path.join(outdir, "pkl", "selected_routes.pkl"), 'rb') as fp:
        selected_routes = pickle.load(fp)
    
    with open(os.path.join(outdir,"parameters.txt")) as params:
        selection = params.readline()[1:-1].split(" ")
        number = params.readline()

    # generating new problem with respect to the user-selection
    new_problem = []
    selected_out = []
    top_n_order_mod = [None]*len(final_routes_top_n_order)
    for i in range(len(problem)):
        # gives an error if i wasn't in the top n routes
        try:
            if selection[final_routes_top_n_order.index(i)] == '-1':
                selected_out.append(i)
            else:
                new_problem.append(problem[i])
                top_n_order_mod[final_routes_top_n_order.index(i)] = i - len(selected_out)
        except:
            new_problem.append(problem[i])

    if number == 'all':
        num_mol_smallTree = len(new_problem)
    else:
        num_mol_smallTree = min(int(number), len(new_problem))
    print("Optimizing again to get best {} routes".format(num_mol_smallTree))

    # reoptimize new problem
    aco = AntColony()
    aco.add_sets(new_problem)
    path = aco.run()
    final_routes = []
    for target, route in enumerate(path):
        final_routes.append(new_problem[target][route])
    
    # This step should not be necessary: if you did not remove the isotopes
    # before running the algorithm then the results will be disappointing 
    # (and the filenames prob. to long for windows).
    # I apply the isotope removal here because I need it for debugging the
    # report generation, but "real" code shouldn't need it
    for idx, route in enumerate(final_routes):
        final_routes[idx] = removeRouteIsotopes(route)

    # save new selected routes in list
    p = 0
    for i in range(len(problem)):
        # gives an error if i wasn't in the top n routes
        try:
            if selection[final_routes_top_n_order.index(i)] != '-1':
                selected_routes[i] = path[p]
                p+=1
        except:
            selected_routes[i] = path[p]
            p+=1
    
    # prepare selection and top_n_order to recalculate order
    i = True
    while i:
        try:
            selection.remove('-1')
            top_n_order_mod.remove(None)
        except:
            i=False
    # Calculate new synthesis order
    synth_order = AntColony.get_right_order(final_routes, num_mol_smallTree, top_n_order_mod, selection)
    final_routes_top_n_order = AntColony.get_top_N_mol(final_routes, synth_order, num_mol=num_mol_smallTree)
    top_n_routes = [final_routes[x] for x in final_routes_top_n_order]
    
    # Testing if find_shorter path improves the Routes
    top_n_routes_test = find_shorter_paths(top_n_routes)
    top_n_score_test = score_tree({'smiles': 'START>>TARGETS', 'children':top_n_routes_test}, 'num_unique_interm')
    top_n_score = score_tree({'smiles':'START>>TARGETS', 'children':top_n_routes}, 'num_unique_interm')
    
    if top_n_score_test < top_n_score:
        top_n_routes = final_small_routes_test
        print("Found shorter routes")

    # write top_n_routes and selected_routes in pikle file
    filename = os.path.join(outdir, "pkl", "final_routes_top_n_order.pkl")
    outfile = open(filename,'wb')
    pickle.dump(final_routes_top_n_order, outfile)
    outfile.close()
    filename = os.path.join(outdir, "pkl", "selected_routes.pkl")
    outfile = open(filename,'wb')
    pickle.dump(selected_routes, outfile)
    outfile.close()

    #write report
    write_multiroute_report(top_n_routes, outdir, synth_order)
    plot_routes(top_n_routes, os.path.join(outdir, "full_graph"))
    plot_routes_order(synth_order, os.path.join(outdir, "synthesis_order.png"))
        
    # Optimize routes for top n targets with AntColony another time:
    top_n_problem = []
    for idx in final_routes_top_n_order:
        top_n_problem.append(new_problem[idx])
    aco2 = AntColony()
    aco2.add_sets(top_n_problem)
    smaller_path = aco2.run()
    

    # list with best routes of top n targets 
    final_small_routes = []
    for idx, winner in enumerate(smaller_path):
        final_small_routes.append(top_n_problem[idx][winner])

    for idx, route in enumerate(final_small_routes):
        final_small_routes[idx] = removeRouteIsotopes(route)

    # Testing if find_shorter path improves the Routes
    final_score_opt = score_tree({'smiles':'START>>TARGETS', 'children':final_small_routes}, 'num_unique_interm')
    final_small_routes_test = final_small_routes.copy()
    final_small_routes_test = find_shorter_paths(final_small_routes)
    final_score_opt_test = score_tree({'smiles': 'START>>TARGETS', 'children':final_small_routes_test}, 'num_unique_interm')
    
    if final_score_opt_test < final_score_opt:
        final_small_routes = final_small_routes_test
        final_score_opt = final_score_opt_test
        print("Found shorter routes")

    # multiroute optimalization
    final_synth_order = AntColony.get_right_order(final_small_routes)
    final_routes_top_n_order_new = aco.get_top_N_mol(final_small_routes, final_synth_order, num_mol=len(final_small_routes))
    print("Tried to optimize routes another time (from score {} to {}).".format(top_n_score,final_score_opt))
    if final_score_opt < top_n_score:
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
        pickle.dump(new_problem, outfile)
        outfile.close()
        outfile = open(os.path.join(outdir_pkl,"selected_routes.pkl"),'wb')
        pickle.dump(path, outfile)
        outfile.close()
        outfile = open(os.path.join(outdir_pkl,"final_routes_top_n_order.pkl"),'wb')
        pickle.dump(final_routes_top_n_order_comb, outfile)
        outfile.close()