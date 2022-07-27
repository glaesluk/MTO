import os
import sys
import pickle
import rdkit.Chem as Chem
from tools.tree_join import removeRouteIsotopes
from tools.optim_colony import get_intermediates
from tqdm import tqdm

home="C:/Users/GNQBG/OneDrive - Bayer/Personal Data/AI4Synthesis_service_calculate_routes/routes/"
path_ending = "/results/routes/mlp5.1_ff1.3_cutoff0.5_1800s_N500_25cores_d7/node0_job0"
enddir = "C:/Users/GNQBG/Documents/AI4Synthesis/routes/"

all_folder = list(os.listdir(home))
for run in all_folder:
    path = home+run
    print(path)
    if os.path.isdir(path):
        route_dir = path
        targetdir = enddir + run
        print("Loading routes from", route_dir)
        all_files = list(os.listdir(route_dir))
        all_files.sort()
        trees = []
        #Loading all molecules
        pbar = tqdm(desc="Loading and cleaning targets", total=len(all_files))
        for file in all_files:
            if file.endswith('pkl'):
                with open(os.path.join(route_dir, file), 'rb') as fp:
                    routes = pickle.load(fp)
                    if len(routes) > 0:
                        # remove "Isotopes"
                        for idx, route in enumerate(routes):
                            routes[idx] = removeRouteIsotopes(route)
                        trees.append(routes)
            pbar.update(1)
        pbar.close()
        sets = []
        for molecule in trees:
            new_row = []
            for idx, route in enumerate(molecule):
                intermediates = set(get_intermediates(route))
                new_row.append(intermediates)
            sets.append(new_row)
        print("Saving problem to", targetdir)
        if not os.path.exists(targetdir):
                os.makedirs(targetdir)

        outfile = open(os.path.join(targetdir,"trees.pkl"),'wb')
        pickle.dump(trees, outfile)
        outfile.close()
        outfile = open(os.path.join(targetdir,"sets.pkl"),'wb')
        pickle.dump(sets, outfile)
        outfile.close()

        