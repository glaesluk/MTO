import graphviz
import matplotlib.pyplot as plt
import os
import re
import shutil
from collections import defaultdict, Counter
from jinja2 import Template
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from tools.tree_join import find_shorter_paths, score_tree, nodeGenerator, connected_components
from tools.tree_join import nodeGenerator
from tools.visualization import draw_route


def plot_routes(all_routes, outfile=None):
    """ Generates the complete graph.

    Parameters
    ----------
    all_routes : list(dict)
        List of all routes in this tree.
    outfile : str
        Path to the final .png file

    Notes
    -----
    It is expected that `all_routes` has been fed through the
    `find_shorter_paths` function first.
    """
    buyables, intermediates, id2id, smiles2id, targets = get_node_relations(all_routes)

    graph = graphviz.Digraph(format='png', engine="dot")
    graph.attr(overlap="false", splines="true")

    graph.node("root", label="Start", fillcolor='beige')
    graph.attr('node', fontname="Arial")
    graph.attr('edge', color="mediumaquamarine", penwidth="2")

    # Declare all individual nodes
    for smiles in smiles2id.keys():
        id = smiles2id[smiles]
        if smiles in targets:
            # Target
            target_idx = targets.index(smiles)
            graph.node(str(id), label='{}'.format(target_idx), peripheries="2",
                       style="filled", fillcolor="steelblue", fontcolor="white",
                       color="#00293c", penwidth="2")
        elif smiles in buyables:
            # Buyable
            graph.node(str(id), label=str(id), style="filled", fillcolor="gray25",
                       color="gray25", fontcolor="white", shape='diamond')
        elif '>>' in smiles:
            # Reaction - we don't draw this one anymore, but we keep track of it.
            # graph.node(str(id), label=str(id), style="filled", fillcolor="gray25",
            #           color="gray25", fontcolor="white", shape='point')
            pass
        else:
            # Molecule
            graph.node(str(id), label=str(id), color="gray25", penwidth="3")
    # Connect all products to the root node
    for smiles in targets:
        graph.edge(str(smiles2id[smiles]), "root")

    # Calculate where the transformations are supposed to go to
    id2smiles = dict([(v, k) for k, v in smiles2id.items()])
    transf2next = dict()
    for source, target in id2id:
        if '>>' in id2smiles[source]:
            transf2next[source] = target

    # Connect all nodes to each other
    for source, target in id2id:
        if target in transf2next:
            graph.edge(str(source), str(transf2next[target]))

    if outfile is not None:
        with open(outfile, "w"):
            graph.render(outfile, format="svg")
    else:
        print(graph.source)


def plot_routes_order(synth_order, outfile):
    """ Generates the graph illustrating the order in which the routes should
    be synthesised.

    Parameters
    ----------
    synth_order : list(set, set)
        Set of targets and the set of new intermediates (since the previous step)
        required to build them.
    outfile : str
        Path to the final .png file
    """
    all_list = []
    interms_list = []
    buyables_list = []
    yvals = []
    mols_acum = 0
    all_acum = 0
    iterm_acum = 0
    buyables_acum = 0
    for mol in synth_order:
        mols_acum += len(mol[0])
        all_acum += len(mol[1])
        buyables_acum += len(mol[2])
        iterm_acum += len(mol[1]) - len(mol[2])
        all_list.append(all_acum)
        buyables_list.append(buyables_acum)
        interms_list.append(iterm_acum)
        yvals.append(mols_acum)
    # Parameters for the graph

    # write csv with the plotted data
    rows = zip(yvals, all_list, buyables_list, interms_list)
    with open(outfile.replace(".png", ".csv"), "w") as f:
        f.write("num_mol,interm_buy,buy,interm\n")
        for rr in rows:
            f.write("{num_mol},{interm_buy},{buy},{interm}\n".format(num_mol=rr[0], interm_buy=rr[1], buy=rr[2], interm=rr[3]))

    # plot number of molecues vs. needed intermediates and buyables
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    ax1.set_title('Buyables and intermediates')
    ax1.set(xlabel='Number of buyables and intermediates', ylabel='Number of targets')
    ax1.set_aspect(aspect='auto', adjustable='datalim')
    ax1.plot(all_list, yvals, 'tab:orange')

    ax2.set_title('Buyables')
    ax2.set(xlabel='Number of buyables', ylabel='Number of targets')
    ax2.set_aspect(aspect='auto', adjustable='datalim')
    ax2.plot(buyables_list, yvals, 'tab:green')

    ax3.set_title('Intermediates')
    ax3.set(xlabel='Number of intermediates', ylabel='Number of targets')
    ax3.set_aspect(aspect='auto', adjustable='datalim')
    ax3.plot(interms_list, yvals, 'tab:blue')

    fig.tight_layout()
    fig.savefig(outfile)


def plot_retro(all_routes, outdir):
    """ Saves all routes as .png in the given directory.
    Each route will be saved to a file called `route_<pos>.png`, where `pos`
    is their index in the `all_routes` list.

    Parameters
    ----------
    all_routes : list(dict)
        List of routes.
    outdir : str
        Path to the directory where the files will be generated.
    """
    for idx, route in enumerate(all_routes):
        draw_route(route, os.path.join(outdir, "route_{}".format(idx)))


def get_node_relations(all_routes):
    """
    Given a list of routes, returns the relations between all nodes.

    Parameters
    ----------
    all_routes : list(dict)
        A list of routes to be related to each other.

    Returns
    -------
    Counter, Counter, set, dict, list
      * A Counter of all the building blocks in the routes and how many times
        do they appear overall.
      * A Counter of all intermediates in the routes and how many times do they
        appear overall.
      * A set of directed edges from one node id to another
      * A dictionary that converts smiles into an internal id
      * A list of all targets in these routes
    """
    # Sets of targets and buyables
    targets = []
    buyables = Counter()
    intermediates = Counter()
    # Converts SMILES into a numeric ID - easier to work with later on
    smiles2id = dict()
    # Lists of directed pairs
    id2id = set()
    # Part 0: Keep track of our targets
    for route in all_routes:
        targets.append(route["smiles"])
    # Part 1: Build a dictionary SMILES -> ID
    for route in all_routes:
        for node in nodeGenerator(route):
            smiles = node["smiles"]
            if smiles not in smiles2id.keys():
                smiles2id[smiles] = len(smiles2id.keys())
            if len(node["children"]) == 0:
                buyables[smiles] += 1
            else:
                intermediates[smiles] += 1
    # Part 2: Build a dictionary of all connections between SMILES
    for route in all_routes:
        for node in nodeGenerator(route):
            target = smiles2id[node["smiles"]]
            for c in node["children"]:
                source = smiles2id[c["smiles"]]
                id2id.add((source, target))
    return buyables, intermediates, id2id, smiles2id, targets


def smiles_to_filename(smiles):
    """ Converts a SMILES into a unique string that can be safely used as a
    filename - no invalid characters whatsoever.

    Parameters
    ----------
    smiles : str
        The smiles that will be converted into a string

    Returns
    -------
    str
        A safe version of the same string

    Notes
    -----
    Code taken from Django's `get_valid_filename`.
    """
    s = str(smiles).lower().strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


def write_multiroute_report(routes, outdir, synth_order):
    """ Generates an HTML report for the given routes in the given directory.
    This function collects all information and then delegates the rendering
    to the `generate_files` function.

    Parameters
    ----------
    routes : list(dict)
        A list of routes for which a report is to be written
    outdir : str
        Directory where all files will be created
    synth_order : list(int, set)
        A list containing the order in which the molecules should be built.

    Preconditions
    -------------
    If your molecules have isotopes, this function will return a lot of garbage
    nodes. Also, it is expected that your routes have been fed through the
    `find_shorter_paths` function first.
    """
    # We need an `img` directory to generate the molecule images
    if not os.path.exists(os.path.join(outdir, "img")):
        os.makedirs(os.path.join(outdir, "img"))

    # We get all information about nodes, edges, intermediates, and so on
    buyables, intermediates, id2id, smiles2id, all_targets = get_node_relations(routes)

    # Collect both individual and general information about the routes
    stats = defaultdict(list)
    route_pages = []
    for idx, route in enumerate(routes):
        target = route["smiles"]
        # Draw the molecule
        filename = "{}.png".format(smiles_to_filename(target))
        full_filename = os.path.join(outdir, "img", filename)
        mol = AllChem.MolFromSmiles(target)
        d2d = Draw.rdMolDraw2D.MolDraw2DSVG(300,300)
        opts = Draw.MolDrawOptions()
        opts.clearBackground=False
        Draw.MolToFile(mol, full_filename, options= opts)
        # Calculate molecule stats
        node_intermediates = 0
        node_buyables = 0
        for node in nodeGenerator(route):
            if len(node["children"]) == 0:
                node_buyables += 1
            elif '>>' not in node['smiles']:
                node_intermediates += 1
        # Save those stats to a dictionary, and add it to the route list
        route_pages.append({"file": "{}.html".format(idx),
                            "title": "Route {}".format(idx),
                            "smiles": target,
                            "node_idx": smiles2id[target],
                            "image": filename,
                            "intermediates": node_intermediates,
                            "buyables": node_buyables})
    # Some general stats
    stats["total_buyables"] = sum(buyables.values())
    stats["total_unique_buyables"] = len(buyables.keys())
    # Remove reactions from the intermediates list
    to_del = list(filter(lambda x: '>>' in x, intermediates.keys()))
    for k in to_del:
        intermediates.pop(k)
    stats["total_interm"] = sum(intermediates.values())
    stats["total_unique_interm"] = len(intermediates.keys())

    # Processes general statistics about buyables and intermediates
    stats["popular_interm"] = []
    stats["popular_buyables"] = []
    counters_to_process = [("popular_buyables", buyables), ("popular_interm", intermediates)]
    for key, counter in counters_to_process:
        for interm in filter(lambda x: x[1] > 1, counter.most_common()):
            smiles = interm[0]
            count = interm[1]
            # Draws a molecule. Note that intermediates are more than one
            # molecule, and therefore need a different function.
            all_mols = list(map(AllChem.MolFromSmiles, smiles.split('>>')))
            filename = "{}.png".format(smiles_to_filename(smiles))
            full_filename = os.path.join(outdir, "img", filename)
            if len(all_mols) == 1:
                Draw.MolToFile(all_mols[0], full_filename, options= opts)
            else:
                img = Draw.MolsToGridImage(all_mols)
                img.save(full_filename)

            # We don't want to show stats for reactions.
            if len(all_mols) == 1:
                stats[key].append({"smiles": smiles,
                                   "node_idx": smiles2id[smiles],
                                   "image": filename,
                                   "count": count})

    # Processes statistics about each individual route
    all_nodes_info = []
    for route in routes:
        steps = []
        for node in nodeGenerator(route):
            smiles = node["smiles"]
            numeric_id = smiles2id[smiles]
            outgoing = list(filter(lambda x: x[0] == numeric_id, id2id))
            # Draws molecules. Same as above.
            all_mols = list(map(AllChem.MolFromSmiles, smiles.split('>>')))
            filename = "{}.png".format(smiles_to_filename(smiles))
            full_filename = os.path.join(outdir, "img", filename)
            if len(all_mols) == 1:
                Draw.MolToFile(all_mols[0], full_filename, options = opts)
            else:
                img = Draw.MolsToGridImage(all_mols)
                img.save(full_filename)

            if len(all_mols) == 1:
                steps.append({"id": numeric_id,
                              "smiles": smiles,
                              "image": filename,
                              "outgoing": len(outgoing)-1})
        node_info = {"target": route["smiles"], "steps": steps}
        all_nodes_info.append(node_info)

    # Processes statistics about the order of the synthesis process
    synthesis_order_dicts = []
    for smiles, interms, buyables in synth_order:
        target_pair = list(map(lambda x: {'file': "{}.png".format(smiles_to_filename(x)),
                                          'smiles': x},
                               smiles))
        synthesis_order_dicts.append({'targets': target_pair,
                                      'interms': list(interms)})

    # Generate .png images for each route
    plot_retro(routes, os.path.join(outdir, "img"))

    # Finally, compile the templates
    generate_files_multitarget(outdir, route_pages, stats, all_nodes_info, synthesis_order_dicts)


def write_job_report(routes, outdir):
    """ Generates an HTML report for the given job in the given directory.
    This function collects all information and then delegates the rendering
    to the `generate_files` function.

    Parameters
    ----------
    routes : list(dict, dict)
        A list of routes for which a report is to be written, stored as a pair
        (route, route_metadata)
    outdir : str
        Directory where all files will be created
    """
    # We need an `img` directory to generate the molecule images
    if not os.path.exists(os.path.join(outdir, "img")):
        os.makedirs(os.path.join(outdir, "img"))

    # We sort the routes first by cluster and then by internal id.
    # This is required by the final report
    routes.sort(key=lambda x: (x[1]['cluster_assignment'], int(x[1]['internal_id'])))

    routes_only = list(map(lambda x: x[0], routes))
    # We get all information about nodes, edges, intermediates, and so on
    buyables, intermediates, id2id, smiles2id, all_targets = get_node_relations(routes_only)

    # Collect both individual and general information about the routes
    # Remember that clusters start from 1
    num_clusters = 1+max(map(lambda x: x[1]['cluster_assignment'], routes))
    route_pages = [[] for _ in range(num_clusters)]

    for idx, route in enumerate(routes):
        target = route[0]["smiles"]
        internal_id = route[1]['internal_id']
        cluster = route[1]['cluster_assignment']
        success_prob = route[1]['success_prob']
        # Draw the molecule
        filename = "{}.png".format(smiles_to_filename(target))
        full_filename = os.path.join(outdir, "img", filename)
        mol = AllChem.MolFromSmiles(target)
        Draw.MolToFile(mol, full_filename, options= opts)
        # Calculate molecule stats
        node_intermediates = 0
        node_buyables = 0
        for node in nodeGenerator(route[0]):
            if len(node["children"]) == 0:
                node_buyables += 1
            elif '>>' not in node['smiles']:
                node_intermediates += 1
        # Save route details to a dictionary, and add it to the route list
        route_pages[cluster].append({"file": "{}.html".format(idx),
                                     "title": "Route {}".format(internal_id),
                                     "smiles": target,
                                     "cluster": cluster,
                                     "success_prob": success_prob,
                                     "node_idx": smiles2id[target],  # Do I need this?
                                     "intermediates": node_intermediates,
                                     "buyables": node_buyables,
                                     "image": filename})

    # Processes statistics about each individual route
    all_nodes_info = []
    for route in routes:
        steps = []
        for node in nodeGenerator(route[0]):
            smiles = node["smiles"]
            numeric_id = smiles2id[smiles]
            # Draws molecules.
            all_mols = list(map(AllChem.MolFromSmiles, smiles.split('>>')))
            filename = "{}.png".format(smiles_to_filename(smiles))
            full_filename = os.path.join(outdir, "img", filename)
            if len(all_mols) == 1:
                Draw.MolToFile(all_mols[0], full_filename, options= opts)
            else:
                img = Draw.MolsToGridImage(all_mols)
                img.save(full_filename)

            # In the multi-target version we don't show transformations, but
            # in the job version we do because it's the only time you get
            # to see the SMILES of a transformation
            steps.append({"id": numeric_id, "smiles": smiles, "image": filename})
        node_info = {"target": route[0]["smiles"], "steps": steps, "internal_id": route[1]['internal_id']}
        all_nodes_info.append(node_info)

    # Generate .png images for each route
    plot_retro(routes_only, os.path.join(outdir, "img"))

    # Finally, compile the templates
    generate_files_job(outdir, route_pages, all_nodes_info)


def generate_files_multitarget(outdir, routes, stats, individual_routes, synthesis_order):
    """ Generates the actual HTML files based on information provided in these
    variables. See the `Notes` section for details on each parameter.
    Parameters
    ----------
    outdir : str
        Directory where all files will be created
    routes : list(dict)
        General information about the routes.
    stats : list(dict)
        General stats about the routes.
    individual_routes : list(dict)
        General stats about every route.
    synthesis_order : list(dict)
        Order in which the routes should be synthesized.

    Notes
    -----
    The JINJA variables are
        * a list named "routes" containing elements with properties
          - file: path to the route file
          - title: title used for this route
        * a dict named "stats" with stats for the general document:
          - total_interm: number of total intermediates
          - total_buyables: number of total buyables
          - total_unique_interm: number of unique intermediates
          - total_unique_buyables: number of unique buyables
          - popular_interm: a list containing dictionaries with the following
            keys:
            - smiles: raw smiles of the intermediate
            - node_idx: integer value for this node in the main graph
            - image: name of the file containing this smiles
            - count: how many times it is used
          - popular_buyables: same as before
        * individual_routes: a list containing dictionaries with the following
          keys:
          - target: the smiles of the route
          - steps: a list of dicts:
            - id: integer in the big graph
            - smiles: smiles of the step
            - image: image of the smiles
            - outgoing: how many other steps use this step
        * synthesis_order: a list containing dictionaries with the following
          keys:
          - targets: a list of pairs containing the smiles of a route and their
            equivalent filename
          - interms: a list of strings, each one the smiles of an intermediate
    """
    template_dir = os.path.join('tools','templates')

    # Create the basic directories
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    try:
        shutil.copytree(os.path.join(template_dir, "css"), os.path.join(outdir, "css"))
    except FileExistsError:
        pass
    try:
        shutil.copytree(os.path.join(template_dir, "vendor"), os.path.join(outdir, "vendor"))
    except FileExistsError:
        pass

    # Open the template for the main page
    with open(os.path.join(template_dir, 'index.html.jinja')) as fp:
        template = Template(fp.read())

    # Generate the pages for each route
    index_file = template.render(routes=routes, stats=stats)
    with open(os.path.join(outdir, "index.html"), "w") as fp:
        print(index_file, file=fp)

    for idx, route in enumerate(individual_routes):
        with open(os.path.join(template_dir, 'route.html.jinja')) as fp:
            template = Template(fp.read())
        route_file = template.render(routes=routes, num_id=idx, steps=route["steps"])
        with open(os.path.join(outdir, "{}.html".format(idx)), "w") as fp:
            print(route_file, file=fp)

    # Open the template for the synthesis order page
    with open(os.path.join(template_dir, 'synthesis-order.html.jinja')) as fp:
        template = Template(fp.read())
    order_file = template.render(routes=routes, synth_order=synthesis_order)
    with open(os.path.join(outdir, "synthesis-order.html"), "w") as fp:
        print(order_file, file=fp)


def generate_files_job(outdir, routes, individual_routes):
    """ Generates the actual HTML files for a single job based on information
    provided in these variables. See the `Notes` section for details on each parameter.

    Parameters
    ----------
    outdir : str
        Directory where all files will be created
    routes : list(dict)
        General information about the routes.
    individual_routes : list(dict)
        General stats about every route.

    Notes
    -----
    The JINJA variables are
        * a list named "routes" containing elements with properties
          - file: path to the route file
          - title: title used for this route
        * a dict named "stats" with stats for the general document:
          - total_interm: number of total intermediates
          - total_buyables: number of total buyables
          - total_unique_interm: number of unique intermediates
          - total_unique_buyables: number of unique buyables
          - popular_interm: a list containing dictionaries with the following
            keys:
            - smiles: raw smiles of the intermediate
            - node_idx: integer value for this node in the main graph
            - image: name of the file containing this smiles
            - count: how many times it is used
          - popular_buyables: same as before
        * individual_routes: a list containing dictionaries with the following
          keys:
          - target: the smiles of the route
          - steps: a list of dicts:
            - id: integer in the big graph
            - smiles: smiles of the step
            - image: image of the smiles
            - outgoing: how many other steps use this step
        * synthesis_order: a list containing dictionaries with the following
          keys:
          - targets: a list of pairs containing the smiles of a route and their
            equivalent filename
          - interms: a list of strings, each one the smiles of an intermediate
    """
    template_dir = "templates"

    # Create the basic directories
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    try:
        shutil.copytree(os.path.join(template_dir, "css"), os.path.join(outdir, "css"))
    except FileExistsError:
        pass
    try:
        shutil.copytree(os.path.join(template_dir, "vendor"), os.path.join(outdir, "vendor"))
    except FileExistsError:
        pass

    # Open the template for the main page
    with open(os.path.join(template_dir, 'index.job.html.jinja')) as fp:
        template = Template(fp.read())

    # Generate the pages for each route
    index_file = template.render(routes=routes)
    with open(os.path.join(outdir, "index.html"), "w") as fp:
        print(index_file, file=fp)

    for idx, route in enumerate(individual_routes):
        with open(os.path.join(template_dir, 'route.nomult.html.jinja')) as fp:
            template = Template(fp.read())
        route_file = template.render(routes=routes, num_id=idx, steps=route['steps'],
                                     title='Route {}'.format(route['internal_id']))
        with open(os.path.join(outdir, "{}.html".format(idx)), "w") as fp:
            print(route_file, file=fp)
