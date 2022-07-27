import os
import subprocess
import rdkit
import rdkit.Chem.Draw
from collections import defaultdict
from math import isnan
from rdkit import Chem

# Quick code to plot a route.
# This file is a condensed version of the route drawing code found in the
# `utils` package. This is intended as a temporary solution only, because I'm
# going on holiday tomorrow and don't have the time to improve any of it.
# TODO: come back to this code and make it nice.

def getAllMolecules(rootMolecule):
    allMolecules = []

    def extractMolecules(molecule):
        allMolecules.append(molecule)
        if not molecule.is_terminal:
            for reactant in molecule.reaction.reactants:
                extractMolecules(reactant)

    extractMolecules(rootMolecule)

    return (allMolecules)


def getReactions(rootMolecule):
    allReactions = []

    def extractReactions(molecule):
        if not molecule.is_terminal:
            allReactions.append(molecule.reaction)

            for reactant in molecule.reaction.reactants:
                extractReactions(reactant)

    extractReactions(rootMolecule)
    return allReactions


def defaultDrawingOptions():
    opts = rdkit.Chem.Draw.DrawingOptions()
    opts.noCarbonSymbols = True
    # opts.selectColor = (1, 0, 0)
    opts.wedgeBonds = True
    
    opts.elemDict = defaultdict(lambda: (0, 0, 0))
    opts.dotsPerAngstrom = 20
    opts.bondLineWidth = 1.5
    opts.atomLabelFontFace = 'sans'
    return opts


def plotMolecule(molecule, filename, filetype='png', mol_width = 200, mol_height = 200):
    opts = defaultDrawingOptions()
    
    if isinstance(molecule, str):
        molecule = rdkit.Chem.MolFromSmiles(molecule)
    
    imgPath = '{}.{}'.format(filename, filetype)
    rdkit.Chem.Draw.MolToFile(molecule, imgPath, size=(mol_width,mol_height), options = opts)
    
    return imgPath
    
def plotReaction(reactionSmiles, filename, graphvizExecutable='dot', filetype='svg', mol_width = 200, mol_height = 200,
                 drawRetro = False, drawVertical = False, doCleanup = True):
    
    dot, imagePaths = reaction2dot(reactionSmiles, 
                                   filetype = filetype, 
                                   drawRetro = drawRetro,
                                   drawVertical = drawVertical,
                                   mol_width = mol_width, 
                                   mol_height = mol_height)
    
    dotPath = '{}.gv'.format(filename)
    with open(dotPath, 'w') as dotFile:
        dotFile.write(dot)
        
    outPath = '{}.{}'.format(filename, filetype)
    command = '{} {} -T{} -o {}'.format(graphvizExecutable, dotPath, filetype, outPath)
    subprocess.call(command, shell=True)
    
    if doCleanup:
        os.remove(dotPath)
        # delete temporary images
        for imagePath in imagePaths:
            if os.path.exists(imagePath):
                os.remove(imagePath)
    
    return outPath

                
def plotRoute(targetMolecule, filename, graphvizExecutable='dot', filetype='svg', mol_width = 300, mol_height = 100,
              terminalColor = 'gray25', intermediateColor = 'grey',
              drawRetro = True, drawVertical = True, doCleanup = True):
    dot, imagePaths = tree2dot(targetMolecule,
                               filetype = filetype, 
                               drawRetro = drawRetro,
                               drawVertical = drawVertical,
                               mol_width = mol_width, 
                               mol_height = mol_height,
                               terminalColor = terminalColor,
                               intermediateColor = intermediateColor)
    
    dotPath = '{}.gv'.format(filename)
    with open(dotPath, 'w') as dotFile:
        dotFile.write(dot)
        
    outPath = '{}.{}'.format(filename, filetype)
    command = '{} {} -T{} -o {}'.format(graphvizExecutable, dotPath, filetype, outPath)
    subprocess.call(command, shell=True)

    if doCleanup:
        os.remove(dotPath)
        
        # delete temporary images
        for imagePath in imagePaths:
            if os.path.exists(imagePath):
                os.remove(imagePath)
    
    return outPath
    
    
def reaction2dot(reactionSmiles, filetype='svg', drawMolecules=True, drawRetro=False, drawVertical=False,
                 mol_width = 200, mol_height = 200):
        
    reactants = reactionSmiles.split('>>')[0].split('.')
    products = reactionSmiles.split('>>')[1].split('.')
    
    imagePaths = []

    # header
    dot = 'digraph G {\n'
    if drawVertical:
        dot += '  rankdir=TB\n'
    else:
        dot += '  rankdir=LR\n'
    dot += '\n'
    
    reactantPrefix = 'r'
    productPrefix = 'p'
    
    # define molecules
    for prefix, molecules in [(reactantPrefix, reactants), (productPrefix, products)]:
        labels = []
        for idx, smiles in enumerate(molecules):
            label = '{}{}'.format(prefix, idx)
            labels.append(label)
            if drawMolecules:
                imgPath = plotMolecule(smiles, label, filetype, mol_width, mol_height)
                imagePaths.append(imgPath)
                
                borderColor = 'invis'
                dot += '  {} [shape=box color={} label=\"\" image=\"{}\" tooltip=\"{}\"]\n'.format(label, borderColor, imgPath, smiles)
            else:
                dot += '  {} [shape=box color={} label=\"{}\"]\n'.format(label, borderColor, smiles)

        # draw invisible edges between molecules to ensure placement on the same rank
        if len(labels) > 1:
            for mol1, mol2 in zip(labels[:-1], labels[1:]):
                if drawRetro:
                    mol1, mol2 = mol2, mol1
                    
                dot += '  {} -> {} [color=invis]\n'.format(mol1, mol2)
        
        
    # reaction arrow
    lastReactant = '{}{}'.format(reactantPrefix, len(reactants)-1)
    firstProduct = '{}0'.format(productPrefix)
    if drawRetro:
        source, target = firstProduct, lastReactant
        arrowStyle = '[color=\"black:invis:black\"]'
    else:
        source, target = lastReactant, firstProduct
        arrowStyle = '[color=mediumaquamarine penwidth=2]'
    
    dot += '  {} -> {} {}\n'.format(source, target, arrowStyle)
    
    dot += '}'
    
    return dot, imagePaths
    
    
def tree2dot(rootMolecule, filetype='svg', drawMolecules=True, drawRetro=True, drawVertical=True,
             mol_width = 300, mol_height = 100,
             terminalColor = 'green', intermediateColor = 'grey'):

    allMolecules = getAllMolecules(rootMolecule)
    allReactions = getReactions(rootMolecule)
    
    molecules = {molecule:'c{}'.format(idx) for idx, molecule in enumerate(allMolecules)}
    reactions = {reaction:'r{}'.format(idx) for idx, reaction in enumerate(allReactions)}
    
    # header
    dot = 'digraph G {\n'
    if drawVertical:
        dot += '  rankdir=TB\n'
    else:
        dot += '  rankdir=LR\n'
    dot += '\n'
    
    imagePaths = []
    
    # define molecules
    for molecule, label in molecules.items():
        if drawMolecules:
            imgPath = plotMolecule(molecule.smiles, label, filetype, mol_width, mol_height)
            imagePaths.append(imgPath)
            
            #borderColor = terminalColor if molecule.is_terminal else (intermediateColor if molecule != rootMolecule else 'invis')
            borderColor = "gray25" if molecule.is_terminal else ("gray25" if molecule != rootMolecule else "#00293c")
            dot += '  {} [shape=box color=\"{}\" label=\"\" image=\"{}\" tooltip=\"{}\" penwidth=\"2\"]\n'.format(label, borderColor, imgPath, molecule.smiles)
        else:
            dot += '  {} [shape=box color={} label=\"{}\"]\n'.format(label, borderColor, molecule.smiles)
    
    # define reactions
    for reaction, label in reactions.items():
        reactionLabel = label
        try:
            if not isnan(reaction.reaxys_id) and reaction.reaxys_id > 0:
                reactionLabel = "{}\\nReaxys ID: {}".format(reactionLabel, reaction.reaxys_id)
                
                if not isnan(reaction.is_patent) and reaction.is_patent > 0:
                    reactionLabel += "\\n(patent)"
        except AttributeError:
            # fail silently in legacy cases where the annotation isn't available
            pass
            
        dot += '  {} [shape=oval label=\"{}\" fontcolor=\"grey25\" color=\"mediumaquamarine\" penwidth=2]\n'.format(label, reactionLabel)
    
    dot += '\n'
    
    # link chemicals to reactions and vice versa
    for reaction, reactionLabel in reactions.items():
        productLabel = molecules[reaction.product]

        src, dest = (productLabel, reactionLabel) if drawRetro else (reactionLabel, productLabel)
        if drawRetro:
            arrowStyle = '[color=\"black:invis:black\"]'
        else:
            arrowStyle = '[color=mediumaquamarine penwidth=2]'
        
        dot += '  {} -> {} {}\n'.format(src, dest, arrowStyle)

        for reactant in reaction.reactants:
            reactantLabel = molecules[reactant]

            src, dest = (reactionLabel, reactantLabel) if drawRetro else (reactantLabel, reactionLabel)
            #lineColor = terminalColor if reactant.is_terminal else intermediateColor
            lineColor = "mediumaquamarine"
            dot += '  {} -> {} [color={} arrowhead=\"none\" penwidth=2]\n'.format(src, dest, lineColor)
        
    dot += '}'
    return dot, imagePaths


def getCanonicalSmiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def getCanonicalReactionSmiles(reactionSmiles):
    parts = reactionSmiles.split('>>')

    reactants = parts[0].split('.')
    products = parts[1].split('.')

    reactants = [getCanonicalSmiles(molecule) for molecule in reactants]
    products = [getCanonicalSmiles(molecule) for molecule in products]

    reactants.sort()
    products.sort()

    result = '{}>>{}'.format('.'.join(reactants), '.'.join(products))

    return result


# Taken from the `utils/routes` package
class Molecule(object):
    def __init__(self, node, depth=0):

        self.smiles = getCanonicalSmiles(node['smiles'])
        self.price = node['ppg']
        self.as_reactant = node['as_reactant']
        self.as_product = node['as_product']
        self.depth = depth
        self.is_terminal = len(node['children']) == 0

        if self.is_terminal:
            self.reaction = None
        else:
            self.reaction = Reaction(node['children'][0], self)


class Reaction(object):
    def __init__(self, node, parent):
        self.smiles = getCanonicalReactionSmiles(node['smiles'])

        self.product = parent
        self.plausibility = node['plausibility']
        self.template_score = node['template_score']
        self.num_examples = node['num_examples']
        self.reactants = [Molecule(child, parent.depth + 1) for child in node['children']]

        self.prd_rct_atom_cnt_diff = node.get('prd_rct_atom_cnt_diff', float('nan'))
        self.tf_id = node.get('tf_id', float('nan'))
        self.reaxys_id = node.get('reaxys_id', float('nan'))
        self.is_patent = node.get('is_patent', float('nan'))


def getReactions(rootMolecule):
    allReactions = []

    def extractReactions(molecule):
        if not molecule.is_terminal:
            allReactions.append(molecule.reaction)

            for reactant in molecule.reaction.reactants:
                extractReactions(reactant)

    extractReactions(rootMolecule)
    return allReactions


def getStartingMolecules(rootMolecule):
    allStartingMolecules = []

    def extractStartingMolecules(molecule):
        if molecule.is_terminal:
            allStartingMolecules.append(molecule)

        else:
            for reactant in molecule.reaction.reactants:
                extractStartingMolecules(reactant)

    extractStartingMolecules(rootMolecule)
    return allStartingMolecules


def draw_route(route, outfile):
    tree = Molecule(route)
    plotRoute(tree, outfile, filetype='png', drawRetro=False, drawVertical=False)
