import subprocess
from datetime import datetime
import xml.etree.ElementTree as ET
import json
import time

def exec_vhts(output_file, querys, actives_db, inactives_db, verbose=0):
    """
    Execute VHTS with the given querys
    :param querys: path to query file
    :param actives_db: path to actives database
    :param inactives_db: path to inactives database
    :return: count of positive and negative hits as separate values tuple(int, int)
    """
    timings = []
    # read the config.json file
    with open(r'C:\Users\kilia\MASTER\rlpharm\src\config.json') as json_file:
        cmd = json.load(json_file)
    # the extractions need to be seperate otherwise the f-string will cry 
    jvm = cmd["java"]
    ilib = cmd["ilib"]
    cp = cmd["cp"]
    # get current time for naming the output file so consistency is ensured
    datestring = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%f")
    output_file = output_file + '-{}.sdf'.format(datestring)  
    # timing of screening process
    start_time = time.time()
    subprocess.run(f"{jvm}{ilib}{cp} --query {querys} --database {actives_db}, {inactives_db} --output {output_file}", capture_output=True, text=True)
    timings.append("Subprocess Call: " + str(time.time() - start_time))

    # read output file from screening and count hits
    poshits = 0
    neghits = 0
    with open(file=output_file.replace("\\\\", "\\"), mode='r') as f:
        for line in f:
            if line.startswith('active'):
                poshits += 1
            if line.startswith('decoy'):
                neghits += 1

    if verbose == 1: print('\n'.join(timings))
    return poshits, neghits

def read_pharmacophores(path):
    """
    Read a Phar file and return the tree
    :param path: path to Phar file
    :return: tree
    """
    tree = ET.parse(path)
    return tree

def set_tol(tree, id, newval, target=None):
    """
    Set tolerance of a feature
    :param tree: tree of Phar file
    :param id: featureId
    :param newval: new tolerance value
    :param target: target or origin, when dealing with HBA and HBD
    :return: updated tree
    """
    newval = str(round(newval, 2))
    elm = tree.find(".//*[@featureId='"+id+"']")
    if (elm.get("name") == "H") or (elm.get("type") == "exclusion"):
        elm.find("./position").set('tolerance', newval)
        return tree
    if target == "target":
        child = elm.find("./target")
        child.set('tolerance', newval)
        return tree
    if target == "origin":
        child = elm.find("./origin")
        child.set('tolerance', newval)
        return tree
    raise ValueError("No valid target specified")

def set_tree(tree, featureIds, newvals):
    """
    Set all elements in tree at once
    :return: tree
    """
    tree = ET.ElementTree()
    root = ET.Element('pharmacophore')
    tree._setroot(root)
    return tree

def set_weight(tree, id, newval):
    """
    Set weight of a feature
    :param tree: tree of Phar file
    :param id: featureId
    :param newval: new weight value
    :return: updated tree    
    """
    newval = str(round(newval, 2))
    elm = tree.find(".//*[@featureId='"+id+"']")
    elm.set("weight", newval)
    return tree

def get_tol(tree, id:str):
    """
    Get tolerance and weight of a feature
    :param tree: tree of Phar file
    :param id: featureId
    :return: tolerance and weight as separate values, when dealing with HBA and HBD, tolerance of target and origin as well as weight are returned
    """
    elm = tree.find(".//*[@featureId='"+str(id)+"']")
    if (elm.get("name") == "H") or (elm.get("type") == "exclusion"):
        child = elm.find("./position")
        return float(child.get("tolerance"))
    else:
        child_target = elm.find("./target")
        child_origin = elm.find("./origin")
        return float(child_target.get('tolerance')), float(child_origin.get('tolerance'))


def action_execution(action, featureIds, tree, initial_tree):
    """
    Execute an action 
    either:
    - add or subtract 0.1 to the tolerance of a feature
    or:
    - add or subtract 0.1 to the weight of a feature
    :param action: action to execute
    :return: Path to modified Phar file
    """
    if tree is None:
        tree = initial_tree
    Hm = 2
    HBAm = 4
    HBDm = 4
    H = len(featureIds[0])*Hm
    HBA = len(featureIds[1])*HBAm
    HBD = len(featureIds[2])*HBDm

    if action < H:
        d = action // Hm #feature number
        r = action % Hm #0: tol+, 1: tol-
        feature = featureIds[0][d] #feature id
        return executor(r, feature, tree)
    if action >= H and action < H + HBA:
        d = (action-H) // HBAm #feature number
        r = (action-H) % HBAm #0: tol+, 1: tol-, 2: tol+, 3: tol-
        feature = featureIds[1][d] #feature id
        return executor(r, feature, tree, True)
    if action >= H + HBA:
        d = (action-H-HBA) // HBDm #feature number
        r = (action-H-HBA) % HBDm #0: tol+, 1: tol-, 2: tol+, 3: tol-
        feature = featureIds[2][d] #feature id
        return executor(r, feature, tree, True)

def executor(r, feature, tree, f=False):
    """
    Outsourcing of execution code
    :param r: action encoding
    :param feature: feature id
    :return: modified tree
    """
    if f:
        tol_target, tol_origin = get_tol(tree, feature)
        match r:
            case 0:
                return set_tol(tree, feature, (float(tol_origin) + 0.1), target="origin")
            case 1:
                return set_tol(tree, feature, (float(tol_origin) - 0.1), target="origin")
            case 2:
                return set_tol(tree, feature, (float(tol_target) + 0.1), target="target")
            case 3:
                return set_tol(tree, feature, (float(tol_target) - 0.1), target="target")
        raise ValueError("No valid action specified")

    else:
        tol = get_tol(tree, feature)
        match r:
            case 0:
                return set_tol(tree, feature, (float(tol) + 0.1))
            case 1:
                return set_tol(tree, feature, (float(tol) - 0.1))
        raise ValueError("No valid action specified")

