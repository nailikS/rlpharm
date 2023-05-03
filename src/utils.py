import subprocess
from datetime import datetime
import xml.etree.ElementTree as ET

def exec_vhts(output_file, querys, actives_db, inactives_db):
    """
    Execute VHTS with the given querys
    :param querys: path to query file
    :param actives_db: path to actives database
    :param inactives_db: path to inactives database
    :return: count of positive and negative hits as separate values
    """
    datestring = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%f")
    output_file = output_file + '-{}.sdf'.format(datestring)  
    subprocess.call([r'C:\Users\kilia\MASTER\data\screen.bat', querys, output_file, actives_db, inactives_db], start_new_session=True)
    poshits = 0
    neghits = 0
    with open(file=output_file, mode='r') as f:
        for line in f:
            if line.startswith('active'):
                poshits += 1
            if line.startswith('decoy'):
                neghits += 1
    #print("Hits(pos: " + str(poshits) + ", neg: " + str(neghits) + ")")
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

def get_tol_and_weight(tree, id:str):
    """
    Get tolerance and weight of a feature
    :param tree: tree of Phar file
    :param id: featureId
    :return: tolerance and weight as separate values, when dealing with HBA and HBD, tolerance of target and origin as well as weight are returned
    """
    elm = tree.find(".//*[@featureId='"+str(id)+"']")
    if (elm.get("name") == "H") or (elm.get("type") == "exclusion"):
        child = elm.find("./position")
        return child.get("tolerance"), elm.get("weight")
    else:
        child_target = elm.find("./target")
        child_origin = elm.find("./origin")
        return child_target.get('tolerance'), child_origin.get('tolerance'), elm.get('weight')


def action_execution(action, featureIds, tree):
    """
    Execute an action 
    either:
    - add or subtract 0.1 to the tolerance of a feature
    or:
    - add or subtract 0.1 to the weight of a feature
    :param action: action to execute
    :return: Path to modified Phar file
    """
    H = len(featureIds[0])*4
    HBA = len(featureIds[1])*6
    HBD = len(featureIds[2])*6

    if action < H:
        d = action // 4 #feature number
        r = action % 4 #0: tol+, 1: tol-, 2: weight+, 3: weight-
        feature = featureIds[0][d] #feature id
        return executor(r, feature, tree)
    if action >= H and action < H + HBA:
        d = (action-H) // 6 #feature number
        r = (action-H) % 6 #0: tol+, 1: tol-, 2: tol+, 3: tol-, 4: weight+, 5: weight-
        feature = featureIds[1][d] #feature id
        return executor(r, feature, tree, True)
    if action >= H + HBA and action < H + HBA + HBD:
        d = (action-H-HBA) // 6 #feature number
        r = (action-H-HBA) % 6 #0: tol+, 1: tol-, 2: tol+, 3: tol-, 4: weight+, 5: weight-
        feature = featureIds[2][d] #feature id
        return executor(r, feature, tree, True)
    if action >= H + HBA + HBD:
        d = (action-H-HBA-HBD) // 4
        r = (action-H-HBA-HBD) % 4
        feature = featureIds[3][d]
        return executor(r, feature, tree)
    

def executor(r, feature, tree, f=False):
    """
    Outsourcing of execution code
    :param r: action encoding
    :param feature: feature id
    :return: modified tree
    """
    if f:
        tol_target, tol_origin, weight = get_tol_and_weight(tree, feature)
        match r:
            case 0:
                return set_tol(tree, feature, (float(tol_origin) + 0.15), target="origin")
            case 1:
                return set_tol(tree, feature, (float(tol_origin) - 0.15), target="origin")
            case 2:
                return set_tol(tree, feature, (float(tol_target) + 0.15), target="target")
            case 3:
                return set_tol(tree, feature, (float(tol_target) - 0.15), target="target")
            case 4:
                return set_weight(tree, feature, (float(weight) + 0.15))
            case 5:
                return set_weight(tree, feature, (float(weight) - 0.15))
    else:
        tol, weight = get_tol_and_weight(tree, feature)
        match r:
            case 0:
                return set_tol(tree, feature, (float(tol) + 0.15))
            case 1:
                return set_tol(tree, feature, (float(tol) - 0.15))
            case 2:
                return set_weight(tree, feature, (float(weight) + 0.15))
            case 3:
                return set_weight(tree, feature, (float(weight) - 0.15))