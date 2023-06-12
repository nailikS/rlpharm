import subprocess
from datetime import datetime
import xml.etree.ElementTree as ET
import json
import time
import re
import math
import numpy as np

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
    subprocess.run(f"{jvm}{ilib}{cp} --query {querys} --database {actives_db}, {inactives_db} --output {output_file} -C 4", capture_output=True, text=True)
    timings.append("Subprocess Call: " + str(time.time() - start_time))

    # read output file from screening and count hits
    hits = []
    pos = 0
    neg = 0
    with open(file=output_file.replace("\\\\", "\\"), mode='r') as f:
        for line in f:
            if line.endswith('active\n'):
                hits.append(1)
                pos += 1
            if line.endswith('decoy\n'):
                hits.append(0)
                neg += 1

    if verbose == 1: print('\n'.join(timings))

    with open(file=output_file.replace("\\\\", "\\"), mode='r') as f:
        document = f.read()
        scores = extract_scores_from_file(document)

    return hits, scores, pos, neg

def extract_scores_from_file(document):
    pattern = r'> <Score>\n(\d+\.?\d*)'  # Regular expression pattern to match the score
    # Find all matches of the pattern in the document
    matches = re.findall(pattern, document)
    scores = [float(match) for match in matches]  # Extract the numbers and convert to floats
    return scores

def insert_elements(i, j):
    """
    Creates an array for TP and FP elements respectively and inserts the TP elements
    evenly distributed into the FP array, then returns that, plus a np.zeros array of the same length
    :param i: number of TP elements to be inserted into FP array
    :param j: number of FP elements
    :return: array with inserted elements plus array of same length filled with zeros
    """
    i = np.ones(i, dtype=int)
    j = np.zeros(j, dtype=int)   
    n = len(i)  # Length of the array to be inserted
    m = len(j)  # Length of the target array
    if n == 0:  # If the array to be inserted is empty, return the target array as it is
        return j
    interval = math.floor(m / n)  # interval to distribute the indices evenly
    index = 0
    for element in i:
        j = np.insert(j, index, element)  # Insert the element
        index += interval + 1 
    return j.tolist(), np.full(len(j), 0).tolist()

def scoring(hits:list, scores:list, ldbi:int, ldba:int):
        """
        Calculate score
        :param hits: list of hit labels (0=FP or 1:TP)
        :param scores: list of pharmacophore fit scores
        :param ldbi: number of inactive compounds in the database
        :param ldba: number of active compounds in the database
        :return: rocAUC of the hitlist
        """     
        a_hits, a_scores = insert_elements(ldba-sum(hits), ldbi-(len(hits)-sum(hits)))
        sorted_hits = sorted(zip(scores.extend(a_scores), hits.extend(a_hits)), key=lambda x: x[0], reverse=True)
        sorted_true_labels = [label for _, label in sorted_hits]

        # Calculate the true positive count and false positive count
        num_positives = sum(sorted_true_labels)
        num_negatives = len(sorted_true_labels) - num_positives

        # Create arrays to store true positive rates and false positive rates
        tpr = np.zeros(len(sorted_true_labels))
        fpr = np.zeros(len(sorted_true_labels))

        # Iterate through the sorted hits to compute TPR and FPR
        tp_count = 0
        fp_count = 0
        for i, (_, label) in enumerate(sorted_hits):
            if label == 1:
                tp_count += 1
            if label == 0:
                fp_count += 1

            tpr[i] = tp_count / num_positives
            fpr[i] = fp_count / num_negatives  

        # Calculate the ROC AUC using the trapezoidal rule
        roc_auc = np.trapz(tpr, fpr)
        return roc_auc

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


def action_execution(action, featureIds, tree, initial_tree, delta, action_space):
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
        return executor(r, feature, tree, False, delta)
    if action >= H and action < H + HBA:
        d = (action-H) // HBAm #feature number
        r = (action-H) % HBAm #0: tol+, 1: tol-, 2: tol+, 3: tol-
        feature = featureIds[1][d] #feature id
        return executor(r, feature, tree, True, delta)
    if action >= H + HBA:
        d = (action-H-HBA) // HBDm #feature number
        r = (action-H-HBA) % HBDm #0: tol+, 1: tol-, 2: tol+, 3: tol-
        feature = featureIds[2][d] #feature id
        return executor(r, feature, tree, True, delta)

def executor(r, feature, tree, f=False, delta=0.1):
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
                return set_tol(tree, feature, (float(tol_origin) + delta), target="origin")
            case 1:
                return set_tol(tree, feature, (float(tol_origin) - delta), target="origin")
            case 2:
                return set_tol(tree, feature, (float(tol_target) + delta), target="target")
            case 3:
                return set_tol(tree, feature, (float(tol_target) - delta), target="target")
        raise ValueError("No valid action specified")

    else:
        tol = get_tol(tree, feature)
        match r:
            case 0:
                return set_tol(tree, feature, (float(tol) + delta))
            case 1:
                return set_tol(tree, feature, (float(tol) - delta))
        raise ValueError("No valid action specified")

