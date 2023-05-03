import subprocess
from rdkit import Chem
import time
from joblib import Parallel, delayed
from datetime import datetime


ACTIVES = r''
INACTIVES = r''
#QUERY = r'C:\Users\kilia\MASTER\data\queries.list'
QUERY = r'C:\Users\kilia\MASTER\data\5AK5_V2Z_pharmakophore.pmz'
OUTPUT = r'C:\Users\kilia\MASTER\data\hitlists\hitlist'


def exec_vhts(output_file, querys, actives_db, inactives_db):
    """
    Execute VHTS with the given querys
    :param querys: path to query file
    :param actives_db: path to actives database
    :param inactives_db: path to inactives database
    :return: path to hitlist
    """
    datestring = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%f")
    output_file = output_file[:-4] + '-{}.sdf'.format(datestring)  
    subprocess.call([r'C:\Users\kilia\MASTER\data\screen.bat', querys, output_file], start_new_session=True)
    
    poshits = 0
    neghits = 0

    # TODO: read lines of hitlist with regex to evaluate hitcounts
    
    return poshits, neghits


## TESTS
t = time.time()
results = Parallel(n_jobs=3)(delayed(exec_vhts)() for _ in range(5))
print(time.time() - t)
print(results)