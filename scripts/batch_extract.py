import os
import tqdm
import pandas as pd
from PDB_extract import *

input_path = './pdbs'
output_path = './pkls'

if not os.path.exists(output_path):
    os.mkdir(output_path)

df = pd.read_csv('./NMR_splits_3.csv')
protein_list = df['name'].tolist()

IDP_list = [f'{i}.pdb' for i in os.listdir('./IDPs')]
IDP_list.extend(protein_list)

missing_list = []
for items in tqdm.tqdm(IDP_list):
    missing = auto_process(os.path.join(input_path, items), output_path)

    if missing != 0:
        missing_list.append(missing)

with open('missing_residue.log', 'w') as f:
    for items in missing_list:
        f.write(f'{items}\n')
