import os
import tqdm
import torch
import esm
from esm_extract import calculate_representation, save_representation
from PDB_extract import *

input_path = './pdbs'
output_path = './processed'
sequence_path = './embeddings'

SEQ_FLAG = "train"  # Change this value if you want to set a different flag
SEQ_THERSHOLD = 1000  # Change this value if you want to set a different threshold
BATCH_SIZE = 8  # Set the batch size
CUDA_DEVICE = 'cuda:0'  # Choose which CUDA device to use

if not os.path.isdir(output_path):
    os.mkdir(output_path)

if not os.path.isdir(sequence_path):
    os.mkdir(sequence_path)

# df = pd.read_csv('./NMR_splits_3.csv')
# protein_list = df['name'].tolist()

IDP_list = [f'{i}.pdb' for i in os.listdir('./IDPs')]
# IDP_list.extend(protein_list)

device = CUDA_DEVICE if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)

# Get PDBs
missing_list = []
to_process_list = []
for items in tqdm.tqdm(IDP_list):
    missing = auto_process(os.path.join(input_path, items), output_path)

    if missing.endswith('.pdb'):
        missing_list.append(missing)
    else:
        to_process_list.append((items, missing))

# ESM embeddings
sequence_labels, sequence_strs, representation = calculate_representation(model, alphabet, to_process_list, device)

for labels, strs, reps in zip(sequence_labels,sequence_strs, representation):
    save_representation(labels, strs, reps, os.path.join(output_path, labels.replace('.pdb', '.pkl')))

# Get missing log
with open('missing_residue.log', 'w') as f:
    for items in missing_list:
        f.write(f'{items}\n')
