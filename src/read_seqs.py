import os
import sys

import hydra
import rootutils
import torch
import esm
from omegaconf import DictConfig

from src.utils.esm_extract import calculate_representation, save_representation

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig,) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    sequence_path = cfg.data.dataset.path_to_seq_embedding
    pdb_path = cfg.data.dataset.path_to_dataset
    input_fasta = cfg.pred_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    to_process_list = []
    with open(input_fasta, 'r') as f:
        lines = f.readlines()

        seq_name, seq = '', ''
        for line in lines:
            if line.startswith('>'):
                seq_name = line[1:].strip()
            else:
                seq = line.strip()
                to_process_list.append((seq_name, seq))

    # set a restype dictionary
    restype_dict = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
                    'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER',
                    'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'}

    # Create virtual pdb files in pdb_path for inference
    for seq_name, seq in to_process_list:
        with open(os.path.join(pdb_path, (seq_name + '.pdb')), 'w') as f:

            for i, aa in enumerate(seq):
                f.write(
                    f'ATOM  {i + 1:>5}  CA  {restype_dict[aa]:>3} A {i + 1:>3}      0.000   0.000   0.000  1.00  0.00           C\n')

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)

    # ESM embeddings
    sequence_labels, sequence_strs, representation = calculate_representation(model, alphabet, to_process_list, device)

    for labels, strs, reps in zip(sequence_labels, sequence_strs, representation):
        save_representation(labels, strs, reps, os.path.join(sequence_path, (labels + '.pkl')))


if __name__ == "__main__":
    main()
