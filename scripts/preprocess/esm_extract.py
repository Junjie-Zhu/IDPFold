import os

import torch
import esm
import numpy as np
import sys
import time
import pickle
import biotite.structure as struc
import biotite.structure.io as strucio

# Set sequence flag
SEQ_FLAG = "train"  # Change this value if you want to set a different flag
SEQ_THERSHOLD = 1000  # Change this value if you want to set a different threshold
BATCH_SIZE = 8  # Set the batch size
CUDA_DEVICE = 'cuda:0'  # Choose which CUDA device to use
START_TIME = time.time()


# Parses fasta file and returns sequence header and sequence in a list
def parse_fasta(filename):
    with open(filename, 'r') as f:
        contents = f.read().split('>')[1:]
        data = []
        for entry in contents:
            lines = entry.split('\n')
            header = lines[0]
            sequence = ''.join(lines[1:])
            sequence = sequence.replace("*", "") if "*" in sequence else sequence
            if len(sequence) <= SEQ_THERSHOLD:
                data.append((header, sequence))
    return data


def parse_single_pdb(filename):
    structure = strucio.load_structure(filename)

    # Get the sequence
    sequence = strucio.pdbx.get_sequence(structure)
    return sequence


# Returns sequence labels, sequences, and their ESM representations
def calculate_representation(model, alphabet, data, device):
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    sequence_labels, sequence_strs, sequence_representations = [], [], []
    total_sequences, num_batches = len(data), len(data) // BATCH_SIZE + (len(data) % BATCH_SIZE != 0)

    for batch in range(num_batches):
        start_idx, end_idx = batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE
        batch_labels, batch_strs, batch_tokens = batch_converter(data[start_idx:end_idx])
        batch_tokens, batch_lens = batch_tokens.to(device), (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            token_representations = model(batch_tokens, repr_layers=[33], return_contacts=True)["representations"][33]

        for i, tokens_len in enumerate(batch_lens):
            # sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0).cpu())
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].cpu())
            sequence_labels.append(batch_labels[i])
            sequence_strs.append(batch_strs[i])

        print(f"Batch {batch + 1}/{num_batches} processed in {time.time() - START_TIME} seconds."
              f" Progress: {100.0 * (batch + 1) / num_batches:.2f}%", end="\r")

        del batch_labels, batch_strs, batch_tokens, token_representations
        torch.cuda.empty_cache()

    return sequence_labels, sequence_strs, sequence_representations


# Saves the ESM representation, sequence labels, and sequences to a pickle file
def save_representation(sequence_labels, sequence_strs, representation, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump({"labels": sequence_labels, "sequences": sequence_strs, "representations": representation}, f)


def main(input_file, output_file):
    start = time.time()

    device = CUDA_DEVICE if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)

    data = parse_fasta(input_file)
    print(f"Read {len(data)} sequences from {input_file}.")

    sequence_labels, sequence_strs, representation = calculate_representation(model, alphabet, data, device)

    save_representation(sequence_labels, sequence_strs, representation, output_file)

    print("Total time taken: ", time.time() - start)


def main_pdb(input_path, output_path):
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    start = time.time()

    device = CUDA_DEVICE if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)

    input_files = os.listdir(input_path)
    print(f"Read {len(input_files)} sequences from {input_path}.")

    data = []
    for filename in input_files:
        if filename.endswith('.pdb'):
            local_data = parse_single_pdb(os.path.join(input_path, filename))
            data.append((filename, local_data))  # (pdb_name.pdb, sequence)

    sequence_labels, sequence_strs, representation = calculate_representation(model, alphabet, data, device)

    for labels, strs, reps in zip(sequence_labels,sequence_strs, representation):
        save_representation(labels, strs, reps, os.path.join(output_path, labels.replace('.pdb', '.pkl')))

    print("Total time taken: ", time.time() - start)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
