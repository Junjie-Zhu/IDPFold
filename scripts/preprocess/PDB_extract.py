import os
import pickle

restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

restype_3to1 = {v: k for k, v in restype_1to3.items()}
restype_3to1.update({"SEP": "S",
                     "MSE": "M",
                     "UNK": "X",
                     "HIE": "H",
                     "SEC": "C"
                     })


def read_pdb(filename):
    atom_data = []
    current_model = None

    with open(filename, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith("MODEL"):
                current_model = []
            elif line.startswith("ATOM"):
                if current_model is not None:
                    atom_name = line[12:16].strip()
                    if atom_name in ['N', 'CA', 'C']:
                        res_type = line[17:20].strip()
                        chain_id = line[21]
                        res_seq = int(line[22:26])
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        current_model.append({
                            'atom_name': atom_name,
                            'residue_name': res_type,
                            'chain_id': chain_id,
                            'residue_serial': res_seq,
                            'coordinates': [x, y, z]
                        })
            elif line.startswith("ENDMDL"):
                if current_model is not None:
                    atom_data.append(current_model)

    return atom_data


def organize_data(atom_data):
    residue_dict = []
    sequence = ''

    index = 0
    for model in atom_data:
        residue_dict.append({})

        single_chain_id = model[0]['chain_id']
        for atom_info in model:
            if atom_info['chain_id'] == single_chain_id:  # Only save single chain structure
                residue_name = restype_3to1[atom_info['residue_name']]
                residue_serial = atom_info['residue_serial']
                residue_info = '%s%d' % (residue_name, residue_serial)

                atom_name = atom_info['atom_name']
                x, y, z = atom_info['coordinates']

                if residue_info not in residue_dict[index]:
                    residue_dict[index][residue_info] = []

                residue_dict[index][residue_info].append([atom_info['residue_name'], atom_name, x, y, z])

                if index == 0:
                    sequence += residue_name
        index += 1
    return residue_dict, sequence


def check_sequence(residue_dict):
    complete = True

    model = residue_dict[0]
    residue_serial = [int(key[1:]) for key in model.keys()]

    if len(residue_serial) != (residue_serial[-1] - residue_serial[0] + 1):
        complete = False
    elif len(residue_serial) <= 10:
        complete = False

    return complete


def save_as_pdb(residue_dict, output_prefix):
    for i in range(len(residue_dict)):
        output_file = f'{output_prefix}_{i}.pdb'
        with open(output_file, 'w') as f:

            for j, residue in enumerate(residue_dict[i]):
                for k, residue_info in enumerate(residue_dict[i][residue]):
                    # Format: ATOM  i  atom  residue chainID  seqNo  x  y  z  occupancy  tempFactor  element
                    f.write(f"ATOM  {3 * j + k + 1: >5}  {residue_info[1]: <4}  {residue_info[0]: <3}  A   {j: >4}    "
                            f"{residue_info[2]: >8.3f}{residue_info[3]: >8.3f}{residue_info[4]: >8.3f}"
                            "  1.00 20.00           "  # Default occupancy and temperature factor
                            f"{residue_info[1][0]}\n")


def save_as_pkl(residue_dict, output_prefix):
    for i in range(len(residue_dict)):
        output_file = f'{output_prefix}_{i}.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(residue_dict[i], f)


def load_as_pkl(input_prefix):
    input_file = f'{input_prefix}.pkl'
    with open(input_file, 'rb') as f:
        residue_dict = pickle.load(f)

    return residue_dict


def auto_process(input_pdb, output_path, check=True):
    output_prefix = os.path.join(output_path, input_pdb.split('/')[-1].split('.')[0])

    atom_data = read_pdb(input_pdb)
    residue_dict, sequence = organize_data(atom_data)

    if check:
        complete = check_sequence(residue_dict)
        if complete:
            save_as_pkl(residue_dict, output_prefix)
        else:
            return input_pdb
    else:
        save_as_pdb(residue_dict, output_prefix)

    return sequence