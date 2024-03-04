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

                residue_dict[index][residue_info].append([x, y, z])
        index += 1
    return residue_dict


def check_sequence(residue_dict):
    complete = True

    try:
        model = residue_dict[0]
    except IndexError:
        return False
    
    residue_serial = [int(key[1:]) for key in model.keys()]

    if len(residue_serial) != (residue_serial[-1] - residue_serial[0] + 1):
         complete = False

    return complete


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


def save_as_pdb(residue_dict, output_prefix):
    for i in range(len(residue_dict)):
        output_file = f'{output_prefix}_{i}.pdb'

        with open(output_file, 'w') as f:
            atom_serial = 1

            for keys, values in residue_dict[i].items():

                residue_type = restype_1to3[keys[0]]
                residue_serial = keys[1:]

                if len(values) == 3:

                    atom_name = ['N', 'CA', 'C']
                    for i in range(3):
                        atom_record = f"ATOM  {atom_serial:5} {atom_name[i]:4} {residue_type:3} A{residue_serial:>4}    {values[i][0]:8.3f}{values[i][1]:8.3f}{values[i][2]:8.3f}  1.00  0.00           {atom_name[i][0]:2}\n"
                        f.write(atom_record)
                        atom_serial += 1

            f.write('ENDMDL\n')
                

def auto_process(input_pdb, output_path, check=True, save_pdb=True):
    output_prefix = os.path.join(output_path, input_pdb.split('/')[-1].split('.')[0])

    atom_data = read_pdb(input_pdb)
    residue_dict = organize_data(atom_data)

    if check:
        complete = check_sequence(residue_dict)
        if complete and not save_pdb:
            save_as_pkl(residue_dict, output_prefix)
        elif complete and save_pdb:
            save_as_pdb(residue_dict, output_prefix)
        else:
            return input_pdb
    else:
        if not save_pdb:
            save_as_pkl(residue_dict, output_prefix)
        elif save_pdb:
            save_as_pdb(residue_dict, output_prefix)

    return 0
