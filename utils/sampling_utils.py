


def output_pdb(coords, reference, filename):
    with open(filename, 'w') as f:
        assert len(coords) == len(reference), 'atom number error!'

        for i in coords.shape[0]:
            f.write()