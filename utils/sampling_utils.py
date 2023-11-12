
def output_pdb(coords, filename):
    coords = coords.squeeze().cpu().numpy()

    with open(filename, 'w') as f:
        # assert len(coords) == len(reference), 'atom number error!'

        a = 0
        for atoms in coords:
            f.write(
                "ATOM  %5d  %2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
                % (a + 1, "CA", "ALA", "A", a + 1, atoms[0], atoms[1], atoms[2], 1, 0)
            )
            a += 1
