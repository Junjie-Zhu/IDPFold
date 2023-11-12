
def output_pdb(coords, filename):
    with open(filename, 'w') as f:
        # assert len(coords) == len(reference), 'atom number error!'

        a = 0
        for x, y, z in coords:
            f.write(
                "ATOM  %5d  %2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
                % (a + 1, "CA", "ALA", "A", a + 1, x, y, z, 1, 0)
            )
            a += 1
