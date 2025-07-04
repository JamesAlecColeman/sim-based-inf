import numpy as np
import utils

def read_alg_mesh(filename):
    """ Loads alg file of form x, y, z, dx, dy, dz, fields, ...
        Note geometry stored as int as geometry assumed to be round numbers """
    with open(filename, 'r') as alg_file:
        # Initialise alg list structure
        first_line = alg_file.readline()
        fields = [f.strip() for f in first_line.split(",")]
        #fields = first_line.split(",")
        alg = [[] for _ in range(len(fields))]

        # Process first line
        for i in range(len(alg)):
            alg[i].append(utils.safe_float(fields[i]))

        # Process the rest of the file
        for line in alg_file:
            fields = [f.strip() for f in line.split(",")]

            for i in range(len(alg)):
                alg[i].append(utils.safe_float(fields[i]))

    alg = [np.array(alg_entry) for alg_entry in alg]

    # Geometry x, y, z, dx, dy, dz stored as int64
    for i in range(6):
        alg[i] = alg[i].astype(np.int64)

    return alg