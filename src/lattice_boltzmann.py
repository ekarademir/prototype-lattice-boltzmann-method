import numpy as np


LATTICE_SIZE = 50
E = [[1.0, 0.0, -1.0,  0.0, 1.0, -1.0, -1.0,  1.0],
     [0.0, 1.0,  0.0, -1.0, 1.0,  1.0, -1.0, -1.0]]
C = 1.0


def calculate_rho(ns):
    r = np.ndarray(ns[0].shape)
    for n in ns:
        r += n
    return r


def calculate_u(ns, rho, axis=0):
    r = np.ndarray(ns[0].shape)
    for i, n in enumerate(ns):
        r += C * E[axis][i] * n / rho
    return r


def main():
    rho = np.ndarray((LATTICE_SIZE, LATTICE_SIZE))
    ux = np.ndarray((LATTICE_SIZE, LATTICE_SIZE))
    uy = np.ndarray((LATTICE_SIZE, LATTICE_SIZE))

    # Discrete probablilities for each nine directions
    ns = []
    ns[0] = np.ndarray((LATTICE_SIZE, LATTICE_SIZE))
    ns[1] = np.ndarray((LATTICE_SIZE, LATTICE_SIZE))
    ns[2] = np.ndarray((LATTICE_SIZE, LATTICE_SIZE))
    ns[3] = np.ndarray((LATTICE_SIZE, LATTICE_SIZE))
    ns[4] = np.ndarray((LATTICE_SIZE, LATTICE_SIZE))
    ns[5] = np.ndarray((LATTICE_SIZE, LATTICE_SIZE))
    ns[6] = np.ndarray((LATTICE_SIZE, LATTICE_SIZE))
    ns[7] = np.ndarray((LATTICE_SIZE, LATTICE_SIZE))
    ns[8] = np.ndarray((LATTICE_SIZE, LATTICE_SIZE))


if __name__ == '__main__':
    main()
