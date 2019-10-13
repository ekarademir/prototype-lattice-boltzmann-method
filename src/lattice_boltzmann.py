from typing import List

import numpy as np


LATTICE_SIZE = 50
#     e1   e2    e3    e4   e5    e6    e7    e8   e0
E = [[1.0, 0.0, -1.0,  0.0, 1.0, -1.0, -1.0,  1.0, 0.0],
     [0.0, 1.0,  0.0, -1.0, 1.0,  1.0, -1.0, -1.0, 0.0]]
DX = 1.0
DT = 1.0
C = DX / DT


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


def stream(ns: List[np.ndarray]):
    r = []
    for i, n in enumerate(ns):
        # row shift means y direction. Also y is inverted
        shift_row, shift_col = -int(E[1][i]), int(E[0][i])
        ncol, nrow = n.shape
        # shift values according to unit vectors
        n = np.roll(n, (shift_row, shift_col), axis=(0, 1))
        # fill in zeros at empty places otherwise roll
        # uses continuous BC
        if shift_col == -1:
            n[:, -1] = 0
        elif shift_col == 1:
            n[:, 0] = 0
        if shift_row == -1:
            n[-1, :] = 0
        elif shift_row == 1:
            n[0, :] = 0
        r.append(n)
    return r


def test_stream():
    specimen = [np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])] * 9
    expected = [
        np.array([
            [0, 1, 2],
            [0, 4, 5],
            [0, 7, 8],
        ]),
        np.array([
            [4, 5, 6],
            [7, 8, 9],
            [0, 0, 0],
        ]),
        np.array([
            [2, 3, 0],
            [5, 6, 0],
            [8, 9, 0],
        ]),
        np.array([
            [0, 0, 0],
            [1, 2, 3],
            [4, 5, 6],
        ]),
        np.array([
            [0, 4, 5],
            [0, 7, 8],
            [0, 0, 0],
        ]),
        np.array([
            [5, 6, 0],
            [8, 9, 0],
            [0, 0, 0],
        ]),
        np.array([
            [0, 0, 0],
            [2, 3, 0],
            [5, 6, 0],
        ]),
        np.array([
            [0, 0, 0],
            [0, 1, 2],
            [0, 4, 5],
        ]),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]),
    ]
    actual = stream(specimen)
    for i in range(len(specimen)):
        np.testing.assert_array_equal(actual[i], expected[i], err_msg=f'For array at index {i}')


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
    neq = np.ndarray((LATTICE_SIZE, LATTICE_SIZE))


def tests():
    test_stream()


if __name__ == '__main__':
    # main()
    tests()
