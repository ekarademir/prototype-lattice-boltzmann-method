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
        ecol, erow = int(E[0][i]), int(E[1][i])
        ncol, nrow = n.shape
        row_fill = np.zeros((1, ncol))
        col_fill = np.zeros((nrow, 1))
        # `erow` can be -1, 0, or 1
        # row direction should be inverted
        row_shift_slice = slice(erow, nrow) if erow >= 0 else slice(0, nrow + erow)
        col_shift_slice = slice(0, ncol - ecol) if ecol >= 0 else slice(1, ncol)
        n = n[row_shift_slice, col_shift_slice]
        # row direction should be inverted
        if erow == -1:
            n = np.concatenate([row_fill, n], axis=0)
        elif erow == 1:
            n = np.concatenate([n, row_fill], axis=0)
        if ecol == 1:
            n = np.concatenate([col_fill, n], axis=1)
        elif ecol == -1:
            n = np.concatenate([n, col_fill], axis=1)
        r.append(n)
    return r


def test_stream():
    specimen = [np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])] * 5
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
