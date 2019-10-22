from typing import List

import numpy as np


LATTICE_SIZE = 50
#     e0   e1   e2    e3    e4   e5    e6    e7    e8  
E = [[0.0, 1.0, 0.0, -1.0,  0.0, 1.0, -1.0, -1.0,  1.0],
     [0.0, 0.0, 1.0,  0.0, -1.0, 1.0,  1.0, -1.0, -1.0]]
W = [4 / 9] + [1 / 9] * 4 + [1 / 36] * 4
DX = 1.0
DT = 1.0
C = DX / DT


def si(i, ui: List[np.ndarray], axis=0):
    wi = W[i]
    udotu = ui * ui
    edotu = E[axis][i] * ui
    return wi * (
            3.0 * edotu / C
            + 9.0 * edotu**2 / (2.0 * C**2)
            - 3.0 * udotu / (2.0 * C**2)
        )


def calculate_rho(ns):
    r = np.zeros(ns[0].shape)
    for n in ns:
        r += n
    return r


def calculate_u(ns, rho: np.ndarray, axis=0):
    r = np.zeros(ns[0].shape)
    for i, n in enumerate(ns):
        r += C * E[axis][i] * n
    return r / rho


def calculate_nieq(i, u):
    pass


def stream(ns: List[np.ndarray]):
    r = []
    ncol, nrow = ns[0].shape
    for i, n in enumerate(ns):
        # row shift means y direction. Also y is inverted
        shift_row, shift_col = -int(E[1][i]), int(E[0][i])
        # shift values according to unit vectors
        n_new = np.roll(n, (shift_row, shift_col), axis=(0, 1))
        # fill in zeros at empty places otherwise roll
        # uses continuous BC
        if shift_col == -1:
            n_new[:, -1] = 0
        elif shift_col == 1:
            n_new[:, 0] = 0
        if shift_row == -1:
            n_new[-1, :] = 0
        elif shift_row == 1:
            n_new[0, :] = 0
        r.append(n_new)
    # mid-grid BC
    # e1 -> e3 at east wall
    r[3][:, -1] += ns[1][:, -1]
    # e2 -> e4 at north wall
    r[4][0, :] += ns[2][0, :]
    # e3 -> e1 at west wall
    r[1][:, 0] += ns[3][:, 0]
    # e4 -> e2 at south wall
    r[2][-1, :] += ns[4][-1, :]
    # e5 -> e7 at east and north wall
    r[7][:, -1] += ns[5][:, -1]
    r[7][0, :-1] += ns[5][0, :-1]
    # e6 -> e8 at west and north wall
    r[8][:, 0] += ns[6][:, 0]
    r[8][0, 1:] += ns[6][0, 1:]
    # e7 -> e5 at west and south wall
    r[5][:, 0] += ns[7][:, 0]
    r[5][-1, 1:] += ns[7][-1, 1:]
    # e8 -> e6 at east and south wall
    r[6][:, -1] += ns[8][:, -1]
    r[6][-1, :-1] += ns[8][-1, :-1]

    return r


def test_si():
    pass


def test_calculate_u():
    specimen = [np.ones((2, 2))] * 9
    expected_ux = np.zeros((2, 2))
    expected_uy = np.zeros((2, 2))
    actual_ux = calculate_u(specimen, 1.0, 0)
    actual_uy = calculate_u(specimen, 1.0, 1)
    np.testing.assert_array_equal(actual_ux, expected_ux, "ux calculation false")
    np.testing.assert_array_equal(actual_uy, expected_uy, "uy calculation false")


def test_calculate_rho():
    specimen = [np.ones((2, 2))] * 9
    expected = np.ones((2, 2)) * 9.0
    actual = calculate_rho(specimen)
    np.testing.assert_array_equal(actual, expected, "rho calculation false")


def test_stream():
    specimen = [np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])] * 9
    expected = [
        np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]),
        np.array([
            [1, 1, 2],
            [4, 4, 5],
            [7, 7, 8],
        ]),
        np.array([
            [4, 5, 6],
            [7, 8, 9],
            [7, 8, 9],
        ]),
        np.array([
            [2, 3, 3],
            [5, 6, 6],
            [8, 9, 9],
        ]),
        np.array([
            [1, 2, 3],
            [1, 2, 3],
            [4, 5, 6],
        ]),
        np.array([
            [1, 4, 5],
            [4, 7, 8],
            [7, 8, 9],
        ]),
        np.array([
            [5, 6, 3],
            [8, 9, 6],
            [7, 8, 9],
        ]),
        np.array([
            [1, 2, 3],
            [2, 3, 6],
            [5, 6, 9],
        ]),
        np.array([
            [1, 2, 3],
            [4, 1, 2],
            [7, 4, 5],
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
    test_calculate_u()
    test_calculate_rho()
    test_si()

if __name__ == '__main__':
    # main()
    tests()
