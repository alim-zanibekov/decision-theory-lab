from functools import reduce
from matplotlib import pyplot as plt
from func import *


def calculate(d, file=None, image=None):
    import sys
    orig_stdout = sys.stdout
    if file:
        sys.stdout = file
    n = d['n']
    nk = d['nk']
    n_min = d['n_min']
    a = d['a']
    b = d['b']
    r = d['r']
    p = d['p']

    m_e = []
    index = 0
    while index * nk <= n:
        m_e.append(n_min + index * nk)
        index += 1
    ni = index
    f = []
    index = 0
    while index * nk <= n:
        f.append(n_min + index * nk)
        index += 1
    nj = index

    matrix = [[0] * nj for _ in range(ni)]

    q = d['q']
    print_vector(m_e)
    print_vector(f)
    print()
    for i in range(ni):
        for j in range(nj):
            matrix[i][j] = (a - r) * min(m_e[i], f[j]) - b * m_e[i] - p

    print_matrix(matrix)

    t_matrix = [list(i) for i in [*zip(*matrix)]]

    risks = [[max(t_matrix[j]) - matrix[i][j] for j in range(nj)] for i in range(ni)]

    mm_calc = [min(i) for i in matrix]
    mm = max(mm_calc)
    mmi = mm_calc.index(mm) + 1

    print('\nWald (MM)', mm, mmi, sep='\t')

    q_mul_sum = []
    for i in range(ni):
        q_mul_sum.append(sum([matrix[i][j] * q[j] for j in range(nj)]))

    bl = max(q_mul_sum)
    bli = q_mul_sum.index(bl) + 1
    print('Bayes-Laplace (BL)', bl, bli, sep='\t')

    s_calc = [max(i) for i in risks]
    s = min(s_calc)
    si = s_calc.index(s) + 1

    # print_matrix(risks)

    print('Savage (S)', s, si, sep='\t')

    lam = 0.5

    hw_calc = [lam * min(i) + (1 - lam) * max(i) for i in matrix]
    hw = max(hw_calc)
    hwi = hw_calc.index(hw) + 1

    print('Hurwitz (HW)', hw, hwi, sep='\t')

    p_calc = [reduce((lambda x, y: x * y), i, 1) for i in matrix]
    p = max(p_calc)
    p_i = p_calc.index(p) + 1

    print('Mul (P)', p, p_i, sep='\t')

    v = 0.5
    hl_calc = [v * q_mul_sum[i] + (1 - v) * min(matrix[i]) for i in range(ni)]
    hl = max(hl_calc)
    hli = hl_calc.index(hl) + 1
    print('Hodge-Lehmann (HL)', hl, hli, sep='\t')

    max_max = max([max(i) for i in matrix])
    g_calc = [min([(matrix[i][j] - max_max - 1) * q[j] for j in range(nj)]) for i in range(ni)]
    g = max(g_calc)
    gi = g_calc.index(g) + 1

    print('Germejer (G)', g, gi, sep='\t')
    print()

    def f_hl(i):
        return lambda _v: _v * q_mul_sum[i] + (1 - _v) * min(matrix[i])

    def f_hw(i):
        return lambda _l: _l * min(matrix[i]) + (1 - _l) * max(matrix[i])

    def get_t_matrix(func, _n):
        m = []
        for i in range(_n):
            m.append([func(i)(0), func(i)(1)])
        return m

    points_hl, path_hl, indexes_hl = get_path_of_lines(get_t_matrix(f_hl, ni), maximize=True)
    points_hw, path_hw, indexes_hw = get_path_of_lines(get_t_matrix(f_hw, ni), maximize=True)

    print('HW')
    print(path_hw)
    print(points_hw)
    print()
    for point in points_hw:
        print('Point ', point)
        for i in range(ni):
            print(f_hw(i)(point[0]))
        print()

    print()
    print('HL')
    print(path_hl)
    print(points_hl)
    print()
    for point in points_hl:
        print('Point ', point)
        for i in range(ni):
            print(f_hl(i)(point[0]))
        print()

    fig = plt.figure(figsize=(10, 12))
    ax = plt.subplot(211)
    ax.grid(True)
    for i in range(ni):
        plt.plot([0, 1], [f_hl(i)(0), f_hl(i)(1)])

    ax = plt.subplot(212)
    ax.grid(True)
    for i in range(ni):
        plt.plot([0, 1], [f_hw(i)(0), f_hw(i)(1)])

    if image:
        fig.savefig(image)
    else:
        plt.show()
    sys.stdout = orig_stdout


if __name__ == '__main__':
    calculate({
        'n': 15,
        'nk': 3,
        'n_min': 2,
        'a': 800,
        'b': 540,
        'r': 45,
        'p': 125,
        'q': [0.2, 0.05, 0.05, 0.1, 0.5, 0.1]
    })
