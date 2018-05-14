from fractions import Fraction
from math import *
F = Fraction


class Fraction(F):
    def __str__(self):
        print(0)
        return "{}/{]".format(self.numerator, self.denominator)


def iteration(matrix):
    t_matrix = [list(i) for i in [*zip(*matrix)]]

    max_min = max([min(i) for i in matrix])
    min_max = min([max(i) for i in t_matrix])
    print(max_min)
    print(min_max)

    mg = [min(i) for i in matrix]
    mx = max(mg)
    m1i = mg.index(mx)

    m2m = min(matrix[m1i])
    m2i = matrix[m1i].index(m2m)

    table = [[m1i, matrix[m1i], m2i, t_matrix[m2i], m2m, max(t_matrix[m2i]), (m2m + max(t_matrix[m2i])) / 2]]

    for k in range(1, 20):
        ans_table = table[-1]
        m1m = max(ans_table[3])
        m1i = ans_table[3].index(m1m)

        m2t = [ans_table[1][i] + matrix[m1i][i] for i in range(len(ans_table[1]))]

        m2m = min(m2t)
        m2i = m2t.index(m2m)

        m1t = [ans_table[3][i] + t_matrix[m2i][i] for i in range(len(ans_table[3]))]

        m1mx = max(m1t)
        m2mi = min(m2t)

        m2k = Fraction(m2mi, (k + 1))
        m1k = Fraction(m1mx, (k + 1))

        table.append([m1i, m2t, m2i, m1t, m2k, m1k, (m2k + m1k) / 2])

    st1 = [0] * len(matrix)
    st2 = [0] * len(matrix[0])
    k = 1
    func_log = [[], []]
    for i in table:
        func_log[0].append([k, i])
        st1[i[0]] += 1
        st2[i[2]] += 1
        k += 1

    st1 = [Fraction(i, len(table)) for i in st1]
    st2 = [Fraction(i, len(table)) for i in st2]
    game_sum: Fraction = table[-1][-1]

    my_sum = []
    my_sum_sum = 0
    my_sum_b = []
    for i, e1 in enumerate(st1):
        my_sum.append(sum([e2 * matrix[i][j] for j, e2 in enumerate(st2)]))
        my_sum_sum += sum([e1 * e2 * matrix[i][j] for j, e2 in enumerate(st2)])

    print_matrix(matrix)

    for j, e2 in enumerate(st2):
        my_sum_b.append(sum([e1 * matrix[i][j] for i, e1 in enumerate(st1)]))

    func_log[1].append('M(X, Y) = ' + str(my_sum_sum))

    for i, e1 in enumerate(my_sum):
        # e1 *= st1[i]
        error = '' if e1 <= my_sum_sum else ' error'
        func_log[1].append('M(a_{}, Y) = {}{}'.format(i+1, calc_fraction(e1), error))

    for i, e1 in enumerate(my_sum_b):
        # e1 *= st2[i]
        error = '' if e1 >= my_sum_sum else ' error'
        func_log[1].append('M(X, b_{}) = {}{}'.format(i + 1, calc_fraction(e1), error))

    return st1, st2, game_sum, func_log


def simplex(__matrix, minimize=False):
    minimization = minimize
    if minimize:
        __matrix = [list(i) for i in [*zip(*__matrix)]]
    min_min = min([min(i) for i in __matrix])
    plus = 0
    if min_min <= 0:
        plus = 0
    matrix = [[1] + [j + plus for j in i] for i in __matrix]
    matrix.append([0] + [-1] * len(__matrix[0]))
    # matrix = [[-1 * j for j in i] for i in matrix]
    n = len(matrix)
    m = len(matrix[0])
    real_m = m

    if minimize:
        for i in range(n):
            for j in range(m):
                matrix[i][j] *= -1

    table = [[0] * (n + m - 1) for _ in range(n)]

    basis = []
    for i in range(n):
        for j in range(m):
            table[i][j] = matrix[i][j]
        if m + i < len(table[i]):
            table[i][m + i] = 1
            basis.append(m + i)

    def is_end():
        f = True
        for j in range(1, m):
            if table[n - 1][j] < 0:
                f = False
                break
        if minimize:
            f = True
            for i in range(n - 1):
                if table[i][0] < 0:
                    f = False
                    break
        return f

    func_log = [table]
    m = len(table[0])
    while not is_end():
        main_column = 1
        main_row = 0
        for j in range(2, m):
            if table[n - 1][j] < table[n - 1][main_column]:
                main_column = j

        for i in range(n - 1):
            if table[i][main_column] > 0:
                main_row = i
                break

        for i in range(main_row + 1, n - 1):
            if table[i][main_column] > 0 and table[i][0] / table[i][main_column] \
                    < table[main_row][0] / table[main_row][main_column]:
                main_row = i

        if minimize:
            main_column = 1
            main_row = 0
            for i in range(1, n - 1):
                if table[i][0] < table[main_row][0]:
                    main_row = i

            for j in range(1, m):
                if table[main_row][j] < 0:
                    main_column = j
                    break
        basis[main_row] = main_column

        print(main_row, main_column)

        new_table = [i.copy() for i in table]

        for j in range(m):
            new_table[main_row][j] = Fraction(table[main_row][j], table[main_row][main_column])

        for i in range(n):
            if i == main_row:
                continue
            for j in range(m):
                new_table[i][j] = table[i][j] - table[i][main_column] * new_table[main_row][j]

        table = new_table
        func_log.append(table)

        if is_end() and minimize:
            minimize = False

    result = []

    for j in range(real_m - 1):
        try:
            k = basis.index(j + 1)
            result.append(table[k][0])
        except ValueError:
            result.append(0)

    ans_line = table[-1]
    game_sum: Fraction = 1 / ans_line[0]

    if minimization:
        game_sum = -game_sum

    st1 = [i * game_sum for i in result]
    st2 = [ans_line[i] * game_sum for i in range(real_m, m)]

    return st2, st1, game_sum - plus, (func_log, result)


def clean_matrix(matrix, only_horizontal=False):
    res = [i.copy() for i in matrix]
    func_log = []

    def get_dom(m, r=False):
        dom = {}
        for i in range(len(m)):
            for k in range(len(m)):
                if i == k:
                    continue
                f, n = True, 0
                for j in range(len(m[0])):
                    if not r:
                        if m[i][j] > m[k][j]:
                            n += 1
                        elif m[i][j] < m[k][j]:
                            f = False
                            break
                    else:
                        if m[i][j] < m[k][j]:
                            n += 1
                        elif m[i][j] > m[k][j]:
                            f = False
                            break
                if f and n > 0:
                    dom[k] = True
                    func_log.append(('B' + str(i + 1) if r else
                                     'A' + str(i + 1), '->', k + 1))
        return reversed(sorted(dom.keys()))

    ast = [i for i in range(len(matrix))]
    bst = [i for i in range(len(matrix[0]))]
    for i in get_dom(matrix):
        del res[i]
        del ast[i]
    res = [list(i) for i in [*zip(*res)]]
    if not only_horizontal:
        for i in get_dom(res, True):
            del res[i]
            del bst[i]
    rem = []
    for i in range(len(res)):
        for o in range(i + 1, len(res)):
            if res[i] == res[o]:
                rem.append(o)
    for i in rem:
        del res[i]
        del bst[i]
    res = [list(i) for i in [*zip(*res)]]
    return res, (ast, bst, len(matrix), len(matrix[0])), func_log


def get_path_of_lines(t_matrix, maximize=False):
    max_max = max([max(i) for i in t_matrix])
    min_min = min([min(i) for i in t_matrix])

    first = 0
    last = 1

    lines = []
    minimal, min_index = max_max + 1, 0
    if maximize:
        minimal = min_min - 1

    for i in t_matrix:
        a = i[0] - i[1]
        b = last - first
        c = i[1] * first - i[0] * last
        lines.append((a, b, c, ((first, i[0]), (last, i[1]))))
        index = len(lines) - 1

        if not maximize and i[0] < minimal or maximize and i[0] > minimal:
            minimal = i[0]
            min_index = index

        if i[0] == minimal:
            if not maximize and t_matrix[min_index][1] > i[1] or maximize and t_matrix[min_index][1] < i[1]:
                minimal = i[0]
                min_index = index

    points = []
    now_line = lines[min_index]
    indexes = [min_index]
    path = []
    end = False
    tmp_index = 0
    while not end:
        a1, b1, c1, line1 = now_line
        end = True
        tmp_line = None
        tmp_index = 0
        min_y = min_min - 1
        for i, (a2, b2, c2, line2) in enumerate(lines):
            if i in indexes:
                continue

            down = (a1 * b2 - a2 * b1)
            if down == 0:
                continue
            x = -(c1 * b2 - c2 * b1) / (a1 * b2 - a2 * b1)
            y = -(a1 * c2 - a2 * c1) / (a1 * b2 - a2 * b1)

            if line1[0][0] <= x <= line1[1][0]:
                if tmp_line is None or ((not maximize and tmp_line[3][0][0] > x and line2[1][1] < min_y)
                                        or (maximize and tmp_line[3][0][0] > x and line2[1][1] > min_y)):
                    min_y = line2[1][1]
                    if equal_points(line1[0], (x, y)):
                        continue
                    if tmp_line is None:
                        path.append((line1[0], (x, y)))
                    else:
                        path[-1] = (line1[0], (x, y))
                    tmp_line = (a2, b2, c2, ((x, y), line2[1]))
                    tmp_index = i

        if tmp_line:
            end = False

            indexes.append(tmp_index)
            points.append(tmp_line[3][0])
            now_line = tmp_line

    indexes.append(tmp_index)
    path.append(now_line[3])
    i = 0
    while i < len(path):
        if equal_points(path[i][0], path[i][1]):
            del path[i]
        else:
            i += 1

    return points, path, indexes


def chart(matrix, ax, ind=(), maximize=False, prefix='', reverse=False):
    import matplotlib.pyplot as plt

    if reverse:
        matrix = [list(i) for i in [*zip(*matrix)]]

    t_matrix = [list(i) for i in [*zip(*matrix)]]

    if maximize:
        matrix, t_matrix = t_matrix, matrix
    if reverse:
        maximize = True
    max_min = max([min(i) for i in matrix])
    min_max = min([max(i) for i in t_matrix])

    max_max = max([max(i) for i in matrix])
    min_min = min([min(i) for i in matrix])

    if min_max == max_min:
        print('Error')
        return

    first = 0
    last = 1

    ax.plot([first, first], [min_min, max_max + 2], color='#000000')
    ax.plot([last, last], [min_min, max_max + 2], color='#000000')

    for index, i in enumerate(t_matrix):
        ax.plot([first, last], [i[0], i[1]], color='#B50000')
        im = ind[1] if not reverse else ind[0]
        ax.text(last + 0.01, i[1], prefix + str(im[index] + 1), fontsize=12).set_zorder(10)

    points, path, indexes = get_path_of_lines(t_matrix, maximize=maximize)

    for x, y in points:
        s = ax.scatter(x, y, marker='.', s=300, color='#0000FF', alpha=1)
        ax.plot([x, x], [y, min_min], 'r--', color='#cccccc')
        ax.plot([x, first], [y, y], 'r--', color='#cccccc')

        s.set_zorder(1)
        ax.text(first + 0.01, y + 0.05, "{:.2}".format(y), fontsize=10).set_zorder(10)
        ax.text(x + 0.01, min_min, "{:.2}".format(x), fontsize=10).set_zorder(10)

    my_point = path[0][0]
    i1, i2 = 0, 0
    for i, (a, b) in enumerate(path):
        p = [*zip(a, b)]
        ax.plot(p[0], p[1], color='#0000BB')
        if not maximize and a[1] > my_point[1] or maximize and a[1] < my_point[1]:
            my_point = a
            i1 = indexes[i]
            i2 = indexes[i + 1]
        if not maximize and b[1] > my_point[1] or maximize and b[1] < my_point[1]:
            my_point = b
            i1 = indexes[i]
            i2 = indexes[i + 1]

    a = [
        t_matrix[i1].copy(),
        t_matrix[i2].copy()
    ]
    if maximize:
        a = transpose(a)
    else:
        a[0].reverse()
        a[1].reverse()
    f_sum = a[0][0] + a[1][1] - a[0][1] - a[1][0]

    _s1t = [(a[1][1] - a[1][0]) / f_sum, (a[0][0] - a[0][1]) / f_sum]
    _s2t = [(a[1][1] - a[0][1]) / f_sum, (a[0][0] - a[1][0]) / f_sum]
    game_sum = (a[1][1] * a[0][0] - a[0][1] * a[1][0]) / f_sum

    if len(ind) != 0:
        ast, bst, n, m = ind
        s2t = [0] * m
        if len(bst) != 2:
            bst = [bst[i1], bst[i2]]
        s2t[bst[0]] = _s2t[0]
        s2t[bst[1]] = _s2t[1]
        s1t = [0] * n
        s1t[ast[0]] = _s1t[0]
        s1t[ast[1]] = _s1t[1]

        return s1t, s2t, game_sum, plt
    return _s1t, _s2t, game_sum, plt


def equal_points(p1, p2):
    return p2[0] - 0.0001 <= p1[0] <= p2[0] + 0.0001 and p2[1] - 0.0001 <= p1[1] <= p2[1] + 0.0001


def n_sqrt(v): return sqrt(sum([j * j for j in v]))


def normalize_sqrt(v): return [j / n_sqrt(v) for j in v]


def normalize(v): return [j / sum(v) for j in v]


def normalize_max_min(v):
    try:
        return [(j - min(v)) / (max(v) - min(v)) for j in v]
    except ZeroDivisionError:
        return v


def transpose(m): return [list(i) for i in [*zip(*m)]]


def dot(a, b): return sum([a[j] * b[j] for j in range(len(a))])


def mul_matrix(a, b):
    n1 = len(a)
    m1 = len(a[0])
    m2 = len(b[0])
    c = [[0] * m2 for _ in range(n1)]
    for i in range(n1):
        for j in range(m2):
            c[i][j] = sum([a[i][r] * b[r][j] for r in range(m1)])
    return c


def mul_matrix_vector(a, b):
    m1 = len(a[0])
    n = len(b)
    c = [0] * n
    for i in range(n):
        c[i] = sum([a[i][r] * b[r] for r in range(m1)])
    return c


def eigenvalue_max(matrix, error, max_iter):
    n = len(matrix)
    y = [1] * n
    x = y.copy()

    def calc():
        nonlocal y, x
        y = mul_matrix_vector(matrix, x)
        _l = dot(y, x) / dot(x, x)
        no = n_sqrt(y)
        x = [j / no for j in y]
        return _l

    l0 = calc()
    l1 = calc()

    i = 0
    flag = False

    while abs(l1 - l0) / abs(l1) > error:
        l0 = l1
        l1 = calc()
        i += 1
        if i > max_iter:
            flag = True
            break

    return x, l1, flag


def calc_fraction(f):
    if type(f) is Fraction:
        if f.numerator == 0:
            return 0
        elif f.denominator == 1:
            return f.numerator
        else:
            return f.numerator / f.denominator
    else:
        return f


def print_simple(pr):
    if type(pr) is Fraction:
        if type(pr.numerator) is Fraction:
            pr.numerator = pr.numerator.numerator / pr.numerator.denominator
        print("{}/{}".format(pr.numerator, pr.denominator) + '\t', end='')
        return
    if isinstance(pr, float):
        print('{}'.format(pr) + '\t', end='')
    else:
        print('{}'.format(pr) + '\t', end='')


def print_fraction(f: Fraction):
    if f.numerator == 0:
        print_simple(0)
    elif f.denominator == 1:
        print_simple(f.numerator)
    else:
        print_simple(f)


def print_vector(vector):
    for g in vector:
        if g is Fraction:
            print_fraction(g)
        else:
            print_simple(g)
    print()


def print_matrix(matrix):
    for z in matrix:
        print_vector(z)
