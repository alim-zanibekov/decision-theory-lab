from functools import reduce
from func import *


def get_os(_l_max, _m):
    ios = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51]
    _is = (_l_max - _m) / (_m - 1)
    return _is, _is / ios[_m - 1]


def calculate(matrix, reverses, w, names=None, alternatives=None, dot_out=None, file=None, name='NAME'):
    import sys
    orig_stdout = sys.stdout
    if file:
        sys.stdout = file

    print('Original Matrix')
    print_matrix(matrix)
    print()
    for key, value in reverses.items():
        for _j in range(len(matrix)):
            matrix[_j][key] = value(matrix[_j][key])

    if names is None:
        names = []
        for _i in range(len(matrix[0])):
            names.append('F{}'.format(_i + 1))

    if alternatives is None:
        alternatives = []
        for _i in range(len(matrix[0])):
            alternatives.append('E{}'.format(_i + 1))

    print('Matrix with reverses')
    print_matrix(matrix)
    print()
    c_matrix, my_inf, my_log = clean_matrix(matrix, only_horizontal=True)
    # real_matrix = matrix
    matrix = c_matrix
    print(my_inf, my_log)

    t_matrix = transpose(matrix)
    # for _i, s in enumerate(t_matrix):
    #     t_matrix[_i] = normalize(s)

    matrix = [list(i) for i in [*zip(*t_matrix)]]

    for o in matrix:
        print(o)
    print()
    n = len(matrix)
    m = len(matrix[0])

    for _i in range(m):
        for _j in range(_i + 1, m):
            if isinstance(w[_i][_j], float):
                w[_j][_i] = Fraction(1, Fraction.from_float(w[_i][_j]))
            else:
                w[_j][_i] = Fraction(1, w[_i][_j])

    vector, l_max, _ = eigenvalue_max(w, 0, 2000)

    print_matrix(w)
    print()
    print('Eigen vector: ')
    print_vector(vector)
    print()
    print('Lmax = ', l_max)
    print('IS[0], OS[1]: ', get_os(l_max, m))

    m_vec = [sum(i) for i in w]
    w_vec = normalize(m_vec)

    def calc():
        g_results = []
        g_vectors = []
        for k in range(m):
            res = []
            for i in range(n):
                line = []
                for j in range(n):
                    line.append(matrix[i][k] / matrix[j][k] if matrix[j][k] != 0 else 0)
                res.append(line)
            print('\nMatrix for ' + names[k])
            print_matrix(res)
            print()

            transposed_res = [line for i, line in enumerate([*zip(*res)])]
            # res = [list(i) for i in [*zip(*transposed_res)]]

            sw = []
            for i, line in enumerate(res):
                sw.append(pow(reduce((lambda x, y: x * y), line, 1), 1 / n))

            ss = 0

            for i in range(len(res)):
                ss += sum(transposed_res[i]) * sw[i]

            vec, lambda_max, _ = eigenvalue_max(res, 0, 2000)
            g_vectors.append(normalize(vec))
            print('Eigen vector: ')
            for v in g_vectors[-1]:
                print(v)
            print('Lmax = ', lambda_max)

            print(ss, lambda_max)
            print([i * lambda_max for i in vec])

            print('IS[0], OS[1]: ', get_os(lambda_max, n))

            g_results.append(res)
        return g_vectors, g_results

    vectors, results = calc()

    my_res = [list(i) for i in [*zip(*vectors)]]

    print('Result matrix:')
    print_matrix(my_res)

    # t_my_res = [list(i) for i in [*zip(*my_res)]]
    result = []

    for s in my_res:
        result.append(sum([w_vec[i] * j for i, j in enumerate(s)]))

    print('Result vector:')
    for _v in result:
        print('{:.6}'.format(_v))
    print(sum(result), result)

    max_mai = max(result)
    max_mai_i = result.index(max_mai)

    print('\nAnswer = ', my_inf[0][max_mai_i] + 1)
    t_matrix = transpose(matrix)

    for i, _ in enumerate(t_matrix):
        t_matrix[i] = normalize(t_matrix[i])
    #
    # sum_matrix = sum([sum(i) for i in matrix])
    #
    # for i, _ in enumerate(t_matrix):
    #     for j, _ in enumerate(t_matrix[i]):
    #         t_matrix[i][j] = t_matrix[i][j] / sum_matrix

    matrix = transpose(t_matrix)
    print_matrix(matrix)

    matrix_mul = [[k * w_vec[j] for j, k in enumerate(s)] for s in matrix]

    p_calc = [reduce((lambda x, y: x * y), i, 1) for i in matrix_mul]
    p = max(p_calc)
    p_i = p_calc.index(p)

    print('\nMul', p, p_i)
    print_vector(p_calc)
    print()
    # matrix = matrix_mul
    # t_matrix = transpose(matrix)
    # min_min = min([min(i) for i in matrix])
    # max_max = max([max(i) for i in matrix])
    # risks = [[-max(t_matrix[j]) + matrix[i][j] - min_min / 100 for j in range(m)] for i in range(n)]
    g_calc = [min([(matrix[i][j]) * w_vec[j] for j in range(m)]) for i in range(n)]
    g = max(g_calc)
    gi = g_calc.index(g)
    print('Germejer', g, gi)
    print_vector(g_calc)
    print()

    def get_ran(p1, p2, method=1, lambdas=None, param=None):
        """
        Eвклид - `method` = 1
        Хеммин/Архимед - `method` = 2
        Чебышев - `method` = 3
        """
        sum_sqr = 0
        _n = len(p1)
        if param:
            method = 0
            _n = param
        if method == 1:
            method = 0
            _n = 2
        if method == 2:
            method = 0
            _n = 1
        arr = []
        if lambdas is None:
            lambdas = [1] * n

        for k, (i, j) in enumerate(zip(p1, p2)):
            if method == 0:
                sum_sqr += lambdas[k] * abs(i - j) ** _n
            if method == 3:
                arr.append(lambdas[k] * abs(i - j))
        if method == 0:
            return sum_sqr ** (1 / _n)
        if method == 3:
            return max(arr)

    a = [max(i) for i in transpose(matrix)]
    print('Great point:')
    print_vector(a)
    w = [get_ran(i, a, method=1, lambdas=w_vec) for i in matrix]
    w_min = min(w)
    wi = w.index(w_min)
    
    print('Euclid', w_min, wi)
    print_vector(w)
    print()

    p = 20
    agg = [sum([(k * w_vec[j]) ** p for j, k in enumerate(s)]) ** (1 / p) for s in matrix]
    agg_max = max(agg)
    aggi = agg.index(agg_max)
    print('Aggregation', agg_max, aggi)
    print_vector(agg)
    print()

    print('\nReal indexes:')
    print('Euclid', my_inf[0][wi] + 1)
    print('Aggregation', my_inf[0][aggi] + 1)
    print('Mul', my_inf[0][p_i] + 1)
    print('Germejer', my_inf[0][gi] + 1)

    sys.stdout = orig_stdout

    if dot_out:
        my_dot = ''
        for i in names:
            my_dot += '\n\t' + name + ' -- ' + i + ';'
            for j in alternatives:
                my_dot += '\n\t' + i + ' -- ' + j + ';'
        my_dot = 'strict graph "" {' + my_dot + '\n}'
        dot_out.write(my_dot)


if __name__ == '__main__':
    def hide():

        # К1 – скорость базового тарифа (МБит/с);
        # К2 – месячный платеж (руб.);
        # К3 – стоимость подключения с учетом дополнительных расходов (тыс. руб.);
        # К4 – негативные отзывы других пользователей об Интернет-провайдере (%);
        # К5 – уровень дополнительных сервисных услуг и возможностей, представляемых
        # абонентам (0-100 баллов, наилучшее значение 100).

        matrix = [
            [100, 650, 15, 26, 70],
            [8, 600, 2, 66, 55],
            [80, 550, 5, 15, 80],
            [100, 350, 18, 78, 40],
            [4, 300, 2.5, 45, 80],
            [30, 499, 7, 29, 60],
            [10, 300, 2, 64, 95],
            [30, 350, 6, 33, 80],
            [10, 300, 3, 64, 90]
        ]

        t_matrix = transpose(matrix)
        names = ['K1', 'K2', 'K3', 'K4', 'K5']
        alternatives = ['П1', 'П2', 'П3', 'П4', 'П5', 'П6', 'П7', 'П8', 'П9']

        # перевод критериев в обратные значения
        # чем больше занчение критерия тем он лучше
        # например чем меньший процент негативных отзывов лучше -> надо перевести негативные в положительные
        # 100 - % негативных = % положительных

        reverses = {
            1: lambda x: max(t_matrix[1]) * 2 - x,
            2: lambda x: max(t_matrix[2]) * 2 - x,
            3: lambda x: max(t_matrix[3]) * 2 - x,
        }

        w = [
            [1, 0.33, 1, 2, 4],
            [0, 1, 2, 4, 4],
            [0, 0, 1, 2, 2],
            [0, 0, 0, 1, 2],
            [0, 0, 0, 0, 1]
        ]

        calculate(matrix, reverses, w, names=names, alternatives=alternatives, name='Провайдер')
    hide()
