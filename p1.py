from func import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def calculate(my_matrix, file=None, image=None):
    import sys
    orig_stdout = sys.stdout
    if file:
        sys.stdout = file

    print('Matrix:')
    print_matrix(my_matrix)

    print()
    s1, s2, gs, log_i = iteration(my_matrix)
    print('\nIteration method:')
    print('A: ', end='')
    print_vector(s1)
    print('B: ', end='')
    print_vector(s2)
    print('V: ', end='')
    print(gs.numerator / gs.denominator)

    print('\nSimplex method:')
    s1, s2, gs, (log_s, result_s) = simplex(my_matrix, minimize=False)
    print('A: ', end='')
    print_vector(s1)
    print('B: ', end='')
    print_vector(s2)
    print('V: ', end='')
    print(gs.numerator / gs.denominator)

    mat, my_indexes, my_log = clean_matrix(my_matrix)
    print('\nMatrix after clean:')
    for f in mat:
        print(f)

    print('\nMatrix clean log:')
    print(my_log)

    mat = [list(i) for i in [*zip(*mat)]]
    my_indexes = (my_indexes[1], my_indexes[0], my_indexes[3], my_indexes[2])

    fig = plt.figure(figsize=(10, 12))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.grid(True)
    ax2.grid(True)

    print('\nChart method:')
    s1, s2, gs, plot = chart(mat, ax1, ind=my_indexes, prefix='A')
    chart(mat, ax2, ind=my_indexes, prefix='B', reverse=True)

    print('A: ', end='')
    print_vector(s1)
    print('B: ', end='')
    print_vector(s2)
    print('V: ', end='')
    print(gs)
    if image:
        fig.savefig(image)
    else:
        plot.show()
    print('\nIteration log:')
    for o in log_i[0]:
        print_simple(o[0])
        s = o[1]
        print_simple(s[0] + 1)
        for i in s[1]:
            print_simple(i)
        print_simple(s[2] + 1)
        for i in s[3]:
            print_simple(i)

        try:
            print_fraction(s[4])
            print_fraction(s[5])
            print_fraction(s[6])
        except AttributeError:
            print_simple(s[4])
            print_simple(s[5])
            print_simple(s[6])
        print()
    print()
    for o in log_i[1]:
        print(o)
    print('\nSimplex log:')

    for _k, o in enumerate(log_s):
        print('Step ' + str(_k) + ':')
        print_matrix(o)
    print('\nSimplex answer:')
    print_vector(result_s)

    sys.stdout = orig_stdout


if __name__ == '__main__':
    calculate([
        [2, 0, 3, 0, 2],
        [-1, 1, 4, 0, 3],
        [1, 0, 2, -3, 0],
        [-2, -2, 0, -2, 3],
        [1, -1, 0, 0, -4],
        [0, 0, -3, 0, 2],
        [1, 2, 4, 2, 3]
    ])
