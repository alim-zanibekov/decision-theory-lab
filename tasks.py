#!/usr/bin/python3

import sys
import p1
import p2
import p3
import json
from func import *


def main(argv):
    def exit_say(code):
        print('tasks.py \n\n-i <input (required)> \n-o <output> \n-m <mode 1..3 (required)> \n'
              '-p <plot (required in mode 1 and 2)> \n-d <dot (for mode 3 dot output file)>')
        sys.exit(code)

    try:
        opts = []
        for i in range(0, len(argv), 2):
            if i + 1 < len(argv):
                opts.append((argv[i], argv[i + 1]))
            else:
                opts.append((argv[i], None))

        input_file, output_file, mode, plot, dot_file \
            = None, None, None, None, None

        for opt, arg in opts:
            if opt == '-h':
                exit_say(0)
            elif opt in ('-i', '--input'):
                input_file = arg
            elif opt in ('-o', '--output'):
                output_file = arg
            elif opt in ('-m', '--mode'):
                mode = arg
            elif opt in ('-p', '--plot'):
                plot = arg
            elif opt in ('-d', '--dot'):
                dot_file = arg

        if input_file is None or mode is None or mode != '3' and plot is None:
            exit_say(2)

        if mode == '1':
            f = open(input_file, 'r')
            data = json.load(f)
            out = open(output_file, 'w') if output_file else None
            p1.calculate(data, file=out, image=plot)
            if out:
                out.close()

        if mode == '2':
            f = open(input_file, 'r')
            data = json.load(f)
            out = open(output_file, 'w') if output_file else None
            p2.calculate(data, file=out, image=plot)
            if out:
                out.close()

        if mode == '3':
            f = open(input_file, 'r')
            data = json.load(f)
            out = open(output_file, 'w') if output_file else None

            matrix = data['matrix']
            t_matrix = transpose(matrix)
            reverses = {}
            get_lambda = lambda _i: lambda x: max(t_matrix[_i]) * 2 - x
            for i in data["reverses"]:
                reverses[i] = get_lambda(i)

            names = data['names'] if data['names'] else None
            alternatives = data['alternatives'] if data['alternatives'] else None
            name = data['name'] if data['name'] else None

            if dot_file:
                dot_file = open(dot_file, 'w')

            p3.calculate(matrix, reverses, data['w'], file=out, name=name,
                         names=names, alternatives=alternatives, dot_out=dot_file)
            if out:
                out.close()
            if dot_file:
                dot_file.close()

        sys.exit()
    except TypeError:
        exit_say(2)
    except RuntimeError:
        exit_say(2)


if __name__ == '__main__':
    # main(['-i', 'p3.json', '-o', 'out.txt', '-m', '3', '-p', 'out.png', '-d', 'my_graph.dot'])
    main(sys.argv[1:])
