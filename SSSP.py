#!/usr/bin/python3

import sys
import calc
import time
import generator
import json


def main(argv):
    def exit_say(code):
        print('SSSP.py \n\n-i <input> \n-o <output (generate, work)> \n-g <generate fill 0.4..1> '
              '\n-c <count (generate)> \n-f <from 1..count (generate, work)> \n-t <to 1..count (generate, work)> '
              '\n-d <dot>')
        sys.exit(code)

    def graph_format(g):
        res = {}
        for item in g.keys():
            res[int(item)] = [(j[0], j[1]) for j in g[item]]
        return res

    try:
        opts = []
        for i in range(0, len(argv), 2):
            if i + 1 < len(argv):
                opts.append((argv[i], argv[i + 1]))
            else:
                opts.append((argv[i], None))
        input_file, output_file, generate, count, f_from, f_to, vertex, visualize \
            = None, None, None, None, None, None, None, None

        for opt, arg in opts:
            if opt == '-h':
                exit_say(0)
            elif opt in ('-d', '--dot'):
                visualize = int(arg)
            elif opt in ('-g', '--generate'):
                generate = float(arg)
            elif opt in ('-c', '--count'):
                count = int(arg)
            elif opt in ('-v', '--vertex'):
                vertex = int(arg)
            elif opt in ('-f', '--from'):
                f_from = int(arg)
            elif opt in ('-t', '--to'):
                f_to = int(arg)
            elif opt in ('-i', '--input'):
                input_file = arg
            elif opt in ('-o', '--output'):
                output_file = arg

        if visualize:
            if input_file:
                import visualizer
                f = open(input_file, 'r')
                data = json.load(f)
                data = [graph_format(item) for item in data]
                f.close()
                visualizer.visualize(data[visualize - 1], output_file)
                sys.exit()
            else:
                raise RuntimeError()

        if generate:
            if generate < 0.4 or not vertex:
                raise RuntimeError()
            arr = []

            for i in range(count):
                graph = generator.generate(vertex, generate, f_from, f_to, int(vertex / 2))
                arr.append(graph)

            f = open(output_file, 'w')
            f.write(json.dumps(arr))
            f.close()
            sys.exit()
        else:
            f = open(input_file, 'r')
            data = json.load(f)
            data = [graph_format(item) for item in data]
            f.close()
            first = f_from
            last = f_to
            f = open(output_file, 'w')
            for graph in data:
                f.write('{\n')
                for key, val in graph.items():
                    f.write('\t' + str(key) + ': ' + str(val) + (',\n' if key < list(graph.keys())[-1] else '\n'))
                f.write('}\n')
                start_time = time.time()
                path, ans = calc.dynamic(graph, first, last)
                my_time = time.time() - start_time
                f.write(('- %.15f seconds dynamic programming\t\t' % my_time) + 'path=' + str(path) + ', ans='
                        + str(ans) + '\n')
                print('--- %.15f seconds ---' % my_time)
                print(path)
                print(ans)

                start_time = time.time()
                path, ans = calc.dijkstra(graph, first, last)
                my_time = time.time() - start_time
                f.write(('- %.15f seconds dijkstra\t\t\t\t' % my_time) + 'path=' + str(path) + ', ans=' + str(ans)
                        + '\n')
                print('--- %.15f seconds ---' % my_time)
                print(path)
                print(ans)
            f.close()
            sys.exit()
    except TypeError:
        exit_say(2)
    except RuntimeError:
        exit_say(2)


if __name__ == '__main__':
    main(sys.argv[1:])
#
# graph = {
#     1: [(2, 3), (3, 4), (4, 2)],
#     2: [(1, 3), (5, 3)],
#     3: [(1, 4), (5, 6)],
#     4: [(1, 2), (5, 3), (6, 1)],
#     5: [(2, 3), (3, 6), (4, 3), (6, 1), (7, 8), (9, 7)],
#     6: [(4, 1), (5, 1), (7, 6), (8, 12)],
#     7: [(5, 8), (6, 6), (10, 14)],
#     8: [(6, 12), (9, 6), (10, 11)],
#     9: [(5, 7), (8, 6), (10, 3)],
#     10: [(7, 14), (8, 11), (9, 3)],
# }
# path, ans = calc.dynamic(graph, 1, 10)
# print(path, ans)
# import visualizer
# visualizer.visualize(graph, 'my_graph.dot')
