import calc
import random


def generate(n, e, v_from, v_to, min_path):
    while True:
        graph = {}
        for i in range(1, n + 1):
            graph[i] = []

        for i in range(1, n + 1):
            for j in range(1, int(e * n)):
                if random.random() < e:
                    to = random.randint(1, n)
                    while to == i:
                        to = random.randint(1, n)
                    flag = True
                    for key, _ in graph[i]:
                        if key == to:
                            flag = False
                            break

                    if flag:
                        for d in graph[to]:
                            if d[0] == i:
                                graph[to].remove(d)
                                break
                        path = random.randint(1, 40)

                        graph[i].append(tuple([to, path]))
                        graph[to].append(tuple([i, path]))

        res, num = calc.dijkstra(graph, v_from, v_to)
        if res is not None and len(res) >= min_path:
            return graph
