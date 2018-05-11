def dynamic(graph, first, last):
    f = {}
    for name, node in graph.items():
        f[name] = [None, None]
    f[first] = [0, 0]
    prev = f
    ans = [prev]

    for i in range(1000):
        data = prev.copy()
        for key, node in graph.items():
            if key == first:
                continue
            minimum, min_to = None, None
            for c in node:
                to = c[0]
                s = c[1]
                if prev[to][0] is not None:
                    if minimum is None:
                        minimum = s + prev[to][0]
                        min_to = to
                    elif s + prev[to][0] < minimum:
                        minimum = s + prev[to][0]
                        min_to = to
            data[key] = [minimum, min_to]

        ans.append(data)

        if str(data) == str(prev):
            break
        prev = data
    ans = ans[-1]
    i = last
    result = [last]
    while i != first:
        result.append(ans[i][1])
        i = ans[i][1]
    result.reverse()
    return result, ans[last][0]


def dijkstra(graph, first, last):
    valid = {key: False for key, _ in graph.items()}
    weight = valid.copy()
    weight[first] = 0
    paths = {first: first}

    for _ in graph.items():
        min_weight = False
        key_min_weight = False
        for key_weight, value_weight in weight.items():
            if not valid[key_weight] and (min_weight is False
                                          or (value_weight is not False and value_weight < min_weight)):
                min_weight = value_weight
                key_min_weight = key_weight

        for key, val in graph[key_min_weight]:
            if weight[key] is False or weight[key_min_weight] + val < weight[key]:
                weight[key] = weight[key_min_weight] + val
                paths[key] = key_min_weight
        valid[key_min_weight] = True

    i = last
    result = [last]
    n = 0
    while i != first:
        if i not in paths.keys():
            return None, None

        result.append(paths[i])
        i = paths[i]
        n += 1
        if n > len(graph) + 1:
            return None, None

    result.reverse()
    return result, weight[last]
