## Decision theory
### Program part

Package requires Python >= 3.6

Before all

```sh
$ pip install -r requirements.txt
```

#### SSSP.py
- -i (--input) Input file
- -o (--output) Output file
- -c (--count)  Number of generated graphs
- -f (--from) Required path in generated grapths 'from' 1..(--vertex)
- -t (--to) Required path in generated grapths 'to' 1..(--vertex)
- -g (--generate)  Generation mode on and coefficient of connectivity of a graph
- -v (--vertex)  Vertex count
- -d (--dot)  Generate dot of \[-d\] graph from input file

Generate *5* graphs, with *10* vertex and required path from *1* to *10* and coefficient of connectivity of a graph = *0.8*, output in **dump.json**

```sh
$ python SSSP.py -o dump.json -c 5 -f 1 -t 10 -g 0.8 -v 10
```

Find minimal path in grapth from **dump.json** from *1* to *10* and store result in **data.txt**

```sh
$ python SSSP.py -i dump.json -o data.txt -f 1 -t 10
```

Generate dot of second graph from file **dump.json** and save result in **data.dot**

```sh
$ python SSSP.py -i dump.json -o data.dot -d 2
```

Convert dot to png

```sh
$ dot data.dot -T png > output.png
```

Simple *5* vertex graph, path from *1* to *5*

```
{
    1: [(2, 5), (3, 20), (4, 20), (5, 37)],
    2: [(1, 5), (4, 38), (3, 4), (5, 16)],
    3: [(2, 4), (5, 19), (1, 20), (4, 16)],
    4: [(2, 38), (1, 20), (3, 16)],
    5: [(3, 19), (2, 16), (1, 37)]
}
```

Output
```
- 0.000025272369385 seconds dynamic programming	path=[1, 2, 5], ans=21
- 0.000011682510376 seconds dijkstra		path=[1, 2, 5], ans=21
```

Dot

![N|Solid](https://raw.githubusercontent.com/AlimZanibekov/DecisionTheoryIT/master/examples/dot.png)

### Practical part

#### tasks.py
- -i (--input) Input file
- -o (--output) Output file
- -m (--mode)  1, 2 or 3
- -p (--plot) Path to output chart (png)
- -d (--d) Path to output graph (for mode 3)

Individual work part 1.1

```sh
$ python tasks.py -m 1 -i ./examples/p1.json -o ./output/p1.txt -p ./output/p1.png
```

Individual work part 1.2

```sh
$ python tasks.py -m 2 -i ./examples/p2.json -o ./output/p2.txt -p ./output/p2.png
```

Individual work part 2

```sh
$ python tasks.py -m 3 -i ./examples/p3.json -o ./output/p3.txt -d ./output/p3.dot
```


