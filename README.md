# Compare algorithms and Decision Tree classiffier
In this repo you can find 2 projects:
1. compare_algos - is a program to compare Kruskal and Prims algorithms.
2. tree_classifier - is a program to build decision tree and predict some result.

### Don't forget to install all the modules you need for proper work of both programs!
~~~bash
pip install -r requirements.txt
~~~

## Compare algorithms
In this project we used two of the most popular algorithms which finds minimum spanning tree.

#### Prim algorithm:
If you tried to use this algorithm on a paper, you now that it is very easy to understand, but code is very ineffective on the big graph. You can check by command 
~~~bash
prim_algorithm(get_info(500, 1))
~~~
And you will see how long it works.

Nevertheless, you can check it on small graphs (less or equal then 200 points)
~~~bash
prim_algorithm(get_info(20, 1))
~~~
As a result you will see something like that:
~~~bash
([(0, 19, 0), (3, 19, 0), (3, 6, 0), (12, 19, 0), (16, 19, 0), (5, 16, 0), (5, 8, 0), (8, 17, 0), (5, 14, 0), (13, 14, 0), (10, 11, 0), (11, 18, 0), (7, 18, 0), (7, 9, 0), (2, 6, 1), (4, 5, 1), (1, 3, 1), (1, 11, 1), (15, 16, 1)], 5)
~~~
You can see tuple here: (tree which is sorted by weight *list of edges*, and weight)

This algorithm works in 3 steps: 
1. Initialize a tree with a single vertex, chosen arbitrarily from the graph.
2. Grow the tree by one edge: of the edges that connect the tree to vertices not yet in the tree, find the minimum-weight edge, and transfer it to the tree.
3. Repeat step 2 (until all vertices are in the tree).

#### Kruskal algorithm
This algorithmis is more suitable for doing in code rather than on the paper.

You can check any graph (within reason) and see that it will take no time...
It took only 5 minutes to run this algorithm for full graph with 1000 nodes. (It is 499500 edges)

№№№
## Decision Tree Classiffier
