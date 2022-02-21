# Compare algorithms and Decision Tree classiffier
In this repo you can find 2 projects:
1. compare_algos - is a program to compare Kruskal and Prims algorithms.
2. tree_classifier - is a program to build decision tree and predict some result.

### Istall the requirements. 
Don't forget to install all of the modules you need.
~~~bash
pip install -r requirements.txt
~~~

## Compare algorithms
In this project we used two of the most popular algorithms which finds minimum spanning tree.

### Prim algorithm:
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

Algorithm: 
1. Initialize a tree with a single vertex, chosen arbitrarily from the graph.
2. Grow the tree by one edge: of the edges that connect the tree to vertices not yet in the tree, find the minimum-weight edge, and transfer it to the tree.
3. Repeat step 2 (until all vertices are in the tree).

### Kruskal algorithm
This algorithmis is more suitable for doing in code rather than on the paper.

You can check any graph (within reason) and see that it will take no time...
It took only 5 minutes to run this algorithm for full graph with 1000 nodes. (It is 499500 edges)

Algorithm:
1. Create a forest F (a set of trees), where each vertex in the graph is a separate tree.
2. Create a set S containing all the edges in the graph.
3. While S is nonempty and F is not yet spanning
    1)Remove an edge with minimum weight from S
    2)If the removed edge connects two different trees then add it to the forest F, combining two trees into a single tree. 
At the termination of the algorithm, the forest forms a minimum spanning forest of the graph. If the graph is connected, the forest has a single component and forms a minimum spanning tree.

### Comparing both algorithm:
You can use function "test_algoritms", which run both algorithms 1000 times for each number of nodes, which return statistics of performance.
To use this function:
~~~bash
test_algorithms()
~~~

### Additional module
There is file generate_plot.py file, which generates three plots with effectivness of algorithms.

Here you can compare both algorithms:

![both](https://user-images.githubusercontent.com/91532556/154964228-5eb2cea4-d313-485c-a791-4d7b193b8295.png)

Plot of Kruskal:

![kruskal](https://user-images.githubusercontent.com/91532556/154964447-2afee2fb-c3e2-44a2-b4a8-47352b28865f.png)


Plot of Prim:

![prim](https://user-images.githubusercontent.com/91532556/154964474-68eeba0c-990f-49a0-824c-3cc476cafd4b.png)


## Decision Tree Classiffier

In this program we implemented Machine Learning algorithm.
It is based on Decision Tree.
It takes train data to train itself and completely different test data to test its performance.

To check this program you should just run the module. There is also a function to find average accuracy. To use it, you should uncomment and run this row:
~~~bash
print(find_mean_accuracy())
~~~
As a result you will get this:
~~~bash
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:25<00:00,  3.88it/s]
0.8922222222222221
~~~
This ^ is an average accuracy of our algorithm based on 100 iterations.

You can find this code in Jupyter file, or in separate module tree_classiffier.py
