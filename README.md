# HighPerfParallelComp
Parallel BFS on TACC's Stampede2 Cluster - Completed in December 2017

This is an implementation of a distributed memory parallel BFS via an optimized 
version of a 1-D graph partitioning algorithm. Some of the optimizations introduced 
in this implementation include: 

- Running the search on multiple keys simultaneously for a given graph
- Use of a "pseudo-global" visited array to minimize data transfered between 
processes
- Minimizing the number of Allreduce collective calls made across the global search 
for n keys
