# Theory Homework 1

Chia Yew Ken (1002675)

## Problem 1

Missionaries and cannibals, state space, initial state, goal test, actions and path cost

Missionaries and cannibals is a classical formal problem, and is generally stated
as follows. Three missionaries and three cannibals are on one side of the river.
They all need to cross in a boat that only holds two people at once. There must
never be a situation where there is a group of missionaries in one place who are
outnumbered by cannibals.

- State space
  - There are 3 locations: At(SideStart), At(SideEnd), At(Boat)
  - There can are 3 missionaries and 3 cannibals
  - State space is any combination of missionaries and cannibals at the 3 locations
- Initial state: 3 missionaries and 3 cannibals At(SideStart)
- Actions: Transition a missionary or cannibal between At(SideStart) & At(Boat)
  or At(Boat) & At(SideEnd)
- Goal test: 3 missionaries and 3 cannibals At(SideEnd)
- Path cost
  - Infinite if at any location, number of missionaries < cannibals, else 0
  - Infinite if at At(Boat) has more than 2 people, else 0

## Problem 2

Tree Search vs Graph Search, BFS, nodes vs states

Name the main difference between tree search and graph search

- Graph search keeps track of a set of explored nodes, and only adds nodes to the frontier
  if they are not in the explored set. This avoid the problem of redundant paths.

What is the difference between nodes and states in terms of a search problem?

- A node is a data structure which is part of a search tree
  - The search tree comprises, state, parent-node, child-node(s), action, path-cost, depth
- A state is a representation of physical configuration, and does not have the other items listed
  above that the search tree has.

Does the explored set keep track of nodes or states? Why is it so?

- It keeps track of states because each node can either be explored or not.

## Problem 3

BFS and DFS, frontier, queue, explored set, tree search

Following the solution format from Problem 2 BFS and DFS examples

Run BFS as a graph search

- Frontier A, explored A
- Frontier AB, AC, explored ABC
- Frontier AC, ABD, explored ABCD
- Frontier AC, ABDX, explored ABCDX
- Solution ABDX

Run DFS as a graph search

- Frontier A, explored A
- Frontier AB, AC, explored AB
- Frontier ABC, ABD, AC, explored ABC
- Frontier ABCD, ABD, AC, explored ABCD
- Frontier ABCDX, ABD, AC, explored ABCDX
- Solution ABCDX

If BFS is run as a tree search (instead of a graph search), what additional
nodes will be inserted? List down 3 such nodes.

- Extra frontiers: ABC, ACB, ABCD

If DFS is run as a tree search (instead of a graph search), what additional
nodes will be inserted? List down 3 such nodes.

- Extra frontiers: ABC ACBD, ABD


## Problem 4

Run BFS as a graph search

- Frontier A, explored A
- Frontier AB, AC, explored ABC
- Frontier ABD, ACE, ACF, explored ABCDEF
- Frontier ABDX, ACE, ACF, explored ABCDEFX
- Solution ABDX

Run DFS as a graph search

- Frontier A, explored A
- Frontier AB, AC, explored AB
- Frontier ABD, AC, explored ABD
- Frontier ABDX, AC, explored ABDX
- Solution ABDX

## Problem 5

Uniform cost search

Run UCS as a graph search

- Frontier A, explored A
- Frontier AC (3), explored AC
- Frontier AC (3), AB (5), explored ABC
- Frontier ACF (8), AB (5), explored ABCF
- Frontier ACFI (11), AB (5), explored ABCI
- Frontier ACFE (13), ACFI (11), AB (5), explored ABCIE
- Frontier ACFIH (14), ACFE (13), AB (5), explored ABCIEH
- Frontier ACFED (15), ACFIH (14), AB (5), explored ABCIEHD
- Frontier ACFEDX (16), ACFIH (14), AB (5), explored ABCIEHDX
- Solution ACFEDX (16)
