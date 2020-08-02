from functools import partial
from random import shuffle
from typing import Dict, List, Any, Tuple

import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from csp import *

"""
Consider a 3×3 array, each of whose entries can be either 1, 2 or 3.
We want tofind an assignment to each of the entries so that the entries in each row, 
in eachcolumn and in one of the diagonals are different.  Note that this will also ensurethat 
these row, colum and diagonals add up to 6 (1 + 2 + 3).  But, note that the”adding to 6” 
constraint is not a ”binary constraint”, that is, it involves morethan  three  variables.   
However,  the  constraint  that  each  pair  of  values  in therow, column or diagonal be 
different is a ”binary constraint

vars={V1,V2...V9}
domains={1,2,3}
constraints:VX!=VY for all X,Y?
”"""


def get_neigbours():
    neighbours = {
        1: [2, 3, 4, 7, 5, 9],
        2: [1, 3, 5, 8],
        3: [1, 2, 6, 9],
        4: [1, 7, 5, 6],
        5: [4, 6, 2, 8, 1, 9],
        6: [4, 5, 3, 9],
        7: [1, 4, 8, 9],
        8: [7, 9, 2, 5],
        9: [7, 8, 3, 6, 1, 5],
    }
    for x, neighs in neighbours.items():
        for n in neighs:
            assert x in neighbours[n]

    def do_format(num: int) -> str:
        return f"V{num}"

    return {
        do_format(k): [do_format(item) for item in v] for k, v in neighbours.items()
    }


def shuffle_dict(d: Dict[str, list]) -> Dict[str, list]:
    new = {}
    keys = list(d.keys())
    shuffle(keys)
    for k in keys:
        lst = list(d[k])
        shuffle(lst)
        new[k] = lst
        assert set(new[k]) == set(d[k])
    assert set(d.keys()) == set(new.keys())
    return new


def solve_semi_magic(algorithm=backtracking_search, verbose=True, **args):
    """ From CSP class in csp.py
        vars        A list of variables; each is atomic (e.g. int or string).
        domains     A dict of {var:[possible_value, ...]} entries.
        neighbors   A dict of {var:[var,...]} that for each variable lists
                    the other variables that participate in constraints.
        constraints A function f(A, a, B, b) that returns true if neighbors
                    A, B satisfy the constraint when they have values A=a, B=b
                    """
    # Use the variable names in the figure
    csp_vars = ["V%d" % d for d in range(1, 10)]

    #########################################
    # Fill in these definitions

    csp_domains = None
    csp_neighbors = None

    csp_domains = {var: [1, 2, 3] for var in csp_vars}
    csp_neighbors = get_neigbours()
    assert set(csp_vars) == set(csp_neighbors.keys())

    shuffle(csp_vars)
    shuffle_dict(csp_domains)
    shuffle_dict(csp_neighbors)

    def csp_constraints(A, a, B, b):
        return a != b

    #########################################

    # define the CSP instance
    csp = CSP(csp_vars, csp_domains, csp_neighbors, csp_constraints)

    # run the specified algorithm to get an answer (or None)
    ans = algorithm(csp, **args)

    if verbose:
        print("number of assignments", csp.nassigns)
        assign = csp.infer_assignment()
        if assign:
            for x in sorted(assign.items()):
                print(x)
    return csp


def benchmark_algorithm(algo=backtracking_search, repeats=100):
    all_assigns = []
    for _ in range(repeats):
        csp = solve_semi_magic(algo, verbose=False)
        all_assigns.append(csp.nassigns)
    avg_num_assigns = sum(all_assigns) / len(all_assigns)
    return dict(algo=algo, avg_num_assigns=avg_num_assigns)


KeyValuePair = Tuple[str, Any]
Kwargs = Dict[str, Any]
GridRow = Dict[str, List[Any]]


from copy import deepcopy


def enumerate_grid(grid: Dict[str, list]) -> List[dict]:
    dicts: List[dict] = [{}]
    for key, values in grid.items():
        temp = []
        for d in dicts:
            for val in values:
                d = deepcopy(d)
                d[key] = val
                temp.append(d)
        dicts = temp
    return dicts


def configure_backtracking_search():
    config = dict(
        select_unassigned_variable=[first_unassigned_variable, mrv],
        order_domain_values=[unordered_domain_values, lcv],
        inference=[no_inference, forward_checking, mac],
    )
    return enumerate_grid(config)


def format_results(r: Dict[str, Any]) -> Dict[str, str]:
    new = {}
    for k, v in r.items():
        try:
            v = v.__name__
        except Exception as e:
            pass
        new[k] = v
    return new


def do_tabulate(df: pd.DataFrame, show_index=True):
    return tabulate(df, tablefmt="github", headers="keys", showindex=show_index)


def main():
    num_repeats = 1000
    print(locals())

    data = []
    for kwargs in tqdm(configure_backtracking_search()):
        algo = partial(backtracking_search, **kwargs)
        results = benchmark_algorithm(algo, num_repeats)
        results.update(kwargs)
        results = format_results(results)
        data.append(results)

    df = pd.DataFrame(data)
    df = df.drop(columns=["algo"])
    df = df.sort_values(by="avg_num_assigns", ascending=False)
    print(do_tabulate(df))


if __name__ == "__main__":
    print(len(configure_backtracking_search()))
    main()


"""
## Experiment Results
A grid search was run over the parameters of:
{"select_unassigned_variable", "order_domain_values", "inference"}
The number of assigns was averaged over 1000 times with different random shuffles of the data 

|    |   avg_num_assigns | select_unassigned_variable   | order_domain_values     | inference        |
|----|-------------------|------------------------------|-------------------------|------------------|
|  0 |            13.523 | first_unassigned_variable    | unordered_domain_values | no_inference     |
|  9 |            13.469 | mrv                          | lcv                     | no_inference     |
|  6 |            13.418 | mrv                          | unordered_domain_values | no_inference     |
|  3 |            13.08  | first_unassigned_variable    | lcv                     | no_inference     |
|  1 |            11.778 | first_unassigned_variable    | unordered_domain_values | forward_checking |
|  4 |            11.557 | first_unassigned_variable    | lcv                     | forward_checking |
|  5 |             9.57  | first_unassigned_variable    | lcv                     | mac              |
|  2 |             9.483 | first_unassigned_variable    | unordered_domain_values | mac              |
|  7 |             9     | mrv                          | unordered_domain_values | forward_checking |
|  8 |             9     | mrv                          | unordered_domain_values | mac              |
| 10 |             9     | mrv                          | lcv                     | forward_checking |
| 11 |             9     | mrv                          | lcv                     | mac              |

The default, pure backtracking search method gives the worst results overall
Using minimum remaining values to select unassigned variable gives little benefit
The same is true of using least constraining values to order the domain values
However, using forward checking can give significant improvement. This is 
    likely due to being able to detect failures and terminate search early
The most effective method uses minimum remaining vlues, least constraining values
    and also Arc consistency checking
Arc consistency checking likely helps because it uses propagation to detect errors that 
    forward checking cannot detect
"""
