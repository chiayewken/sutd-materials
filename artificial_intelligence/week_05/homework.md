## Usage

pip install requirements.txt

python run main.py

## Experiment Results

A grid search was run over the parameters of:
{"select_unassigned_variable", "order_domain_values", "inference"}
The number of assigns was averaged over 1000 times with different random shuffles of the data

|     | avg_num_assigns | select_unassigned_variable | order_domain_values     | inference        |
| --- | --------------- | -------------------------- | ----------------------- | ---------------- |
| 0   | 13.523          | first_unassigned_variable  | unordered_domain_values | no_inference     |
| 9   | 13.469          | mrv                        | lcv                     | no_inference     |
| 6   | 13.418          | mrv                        | unordered_domain_values | no_inference     |
| 3   | 13.08           | first_unassigned_variable  | lcv                     | no_inference     |
| 1   | 11.778          | first_unassigned_variable  | unordered_domain_values | forward_checking |
| 4   | 11.557          | first_unassigned_variable  | lcv                     | forward_checking |
| 5   | 9.57            | first_unassigned_variable  | lcv                     | mac              |
| 2   | 9.483           | first_unassigned_variable  | unordered_domain_values | mac              |
| 7   | 9               | mrv                        | unordered_domain_values | forward_checking |
| 8   | 9               | mrv                        | unordered_domain_values | mac              |
| 10  | 9               | mrv                        | lcv                     | forward_checking |
| 11  | 9               | mrv                        | lcv                     | mac              |

The default, pure backtracking search method gives the worst results overall
Using minimum remaining values to select unassigned variable gives little benefit
The same is true of using least constraining values to order the domain values
However, using forward checking can give significant improvement. This is
likely due to being able to detect failures and terminate search early
The most effective method uses minimum remaining vlues, least constraining values
and also Arc consistency checking
Arc consistency checking likely helps because it uses propagation to detect errors that
forward checking cannot detect
