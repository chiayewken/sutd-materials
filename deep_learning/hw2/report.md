### Task 1 

Decompose formula into a sequence of computations. How to compute a term norm(X(i,:)-T(j,:))^2 in pytorch? How to reshape X, T such that you can use broadcasting to get dij = norm(X(i,:)-T(j,:))^2 in pytorch?
```
def get_norm_square(X, T):
    n, d1 = X.shape
    p, d2 = T.shape
    assert d1 == d2
    X = X.view(n, 1, d1)
    T = T.view(1, p, d2)
    norm = torch.sub(X, T)
    norm = torch.pow(norm, exponent=2)
    norm = torch.sum(norm, dim=-1)
    return norm
```
Timings
* For-loop: 4.5 s
* Numpy broadcasting: 17.1 s
* Pytorch broadcasting: 2.6 s

### Task 2
Visualizations are attached (png images)
