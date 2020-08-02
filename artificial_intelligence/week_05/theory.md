# Theory Homework Week 4

Chia Yew Ken (1002675)

## Logistic Problem 1

### Part A

- PackageAt(x): Package at location x in {a,b,c}
- TruckAt(x): Truck at location x in {a,b,c}
- TruckLoaded: Truck is loaded
- TruckEmpty: Truck is not loaded

### Part B

move(x, y)

- pre: TruckAt(x)
- post: add TruckAt(y), delete TruckAt(x)

load(x)

- pre: TruckAt(x), PackageAt(x), TruckEmpty
- post: add TruckLoaded, delete PackageAt(x), delete TruckEmpty

unload(x)

- pre: TruckAt(x), TruckLoaded
- post: add TruckEmpty, PackageAt(x), delete TruckLoaded

Initial: TruckAt(a), PackageAt(c), TruckEmpty

Goal: PackageAt(b)

## Logistic Problem 2

### Part A

move(a ,b), move(b, c), load(c), move(c, b), unload(b)

### Part B

For all the actions, remove all delete effects

- move(x, y), remove (delete TruckAt(x))
- load(x), remove (delete PackageAt(x), delete TruckEmpty)
- unload(x): remove (delete TruckLoaded)

### Part C

- 0: Facts=TruckAt(a), PackageAt(c), TruckEmpty, Actions=move(a,b)
- 1: Facts=Same as #0 + TruckAt(b), Actions=move(b, a), move(b, c)
- 2: Facts=Same as #1 + TruckAt(c), Actions=move(c, b), load(c)
- 3: Facts=Same as #2 + TruckLoaded, Actions=unload(a), unload(b), unload(c)
- 4: Facts=Same as #3 + PackageAt(b), PackageAt(a), Actions=NIL

## Logistic Problem 3

### Part A

Using h+ heuristic, the solution is move(a, b), move(b, c), load(c), unload(b)

### Part B

Goal state is in level 4 and level 0 is TruckEmpty. Thus, hadd=4+0=4

### Part C

Goal state is in level 4 and level 0 is TruckEmpty. Thus, hmax=max(4,0)=4

## Generic Planning 1

- 0: Facts=m, Actions=A
- 1: Facts=Same as #0 + n,o, Actions=B,D
- 2: Facts=Same as #1 + p, Actions=C

### Part A

Given delete-relaxed case, best solution can be A,B or A,D, two actions thus h+ = 2

### Part B

For m,n,o,p, they occur at layer 0,1,1,2 thus hadd = 0+1+1+2 = 4

### Part C

hmax = max(0,1,1,2) = 2

## Generic Planning 2

- 0: Facts=p, Actions=C
- 1: Facts=Same as #0 + m, Actions=A
- 2: Facts=Same as #1 + n,o, Actions=B,D

### Part A

Given delete-relaxed case, best solution can be C,A, two actions thus h+ = 2

### Part B

For m,n,o,p, they occur at layer 1,2,2,0 thus hadd = 1+2+2+0 = 4

### Part C

hmax = max(1,2,2,0) = 2
