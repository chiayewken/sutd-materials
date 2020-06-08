# Homework

Chia Yew Ken (1002675)

## Question 2 Part 4 Answers

- Will this strategy find the "soonest" path?
  - Yes, assuming time > 0 and the deadline increments "one-by-one", eventually it will
    find the path that arrives earliest
- How many calls for a problem with shortest path length=200?
  - Because the deadline is incremented "one-by-one", for a problem with
    shortest path length = x, it will take x - 1 calls. Hence, for 200, it will take 199 calls.

## Code

The code for questions 1 and 2 are in `main.py`

## Usage

- Requirements: Python 3.8.3
- Run `python main.py`
