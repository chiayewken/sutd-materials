from typing import Tuple, List

import search

WordLadderAction = Tuple[int, str]
FlightState = Tuple[str, int]


class ValidWords:
    def __init__(self, txt_word: str):
        with open(txt_word) as f:
            self.words = set(w.lower().strip() for w in f)

    def check(self, word: str) -> bool:
        return word in self.words


class WordLadderProblem(search.Problem):
    def __init__(self, initial="cold", goal="warm", checker: ValidWords = None):
        assert checker
        self.checker = checker
        super().__init__(initial, goal)

    def actions(self, state: str) -> List[WordLadderAction]:
        options = []
        for i in range(len(state)):
            if state[i] == self.goal[i]:
                continue
            a: WordLadderAction = (i, self.goal[i])
            if self.checker.check(self.result(state, a)):
                options.append(a)
        return options

    def result(self, state: str, action: WordLadderAction) -> str:
        state_split = list(state)
        i, char_new = action
        state_split[i] = char_new
        return "".join(state_split)

    def goal_test(self, state: str) -> bool:
        return state == self.goal

    def path_cost(
        self, cost: int, state1: str, action: WordLadderAction, state2: str
    ) -> int:
        penalty = int(1e9)
        if not self.checker.check(state2):
            return penalty
        else:
            return cost + 1

    def value(self, state: str) -> int:
        score = 0
        for i in range(len(state)):
            if state[i] == self.goal[i]:
                score += 1
        return score


class Flight:
    def __init__(self, start_city: str, start_time: int, end_city: str, end_time: int):
        self.end_time = end_time
        self.end_city = end_city
        self.start_time = start_time
        self.start_city = start_city

    def __str__(self):
        return f"{self.start_city},{self.start_time}->{self.end_city},{self.end_time}"

    def __repr__(self):
        return str(self)

    def matches(self, a: FlightState) -> bool:
        city, time = a
        return city == self.start_city and time <= self.start_time


class Places:
    rome = "Rome"
    paris = "Paris"
    madrid = "Madrid"
    istanbul = "Istanbul"
    london = "London"
    oslo = "Oslo"
    rabat = "Rabat"
    constantinople = "Constantinople"


class FlightProblem(search.Problem):
    def __init__(self, initial: FlightState, goal: FlightState, flights: List[Flight]):
        self.flights = flights
        super().__init__(initial, goal)

    def actions(self, state: FlightState) -> List[Flight]:
        return [f for f in self.flights if f.matches(state)]

    def result(self, state: FlightState, action: Flight) -> FlightState:
        return action.end_city, action.end_time

    def goal_test(self, state: FlightState) -> bool:
        state_city, state_time = state
        goal_city, goal_time = self.goal
        return state_city == goal_city and state_time <= goal_time

    def path_cost(
        self, cost: int, state1: FlightState, action: Flight, state2: FlightState
    ) -> int:
        return cost + 1

    def value(self, state: FlightState) -> int:
        state_city, state_time = state
        goal_city, goal_time = self.goal
        return goal_time - state_time


def get_solution(problem: search.Problem, verbose=True) -> Tuple[list, list]:
    solution, path = [], []
    node: search.Node = search.depth_first_tree_search(problem)
    if node:
        solution = node.solution()
        path = node.path()
        if verbose:
            print(dict(solution=solution))
            print(dict(path=path))
            print()
    return solution, path


def main_word_ladder():
    checker = ValidWords(txt_word="words2.txt")
    for initial, goal in [("cold", "warm"), ("cars", "cats"), ("best", "math")]:
        problem = WordLadderProblem(initial, goal, checker)
        get_solution(problem)


def find_itinerary(
    start_city=Places.rome,
    start_time=1,
    end_city=Places.constantinople,
    deadline=10,
    verbose=True,
) -> List[FlightState]:
    flights = [
        Flight(Places.rome, 1, Places.paris, 4),
        Flight(Places.rome, 3, Places.madrid, 5),
        Flight(Places.rome, 5, Places.istanbul, 10),
        Flight(Places.paris, 2, Places.london, 4),
        Flight(Places.paris, 5, Places.oslo, 7),
        Flight(Places.paris, 5, Places.istanbul, 9),
        Flight(Places.madrid, 7, Places.rabat, 10),
        Flight(Places.madrid, 8, Places.london, 10),
        Flight(Places.istanbul, 10, Places.constantinople, 10),
    ]
    initial = (start_city, start_time)
    goal = (end_city, deadline)
    problem = FlightProblem(initial=initial, goal=goal, flights=flights)
    solution, path = get_solution(problem, verbose=verbose)
    return [n.state for n in path]


def find_shortest_itinerary(
    start_city=Places.rome, end_city=Places.constantinople
) -> List[FlightState]:
    start_time = 1
    deadline = 1
    while True:
        itinerary = find_itinerary(
            start_city=start_city,
            start_time=start_time,
            end_city=end_city,
            deadline=deadline,
            verbose=False,
        )
        if itinerary:
            return itinerary
        else:
            deadline += 1


def main_flights():
    find_itinerary()
    find_itinerary(start_time=0, end_city=Places.istanbul)
    find_itinerary(start_time=0, end_city=Places.istanbul, deadline=8)

    print(find_shortest_itinerary())
    print(find_shortest_itinerary(end_city=Places.istanbul))


if __name__ == "__main__":
    main_word_ladder()
    main_flights()
