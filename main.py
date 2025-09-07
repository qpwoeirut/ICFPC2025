import random
import subprocess
import sys

import dotenv
import requests

config = dotenv.dotenv_values()

BASE_URL = "https://31pwr5t6ij.execute-api.eu-west-2.amazonaws.com"
TEAM_ID = config["TEAM_ID"]

PROBLEMS = {3: "probatio", 6: "primus", 12: "secundus", 18: "tertius", 24: "quartus", 30: "quintus"}


def register(name: str, pl: str, email: str):
    resp = requests.post(f"{BASE_URL}/register", json={
        "name": name,
        "pl": pl,
        "email": email
    })
    return resp.json()


def select(problem_name: str):
    resp = requests.post(f"{BASE_URL}/select", json={
        "id": TEAM_ID,
        "problemName": problem_name
    })
    assert resp.json() == {"problemName": problem_name}


def explore(plans: list[str]) -> dict[str, list[list[int]] | str]:
    resp = requests.post(f"{BASE_URL}/explore", json={
        "id": TEAM_ID,
        "plans": plans
    })
    return resp.json()


def guess(rooms: list[int], starting_room: int, connections: list[dict]):
    resp = requests.post(f"{BASE_URL}/guess", json={
        "id": TEAM_ID,
        "map": {
            "rooms": rooms,
            "startingRoom": starting_room,
            "connections": connections
        }
    })
    return resp.json()


def parse_graph() -> tuple[list[int], int, list[dict[str, dict[str, int]]]]:
    with open("graph.txt", 'r') as f:
        labels = [int(x) for x in f.readline().split()]
        start = int(f.readline())
        connections = []
        for line in f:
            src_room, src_door, dst_room, dst_door = map(int, line.split())
            connections.append({
                "from": {
                    "room": src_room,
                    "door": src_door
                },
                "to": {
                    "room": dst_room,
                    "door": dst_door
                }
            })
    return labels, start, connections


def get_problem(N: int, routes: list[str]):
    assert all(len(route) == 18 * N for route in routes)
    select(PROBLEMS[N])
    output = explore(routes)
    results = output["results"]
    with open("route.txt", 'w') as f:
        f.write(f"{N}\n{' '.join(routes[0])}\n{' '.join(map(str, results[0]))}")


def submit_solution():
    labels, start, connections = parse_graph()
    print(guess(labels, start, connections))


def attempt_one_shot(N: int, K: int):
    route = (''.join('0' + str(random.randint(0, 5)) for _ in range((18 - K) * N // 2)) +
             ''.join(str(random.randint(0, 5)) for _ in range(N * K)))
    get_problem(N, [route])
    subprocess.run(["./solve.sh"], stdout=sys.stdout, stderr=sys.stderr)
    submit_solution()


if __name__ == '__main__':
    attempt_one_shot(12, 13)  # Should work within 4s if we're lucky w/ input
