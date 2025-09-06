import random

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


if __name__ == '__main__':
    N = 6

    select(PROBLEMS[N])
    random_route = ''.join(str(random.randint(0, 5)) for _ in range(18 * N))
    output = explore([random_route])
    results = output["results"]
    with open("route.txt", 'w') as f:
        f.write(f"{N}\n{' '.join(random_route)}\n{' '.join(map(str, results[0]))}")
