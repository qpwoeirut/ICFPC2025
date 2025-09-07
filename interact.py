import dotenv
import requests

config = dotenv.dotenv_values()

BASE_URL = "https://31pwr5t6ij.execute-api.eu-west-2.amazonaws.com"
TEAM_ID = config["TEAM_ID"]

# All problem sets allow charcoal marking.
# Problem set 1 has a route limit of 18n doors.
# Problem sets 2 and 3 have a route limit of 6n doors.
PROBLEMS = {3: "probatio", 6: "primus", 12: "secundus", 18: "tertius", 24: "quartus", 30: "quintus"}
PROBLEMS_2 = {12: "aleph", 24: "beth", 36: "gimel", 48: "daleth", 60: "he"}
PROBLEMS_3 = {18: "vau", 36: "zain", 54: "hhet", 72: "teth", 90: "iod"}


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
