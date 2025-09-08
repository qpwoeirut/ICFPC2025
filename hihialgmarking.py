import requests
import json
import random
import copy

BASE_URL = "https://31pwr5t6ij.execute-api.eu-west-2.amazonaws.com"
team_id = "[redacted]"

def register(name: str, pl: str, email: str):
    resp = requests.post(f"{BASE_URL}/register", json={
        "name": name,
        "pl": pl,
        "email": email
    })
    return resp.json()

def select(problem_name: str):
    resp = requests.post(f"{BASE_URL}/select", json={
        "id": team_id,
        "problemName": problem_name
    })
    return resp.json()

def explore(plans: list[str]):
    resp = requests.post(f"{BASE_URL}/explore", json={
        "id": team_id,
        "plans": plans
    })
    return resp.json()

def guess(rooms: list[int], starting_room: int, connections: list[dict]):
    resp = requests.post(f"{BASE_URL}/guess", json={
        "id": team_id,
        "map": {
            "rooms": rooms,
            "startingRoom": starting_room,
            "connections": connections
        }
    })
    return resp.json()

def interact():
    while True:
        cmd = input("select[s], guess[g], explore[e]: ")
        if cmd == "s":
            problem = input("problem: ")
            print(select(problem))
        elif cmd == "e":
            path = input("path: ")
            print(explore([path]))
        elif cmd == "g":
            rooms = list(map(int, input("rooms: ").strip().split(' ')))
            starting_room = int(input("starting room: "))
            connections = []
            while True:
                inp = input("room door room door: ")
                print(inp)
                if len(inp) == 0:
                    break
                inp = list(map(int, inp.strip().split(' ')))
                connections.append(
                    {
                        "from": {
                            "room": inp[0],
                            "door": inp[1]
                        },
                        "to": {
                            "room": inp[2],
                            "door": inp[3]
                        }
                    }
                )
            print(guess(rooms, starting_room, connections))
        else:
            break
        
def randstr(sample, n, id):
    res = ""
    realquery = ""
    while len(realquery) < n:
        e = random.choice(sample)
        if len(res) == 0:
            e = '6'
        res += e
        if e == "6":
            realquery += id
        else:
            realquery += e
    return [realquery, res]

def randstr2(sample, n):
    res = "" #6789 = change to 0123
    realquery = ""
    cnt = 0
    while cnt < n:
        if len(res)%2 == 0:
            # if len(res) == 0:
            #     e = random.choice("6")
            # else:
            #     e = random.choice("6789")
            e = random.choice("6789")
        else:
            e = random.choice(sample)
        if int(e) > 5:
            realquery += "[" + str(int(e)-6) + "]"
        else:
            realquery += e
            cnt += 1
        res += e
    return [realquery, res]

def done(edges):
    for i in edges:
        for j in i:
            if j == -1:
                return False
    return True

def nextfree(nodes):
    for i in range(len(nodes)):
        if nodes[i]["value"] == -1:
            return i
    return -1

def parse(edges):
    pairs = []
    seen = set()
    n = len(edges)
    for i in range(n):
        for j in range(6):
            k = edges[i][j]
            if tuple([i,j]) in seen:
                continue
            m = None
            for d in range(6):
                if edges[k][d] == i and not tuple([k,d]) in seen:
                    m = d
                    break

            if m is None:
                continue

            seen.add(tuple([k,m]))
            seen.add(tuple([i,j]))

            pairs.append({
                "from": {"room": i, "door": j},
                "to": {"room": k, "door": m}
            })
    return pairs

def countneg(edges):
    cnt = 0
    for i in edges:
        for j in i:
            if j == -1:
                cnt += 1
    return cnt



if __name__ == "__main__":
    interact()
    problem = "he"
    n = 60
    lenid = 15
    lenstart = 20
    batches = 1000
    multiplier = 6
    sample = "01234566666666"


    total_correct = 0
    total_tried = 0
    while True:
        select(problem)
        id = ""
        for i in range(lenid):
            id += random.choice("012345")

        gen1 = randstr2("012345", lenstart)
        start = gen1[0]
        startpath = gen1[1]

        nodes = [] #nodes[i]['id'] = id of i, nodes[i]['value'] = value of node
        edges = []
        starting_room = 0
        for i in range(n):
            nodes.append([])
            edges.append([])
            nodes[i] = {
                    "id": [],
                    "value": -1,
                    "original": -1
                }
            for j in range(7):
                edges[i].append(-1)
        
        sad = False
        while not done(edges):
            send_query = []
            our_queries = []
            for _ in range(batches):
                gen = randstr(sample, n*multiplier-len(id) - lenstart, id)
                send_query.append(start + gen[0])
                our_queries.append(gen[1])
            res = explore(send_query)
            while True:
                prevedge = countneg(edges)
                prevnode = copy.deepcopy(nodes)
                for _ in range(batches):
                    query = our_queries[_]
                    try:
                        results = res['results'][_][len(startpath):]
                        startresults = res['results'][_][:len(startpath)]
                    except:
                        print("i died :(")
                        while True:
                            exec(input())
                    starting_room = startresults[0]
                    prevat = -1
                    prevdoor = -1
                    at = -1
                    result_index = 0
                    for i in range(len(query)):
                        if query[i] == '6':
                            foundid = results[result_index:result_index + len(id)]
                            value = results[result_index]
                            at = -1
                            for j in range(len(nodes)):
                                if nodes[j]['id'] == foundid:
                                    at = j
                                    break
                            if at == -1:
                                at = nextfree(nodes)
                                nodes[at]['id'] = foundid
                                nodes[at]['value'] = value
                            if prevat != -1:
                                edges[prevat][prevdoor] = at
                            prevat = at
                            prevdoor = 6
                            at = edges[at][6]
                            result_index += len(id)
                        else:
                            if at == -1:
                                prevat = -1
                                prevdoor = -1
                            else:
                                prevat = at
                                prevdoor = int(query[i])  
                                at = edges[at][prevdoor]
                            result_index += 1
                if countneg(edges) == prevedge and nodes == prevnode:
                    break
                else:
                    pass
                    # print(edges)
                    # print(prevedge)
                    # print(nodes)
                    # print(prevnode)
                # break
            print(edges)
            print(nodes)
            print(query)
            if nodes[-1]['value'] == -1:
                sad = True
                break
        if total_tried != 0:
            print(total_correct, total_tried, total_correct/total_tried)
        if sad:
            continue
        starting_room = -1
        for i in range(len(nodes)):
            startat = i
            for j in range(len(startpath)):
                if int(startpath[j]) > 5:
                    continue
                startat = edges[startat][int(startpath[j])]
            if startat == 0:
                starting_room = i
        if (starting_room == -1):
            print("no starting room found")
            while True:
                pass
        print(starting_room)
        startat = starting_room
        for i in range(len(startpath)):
            if int(startpath[i]) > 5:
                if nodes[startat]['original'] == -1:
                    nodes[startat]['original'] = startresults[i]
            else:
                startat = edges[startat][int(startpath[i])]
        print(parse(edges))
        rooms = []
        for i in nodes:
            if (i['original'] != -1):
                rooms.append(i['original'])
            else:
                rooms.append(i['value'])
        ansres = guess(rooms, starting_room, parse(edges))
        print(ansres)
        if ansres['correct']:
            total_correct += 1
        total_tried += 1
        print(startpath)
        print(rooms)
        # break
