import random

from interact import *

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
        res += e
        if e == "6":
            realquery += id
        else:
            realquery += e
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
            m = None
            for d in range(6):
                if edges[k][d] == i:
                    m = d
                    break

            if m is None:
                continue

            key = tuple(sorted([(i, j), (k, m)]))
            if key in seen:
                continue
            seen.add(key)

            pairs.append({
                "from": {"room": i, "door": j},
                "to": {"room": k, "door": m}
            })
    return pairs


if __name__ == "__main__":
    problem = "secundus"
    n = 12
    lenid = 6
    batches = 50


    while True:
        select(problem)
        id = ""
        for i in range(lenid):
            id += random.choice("012345")

        nodes = [] #nodes[i]['id'] = id of i, nodes[i]['value'] = value of node
        edges = []
        starting_room = 0
        for i in range(n):
            nodes.append([])
            edges.append([])
            nodes[i] = {
                    "id": [],
                    "value": -1,
                }
            for j in range(7):
                edges[i].append(-1)
        
        sad = False
        while not done(edges):
            send_query = []
            our_queries = []
            for _ in range(batches):
                gen = randstr("012345666", n*18-len(id), id)
                send_query.append(gen[0])
                our_queries.append(gen[1])
            res = explore(send_query)
            for _ in range(batches):
                query = our_queries[_]
                results = res['results'][_]
                starting_room = results[0]
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
                print(edges)
                print(nodes)
                print(query)
            if nodes[-1]['value'] == -1:
                sad = True
                break
        print("e")
        if sad:
            continue
        print(parse(edges))
        rooms = []
        for i in nodes:
            rooms.append(i['value'])
        print(guess(rooms, starting_room, parse(edges)))
        # break