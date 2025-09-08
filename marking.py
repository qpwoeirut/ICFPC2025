import itertools
import random

import interact

N = 12
K = 6
PROBLEM = interact.PROBLEMS_2[N]


def create_modified_labels(labels: list[int]) -> list[int]:
    """
    Takes a list of labels and returns a new list of labels where each label is unique and all the labels have changed.

    :param labels: initial list of labels
    :return: list of labels where each is unique and old[i] != new[i]
    """
    assert len(labels) <= 4
    for perm in itertools.permutations(range(4)):
        if all(x != y for x, y in zip(perm, labels)):
            return list(perm)[:len(labels)]

    raise ValueError("All the labels were the same :( generate_routes should make sure this doesn't happen")


def generate_routes(initial_route: list[int], initial_result: list[int]) -> tuple[
    list[list[tuple[int, int]]], list[str]]:
    """
    Generates modified routes based on an initial route.

    Modifies up to 4 labels in each route, so that the changes can be tracked in the query result.

    TODO: we probably don't need to modify every index. most, if not all rooms, should be visited more than twice. maybe query in batches

    :param initial_route: the initial route that was explored, without any charcoal markings
    :param initial_result: the result of exploring the initial route
    :return: the indexes of the marked rooms and which values they were marked as, and the corresponding routes
    """
    modifications = []
    routes = []

    to_be_modified = [*range(len(initial_result[:-1]))]
    while len(to_be_modified) > 0:
        selected_idxs, to_be_modified = to_be_modified[:3], to_be_modified[3:]
        selected_labels = [initial_result[i] for i in selected_idxs]

        # Make sure we don't pick four of the same label
        for i, idx in enumerate(to_be_modified):
            label = initial_result[idx]
            if len({*selected_labels, label}) > 1:
                selected_idxs.append(to_be_modified.pop(i))
                selected_labels.append(label)
                break

        route = [*map(str, initial_route)]

        new_labels = create_modified_labels(selected_labels)
        for idx, label in zip(selected_idxs, new_labels):
            route[idx] = f"[{label}]" + route[idx]

        modifications.append([*zip(selected_idxs, new_labels)])
        routes.append(''.join(route))

    return modifications, routes


def clean_results(modifications: list[list[tuple[int, int]]], results: list[list[int]]) -> list[list[int]]:
    """
    Removes the parts of the result created by applying charcoal marks

    :param modifications: the modifications made to the initial route
    :param results: the output of the modified routes
    :return: the output of the modified routes, without the numbers from applying charcoal
    """
    assert len(modifications) == len(results)

    cleaned_results = []
    for mods, result in zip(modifications, results):
        result = [*result]
        for idx, _ in sorted(mods):
            result.pop(idx + 1)
        cleaned_results.append(result)
    return cleaned_results


def construct_graph(initial_route: list[int], initial_result: list[int], modifications: list[list[tuple[int, int]]],
                    results: list[list[int]]) -> tuple[list[int], list[list[int]], dict[int, int]]:
    """
    Construct the full graph. Some edges may lead to (-1, -1) if there is insufficient information.

    For each modified route, find which labels have changed from the original result. The indexes with those changed
    labels must belong to the same room, and we can merge them.

    :param initial_route: initial route without any charcoal markings
    :param initial_result: result of querying the initial route
    :param modifications: a list of what modifications were made to the initial route, in the form [index changed, new label]
    :param results: a list of the results after the modification
    :return: room IDs for each visited node, the destination room of each door, and the ID-to-index mapping for each room
    """
    assert len(modifications) == len(results)

    # Merge nodes into rooms using DSU.
    root = [*range(len(initial_result))]

    def get_root(u):
        if root[u] != u:
            root[u] = get_root(root[u])
        return root[u]

    def join(u, v):
        ru = get_root(u)
        rv = get_root(v)
        ru, rv = min(ru, rv), max(ru, rv)
        root[v] = root[rv] = ru

    for mods, result in zip(modifications, results):
        mods = {label: idx for idx, label in mods}
        assert len(initial_result) == len(result), f"{initial_result = }\n{len(initial_result) = }\n{result = }\n{len(result) = }"
        for i in range(len(result)):
            if initial_result[i] != result[i]:
                root_idx = mods[result[i]]
                join(root_idx, i)

    root = [get_root(u) for u in root]
    room_idx = {idx: i for i, idx in enumerate(sorted(set(root)))}
    assert len(room_idx) == N, room_idx

    # Calculate the destination rooms.
    graph = [[-1 for _ in range(6)] for _ in range(N)]
    cur = get_root(0)
    for i, d in enumerate(initial_route):
        dst = get_root(i + 1)
        graph[room_idx[cur]][d] = room_idx[dst]
        cur = dst

    return root, graph, room_idx


def convert_graph(
        initial_result: list[int], graph: list[list[int]], room_idx: dict[int, int]
) -> tuple[list[int], int, list[dict[str, dict[str, int]]]]:
    """
    Converts the graph into the format expected by the server.

    If there is insufficient information, it will raise a warning and assign doors randomly.

    :param initial_result: the result of the initial route
    :param graph: the destination room of each door
    :param room_idx: the index of each room ID
    :return: the room labels, the index of the starting room, and the edges
    """

    doors = [[(v, -1) for v in room] for room in graph]

    # Match doors where both sides agree.
    for src_idx, src_door in itertools.product(range(N), range(6)):
        if doors[src_idx][src_door][1] != -1 or doors[src_idx][src_door][0] == -1:
            continue
        for dst_idx, dst_door in itertools.product(range(N), range(6)):
            if doors[dst_idx][dst_door][1] != -1:
                continue
            if doors[src_idx][src_door][0] == dst_idx and doors[dst_idx][dst_door][0] == src_idx:
                doors[src_idx][src_door] = dst_idx, dst_door
                doors[dst_idx][dst_door] = src_idx, src_door
                break

    # Assign doors where one side matches.
    for src_idx, src_door in itertools.product(range(N), range(6)):
        if doors[src_idx][src_door][1] != -1 or doors[src_idx][src_door][0] == -1:
            continue
        print(f"Door {src_idx, src_door} with dst={doors[src_idx][src_door][0]} is unassigned")
        for dst_idx, dst_door in itertools.product(range(N), range(6)):
            if doors[dst_idx][dst_door][1] != -1:
                continue
            if doors[src_idx][src_door][0] == dst_idx:
                doors[src_idx][src_door] = dst_idx, dst_door
                doors[dst_idx][dst_door] = src_idx, src_door
                print(f"Assigned to {dst_idx, dst_door}")
                break

    # Point remaining doors to themselves.
    for src_idx, src_door in itertools.product(range(N), range(6)):
        if doors[src_idx][src_door][1] != -1:
            continue
        print(f"Assigning door {src_idx, src_door} to itself")
        doors[src_idx][src_door] = src_idx, src_door

    idx_to_id = {v: k for k, v in room_idx.items()}
    labels = [initial_result[idx_to_id[i]] for i in range(N)]
    connections = []
    for src_idx, src_door in itertools.product(range(N), range(6)):
        dst_idx, dst_door = doors[src_idx][src_door]
        connections.append({
            "from": {"room": src_idx, "door": src_door},
            "to": {"room": dst_idx, "door": dst_door}
        })
    return labels, room_idx[0], connections


def main():
    print(f"Trying to solve {PROBLEM} ({N = })")
    interact.select(PROBLEM)

    initial_route = [random.randint(0, 5) for _ in range(N * K)]
    # print(initial_route)
    initial_resp = interact.explore([''.join(map(str, initial_route))])
    # print(initial_resp)
    initial_result = initial_resp["results"][0]

    modifications, routes = generate_routes(initial_route, initial_result)
    # print(routes)
    resp = interact.explore(routes)
    # print(resp)
    queries = resp["queryCount"]
    results = clean_results(modifications, resp["results"])

    room_ids, graph, room_idx = construct_graph(initial_route, initial_result, modifications, results)
    # print(graph)
    rooms, starting_room, connections = convert_graph(initial_result, graph, room_idx)

    verdict = interact.guess(rooms, starting_room, connections)
    print(verdict, f"{queries = }")


if __name__ == '__main__':
    main()
