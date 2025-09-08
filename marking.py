import collections
import datetime
import itertools
import random

import interact


class RoomCountException(Exception):
    def __init__(self, expected_count: int, actual_count: int):
        self.expected_count = expected_count
        self.actual_count = actual_count


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


def generate_routes(initial_route: list[int], initial_result: list[int], n: int = None) -> tuple[
    list[list[tuple[int, int]]], list[str]]:
    """
    Generates modified routes based on an initial route.

    Modifies up to 4 labels in each route, so that the changes can be tracked in the query result.

    TODO: we probably don't need to modify every index. most, if not all rooms, should be visited more than twice. maybe query in batches

    :param initial_route: the initial route that was explored, without any charcoal markings
    :param initial_result: the result of exploring the initial route
    :param n: the number of indexes that need to be identified to find all N rooms
    :return: the indexes of the marked rooms and which values they were marked as, and the corresponding routes
    """
    if n is None:
        n = len(initial_result) - 1

    modifications = []
    routes = []

    to_be_modified = [*range(n)]
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


def construct_graph(N: int, initial_route: list[int], initial_result: list[int],
                    modifications: list[list[tuple[int, int]]],
                    results: list[list[int]]) -> tuple[list[int], list[list[int]]]:
    """
    Construct the full graph. Some edges may lead to (-1, -1) if there is insufficient information.

    For each modified route, find which labels have changed from the original result. The indexes with those changed
    labels must belong to the same room, and we can merge them.

    :param N: number of rooms in the graph
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
        assert len(initial_result) == len(
            result), f"{initial_result = }\n{len(initial_result) = }\n{result = }\n{len(result) = }"
        for i in range(len(result)):
            if initial_result[i] != result[i]:
                root_idx = mods[result[i]]
                join(root_idx, i)

    root = [get_root(u) for u in root]
    all_roots = [*sorted(set(root))]
    room_idx = {idx: i for i, idx in enumerate(all_roots)}

    # TODO: this fails sometimes with len(all_roots) == len(room_idx) == N - 1
    if not (len(all_roots) == len(room_idx) == N):
        raise RoomCountException(expected_count=N, actual_count=len(all_roots))

    labels = [initial_result[idx] for idx in all_roots]

    # Calculate the destination rooms.
    graph = [[-1 for _ in range(6)] for _ in range(N)]
    cur = get_root(0)
    for i, d in enumerate(initial_route):
        dst = get_root(i + 1)
        graph[room_idx[cur]][d] = room_idx[dst]
        cur = dst

    return labels, graph


def convert_graph_to_connections(N: int, graph: list[list[int]], dry_run=True) -> tuple[list[dict[str, dict[str, int]]], int, int]:
    """
    Converts the graph into the format expected by the server.

    If there is insufficient information, it will raise a warning and assign doors randomly.

    :param N: number of rooms in the graph
    :param graph: the destination room of each door
    :param dry_run: whether to print warnings if there are single-side matches or guesses
    :return: the edges, as a list of dicts
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
    single_matches = 0
    for src_idx, src_door in itertools.product(range(N), range(6)):
        if doors[src_idx][src_door][1] != -1 or doors[src_idx][src_door][0] == -1:
            continue
        single_matches += 1
        if not dry_run:
            print(f"Door {src_idx, src_door} with dst={doors[src_idx][src_door][0]} is unassigned")
        for dst_idx, dst_door in itertools.product(range(N), range(6)):
            if doors[dst_idx][dst_door][1] != -1:
                continue
            if doors[src_idx][src_door][0] == dst_idx:
                doors[src_idx][src_door] = dst_idx, dst_door
                doors[dst_idx][dst_door] = src_idx, src_door
                if not dry_run:
                    print(f"Assigned to {dst_idx, dst_door}")
                break

    # Point remaining doors to themselves.
    guesses = 0
    for src_idx, src_door in itertools.product(range(N), range(6)):
        if doors[src_idx][src_door][1] != -1:
            continue
        guesses += 1
        if not dry_run:
            print(f"Assigning door {src_idx, src_door} to itself")
        doors[src_idx][src_door] = src_idx, src_door

    connections = []
    for src_idx, src_door in itertools.product(range(N), range(6)):
        dst_idx, dst_door = doors[src_idx][src_door]
        connections.append({
            "from": {"room": src_idx, "door": src_door},
            "to": {"room": dst_idx, "door": dst_door}
        })
    return connections, single_matches, guesses


def find_traversal(graph: list[list[int]]) -> tuple[list[int], list[int]]:
    """
    Find a minimal or near-minimal walk that visits all nodes in the graph at least once, starting from room index 0.

    Assumptions:
    - graph[u] is a list of door destinations for node u with length 6,
      where -1 denotes a non-existent/unknown edge and should be ignored.
    - The graph must be connected from node 0 via existing (non -1) edges; otherwise an error is raised.
    - N is small (<= 90), so an all-pairs BFS approach is acceptable.

    Strategy:
    - Build adjacency lists ignoring -1 edges.
    - Verify connectivity from start node 0 using BFS.
    - Run BFS from every node to obtain pairwise shortest paths and parents for reconstruction.
    - Greedily walk from the current node to the nearest unvisited node, appending the shortest path.
      Mark every node encountered along the appended path as visited.

    Written by GPT-5/Cascade

    :param graph: adjacency via door destinations, potentially containing -1 entries.
    :return: (traversal_room_indexes, traversal_doors)
    """

    N = len(graph)

    # Build adjacency list ignoring -1s and also remember door mapping
    adj: list[list[tuple[int, int]]] = [[] for _ in range(N)]  # (v, door)
    for u in range(N):
        for d, v in enumerate(graph[u]):
            if v != -1:
                adj[u].append((v, d))

    # Verify connectivity from 0 using BFS
    q = collections.deque([0])
    seen = [False] * N
    seen[0] = True
    while q:
        u = q.popleft()
        for v, _d in adj[u]:
            if not seen[v]:
                seen[v] = True
                q.append(v)
    if not all(seen):
        # The known graph is not fully connected yet; restrict to the reachable set.
        # However, per our usage, the initial route should have visited all rooms,
        # so this should rarely trigger.
        reachable = sum(seen)
        raise ValueError(f"Known graph is not fully connected from 0 ({reachable}/{N} reachable).")

    # Precompute BFS from each node to reconstruct shortest paths with doors
    def bfs_from(src: int):
        dq = collections.deque([src])
        dist = [-1] * N
        parent = [-1] * N
        parent_door = [-1] * N
        dist[src] = 0
        while dq:
            u = dq.popleft()
            for v, d in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    parent_door[v] = d
                    dq.append(v)
        return dist, parent, parent_door

    all_bfs = [bfs_from(i) for i in range(N)]

    visited = set([0])
    cur = 0
    traversal_idxs: list[int] = [0]
    traversal_doors: list[int] = []

    while len(visited) < N:
        dist, parent, parent_door = all_bfs[cur]
        # Find nearest unvisited node
        target = None
        best = 10**9
        for v in range(N):
            if v in visited:
                continue
            if dist[v] != -1 and dist[v] < best:
                best = dist[v]
                target = v
        if target is None:
            # Should not happen due to connectivity check above
            break

        # Reconstruct path cur -> target
        path_nodes = []
        node = target
        while node != cur:
            path_nodes.append(node)
            node = parent[node]
        path_nodes.reverse()

        # Append corresponding doors and nodes
        u = cur
        for v in path_nodes:
            # door from u to v is parent_door[v]
            d = parent_door[v]
            traversal_doors.append(d)
            traversal_idxs.append(v)
            u = v

        cur = target
        visited.update(path_nodes)

    return traversal_idxs, traversal_doors


def augment_graph(
        graph: list[list[int]], traversal_idxs: list[int], traversal_route: list[int], traversal_result: list[int],
        traversal_modifications: list[list[tuple[int, int]]], traversal_results: list[list[int]]
) -> list[list[int]]:
    """
    Augments the graph by filling in missing edges.

    Identifies each node in the graph by following the traversal_idxs.
    Uses similar logic to construct_graph to fill in missing edges.

    Written by GPT-5/Cascade

    :param graph: the original graph
    :param traversal_idxs: the room indexes in the traversal
    :param traversal_route: the doors used in the traversal
    :param traversal_result: the result of the traversal
    :param traversal_modifications: the modifications made to the traversal
    :param traversal_results: the results of the modified routes
    :return: the augmented graph
    """

    # Build DSU across all visit positions using the traversal_result and modified results.
    L = len(traversal_result) - 1  # number of moves; also len(traversal_route)
    assert L == len(traversal_route), (L, len(traversal_route))
    root = list(range(L + 1))

    def get_root(u: int) -> int:
        if root[u] != u:
            root[u] = get_root(root[u])
        return root[u]

    def join(u: int, v: int):
        ru, rv = get_root(u), get_root(v)
        if ru == rv:
            return
        if ru > rv:
            ru, rv = rv, ru
        root[rv] = ru

    # Use the same logic as construct_graph: changed labels identify same room indices.
    for mods, result in zip(traversal_modifications, traversal_results):
        # mods is list[(idx, new_label)]
        mods_map = {label: idx for idx, label in mods}
        assert len(traversal_result) == len(result)
        for i in range(len(result)):
            if traversal_result[i] != result[i]:
                root_idx = mods_map[result[i]]
                join(root_idx, i)

    root = [get_root(u) for u in root]

    # Map DSU root -> known room index using the planned traversal prefix
    # traversal_idxs has length prefix_len+1 (rooms sequence of planned traversal)
    prefix_len = len(traversal_idxs) - 1
    assert prefix_len >= 0
    root_to_room: dict[int, int] = {}
    for pos in range(prefix_len + 1):
        r = root[pos]
        room = traversal_idxs[pos]
        if r not in root_to_room:
            root_to_room[r] = room
        else:
            # Ensure consistency if we already mapped it
            pass

    # Fill in edges where both endpoints are identified
    for i in range(L):
        src_root = root[i]
        dst_root = root[i + 1]
        if src_root in root_to_room and dst_root in root_to_room:
            src_room = root_to_room[src_root]
            dst_room = root_to_room[dst_root]
            door = traversal_route[i]
            if graph[src_room][door] == -1:
                graph[src_room][door] = dst_room

    return graph


def solve_problem(N, K, problem):
    print(f"Trying to solve {problem} ({N = }, {K = })")
    interact.select(problem)

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

    labels, graph = construct_graph(N, initial_route, initial_result, modifications, results)
    # print(graph)

    _, single_matches, guesses = convert_graph_to_connections(N, graph)
    while single_matches > 1 or guesses > 0:
        traversal_idxs, traversal_doors = find_traversal(graph)
        # print(traversal_idxs)
        # print(traversal_doors)

        traversal_route = traversal_doors + [random.randint(0, 5) for _ in range(N * K - len(traversal_doors))]
        traversal_resp = interact.explore([''.join(map(str, traversal_route))])
        print(traversal_resp)
        traversal_result = traversal_resp["results"][0]

        traversal_modifications, traversal_routes = generate_routes(traversal_route, traversal_result,
                                                                    n=len(traversal_doors) + 1)

        traversal_resp = interact.explore(traversal_routes)
        queries = traversal_resp["queryCount"]
        traversal_results = clean_results(traversal_modifications, traversal_resp["results"])

        graph = augment_graph(graph, traversal_idxs, traversal_route, traversal_result, traversal_modifications,
                              traversal_results)
        _, single_matches, guesses = convert_graph_to_connections(N, graph)

    connections, single_matches, guesses = convert_graph_to_connections(N, graph, dry_run=False)

    verdict = interact.guess(labels, 0, connections)
    print(verdict, f"{queries = }. {single_matches = }. {guesses = }")

    with open("data/runs.txt", 'a') as f:
        f.write(f"{datetime.datetime.now()} {problem} {N} {K} {verdict["correct"]} {queries} {single_matches} {guesses}\n")


def main():
    for N, problem in interact.PROBLEMS.items():
        solve_problem(N, 18, problem)
    while True:
        for N, problem in [*interact.PROBLEMS_2.items(), *interact.PROBLEMS_3.items()]:
            K = 6
            for _ in range(100):
                try:
                    solve_problem(N, K, problem)
                    break
                except RoomCountException as e:
                    with open("data/runs.txt", 'a') as f:
                        f.write(f"{datetime.datetime.now()} {problem} {N} {K} {None} {e.actual_count} {e.expected_count}\n")


if __name__ == '__main__':
    main()
