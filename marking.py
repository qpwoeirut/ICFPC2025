import collections
import datetime
import itertools
import random
from typing import Optional

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


def generate_routes(initial_route: list[int], initial_result: list[int], traversal_idxs: list[int] = None) -> tuple[
    list[list[tuple[int, int]]], list[str]]:
    """
    Generates modified routes based on an initial route.

    Modifies up to 4 labels in each route, so that the changes can be tracked in the query result.

    TODO: we probably don't need to modify every index. most, if not all rooms, should be visited more than twice. maybe query in batches

    :param initial_route: the initial route that was explored, without any charcoal markings
    :param initial_result: the result of exploring the initial route
    :param traversal_idxs: the traversal this route is based on, if any
    :return: the indexes of the marked rooms and which values they were marked as, and the corresponding routes
    """
    if traversal_idxs is None:
        to_be_modified = [*range(len(initial_result) - 1)]
    else:
        to_be_modified = [i for i, v in enumerate(traversal_idxs) if traversal_idxs.index(v) == i]

    modifications = []
    routes = []

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


def _init_to_be_modified(initial_result: list[int], traversal_idxs: Optional[list[int]]) -> list[int]:
    """
    Helper to initialize the list of indices that still need to be identified.

    Mirrors the selection logic in generate_routes.
    """
    if traversal_idxs is None:
        return [*range(len(initial_result) - 1)]
    else:
        return [i for i, v in enumerate(traversal_idxs) if traversal_idxs.index(v) == i]


def _pick_batch_indices(initial_result: list[int], to_be_modified: list[int]) -> tuple[list[int], list[int], list[int]]:
    """
    Pick up to 4 indices to modify, ensuring label diversity when possible.

    Returns (selected_idxs, selected_labels, remaining_to_modify)
    """
    selected_idxs = to_be_modified[:3]
    remaining = to_be_modified[3:]
    selected_labels = [initial_result[i] for i in selected_idxs]

    # Try to add a 4th index with a different label set if possible
    for j, idx in enumerate(remaining):
        label = initial_result[idx]
        if len({*selected_labels, label}) > 1:
            selected_idxs.append(remaining.pop(j))
            selected_labels.append(label)
            break

    return selected_idxs, selected_labels, remaining


def _build_route_with_mods(initial_route: list[int], selected_idxs: list[int], new_labels: list[int]) -> tuple[list[tuple[int, int]], str]:
    """
    Build a single modified route string and corresponding modifications list.
    """
    route = [*map(str, initial_route)]
    for idx, label in zip(selected_idxs, new_labels):
        route[idx] = f"[{label}]" + route[idx]
    modifications = [*zip(selected_idxs, new_labels)]
    return modifications, ''.join(route)


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


def count_possible_assignments(N: int, graph: list[list[int]]) -> int:
    """
    Count the number of valid assignments for the remaining unmatched doors
    (after pairing all mutually-known edges).

    Model mirrors the logic in `convert_graph_to_connections` up to the
    "both sides agree" phase, then counts how many single-side matches remain
    per destination room and how many door slots are available there. The
    number of possible completions is the product over rooms v of P(slots_v, needs_v),
    where:
      - needs_v = number of incoming single matches targeting v
      - slots_v = number of v's doors not already committed by mutual pairs

    Notes:
    - Unknown edges (value -1) are ignored for single-match counting.
    - Self-loops (u -> u) consume two doors per mutual pair and contribute
      floor(count[u->u] / 2) to the mutual-pair count for that room.
    - If for any room needs_v > slots_v, the count is 0 (infeasible).

    :param N: number of rooms
    :param graph: adjacency where graph[u][d] is destination room or -1
    :return: the exact number of possible assignments as an integer
    """

    # Count multiplicities of directed edges between room pairs.
    # counts[u][v] = number of doors in u leading to v (v != -1)
    counts: list[dict[int, int]] = [collections.defaultdict(int) for _ in range(N)]
    in_deg = [0] * N
    for u in range(N):
        for v in graph[u]:
            if v != -1:
                counts[u][v] += 1
                in_deg[v] += 1

    # Compute how many doors in each room are already fixed by mutual agreement.
    # For u != v: mutual pairs contribute min(count[u][v], count[v][u]) doors at each of u and v.
    # For u == v: each mutual pair consumes 2 self-loop doors in the same room, up to floor(count[v][v]/2).
    mutual_used = [0] * N  # doors already committed by mutual agreements per room

    # Handle u != v pairs by iterating pairs once.
    for u in range(N):
        for v, c_uv in counts[u].items():
            if v < 0 or v >= N:
                continue
            if v <= u:
                continue  # ensure unordered pair processed once where u < v
            c_vu = counts[v].get(u, 0)
            m = min(c_uv, c_vu)
            if m:
                mutual_used[u] += m
                mutual_used[v] += m

    # Handle self-loops
    for v in range(N):
        c_vv = counts[v].get(v, 0)
        if c_vv:
            m_self = c_vv // 2  # floor
            if m_self:
                mutual_used[v] += m_self

    # After mutual agreements, per room v:
    #  - needs_v = remaining incoming singles to v = in_deg[v] - mutual_used[v]
    #  - slots_v = remaining door slots in v that are not mutually committed = 6 - mutual_used[v]
    # The number of assignments is product_v P(slots_v, needs_v).
    ways = 1
    for v in range(N):
        needs_v = in_deg[v] - mutual_used[v]
        if needs_v <= 0:
            continue
        slots_v = 6 - mutual_used[v]
        if needs_v > slots_v:
            # Inconsistent: not enough available doors to accommodate singles.
            return 0
        # Multiply permutations m * (m-1) * ... for k terms.
        m = slots_v
        k = needs_v
        for t in range(k):
            ways *= (m - t)

    return ways

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
    # Use server-reported query count
    queries = initial_resp.get("queryCount", 0)

    # Batched querying: only query remaining unknown nodes after each modification batch
    to_be_modified = _init_to_be_modified(initial_result, traversal_idxs=None)
    modifications: list[list[tuple[int, int]]] = []
    routes: list[str] = []
    results: list[list[int]] = []

    while len(to_be_modified) > 0:
        selected_idxs, selected_labels, remaining = _pick_batch_indices(initial_result, to_be_modified)
        new_labels = create_modified_labels(selected_labels)
        mods, route_str = _build_route_with_mods(initial_route, selected_idxs, new_labels)

        # Issue query for just this batch
        resp = interact.explore([route_str])
        cleaned = clean_results([mods], [resp["results"][0]])[0]
        queries = resp.get("queryCount", queries)

        # Record
        modifications.append(mods)
        routes.append(route_str)
        results.append(cleaned)

        # Prune indices whose label changed (now identified)
        to_be_modified = [i for i in remaining if initial_result[i] == cleaned[i]]

    # queries already reflects the last server-reported query count

    labels, graph = construct_graph(N, initial_route, initial_result, modifications, results)
    # print(graph)

    # Use the exact assignment-count check instead of a dry-run conversion
    while count_possible_assignments(N, graph) > 4:
        traversal_idxs, traversal_doors = find_traversal(graph)
        # print(traversal_idxs)
        # print(traversal_doors)

        traversal_route = traversal_doors + [random.randint(0, 5) for _ in range(N * K - len(traversal_doors))]
        traversal_resp = interact.explore([''.join(map(str, traversal_route))])
        print(traversal_resp)
        traversal_result = traversal_resp["results"][0]
        queries = traversal_resp.get("queryCount", queries)  # baseline traversal query

        # Batched querying for traversal augmentation
        trav_to_be_modified = _init_to_be_modified(traversal_result, traversal_idxs)
        traversal_modifications: list[list[tuple[int, int]]] = []
        traversal_routes: list[str] = []
        traversal_results: list[list[int]] = []

        while len(trav_to_be_modified) > 0:
            sel_idxs, sel_labels, trav_remaining = _pick_batch_indices(traversal_result, trav_to_be_modified)
            new_labels = create_modified_labels(sel_labels)
            mods, route_str = _build_route_with_mods(traversal_route, sel_idxs, new_labels)

            trav_resp = interact.explore([route_str])
            cleaned = clean_results([mods], [trav_resp["results"][0]])[0]
            queries = trav_resp.get("queryCount", queries)

            traversal_modifications.append(mods)
            traversal_routes.append(route_str)
            traversal_results.append(cleaned)

            trav_to_be_modified = [i for i in trav_remaining if traversal_result[i] == cleaned[i]]

        graph = augment_graph(graph, traversal_idxs, traversal_route, traversal_result, traversal_modifications,
                              traversal_results)

    connections, single_matches, guesses = convert_graph_to_connections(N, graph, dry_run=False)

    verdict = interact.guess(labels, 0, connections)
    print(verdict, f"{queries = }. {single_matches = }. {guesses = }")

    with open("data/runs.txt", 'a') as f:
        f.write(f"{datetime.datetime.now()} {problem} {N} {K} {verdict['correct']} {queries} {single_matches} {guesses}\n")


def main():
    while True:
        for N, problem in interact.PROBLEMS.items():
            if N >= 18:
                solve_problem(N, 18, problem)
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
