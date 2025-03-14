import copy
import random
from collections import defaultdict, Counter

import community  # install via `pip install python-louvain`
import networkx as nx

MUTUAL_CONNECTION_WEIGHT = 2
ONE_WAY_CONNECTION_WEIGHT = 1


def calculate_max_possible_score(preferences: dict):
    seen_pairs = set()
    mutuals = 0
    one_way = 0
    likes = 0

    for person, liked in preferences.items():
        likes += len(liked)
        for other in liked:
            if (other, person) in seen_pairs:
                # Already counted as a mutual
                continue
            if other in preferences and person in preferences[other]:
                mutuals += 1
                seen_pairs.add((person, other))
                seen_pairs.add((other, person))
            else:
                one_way += 1

    print(f"There are {len(preferences)} people to arrange")
    print(f"There are {likes} preferences")
    print(f"The max possible score is {(mutuals * MUTUAL_CONNECTION_WEIGHT) + (one_way * ONE_WAY_CONNECTION_WEIGHT)}")

def build_graph(preferences):
    graph = nx.Graph()
    for person, prefs in preferences.items():
        graph.add_node(person)
        for p in prefs:
            weight = MUTUAL_CONNECTION_WEIGHT if person in preferences.get(p, []) else ONE_WAY_CONNECTION_WEIGHT
            if graph.has_edge(person, p):
                graph[person][p]['weight'] += weight
            else:
                graph.add_edge(person, p, weight=weight)
    return graph


def detect_communities(graph):
    partition = community.best_partition(graph, weight='weight')
    clusters = defaultdict(list)
    for person, group in partition.items():
        clusters[group].append(person)
    return list(clusters.values())


def adjust_cluster_sizes(clusters, available_table_sizes):
    adjusted = []
    leftovers = []

    # Keep track of remaining tables using Counter
    table_counter = Counter(available_table_sizes)

    for cluster in clusters:
        # While the cluster is too large to fit into any table
        while len(cluster) > max(table_counter.elements(), default=0):
            # Try to find the biggest available table size that we still have
            possible_sizes = [s for s in table_counter if table_counter[s] > 0 and len(cluster) >= s]
            if not possible_sizes:
                break  # Can't split anymore, will move to leftovers
            best_size = max(possible_sizes)
            group = cluster[:best_size]
            adjusted.append(group)
            table_counter[best_size] -= 1
            cluster = cluster[best_size:]

        # Try to fit remaining cluster as-is
        if len(cluster) in table_counter and table_counter[len(cluster)] > 0:
            adjusted.append(cluster)
            table_counter[len(cluster)] -= 1
        else:
            leftovers.extend(cluster)

    # Try to pack leftovers greedily into remaining tables
    for size in sorted(table_counter.elements(), reverse=True):
        if len(leftovers) >= size:
            group = leftovers[:size]
            adjusted.append(group)
            leftovers = leftovers[size:]
        else:
            continue

    # If still any people left, assign them to smallest available tables
    for size in sorted(table_counter.elements()):
        if not leftovers:
            break
        group = leftovers[:size]
        adjusted.append(group)
        leftovers = leftovers[size:]

    # If still leftovers exist (edge case), group them all in one table
    if leftovers:
        adjusted.append(leftovers)

    return adjusted


def compute_satisfaction_score(group, preferences):
    score = 0
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            a, b = group[i], group[j]
            if b in preferences.get(a, []) and a in preferences.get(b, []):
                score += MUTUAL_CONNECTION_WEIGHT
                break
            if b in preferences.get(a, []) or a in preferences.get(b, []):
                score += ONE_WAY_CONNECTION_WEIGHT
    return score


def evaluate_all_tables(tables, preferences):
    return sum(compute_satisfaction_score(t, preferences) for t in tables)


def print_table_summary(tables):
    print("Seating Arrangement:")
    for i, t in enumerate(tables):
        print(f"Table {i + 1} ({len(t)} seats): {', '.join(t)}")


def simulated_annealing(tables, preferences, table_sizes, max_iter=1000, temp=100.0, cooling_rate=0.98):
    current = copy.deepcopy(tables)
    current_score = evaluate_all_tables(current, preferences)
    best = current
    best_score = current_score

    for _ in range(max_iter):
        temp *= cooling_rate
        new_tables = copy.deepcopy(current)

        # Pick two random tables and try to swap one person from each
        t1, t2 = random.sample(range(len(new_tables)), 2)
        if not new_tables[t1] or not new_tables[t2]:
            continue

        i = random.randint(0, len(new_tables[t1]) - 1)
        j = random.randint(0, len(new_tables[t2]) - 1)

        new_tables[t1][i], new_tables[t2][j] = new_tables[t2][j], new_tables[t1][i]

        if len(new_tables[t1]) not in table_sizes or len(new_tables[t2]) not in table_sizes:
            continue  # invalid swap

        new_score = evaluate_all_tables(new_tables, preferences)
        delta = new_score - current_score

        if delta > 0 or random.random() < pow(2.718, delta / temp):
            current = new_tables
            current_score = new_score
            if new_score > best_score:
                best = new_tables
                best_score = new_score

    return best


def check_number_of_available_seats(preferences, available_table_sizes):
    seats = sum(available_table_sizes)
    people = set()
    for person, liked in preferences.items():
        people.add(person)
        for other in liked:
            people.add(other)

    if len(people) > seats:
        print(f"WARNING: there are {seats} seats available, but at least {len(people)} people needing a seat. Fix input please.")
        exit(1)


if __name__ == "__main__":
    preferences = {
        'Alice': ['Bob', 'Charlie', 'Dana'],
        'Bob': ['Charlie'],  # one-way to Charlie
        'Charlie': ['Alice'],  # one-way to Alice
        'Dana': ['Eve'],
        'Eve': ['Frank'],
        'Frank': ['Eve', 'Grace'],  # mutual with Eve
        'Grace': ['Helen'],  # one-way
        'Helen': ['Grace'],  # mutual with Grace
        'Ian': ['Jack', 'Kelly'],
        'Jack': ['Ian'],  # mutual
        'Kelly': ['Leo'],  # one-way
        'Leo': ['Maya'],
        'Maya': ['Nina'],  # one-way
        'Nina': ['Oscar'],
        'Oscar': ['Paul'],  # one-way
        'Paul': [],
        'Quinn': ['Riley'],
        'Riley': ['Sam'],
        'Sam': ['Quinn'],  # mutual
        'Tina': ['Uma', 'Vince'],
        'Uma': [],
        'Vince': ['Tina']  # one-way
    }
    calculate_max_possible_score(preferences)

    available_table_sizes = [6, 6, 7, 3]
    check_number_of_available_seats(preferences, available_table_sizes)

    G = build_graph(preferences)

    clusters = detect_communities(G)
    tables = adjust_cluster_sizes(clusters, available_table_sizes)

    print("Initial Seating:")
    print_table_summary(tables)
    print(f"Initial Total Score: {evaluate_all_tables(tables, preferences)}\n")

    tables = simulated_annealing(tables, preferences, available_table_sizes)
    print("After Optimization:")
    print_table_summary(tables)
    print(f"Optimized Total Score: {evaluate_all_tables(tables, preferences)}")
