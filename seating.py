import copy
import random
from collections import defaultdict

import community  # install via `pip install python-louvain`
import networkx as nx


def build_graph(preferences):
    graph = nx.Graph()
    for person, prefs in preferences.items():
        graph.add_node(person)
        for p in prefs:
            weight = 2 if person in preferences.get(p, []) else 1
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

    for cluster in clusters:
        while len(cluster) > max(available_table_sizes):
            new_group = cluster[:max(available_table_sizes)]
            adjusted.append(new_group)
            cluster = cluster[max(available_table_sizes):]
        if len(cluster) in available_table_sizes:
            adjusted.append(cluster)
        else:
            leftovers.extend(cluster)

    def best_fit_group(group, table_sizes):
        for size in sorted(table_sizes, reverse=True):
            if len(group) >= size:
                return group[:size], group[size:]
        return group, []

    while len(leftovers) >= min(available_table_sizes):
        group, leftovers = best_fit_group(leftovers, available_table_sizes)
        adjusted.append(group)

    if leftovers:
        adjusted.append(leftovers)

    return adjusted


def compute_satisfaction_score(group, preferences):
    score = 0
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            a, b = group[i], group[j]
            if b in preferences.get(a, []):
                score += 1
            if a in preferences.get(b, []):
                score += 1
    return score


def evaluate_all_tables(tables, preferences):
    return sum(compute_satisfaction_score(t, preferences) for t in tables)


def print_table_summary(tables):
    print("Seating Arrangement:")
    for i, t in enumerate(tables):
        print(f"Table {i + 1} ({len(t)} seats): {', '.join(t)}")


def prioritize_strong_mutual_clusters(preferences):
    mutual_clusters = []
    used = set()
    for a in preferences:
        for b in preferences[a]:
            if a in preferences.get(b, []) and (a, b) not in used and (b, a) not in used:
                mutual_clusters.append([a, b])
                used.add((a, b))
    return mutual_clusters


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


if __name__ == "__main__":
    preferences = {
        'Alice': ['Bob', 'Carol'],
        'Bob': ['Alice', 'Carol'],
        'Carol': ['Bob'],
        'Dave': ['Eve', 'Frank'],
        'Eve': ['Dave', 'Grace'],
        'Frank': ['Dave'],
        'Grace': ['Eve'],
        'Heidi': ['Ivan'],
        'Ivan': ['Heidi'],
        'Judy': [],
        'Mallory': [],
        'Niaj': ['Oscar'],
        'Oscar': ['Niaj'],
        'Peggy': ['Sybil'],
        'Sybil': ['Peggy']
    }

    available_table_sizes = [6, 7, 8, 9]

    G = build_graph(preferences)
    mutual_groups = prioritize_strong_mutual_clusters(preferences)
    clusters = detect_communities(G)
    tables = adjust_cluster_sizes(clusters, available_table_sizes)

    print("Initial Score:")
    print_table_summary(tables)
    print(f"Initial Total Score: {evaluate_all_tables(tables, preferences)}\n")

    tables = simulated_annealing(tables, preferences, available_table_sizes, max_iter=500)

    print("After Optimization:")
    print_table_summary(tables)
    print(f"Optimized Total Score: {evaluate_all_tables(tables, preferences)}")
