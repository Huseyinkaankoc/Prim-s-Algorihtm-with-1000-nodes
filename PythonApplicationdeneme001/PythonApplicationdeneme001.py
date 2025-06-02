import random
import heapq
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import imageio.v2 as imageio  # GIF üretimi için

# 0. Önceki görselleri temizleyen yardımcı fonksiyon
def clear_output_folder(folder):
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.endswith(".png"):
                os.remove(os.path.join(folder, filename))

# 1. Rastgele bağlı graf oluşturma
def generate_random_graph(num_nodes, min_neighbors=2, max_neighbors=4, max_weight=100):
    graph = defaultdict(list)
    for node in range(num_nodes):
        possible_neighbors = [n for n in range(num_nodes) if n != node and n not in [v for v, _ in graph[node]]]
        num_edges = min(random.randint(min_neighbors, max_neighbors), len(possible_neighbors))
        neighbors = random.sample(possible_neighbors, num_edges)

        for neighbor in neighbors:
            weight = random.randint(1, max_weight)
            graph[node].append((neighbor, weight))
            graph[neighbor].append((node, weight))  # undirected
    return graph


# 2. Prim's algoritması
def prim(graph, start_node):
    visited = set()
    mst = []
    min_heap = [(0, start_node, -1)]  # (weight, target, source)

    while min_heap and len(visited) < len(graph):
        weight, current, prev = heapq.heappop(min_heap)
        if current not in visited:
            visited.add(current)
            if prev != -1:
                mst.append((prev, current, weight))

            for neighbor, edge_weight in graph[current]:
                if neighbor not in visited:
                    heapq.heappush(min_heap, (edge_weight, neighbor, current))
    return mst

# 3. MST adımlarını .png olarak kaydetme
def visualize_mst_steps(graph_dict, mst_edges, output_folder="mst_output_200"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    G_full = nx.Graph()
    for node, neighbors in graph_dict.items():
        for neighbor, weight in neighbors:
            G_full.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(G_full, seed=42)
    G_mst = nx.Graph()

    for i, (u, v, w) in enumerate(mst_edges):
        G_mst.add_edge(u, v, weight=w)

        plt.figure(figsize=(12, 8))
        plt.title(f"Step {i+1}: Edge added ({u} -- {v}, weight: {w})")

        #nx.draw(G_full, pos, node_size=10, edge_color='lightgray', with_labels=False)
        nx.draw_networkx_edges(G_mst, pos, edge_color='blue', width=2)
        nx.draw_networkx_edges(G_mst, pos, edgelist=[(u, v)], edge_color='red', width=3)

        plt.savefig(f"{output_folder}/step_{i+1:04d}.png")
        plt.close()

    print(f"Saved {len(mst_edges)} images to '{output_folder}' folder.")

# 4. Görsellerden GIF oluşturma
def create_gif_from_images(folder="mst_output_200", output_name="mst_animation_200.gif", duration=0.08):
    image_files = sorted(
        [f for f in os.listdir(folder) if f.endswith(".png")],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )

    images = []
    for filename in image_files:
        file_path = os.path.join(folder, filename)
        images.append(imageio.imread(file_path))

    imageio.mimsave(output_name, images, duration=duration)
    print(f"GIF saved as '{output_name}'")

# 5. Ana program
if __name__ == "__main__":
    output_folder = "mst_output_200"
    clear_output_folder(output_folder)

    random.seed(42)
    graph = generate_random_graph(200)
    print(f"Generated graph with {len(graph)} nodes")

    mst = prim(graph, 0)
    print("Total edges in MST:", len(mst))
    print("First 10 edges:")
    for u, v, w in mst[:10]:
        print(f"{u} -- {v}  (weight: {w})")

    visualize_mst_steps(graph, mst, output_folder=output_folder)
    create_gif_from_images(folder=output_folder, output_name="mst_animation_200.gif", duration=0.08)
