import networkx as nx
import matplotlib.pyplot as plt

def visualise_graph(edges):
    # Create graph
    G = nx.Graph()

    # Add edges with weights
    for (node1, node2), weight in edges.items():
        G.add_edge(node1, node2, weight=weight)

    # Force-directed layout (spring layout)
    pos = nx.spring_layout(G, weight='weight')  # 'weight' affects force pull

    # Draw with adjusted parameters
    plt.figure(figsize=(12, 8))  # Larger figure size
    nx.draw(G, pos, with_labels=True, 
            node_color='skyblue', 
            edge_color='gray', 
            node_size=2000,  # Bigger nodes
            font_size=12,    # Larger font
            font_weight='bold',
            width=2)  # Thicker edges
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    # nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=1500, font_size=10)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['weight'] for u, v, d in G.edges(data=True)})

    plt.title("Force-Directed Graph")
    plt.show()
