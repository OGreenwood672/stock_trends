from collections import defaultdict
from k_means import perform_kmeans
from autoencoder import load_features
from GLOBALS import SHAPE_FEATURES, GROUPS_CSV
from tqdm import tqdm

def create_graph():

    features = load_features(SHAPE_FEATURES)

    graph = defaultdict(int)
    progress_bar = tqdm(range(len(features), 0, -5), desc="Processing clusters")
    for n_clusters in progress_bar:

        results, centers = perform_kmeans(features, n_clusters=n_clusters)

        results = results.sort_values(by=['Cluster', 'ticketer'], kind='mergesort')

        i = 0
        while i < len(results):
            current_cluster = results.iloc[i]
            j = i + 1
            while j < len(results) and current_cluster['Cluster'] == results.iloc[j]['Cluster']:
                graph[(current_cluster['ticketer'], results.iloc[j]['ticketer'])] += 1
                j += 1
            
            i += 1


    print("Graph created")
    return graph
