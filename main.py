import os
import pickle

from process_data import process_data
from get_data import get_ticketers, get_data, save_data
from autoencoder import create_autoencoder, save_features, load_features
import argparse
from calculate_correlations import create_graph
from visualise import visualise_graph

from GLOBALS import STOCKS_PARQUET, START_DATE, END_DATE, NORMALISED_STOCKS_PARQUET, SHAPE_FEATURES, GROUPS_CSV, CORRELATIONS_PICKLE

def main():

    parser = argparse.ArgumentParser(description='Finance data processing')
    parser.add_argument('--load-data', action='store_true', help='Get financial data')
    args = parser.parse_args()

    # Generate folders if they do not exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/stocks", exist_ok=True)

    if args.load_data:
        ticketers = get_ticketers()
        data = get_data(ticketers, START_DATE, END_DATE)
        if data is None:
            return

        process_data()

        create_autoencoder(NORMALISED_STOCKS_PARQUET)
        save_features(NORMALISED_STOCKS_PARQUET, SHAPE_FEATURES)
    
    # graph = create_graph()
    
    # pickle.dump(graph, open(CORRELATIONS_PICKLE, "wb"))
    # print("Graph saved to", CORRELATIONS_PICKLE)

    graph = pickle.load(open(CORRELATIONS_PICKLE, "rb"))
    print("Graph loaded from", CORRELATIONS_PICKLE)
    
    print("Graph before filtering:", len(graph), "edges")
    graph = {k: v for k, v in graph.items() if v >= 25}
    print(len(graph), "edges in the graph after filtering")

    # Visualise the graph
    visualise_graph(graph)

if __name__ == "__main__":
    main()