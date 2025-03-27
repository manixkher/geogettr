#!/usr/bin/env python3
import pandas as pd

def filter_quadtree(full_quadtree_csv, reduced_train_csv, output_csv):

    quadtree_df = pd.read_csv(full_quadtree_csv)
    print(f"Full quadtree CSV loaded: {len(quadtree_df)} rows")


    reduced_train_df = pd.read_csv(reduced_train_csv)
    print(f"Reduced training CSV loaded: {len(reduced_train_df)} rows")

    # Get the unique cluster ids from the reduced training CSV.
    valid_ids = set(reduced_train_df["quadtree_10_1000"].unique())
    print(f"Found {len(valid_ids)} unique cluster ids in the reduced training CSV")

    # Filter the quadtree DataFrame to only include rows where the cluster_id is in valid_ids
    filtered_df = quadtree_df[quadtree_df["cluster_id"].isin(valid_ids)]
    print(f"Filtered quadtree CSV contains {len(filtered_df)} rows")

    # Save the filtered DataFrame to the output CSV
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered quadtree CSV saved to: {output_csv}")

if __name__ == "__main__":
    full_quadtree_csv = "quadtree_10_1000.csv"
    reduced_train_csv = "/home/s2751435/Work/geogettr/my_datasets/osv5m/reduced_train_europe.csv"
    output_csv = "reduced_quadtree_10_1000.csv"

    filter_quadtree(full_quadtree_csv, reduced_train_csv, output_csv)
