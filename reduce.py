import pandas as pd

# Load metadata
metadata_path = "/home/s2089339/Work/geogettr/my_datasets/osv5m/train.csv"

df = pd.read_csv(metadata_path, low_memory=False)
# print(df.head())

# Europe bounding box
EUROPE_BOUNDS = {
    "min_lat": 35.0,
    "max_lat": 71.0,
    "min_lon": -25.0,
    "max_lon": 45.0,
}

# Filter dataset to only include images inside European geocells
european_geocells = df[
    (df["latitude"] >= EUROPE_BOUNDS["min_lat"]) &
    (df["latitude"] <= EUROPE_BOUNDS["max_lat"]) &
    (df["longitude"] >= EUROPE_BOUNDS["min_lon"]) &
    (df["longitude"] <= EUROPE_BOUNDS["max_lon"])
]["quadtree_10_1000"].unique()

print(f"Identified {len(european_geocells)} European geocells")

# Filter dataset to keep only rows with European geocells
df_europe = df[df["quadtree_10_1000"].isin(european_geocells)]

# Save the filtered dataset
filtered_path = "/home/s2089339/Work/geogettr/my_datasets/osv5m/train_europe.csv"
df_europe.to_csv(filtered_path, index=False)

print(f"Saved filtered dataset with {len(df_europe)} rows to {filtered_path}")

# Extract only relevant image IDs
image_ids = set(df_europe["id"].astype(str))  # Convert to string to match filenames

# Save this list to a text file for extraction
image_id_path = "/home/s2089339/Work/geogettr/my_datasets/osv5m/europe_image_ids.txt"
with open(image_id_path, "w") as f:
    for img_id in image_ids:
        f.write(img_id + "\n")

print(f"Identified {len(image_ids)} images for extraction.")
