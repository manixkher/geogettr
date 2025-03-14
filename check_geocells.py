import csv
from collections import Counter

# Path to the CSV file
csv_path = "/home/s2751435/Work/geogettr/my_datasets/osv5m/reduced_train_europe.csv"

# Correct column name
quadtree_column = "quadtree_10_1000"

# # Open the CSV file safely
with open(csv_path, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    
    # Count occurrences of each geocell
    geocell_counts = Counter(row[quadtree_column] for row in reader)

# # Print the top and bottom 20 geocells by image count
print("\n📊 **Top 20 Geocells (Most Images):**")
for geocell, count in geocell_counts.most_common(20):
    print(f"{geocell}: {count}")

print(f"Total geocells: {len(geocell_counts)}")
print(f"geocell 10818:  {geocell_counts}")
# print("\n📉 **Bottom 20 Geocells (Fewest Images):**")
# for geocell, count in geocell_counts.most_common()[-20:]:
#     print(f"{geocell}: {count}")
# Threshold for "small" geocells


# 1) First pass: Count how many images per geocell


# 2) Determine which geocells are “small”
# small_cells = {cell for cell, cnt in geocell_counts.items() if cnt < THRESHOLD}


# print(f"Geocells with fewer than {THRESHOLD} images: {len(small_cells)}")


# 3) Optional: Print a sample of small cells
# small_sample = list(small_cells)[:20]
# if small_sample:
#     print("\nSample of small geocells (fewer than 100 images):")
#     for sc in small_sample:
#         print(f"  Geocell {sc}: {geocell_counts[sc]} images")

# 4) Second pass: Count how many rows would remain if we remove small cells
# new_length = 0
# with open(csv_path, "r", encoding="utf-8") as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         if row[quadtree_column] not in small_cells:
#             new_length += 1

# print(f"\nIf you remove those 'small' geocells, "
#       f"you'd have {new_length} rows left in the dataset.")