import csv
from collections import Counter


csv_path = "/home/s2751435/Work/geogettr/my_datasets/osv5m/reduced_train_europe.csv"


quadtree_column = "quadtree_10_1000"


with open(csv_path, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    
    # Count occurrences of each geocell
    geocell_counts = Counter(row[quadtree_column] for row in reader)


print("\n📊 **Top 20 Geocells (Most Images):**")
for geocell, count in geocell_counts.most_common(20):
    print(f"{geocell}: {count}")

print(f"Total geocells: {len(geocell_counts)}")
print(f"geocell 10818:  {geocell_counts}")
# print("\n📉 **Bottom 20 Geocells (Fewest Images):**")
# for geocell, count in geocell_counts.most_common()[-20:]:
#     print(f"{geocell}: {count}")
# Threshold for "small" geocells

