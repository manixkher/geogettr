import csv
from collections import Counter


input_csv = "/home/s2089339/Work/geogettr/my_datasets/osv5m/train_europe.csv"
output_csv = "/home/s2089339/Work/geogettr/my_datasets/osv5m/reduced_train_europe.csv"

# Column name for geocell
quadtree_column = "quadtree_10_1000"

THRESHOLD = 500

def main():
    # Count how many images per geocell
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        geocell_counts = Counter(row[quadtree_column] for row in reader)

  
    small_cells = {cell for cell, cnt in geocell_counts.items() if cnt < THRESHOLD}
    print(f"Found {len(small_cells)} geocells with fewer than {THRESHOLD} images.")

    with open(input_csv, "r", encoding="utf-8") as fin, \
         open(output_csv, "w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        kept_count = 0
        removed_count = 0

        for row in reader:
            if row[quadtree_column] not in small_cells:
                writer.writerow(row)
                kept_count += 1
            else:
                removed_count += 1

    print(f"New dataset written to {output_csv}.")
    print(f"Rows kept:   {kept_count}")
    print(f"ßRows removed:{removed_count}")

if __name__ == "__main__":
    main()
