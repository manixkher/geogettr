import csv

# Adjust paths as needed
reduced_train_csv = "/home/s2751435/Work/geogettr/my_datasets/osv5m/reduced_train_europe.csv"
original_test_csv = "/home/s2751435/Work/geogettr/my_datasets/osv5m/test.csv"
filtered_test_csv = "/home/s2751435/Work/geogettr/my_datasets/osv5m/reduced_test_europe.csv"

# Column name for the geocell ID
quadtree_column = "quadtree_10_1000"

def main():
    # 1) Read the geocells from the reduced training set
    allowed_geocells = set()
    with open(reduced_train_csv, "r", encoding="utf-8") as f_train:
        train_reader = csv.DictReader(f_train)
        for row in train_reader:
            allowed_geocells.add(row[quadtree_column])

    print(f"✅ Reduced training set has {len(allowed_geocells)} unique geocells.")

    # 2) Go through test.csv and keep only rows with geocells in allowed_geocells
    kept_count = 0
    removed_count = 0
    with open(original_test_csv, "r", encoding="utf-8") as f_test, \
         open(filtered_test_csv, "w", encoding="utf-8", newline="") as f_out:

        test_reader = csv.DictReader(f_test)
        fieldnames = test_reader.fieldnames

        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in test_reader:
            if row[quadtree_column] in allowed_geocells:
                writer.writerow(row)
                kept_count += 1
            else:
                removed_count += 1

    print(f"Wrote filtered test CSV to: {filtered_test_csv}")
    print(f"Rows kept:    {kept_count}")
    print(f"Rows removed: {removed_count}")

if __name__ == "__main__":
    main()
