#!/bin/bash

path="/lmh_data/data/sclab/GSE223917"
hic_path="$path"/"hic"
cool_path="$path"/"cool"

find "$hic_path" -type f -name "*.pairs.gz" -print0 | while IFS= read -r -d '' file; do
    filename=$(basename "$file")
    segments=(${filename//[_\.]/ })
    sample_name="${segments[1]}"

    cooler cload pairs -c1 2 -p1 3 -c2 4 -p2 5 "$path"/chrom.sizes:10000 "$hic_path"/"$filename" "$cool_path"/"$sample_name".cool

    echo "$sample_name"

done
