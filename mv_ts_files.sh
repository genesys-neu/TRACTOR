#!/bin/bash

target_dir=$1
source_dir=$2
start_ts=$3
end_ts=$4

# Timestamp range (in milliseconds)
start_timestamp=$start_ts
end_timestamp=$end_ts

# Source directory and target directory
source_dir=$source_dir
target_dir=$target_dir
# Loop through files in source directory
file_list=`ls $source_dir/*.pkl`
for file in $file_list; do
    # Get the filename without the path
    filename=$(basename -- "$file")
    # Extract the timestamp (in milliseconds) from the filename
    timestamp=$(echo "$filename" | grep -oE '[0-9]+')
    # Check if the timestamp is within the range
    if [[ "$timestamp" -ge "$start_timestamp" && "$timestamp" -le "$end_timestamp" ]]; then
        echo "Moving $file"
        # Copy the file to the target directory
        mv ${file} ${target_dir}
    fi
done
