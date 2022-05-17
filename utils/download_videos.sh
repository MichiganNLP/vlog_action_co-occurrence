#!/usr/bin/env bash

set -e

if [ "$#" -ne 2 ]; then
  echo "Illegal number of parameters"
  exit
fi

input_file="$1"
output_folder="$2"

echo "$input_file"
mkdir -p "$output_folder"

mapfile -t video_ids < <(jq --raw-output 'to_entries | .[].value | .[].video' "$input_file")

n=${#video_ids[@]}

cd "$output_folder"

for i in "${!video_ids[@]}"; do
  video_id=${video_ids[$i]}
  yt-dlp -f 'worstvideo[height>=224][width>=224]' --id -c --ignore-errors -- "${video_id}"
done | tqdm --total "$n" --desc "Downloading videos" >/dev/null
