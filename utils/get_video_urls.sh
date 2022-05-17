#!/usr/bin/env bash

set -e

if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters"
  exit
fi

input_file=$1

mapfile -t video_ids < <(jq --raw-output 'to_entries | .[].value | .[].video' "$input_file")

n=${#video_ids[@]}

# `youtube-dl` outputs the URL for the video, or an error message. This error message starts with "ERROR" and it can
# have more information. However, sometimes it prints multiple lines when it describes the error, thus breaking the
# alignment of one URL per line. We then only print the first error line or the URL.

youtube-dl -f 'worstvideo[height>=224][width>=224]' --ignore-errors --get-url -a <(
  IFS=$'\n'
  echo "${video_ids[*]}"
) |& awk '{ if ( $0 ~ /^(ERROR|http)/ ) { print $0 } }' | tqdm --total "$n" --desc "Getting URLs"
