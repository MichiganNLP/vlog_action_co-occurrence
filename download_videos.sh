#!/usr/bin/env bash

input_file=data/dict_action_clips_sample.json
output_folder=data/videos_sample

mkdir -p "$output_folder"

mapfile -t video_ids < <(jq --raw-output 'to_entries | .[].value | .[].video' "$input_file")

n=${#video_ids[@]}

# Pairs of video and audio URLs.
mapfile -t video_urls < <(youtube-dl -f 'worstvideo[height>=224][width>=224]' --get-url -a <(
  IFS=$'\n'
  echo "${video_ids[*]}"
) |
  tqdm --total "$n" --desc "Getting URLs")

mapfile -t start_times < <(jq --raw-output 'to_entries | .[].value | .[].time_s' "$input_file")
mapfile -t end_times < <(jq --raw-output 'to_entries | .[].value | .[].time_e' "$input_file")

# From https://unix.stackexchange.com/a/426827:

# converts HH:MM:SS.sss to fractional seconds
codes2seconds() (
  local hh=${1%%:*}
  local rest=${1#*:}
  local mm=${rest%%:*}
  local ss=${rest#*:}
  printf "%s" "$(bc <<<"$hh * 60 * 60 + $mm * 60 + $ss")"
)

# converts fractional seconds to HH:MM:SS.sss
seconds2codes() (
  local seconds=$1
  local hh
  hh=$(bc <<<"scale=0; $seconds / 3600")
  local remainder
  remainder=$(bc <<<"$seconds % 3600")
  local mm
  mm=$(bc <<<"scale=0; $remainder / 60")
  local ss
  ss=$(bc <<<"$remainder % 60")
  printf "%02d:%02d:%06.3f" "$hh" "$mm" "$ss"
)

subtract_times() (
  local t1sec
  t1sec=$(codes2seconds "$1")
  local t2sec
  t2sec=$(codes2seconds "$2")
  printf "%s" "$(bc <<<"$t2sec - $t1sec")"
)

for i in "${!video_ids[@]}"; do
  video_id=${video_ids[$i]}
  video_url=${video_urls[$i]}
  start_time=${start_times[$i]}
  end_time=${end_times[$i]}

  duration=$(subtract_times "$start_time" "$end_time")
  {
      ffmpeg \
        -ss "$start_time" \
        -i "$video_url" \
        -t "$duration" \
        -n \
        "$output_folder/$video_id+${start_time%%.*}+${end_time%%.*}.mp4" >/dev/null 2>&1
        # Note this uses floor to round the durations in the filename, not round.

      echo "$i"
    } ||
    {
      continue
    }

done | tqdm --total "$n" --desc "Downloading videos" >/dev/null
