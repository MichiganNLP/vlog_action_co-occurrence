#!/usr/bin/env bash

jq --raw-output 'to_entries | .[].value | .[].video' "$1"
