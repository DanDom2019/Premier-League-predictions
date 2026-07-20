#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || -z $1 || $1 == "/" || $1 == "." ]]; then
  echo "usage: build_pages_artifact.sh <empty-output-directory>" >&2
  exit 2
fi

pages_artifact_dir=$1
mkdir -p "$pages_artifact_dir"

if find "$pages_artifact_dir" -mindepth 1 -print -quit | grep -q .; then
  echo "output directory must be empty: $pages_artifact_dir" >&2
  exit 2
fi

mkdir -p "$pages_artifact_dir/static/foundationData" "$pages_artifact_dir/icons"
cp index.html "$pages_artifact_dir/"
cp static/script.js static/styles.css "$pages_artifact_dir/static/"
cp static/foundationData/mainLeaguesTeams.json "$pages_artifact_dir/static/foundationData/"
cp icons/* "$pages_artifact_dir/icons/"
