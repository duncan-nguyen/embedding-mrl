#!/usr/bin/env bash
# Flags go to the training CLI; anything else (bash, pytest, python) runs as-is.
set -euo pipefail

if [ "$#" -eq 0 ]; then
    exec embedding-mrl --help
fi

case "$1" in
    -*) exec embedding-mrl "$@" ;;
    *)  exec "$@" ;;
esac
