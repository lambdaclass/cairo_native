#!/usr/bin/env bash

# Script to check the cairo2 binaries version matches.

version=$(cairo2/bin/cairo-compile --version | cut -d' ' -f 2)

if ! [ "$version" = "$1" ]; then
  echo "cairo2 version mismatch, please re-run 'make deps'"
  exit 1
fi
