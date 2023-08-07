#!/usr/bin/env bash

# Script to invoke the dev version of the jit runner to run programs easily.

echo $3 | cargo r --profile optimized-dev --all-features --bin sierrajit $1 $2 --inputs - --outputs -
