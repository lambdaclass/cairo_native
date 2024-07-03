#!/usr/bin/env bash

# Script to invoke the dev version of the jit runner to run programs easily.

echo $3 | cargo r --profile optimized-dev --all-features --bin cairo-native-run $1 $2 --inputs - --outputs -
# PLT: ACK
