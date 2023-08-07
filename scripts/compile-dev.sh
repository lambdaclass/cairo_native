#!/usr/bin/env bash

# Script to compile a cairo program natively, for development purposes.

cargo r --profile optimized-dev --all-features --bin cairo-native-dump $1
