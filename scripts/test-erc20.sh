#!/usr/bin/env bash

# Donwload the ERC20 contract from the Cairo repository and compile with the
# Sierra compiler. The output is a MLIR file.

curl https://raw.githubusercontent.com/starkware-libs/cairo/b4e049a13b62dc493ac378747af8e0908c1b86b7/crates/cairo-lang-starknet/test_data/erc20.sierra > erc20.sierra

cargo run --profile=optimized-dev \
    --features=build-cli,with-runtime \
    --bin="cairo-native-dump" -- erc20.sierra

rm erc20.sierra
