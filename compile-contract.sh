#!/usr/bin/env bash

CLASS_HASH=$1
NETWORK=$2

if [[ $NETWORK != "sepolia" ]] && [[ $NETWORK != "mainnet" ]]; then
    exit 1
fi

starkli class-by-hash --network $NETWORK "$CLASS_HASH" > "$CLASS_HASH".contract_class.json
jq 'del(.abi)' "$CLASS_HASH".contract_class.json > "$CLASS_HASH".json
rm "$CLASS_HASH".contract_class.json
echo "Compiling $CLASS_HASH"
NATIVE_DEBUG_DUMP=true time cargo run --bin starknet-native-compile -- "$CLASS_HASH".json out.dylib --opt-level 2
