#!/usr/bin/env bash

circuit_contracts_mainnet=(
    0x79666cdb4fc3cbcafbd74f4ea4e2855bf455c5a7c70915f5679325c54032771
    0xa40ecaa08eaef629d433b15f6b017461df8bea9b4e7299eb8ba3632e32b5a5
    0x4ffeed293927cd56686a9038a10026a2d3b9602f789d1f163c1c4ac9a822a82
    0x2269858a40ea0535cb373b0c981c91b907466edf6d65bcaf669760bbee0ae4d
    0x5ff378cb2f16804539ecb92e84f273aafbab57d450530e9fe8e87771705a673
    0x4edde37ca59d9dff8f4ac8945b1c4860b606abd61d74727904ad7494fccdfa9
) 

circuit_contracts_testnet=(
    0x1b5fbe104c033025dbb7fb37011781cc9344e881b4828cdaa023a80fecafde4
    0x3100defca27214e5f78f25e48a5b05e45899c6834cb4d34f48384c18e14dff7

)

for contract_class in "${circuit_contracts_mainnet[@]}"
do
    starkli class-by-hash $contract_class --rpc https://mainnet.juno.internal.lambdaclass.com > contracts2/$contract_class.contract_class.json
    jq 'del(.abi)' contracts2/$contract_class.contract_class.json > contracts2/$contract_class.json
    cargo run --bin starknet-native-compile contracts2/$contract_class.json contracts2/$contract_class.out.dylib --stats contracts2/$contract_class.stats.json
    sleep 2
done

for contract_class in "${circuit_contracts_testnet[@]}"
do
    starkli class-by-hash $contract_class --rpc https://sepolia.juno.internal.lambdaclass.com > contracts2/$contract_class.contract_class.json
    jq 'del(.abi)' contracts2/$contract_class.contract_class.json > contracts2/$contract_class.json
    cargo run --bin starknet-native-compile contracts2/$contract_class.json contracts2/$contract_class.out.dylib --stats contracts2/$contract_class.stats.json
    sleep 2
done
