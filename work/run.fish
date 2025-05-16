for file in *.sierra.json
    set name $(path change-extension '' $(path change-extension '' $file))
    cargo run --release --bin starknet-native-compile $file "$name.out" -O3
end
