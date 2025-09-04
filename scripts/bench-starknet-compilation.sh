#!/bin/sh

print_usage() {
cat <<EOF
Usage: $0 (<class_hash> <net>)+

Benchmarks the compilation of the given classes. Receives any number of pairs of
class_hash and network, separated by whitespace.
EOF
}

if ! [ "$#" -ge "2" ]; then
	print_usage
fi

mkdir -p bench_data

echo "Building starknet-native-compile"
cargo build --release --bin starknet-native-compile 2>/dev/null

while [ "$#" -ge "2" ]; do
	class="$1"
	net="$2"
	shift 2

	raw_class_path="bench_data/raw-$class.json"
	stats_class_path="bench_data/stats-$class.json"
	compiled_class_path="bench_data/compiled-$class.out"

	if ! [ -s "$raw_class_path" ];  then
		if [ "$net" = "mainnet" ] ; then
			rpc="$MAINNET_STARKNET_RPC"
		elif [ "$net" = "sepolia" ] ; then
			rpc="$SEPOLIA_STARKNET_RPC"
		else
			echo "invalid network: $net"
			continue
		fi
		
		echo "Fetching $class"
		starkli class-by-hash "$class" --rpc "$rpc" > "$raw_class_path"
		jq 'del(.abi)' "$raw_class_path" > bench_data/tmp
		mv bench_data/tmp "$raw_class_path" 
	else
		echo "Already fetched $class"
	fi

	if [ -s "$raw_class_path" ];  then
		echo "Compiling $class"
	  ./target/release/starknet-native-compile -O2 \
			"$raw_class_path" "$compiled_class_path" \
			--stats "$stats_class_path"

		time=$(jq '.compilation_total_time_ms' "$stats_class_path" --raw-output)
		echo "Took $time ms"
	else
		echo "Skipping $class"
	fi
done

