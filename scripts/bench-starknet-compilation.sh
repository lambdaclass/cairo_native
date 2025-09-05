#!/bin/sh

WORK="bench_starknet_compilation_data"
OUTPUT=$WORK/data.csv
NATIVE_OPT_LEVEL="${NATIVE_OPT_LEVEL:-2}"

usage() {
cat <<EOF 
usage: $0 (<class-hash> <network>)+

Receives pairs of class-hash and network, and benchmarks their compilation. For example, the following command benchmarks the compilation of four mainnet
contract classes:

\`\`\`
./scripts/bench-starknet-compilation.sh \\
    0x01ad981ba6707c4f518467704f33e415832588b5029ff9dc53118d3dff84b599 mainnet \\
    0x01821775e6fc1c2e4cd1452dd7b01f35ca3de211e66a916a6dd3bd1a7d257a6b mainnet \\
    0x00609091b885073373cd050d2e5c7a2791a750abc3733ce5f98e3e1b9aeab6b8 mainnet \\
    0x0465991ec820cf53dbb2b27474b6663fb6f0c8bf3dac7db3991960214fad97f5 mainnet
\`\`\`

The output of this command is saved to the $WORK directory
- data.csv - table with the total compilation time of each class
- raw-*.json - raw contract classes
- compiled-* - compilation artifacts
- stats-*.json - compilation statistics

The RPC endpoints are taken from mandatory environment variables:
- RPC_ENDPOINT_MAINNET: RPC endpoint of mainnet network.
- RPC_ENDPOINT_TESTNET: RPC endpoint of sepolia network.

The optimization level can be configured with the NATIVE_OPT_LEVEL environment
variable. By default, level 2 is used.
EOF
}

set -eu

if ! [ "$#" -ge "2" ]; then
	echo "error: expected at least two arguments"
	echo
	usage
	exit 1
fi

mkdir -p $WORK

echo "Building starknet-native-compile"
echo
cargo build --release --bin starknet-native-compile --color never --quiet

download() {
	while [ "$#" -ge "2" ]; do
		class="$1"
		net="$2"
		shift 2

		raw_class_path="$WORK/raw-$class.json"

		if ! [ -s "$raw_class_path" ];  then
			if [ "$net" = "mainnet" ] ; then
				rpc="$RPC_ENDPOINT_MAINNET"
			elif [ "$net" = "sepolia" ] ; then
				rpc="$RPC_ENDPOINT_TESTNET"
			else
				echo "invalid network: $net"
				exit 1
			fi
		
			echo "Fetching $class"
			starkli class-by-hash "$class" --rpc "$rpc" > "$raw_class_path"
			jq 'del(.abi)' "$raw_class_path" > $WORK/tmp
			mv $WORK/tmp "$raw_class_path" 
		else
			echo "Already fetched $class"
		fi
	done
}

compile() {
	echo "class_hash,network,time" > "$OUTPUT"

	while [ "$#" -ge "2" ]; do
		class="$1"
		net="$2"
		shift 2

		raw_class_path="$WORK/raw-$class.json"
		stats_class_path="$WORK/stats-$class.json"
		compiled_class_path="$WORK/compiled-$class.out"

		echo "Compiling $class"
	  target/release/starknet-native-compile -O "$NATIVE_OPT_LEVEL" \
			"$raw_class_path" "$compiled_class_path" \
			--stats "$stats_class_path"
		time=$(jq '.compilation_total_time_ms' "$stats_class_path" --raw-output)

		echo "$class,$net,$time" >> "$OUTPUT"
	done
}

echo "Fetching classes"
echo
download "$@"
echo

echo "Compiling classes"
echo
compile "$@"
echo

echo "Summary"
echo
column -ts, "$OUTPUT"
echo
