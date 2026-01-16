#!/bin/bash

set -eu

cargo build --package cairo-native-test-2
echo

find test_data_artifacts -name "*.tests.json" -print0 |
while IFS= read -r -d '' test_file; do
	echo "testing $test_file..."
	echo
	target/debug/cairo-native-test-2 $test_file -O2 --compare-with-cairo-vm
done
