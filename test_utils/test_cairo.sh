#!/bin/bash

set -eu
shopt -s globstar

cargo build --package cairo-native-test-2
echo

for test_file in test_data_artifacts/**/*.tests.json; do
	echo "testing $test_file..."
	echo
	target/debug/cairo-native-test-2 $test_file -O2 --compare-with-cairo-vm
done
