#!/usr/bin/env bash


echo "# Benchmarking results" > bench-hyperfine.md
for path in $(find target/bench-outputs/*.md)
do
    base_path=$(basename -s .md "$path")

    {
        echo "## Benchmark for program \`$base_path\`"
        echo "<details><summary>Open benchmarks</summary><br>"
        cat $path
        echo "</details>"
    } >> bench-hyperfine.md

done
