#!/usr/bin/env bash


echo "# Benchmarking results" > bench-hyperfine.md
for path in $(find target/bench-outputs/*.md)
do
    base_path=$(basename -s .md $path)

    echo "## Benchmark for programm \`$base_path\`" >> bench-hyperfine.md
    cat $path >> bench-hyperfine.md
done
