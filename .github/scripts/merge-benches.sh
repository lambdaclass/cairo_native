#!/usr/bin/env bash


echo "# Benchmarking results" > bench-hyperfine.md
for path in $(find target/bench-outputs/*.md)
do
    base_path=$(basename -s .md $path)

    echo "## Benchmark for program \`$base_path\`" >> bench-hyperfine.md
    echo "<details><summary>Open benchmarks</summary><br>" >> bench-hyperfine.md
    cat $path >> bench-hyperfine.md
    echo "</details>" >> bench-hyperfine.md
done
