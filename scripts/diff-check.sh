#!/usr/bin/env bash

diffing=0
skipping=0

for vm_dump in state_dumps/vm/*/*.json; do
  [ -f "$vm_dump" ] || continue

  native_dump="${vm_dump//vm/native}"

  # Check if the corresponding native_dump file exists, if not, skip
  if [ ! -f "$native_dump" ]; then
    echo "Skipping: $native_dump (file not found)"
    skipping=$((skipping+1))
    continue
  fi

  base=$(basename "$vm_dump")

  if ! cmp -s \
      <(sed '/"reverted": /d' "$native_dump") \
      <(sed '/"reverted": /d' "$vm_dump")
  then
    echo "NATIVE DIFFING IN TX: $native_dump"
    diffing=1
  fi
done

local -a results=($diffing $skipping)

echo $results
