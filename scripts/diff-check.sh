#!/usr/bin/env bash

diffing=0

for vm_dump in state_dumps/vm/*/*.json; do
  [ -f "$vm_dump" ] || continue

  native_dump="${vm_dump//vm/native}"

  base=$(basename "$native_dump")

  if ! cmp -s \
      <(sed '/"reverted": /d' "$native_dump") \
      <(sed '/"reverted": /d' "$vm_dump")
  then
    echo "NATIVE DIFFING IN TX: $native_dump"
    diffing=1
  fi
done

echo $diffing
