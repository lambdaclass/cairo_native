#!/usr/bin/env bash

# Compares state dump files between two directories: 'state_dumps/vm' and 'state_dumps/native'.
# It iterates over all JSON files in the 'state_dumps/vm' directory and checks if the corresponding
# file exists in 'state_dumps/native'.
# If the corresponding file does not exist, it skips the comparison and counts the missing files.
# For existing pairs, it compares the contents, ignoring the lines containing the "reverted" field, because of error message diference in Native and VM.
# It counts and displays the number of matching, differing, and missing state dumps.

matching=0
diffing=0
missing=0

# Iterate over state_dumps/vm dumps
for vm_dump in state_dumps/vm/*/*.json; do
  [ -f "$vm_dump" ] || continue

  native_dump="${vm_dump//vm/native}"

  # Check if the corresponding native_dump file exists, if not, skip
  if [ ! -f "$native_dump" ]; then
    echo "Missing: $native_dump (file not found)"
    missing=$((missing+1))
    continue
  fi

  tx_name=$(basename "$vm_dump")
  tx=${tx_name//.*/}
  block_name=$(basename "$(dirname "$vm_dump")")
  block=${block_name//block/}

  if ! cmp -s \
      <(sed '/"revert_error": /d' "$native_dump") \
      <(sed '/"revert_error": /d' "$vm_dump")
  then
    echo "Diff at block $block, tx $tx"
    diffing=$((diffing+1))
  else
    matching=$((matching+1))
  fi
done

echo
echo "Finished comparison"
echo "- Matching: $matching"
echo "- Diffing:  $diffing"
echo "- Missing: $missing"

if ! [[ $diffing -eq 0 && $missing -eq 0 ]] ; then
  exit 1
fi
