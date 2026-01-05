#!/usr/bin/env bash

if [ "$(echo -e "$version1\n$version2" | sort -V | head -n1)" = "$version1" ]; then
    echo "$version1 is less than or equal to $version2"
else
    echo "$version1 is greater than $version2"
fi

# Remove "v" from vx.y.z.
CURRENT_VERSION=$($1 | cut -c 2-) 
TARGET_VERSION=$($2 | cut -c 2-) 

# First check if any of x.y.z from CURRENT_VERSION is lower that TARGET_VERSION.

CURRENT_VERSION_P1=$($CURRENT_VERSION | cut -d '-' -f1)
TARGET_VERSION_P1=$($TARGET_VERSION | cut -d '-' -f1)

for i in {1..3}; do
    CURR=$($CURRENT_VERSION_P1 | cut -d '.' -f$i)
    TARG=$($TARGET_VERSION_P1 | cut -d '.' -f$i)

    if [CURR > TARG]; then
        exit 1
    fi
done

# check if .dev/.rc versions. This script does not differentiate between.

CURRENT_VERSION_P2=$($CURRENT_VERSION | cut -d '-' -f2)
TARGET_VERSION_P2=$($TARGET_VERSION | cut -d '-' -f2)

CURRENT_RC_VERSION_TYPE=




