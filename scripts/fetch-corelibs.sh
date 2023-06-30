#!/usr/bin/env bash

set -e


git clone \
    --depth 1 \
    --branch v2.0.0-rc5 \
    https://github.com/starkware-libs/cairo.git \
    starkware-cairo

cp -r starkware-cairo/corelib .
rm -rf starkware-cairo/
