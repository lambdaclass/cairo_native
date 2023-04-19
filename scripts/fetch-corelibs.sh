#!/bin/bash

set -e


git clone \
    --depth 1 \
    --branch v1.0.0-alpha.7 \
    https://github.com/starkware-libs/cairo.git \
    starkware-cairo

cp -r starkware-cairo/corelib .
rm -rf starkware-cairo/
