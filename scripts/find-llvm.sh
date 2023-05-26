#!/usr/bin/env bash


function find_llvm_config() {
    # Find LLVM by the prefix environment variable.
    LLVM_PREFIX=$(printenv MLIR_SYS_160_PREFIX)
    if [[ $? -eq 0 ]]; then
        if [[ -e "$LLVM_PREFIX/bin/llvm-config" ]]; then
            LLVM_VERSION=$($LLVM_PREFIX/bin/llvm-config --version | cut -d '.' -f 1)
            if [[ $LLVM_VERSION == 16 ]]; then
                echo "$LLVM_PREFIX/bin/llvm-config"
                return 0
            fi
        fi
    fi

    # Find system LLVM using its versioned alias (llvm-config-16).
    LLVM_CONFIG_BIN=$(which llvm-config-16)
    if [[ -n "$LLVM_CONFIG_BIN" ]]; then
        echo "$LLVM_CONFIG_BIN"
        return 0
    fi

    # Find system LLVM using its standard alias (llvm-config).
    LLVM_CONFIG_BIN=$(which llvm-config)
    if [[ -n "$LLVM_CONFIG_BIN" ]]; then
        LLVM_VERSION=$($LLVM_PREFIX/bin/llvm-config --version | cut -d '.' -f 1)
        if [[ $LLVM_VERSION == 16 ]]; then
            echo "$LLVM_CONFIG_BIN"
            return 0
        fi
    fi

    # A valid LLVM toolchain was not found.
    return 1
}

# Args:
#   - LLVM_CONFIG_BIN: Path to llvm-config.
function find_llvm_prefix() {
    echo "$($1 --prefix)"
    return 0
}


LLVM_CONFIG_BIN=$(find_llvm_config)
find_llvm_prefix "$LLVM_CONFIG_BIN"
