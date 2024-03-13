#!/usr/bin/env bash


mlir-opt-17 --test-lower-to-llvm sample.mlir | mlir-translate-17 --mlir-to-llvmir | lli-17 --entry-function sample::sample::run_test
