# Cairo Native Development Guide
This is a guide to get you started into being a Cairo Native developer!
Here you will learn about the code layout, MLIR and more.

## Getting started
First make sure you have a working environment and are able to compile the project without issues. Make sure to follow the [setup](/README.md#setup) guide on steps on how to do this.

It is generally recommended to use the `optimized-dev` cargo profile when testing or running programs, the make target `make build-dev` will be useful for this.

To aid with development, there are 2 scripts that invoke cargo for you:

```bash
# Invokes the jit runner with the given program, entry point and json input.
./scripts/run-jit-dev.sh <program.cairo> <entry point> '[json input]'

# Example invocation of run-jit-dev.sh
./scripts/run-jit-dev.sh programs/print.cairo print::print::main '[]'

# Dumps the generated MLIR of a given cairo program
./scripts/compile-dev.sh <program.cairo>
```

### Other tools
It is also recommended you have `cairo-compile` and `cairo-run` installed to check how
the generated sierra code looks like, and to compare results manually (when required) which will help greatly when implementing functionality into Cairo Native.

You can check the [cairo](https://github.com/starkware-libs/cairo) repository for more info on how to get those tools.

## Basic Workflow
After having implemented your desired feature or bug fix, you should check it passes all tests and lints, also make sure to add any needed test cases for the added code.

```bash
# Check it passes all lints
make check

# Check it passes all tests
make test
```

Then you are free to go and make a PR!

## High level project overview
This will explain how the project is structured, without going into much details yet:

### Build script
We have a build script to cover a small missing functionality from `melior`, it's quite simple and the compiled cpp code is under `src/ffi.cpp`.

---
/// ## Compiling from Sierra to MLIR to a standalone native binary
///
/// ```bash
/// # to mlir with llvm dialect
/// sierra2mlir program.sierra -o program.mlir
///
/// # translate all dialects to the llvm dialect
/// "$MLIR_SYS_180_PREFIX/bin/mlir-opt" \
///         --canonicalize \
///         --convert-scf-to-cf \
///         --canonicalize \
///         --cse \
///         --expand-strided-metadata \
///         --finalize-memref-to-llvm \
///         --convert-func-to-llvm \
///         --convert-index-to-llvm \
///         --reconcile-unrealized-casts \
///         "program.mlir" \
///         -o "program-llvm.mlir"
///
/// # translate mlir to llvm-ir
/// "$MLIR_SYS_180_PREFIX"/bin/mlir-translate --mlir-to-llvmir program-llvm.mlir -o program.ll
///
/// # compile natively
/// "$MLIR_SYS_180_PREFIX"/bin/clang program.ll -Wno-override-module \
///     -L "$MLIR_SYS_180_PREFIX"/lib -L"./target/release/" \
///     -lcairo_native_runtime -lmlir_c_runner_utils \
///     -Wl,-rpath "$MLIR_SYS_180_PREFIX"/lib \
///     -Wl,-rpath ./target/release/ \
///     -o program
///
/// ./program
/// ```

---
/// # Ahead of time Compilation Remarks
///
/// To use the AOT executor, it needs to know where the static runtime library is located, this can be configured using the following
/// env var pointing to the absolute path to the library:
///
/// ```bash
/// export CAIRO_NATIVE_RUNTIME_LIBRARY=/absolute/path/to/libcairo_native_runtime.a
/// ```
