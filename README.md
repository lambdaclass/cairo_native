<div align="center">

### ⚡ Cairo Native ⚡

A compiler to convert Cairo's intermediate representation "Sierra" code <br>
to machine code via MLIR and LLVM.

[Report Bug](https://github.com/lambdaclass/cairo_native/issues/new) · [Request Feature](https://github.com/lambdaclass/cairo_native/issues/new)

[![Telegram Chat][tg-badge]][tg-url]
[![rust](https://github.com/lambdaclass/cairo_native/actions/workflows/ci.yml/badge.svg)](https://github.com/lambdaclass/cairo_native/actions/workflows/ci.yml)
[![codecov](https://img.shields.io/codecov/c/github/lambdaclass/cairo_native)](https://codecov.io/gh/lambdaclass/cairo_native)
[![license](https://img.shields.io/github/license/lambdaclass/cairo_native)](/LICENSE)
[![pr-welcome]](#-contributing)

[tg-badge]: https://img.shields.io/endpoint?url=https%3A%2F%2Ftg.sumanjay.workers.dev%2FLambdaStarkNet%2F&logo=telegram&label=chat&color=neon
[tg-url]: https://t.me/LambdaStarkNet
[pr-welcome]: https://img.shields.io/static/v1?color=orange&label=PRs&style=flat&message=welcome

</div>

For in-depth documentation, see the [DeveloperDocumentation][].

## ⚠️ Disclaimer
🚧 `cairo-native` is still being built therefore API breaking changes might happen often so use it at your own risk. 🚧

For versions under `1.0` `cargo` doesn't comply with [semver](https://semver.org/), so we advise to pin the version the version you use. This can be done by adding `cairo-native = "0.1.0"` to your Cargo.toml

## Getting Started

### Dependencies
- Linux or macOS (aarch64 included) only for now
- LLVM 18 with MLIR: On debian you can use [apt.llvm.org](https://apt.llvm.org/), on macOS you can use brew
- Rust 1.78.0 or later, since we make use of the u128 [abi change](https://blog.rust-lang.org/2024/03/30/i128-layout-update.html).
- Git

### Setup
> This step applies to all operating systems.

Run the following make target to install the dependencies (**both Linux and macOS**):

```bash
make deps
```

#### Linux
Since Linux distributions change widely, you need to install LLVM 18 via your package manager, compile it or check if the current release has a Linux binary.

If you are on Debian/Ubuntu, check out the repository https://apt.llvm.org/
Then you can install with:

```bash
sudo apt-get install llvm-18 llvm-18-dev llvm-18-runtime clang-18 clang-tools-18 lld-18 libpolly-18-dev libmlir-18-dev mlir-18-tools
```

If you decide to build from source, here are some indications:

<details><summary>Install LLVM from source instructions</summary>

```bash
# Go to https://github.com/llvm/llvm-project/releases
# Download the latest LLVM 18 release:
# The blob to download is called llvm-project-18.x.x.src.tar.xz

# For example
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.7/llvm-project-18.1.7.src.tar.xz
tar xf llvm-project-18.1.7.src.tar.xz

cd llvm-project-18.1.7.src.tar
mkdir build
cd build

# The following cmake command configures the build to be installed to /opt/llvm-18
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang;clang-tools-extra;lld;polly" \
   -DLLVM_BUILD_EXAMPLES=OFF \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_INSTALL_PREFIX=/opt/llvm-18 \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DLLVM_PARALLEL_LINK_JOBS=4 \
   -DLLVM_ENABLE_BINDINGS=OFF \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
   -DLLVM_ENABLE_ASSERTIONS=OFF

ninja install
```

</details>

Setup a environment variable called `MLIR_SYS_180_PREFIX`, `LLVM_SYS_181_PREFIX` and `TABLEGEN_180_PREFIX` pointing to the llvm directory:

```bash
# For Debian/Ubuntu using the repository, the path will be /usr/lib/llvm-18
export MLIR_SYS_180_PREFIX=/usr/lib/llvm-18
export LLVM_SYS_181_PREFIX=/usr/lib/llvm-18
export TABLEGEN_180_PREFIX=/usr/lib/llvm-18
```

Alternatively, if installed from Debian/Ubuntu repository, then you can use `env.sh` to automatically setup the environment variables.

```bash
source env.sh
```

#### MacOS
The makefile `deps` target (which you should have ran before) installs LLVM 18 with brew for you, afterwards you need to execute the `env.sh` script to setup the needed environment variables.

```bash
source env.sh
```

### Make commands:
Running `make` by itself will check whether the required LLVM installation and corelib is found, and then list available targets.

```
% make
LLVM is correctly set at /opt/homebrew/opt/llvm.
./scripts/check-corelib-version.sh 2.6.4
Usage:
    deps:         Installs the necesary dependencies.
    build:        Builds the cairo-native library and binaries in release mode.
    build-native: Builds cairo-native with the target-cpu=native rust flag.
    build-dev:    Builds cairo-native under a development-optimized profile.
    runtime:      Builds the runtime library required for AOT compilation.
    check:        Checks format and lints.
    test:         Runs all tests.
    proptest:     Runs property tests.
    coverage:     Runs all tests and computes test coverage.
    doc:          Builds documentation.
    doc-open:     Builds and opens documentation in browser.
    bench:        Runs the hyperfine benchmark script.
    bench-ci:     Runs the criterion benchmarks for CI.
    install:      Invokes cargo to install the cairo-native tools.
    clean:        Cleans the built artifacts.
    stress-test   Runs a command which runs stress tests.
    stress-plot   Plots the results of the stress test command.
    stress-clean  Clean the cache of AOT compiled code of the stress test command.
```

## Tooling Command Line Interface

Aside from the compilation and execution engine library, Cairo Native includes a few command-line tools to aid development.
These are:
- `cairo-native-compile`
- `cairo-native-dump`
- `cairo-native-run`
- `cairo-native-stress`
- `cairo-native-test`

### `cairo-native-compile`
```
Compiles a Cairo project outputting the generated MLIR and the shared library.
Exits with 1 if the compilation or run fails, otherwise 0.

Usage: cairo-native-compile [OPTIONS] <PATH> [OUTPUT_MLIR] [OUTPUT_LIBRARY]

Arguments:
  <PATH>            The Cairo project path to compile and run its tests
  [OUTPUT_MLIR]     The output path for the mlir, if none is passed, out.mlir will be the default
  [OUTPUT_LIBRARY]  If a path is passed, a dynamic library will be compiled and saved at that path

Options:
  -s, --single-file            Whether path is a single file
      --allow-warnings         Allows the compilation to succeed with warnings
  -r, --replace-ids            Replaces sierra ids with human-readable ones
  -O, --opt-level <OPT_LEVEL>  Optimization level, Valid: 0, 1, 2, 3. Values higher than 3 are considered as 3 [default: 0]
  -h, --help                   Print help
  -V, --version                Print version
```

###  `cairo-native-dump`
```
Usage: cairo-native-dump [OPTIONS] <INPUT>

Arguments:
  <INPUT>

Options:
  -o, --output <OUTPUT>  [default: -]
      --starknet         Compile a starknet contract
  -h, --help             Print help
```

### `cairo-native-run`
This tool allows to run programs using the JIT engine, like the `cairo-run` tool, the parameters can only be felt values.

Example: `echo '1' | cairo-native-run 'program.cairo' 'program::program::main' --inputs - --outputs -`

```
Exits with 1 if the compilation or run fails, otherwise 0.

Usage: cairo-native-run [OPTIONS] <PATH>

Arguments:
  <PATH>  The Cairo project path to compile and run its tests

Options:
  -s, --single-file                    Whether path is a single file
      --allow-warnings                 Allows the compilation to succeed with warnings
      --available-gas <AVAILABLE_GAS>  In cases where gas is available, the amount of provided gas
      --run-mode <RUN_MODE>            Run with JIT or AOT (compiled) [default: jit] [possible values: aot, jit]
  -O, --opt-level <OPT_LEVEL>          Optimization level, Valid: 0, 1, 2, 3. Values higher than 3 are considered as 3 [default: 0]
  -h, --help                           Print help
  -V, --version                        Print version
```

### `cairo-native-stress`
```
A stress tester for Cairo Native

It compiles Sierra programs with Cairo Native, caches, and executes them with AOT runner. The compiled dynamic libraries are stored in `AOT_CACHE_DIR` relative to the current working directory.

Usage: cairo-native-stress [OPTIONS] <ROUNDS>

Arguments:
  <ROUNDS>
          Amount of rounds to execute

Options:
  -o, --output <OUTPUT>
          Output file for JSON formatted logs

  -h, --help
          Print help (see a summary with '-h')
```

### `cairo-native-test`
```
Compiles a Cairo project and runs all the functions marked as `#[test]`.
Exits with 1 if the compilation or run fails, otherwise 0.

Usage: cairo-native-test [OPTIONS] <PATH>

Arguments:
  <PATH>  The Cairo project path to compile and run its tests

Options:
  -s, --single-file            Whether path is a single file
      --allow-warnings         Allows the compilation to succeed with warnings
  -f, --filter <FILTER>        The filter for the tests, running only tests containing the filter string [default: ]
      --include-ignored        Should we run ignored tests as well
      --ignored                Should we run only the ignored tests
      --starknet               Should we add the starknet plugin to run the tests
      --run-mode <RUN_MODE>    Run with JIT or AOT (compiled) [default: jit] [possible values: aot, jit]
  -O, --opt-level <OPT_LEVEL>  Optimization level, Valid: 0, 1, 2, 3. Values higher than 3 are considered as 3 [default: 0]
  -h, --help                   Print help
  -V, --version                Print version
```

[DeveloperDocumentation]: https://lambdaclass.github.io/cairo_native/cairo_native/docs/index.html
