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

- [Getting Started](#getting-started)
- [Included Tools](#included-tools)
  - [Scripts](#scripts)
  - [cairo-native-compile](#cairo-native-compile)
  - [cairo-native-dump](#cairo-native-dump)
  - [cairo-native-run](#cairo-native-run)
  - [cairo-native-test](#cairo-native-test)
  - [cairo-native-stress](#cairo-native-stress)
  - [scarb-native-dump](#scarb-native-dump)
  - [scarb-native-test](#scarb-native-test)
- [Benchmarking](#benchmarking)

For in-depth documentation, see the [developer documentation][].

## Disclaimer
🚧 Cairo Native is still being built therefore API breaking changes might happen
often so use it at your own risk. 🚧

For versions under `1.0` `cargo` doesn't comply with
[semver](https://semver.org/), so we advise to pin the version the version you
use. This can be done by adding `cairo-native = "0.1.0"` to your Cargo.toml

## Getting Started

### Dependencies
- Linux or macOS (aarch64 included) only for now
- LLVM 19 with MLIR: On debian you can use [apt.llvm.org](https://apt.llvm.org/),
  on macOS you can use brew
- Rust 1.78.0 or later, since we make use of the u128
  [abi change](https://blog.rust-lang.org/2024/03/30/i128-layout-update.html).
- Git

### Setup
> This step applies to all operating systems.

Run the following make target to install the dependencies (**both Linux and macOS**):

```bash
make deps
```

#### Linux
Since Linux distributions change widely, you need to install LLVM 19 via your
package manager, compile it or check if the current release has a Linux binary.

If you are on Debian/Ubuntu, check out the repository https://apt.llvm.org/
Then you can install with:

```bash
sudo apt-get install llvm-19 llvm-19-dev llvm-19-runtime clang-19 clang-tools-19 lld-19 libpolly-19-dev libmlir-19-dev mlir-19-tools
```

If you decide to build from source, here are some indications:

<details><summary>Install LLVM from source instructions</summary>

```bash
# Go to https://github.com/llvm/llvm-project/releases
# Download the latest LLVM 19 release:
# The blob to download is called llvm-project-19.x.x.src.tar.xz

# For example
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-19.1.0/llvm-project-19.1.0.src.tar.xz
tar xf llvm-project-19.1.0.src.tar.xz

cd llvm-project-19.1.0.src.tar
mkdir build
cd build

# The following cmake command configures the build to be installed to /opt/llvm-19
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang;clang-tools-extra;lld;polly" \
   -DLLVM_BUILD_EXAMPLES=OFF \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_INSTALL_PREFIX=/opt/llvm-19 \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DLLVM_PARALLEL_LINK_JOBS=4 \
   -DLLVM_ENABLE_BINDINGS=OFF \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
   -DLLVM_ENABLE_ASSERTIONS=OFF

ninja install
```

</details>

Setup a environment variable called `MLIR_SYS_190_PREFIX`, `LLVM_SYS_191_PREFIX`
and `TABLEGEN_190_PREFIX` pointing to the llvm directory:

```bash
# For Debian/Ubuntu using the repository, the path will be /usr/lib/llvm-19
export MLIR_SYS_190_PREFIX=/usr/lib/llvm-19
export LLVM_SYS_191_PREFIX=/usr/lib/llvm-19
export TABLEGEN_190_PREFIX=/usr/lib/llvm-19
```

Alternatively, if installed from Debian/Ubuntu repository, then you can use
`env.sh` to automatically setup the environment variables.

```bash
source env.sh
```

#### MacOS
The makefile `deps` target (which you should have ran before) installs LLVM 19
with brew for you, afterwards you need to execute the `env.sh` script to setup
the needed environment variables.

```bash
source env.sh
```

### Make targets:
Running `make` by itself will check whether the required LLVM installation and
corelib is found, and then list available targets.

```bash
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

## Included Tools
Aside from the compilation and execution engine library, Cairo Native includes
a few command-line tools to aid development, and some useful scripts.

These are:
- The contents of the `/scripts/` folder
- `cairo-native-compile`
- `cairo-native-dump`
- `cairo-native-run`
- `cairo-native-test`
- `cairo-native-stress`
- `scarb-native-dump`
- `scarb-native-test`

### `cairo-native-compile`
```bash
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
```bash
Usage: cairo-native-dump [OPTIONS] <INPUT>

Arguments:
  <INPUT>

Options:
  -o, --output <OUTPUT>  [default: -]
      --starknet         Compile a starknet contract
  -h, --help             Print help
```

### `cairo-native-run`
This tool allows to run programs using the JIT engine, like the `cairo-run`
tool, the parameters can only be felt values.

Example: `echo '1' | cairo-native-run 'program.cairo' 'program::program::main' --inputs - --outputs -`

```bash
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

### `cairo-native-test`
This tool mimics the `cairo-test`
[tool](https://github.com/starkware-libs/cairo/tree/main/crates/cairo-lang-test-runner)
and is identical to it in interface, the only feature it doesn't have is the profiler.

```bash
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

For single files, you can use the `-s, --single-file` option.

For a project, it needs to have a `cairo_project.toml` specifying the
`crate_roots`. You can find an example under the `cairo-tests/` folder, which
is a cairo project that works with this tool.

```bash
cairo-native-test -s myfile.cairo

cairo-native-test ./cairo-tests/
```

This will run all the tests (functions marked with the `#[test]` attribute).

### `cairo-native-stress`
This tool runs a stress test on Cairo Native.

```bash
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

To quickly run a stress test and save logs as json, run:
```bash
make stress-test
```

This takes a lot of time to finish (it will probably crash first), you can kill
the program at any time.

To plot the results, run:
```bash
make stress-plot
```

To clear the cache directory, run:
```bash
make stress-clean
```

### `scarb-native-dump`
This tool mimics the `scarb build` [command](https://github.com/software-mansion/scarb/tree/main/extensions/scarb-cairo-test).
You can download it on our [releases](https://github.com/lambdaclass/cairo_native/releases) page.

This tool should be run at the directory where a `Scarb.toml` file is and it will
behave like `scarb build`, leaving the MLIR files under the `target/` folder
besides the generated JSON sierra files.

### `scarb-native-test`
This tool mimics the `scarb test` [command](https://github.com/software-mansion/scarb/tree/main/extensions/scarb-cairo-test).
You can download it on our [releases](https://github.com/lambdaclass/cairo_native/releases) page.

```bash
Compiles all packages from a Scarb project matching `packages_filter` and
runs all functions marked with `#[test]`. Exits with 1 if the compilation
or run fails, otherwise 0.

Usage: scarb-native-test [OPTIONS]

Options:
  -p, --package <SPEC>         Packages to run this command on, can be a concrete package name (`foobar`) or a prefix glob (`foo*`) [env: SCARB_PACKAGES_FILTER=] [default: *]
  -w, --workspace              Run for all packages in the workspace
  -f, --filter <FILTER>        Run only tests whose name contain FILTER [default: ]
      --include-ignored        Run ignored and not ignored tests
      --ignored                Run only ignored tests
      --run-mode <RUN_MODE>    Run with JIT or AOT (compiled) [default: jit] [possible values: aot, jit]
  -O, --opt-level <OPT_LEVEL>  Optimization level, Valid: 0, 1, 2, 3. Values higher than 3 are considered as 3 [default: 0]
  -h, --help                   Print help
  -V, --version                Print version
```

## Benchmarking

### Requirements
- [hyperfine](https://github.com/sharkdp/hyperfine): `cargo install hyperfine`
- [cairo 2.8.2](https://github.com/starkware-libs/cairo)
- Cairo Corelibs
- LLVM 19 with MLIR

You need to setup some environment variables:

```bash
$MLIR_SYS_190_PREFIX=/path/to/llvm19  # Required for non-standard LLVM install locations.
$LLVM_SYS_191_PREFIX=/path/to/llvm19  # Required for non-standard LLVM install locations.
$TABLEGEN_190_PREFIX=/path/to/llvm19  # Required for non-standard LLVM install locations.
```

You can then run the `bench` makefile target:

```bash
make bench
```

The `bench` target will run the `./scripts/bench-hyperfine.sh` script.
This script runs hyperfine commands to compare the execution time of programs in the `./programs/benches/` folder.
Each program is compiled and executed via the execution engine with the `cairo-native-run` command and via the cairo-vm with the `cairo-run` command provided by the `cairo` codebase.
The `cairo-run` command should be available in the `$PATH` and ideally compiled with `cargo build --release`.
If you want the benchmarks to run using a specific build, or the `cairo-run` commands conflicts with something (e.g. the cairo-svg package binaries in macos) then the command to run `cairo-run` with a full path can be specified with the `$CAIRO_RUN` environment variable.

[developer documentation]: https://lambdaclass.github.io/cairo_native/cairo_native/docs/index.html
