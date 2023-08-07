# Cairo Native Development

This is a guide to get you started into being a full fledge Cairo Native developer!

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
