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

### Project dependencies

The major dependencies of the project are the following:

- Melior: This is the crate that abstracts away most of the interfacing with MLIR,
our compilation target, it uses mlir-sys and tries to safely abstract MLIR in Rust.
- Cairo: We use the cairo crates to keep a close tie to the API contracts of the language, they provide a really nice way to know what features the language has and aids with codegen. For example, most library functions are under enumerations, and thanks to Rust exhaustive pattern matching we can't miss any.
- Runtime: The JIT runner and compiler depend on a "runtime" that lives on this repository too, it aids with more complex stuff like `pedersen`, `keccak` and dictionaries that would be quite complex to implement from the ground up in MLIR (Basically would be like coding a complex hash function in pseudo assembly).

### Build script

We have a build script to cover a small missing functionality from `melior`, it's quite simple and the compiled cpp code is under `src/ffi.cpp`.

### General flow

If you check `lib.rs` you will see the most high level modules of the project.

The compiler module is what glues together everything. You should read its module level documentation.
But the basic flow is like this:

- We take a sierra `Program` and iterate over its functions.
- On each function, we create a MLIR region and a block for each statement (a.k.a library function call), taking into account possible branches.
- On each statement we call the library function implementation, which appends MLIR code to the given block, and with helper methods, it handles possible branches and input/output variables.


### What is a library function

Sierra uses a list of builtin functions that implement the language functionality, those are called library functions, short: libfuncs. Basically every statement in a sierra program is a call to a libfunc, thus they are the core of Cairo Native.

Each libfunc takes input variables and outputs some other variables. Note that in cairo a function that has 2 arguments may have more in sierra, due to "implicits" / "builtins", which are arguments passed hidden from the user, such as the `GasBuiltin`.

### A libfunc implementation.

A libfunc usually works with a `type`, such as `felt252`. The compiler needs to have information on this type, such as its layout and size.
This is defined in `src/types.rs` and `src/types/{typename}.rs`.

On each `src/types/{typename}.rs` such as ``src/types/felt252.rs` you will find a `build` function, this has all the necessary arguments
to generate the proper type and return a MLIR type, such as `IntegerType::new(context, 252)` (a 252 bit integer).

When adding a type, we also need to add the **serialization** and **deserialization** functionality, so we can use with the JIT runner.

You can find this functionality under `src/values.rs` and `src/values/{typename}.rs`. As you can see, the project is quite organized if you have a feel of its layout.

Serialization is done using `Serde`, and each type provides a `deserialize` and `serialize` function. The inner workings of such functions can be a bit complex due to how the JIT runner works. You need to work with pointers and unsafe rust.

#### Deserializing a type
When **deserializing** (a.k.a converting the inputs so the JIT runner accepts them), you are passed a bump allocator arena from `Bumpalo`, the general idea is to get the layout and size of the type, allocate it under the arena, get a pointer, and return it. Which will later be passed to the MLIR JIT runner. It is important the pointers passed are allocated by the arena and not Rust itself.

#### Serializing a type
When **serializing** a type, you will get a `ptr: NonNull<()>` (non null pointer), which you will have to cast, dereference and then deserialize.

For a simple type to learn how it works, we recommend checking `src/values/uint8.rs`, for more complex types, check `src/values/felt252.rs`. The hardest types to understand are the enums, dictionaries and arrays, since they are complex types.
