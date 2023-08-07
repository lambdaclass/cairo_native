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

In `src/types.rs` we need to declare the type layout, for example the felt252 would have the layout returned by `get_integer_layout(252)`.
A type that doesn't have size would be `Layout::new::<()>()`, or if the type is a pointer like box: `Layout::new::<*mut ()>()`

When adding a type, we also need to add the **serialization** and **deserialization** functionality, so we can use with the JIT runner.

You can find this functionality under `src/values.rs` and `src/values/{typename}.rs`. As you can see, the project is quite organized if you have a feel of its layout.

Serialization is done using `Serde`, and each type provides a `deserialize` and `serialize` function. The inner workings of such functions can be a bit complex due to how the JIT runner works. You need to work with pointers and unsafe rust.

In `values.rs` we should also declare whether the type is complex under `is_complex` in the `ValueBuilder` trait implmentation.

> Complex types are always passed by pointer (both as params and return values) and require a stack allocation. Examples of complex values include structs and enums, but not felts since LLVM considers them integers.

#### Deserializing a type
When **deserializing** (a.k.a converting the inputs so the JIT runner accepts them), you are passed a bump allocator arena from `Bumpalo`, the general idea is to get the layout and size of the type, allocate it under the arena, get a pointer, and return it. Which will later be passed to the MLIR JIT runner. It is important the pointers passed are allocated by the arena and not Rust itself.

Then we need to hookup de `deserialize` method in `values.rs` `deserialize` method.

#### Serializing a type
When **serializing** a type, you will get a `ptr: NonNull<()>` (non null pointer), which you will have to cast, dereference and then deserialize.

For a simple type to learn how it works, we recommend checking `src/values/uint8.rs`, for more complex types, check `src/values/felt252.rs`. The hardest types to understand are the enums, dictionaries and arrays, since they are complex types.

Then we need to hookup de `serialize` method in `values.rs` `serialize` method.

#### Implementing the library function

Libfuncs are implemented under `src/libfuncs.rs` and `src/libfuncs/{libfunc_name}.rs`. Just like types.

Using the `src/libfuncs/felt252.rs` libfuncs as a aid:

```rust
/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Felt252Concrete,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        Felt252Concrete::BinaryOperation(info) => {
            build_binary_operation(context, registry, entry, location, helper, metadata, info)
        }
        Felt252Concrete::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        Felt252Concrete::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, metadata, info)
        }
    }
}
```

You can see it also defines a build function, in this case the last method is a selector, this means in this case we have a group of related
libfuncs, which we will implement in this same file. This is where calls to melior (MLIR) and most MLIR code is located, where one gets their hands dirty.

After implementing the libfuncs, we need to hookup the `build` method in the `src/libfuncs.rs` match statement.

#### Example libfunc implementation: u8_to_felt252

An example libfunc, converting a u8 to a felt252, extensively commented:

```rust
/// Generate MLIR operations for the `u8_to_felt252` libfunc.
pub fn build_to_felt252<'ctx, 'this, TType, TLibfunc>(
    // The Context from MLIR, this is like the heart of the MLIR API, its required to create most stuff like types.
    context: &'ctx Context,
    // This is the sierra program registry, it aids us at finding types, functions, etc.
    registry: &ProgramRegistry<TType, TLibfunc>,
    // This is the MLIR entry block for this libfunc. Remember we append operations to blocks.
    entry: &'this Block<'ctx>,
    // The already created MLIR location for this libfunc, we need to pass this to all the MLIR operations.
    location: Location<'ctx>,
    // A helper, which also works as a MLIR Module, it has useful functions for stuff like branching to other libfuncs.
    helper: &LibfuncHelper<'ctx, 'this>,
    // The metadata storage, contains extra information needed on some libfuncs. Check out `src/metadata.rs` to learn how it works.
    metadata: &mut MetadataStorage,
    // The sierra information for this specific library function. This libfunc only contains signature information, but
    // others which are generic over a type will contain information about that type, for example array related libfuncs.
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    // We retrieve the felt252 type from the registry and call the "build" method to create the MLIR type.
    // We could also just call get_type() to hold on to the sierra type, and then `.layout(registry)` to get the type layout,
    // which is needed in some libfuncs doing more complex stuff.
    let felt252_ty = registry
        .get_type(&info.branch_signatures()[0].vars[0].ty)?
        .build(context, helper, registry, metadata)?;

    // Retrieve the first argument passed to this library function, in this case its the u8 value we need to convert.
    let value: Value = entry.argument(0)?.into();

    // We create a "extui" operation from the "arith" dialect, which basically zero extends the value to have the same bits as the given type.
    let op = entry.append_operation(arith::extui(value, felt252_ty, location));

    // Get  the result from the operation, in this case it's the extended value
    let result = op.result(0)?.into();

    // Using the helper argument, append the branching operation to the next statement, passing result as our output variable.
    entry.append_operation(helper.br(0, &[result], location));

    Ok(())
}
```

More info on the `extui` operation: <https://mlir.llvm.org/docs/Dialects/ArithOps/#arithextui-arithextuiop>

## How MLIR Works

MLIR is composed of **dialects**, which is like a IR of it's own, and this IR can be converted to another dialect IR (if the functionality exists). This is what makes MLIR shine.

Some commonly used dialects in this project:

#### The arith dialect

It contains arithmetic operations, such as `addi`, `subi` for addition and substraction.

#### The cf dialect

It contains basic control flow operations, such as the `br` and `cond_br`, which are unconditional and conditional jumps.


### The IR

The MLIR IR is composed recursively like this:

`Operation -> Region -> Block -> Operations`

Each operation has 1 or more region, each region has 1 or more blocks, each block has 1 or more operations.

This way a MLIR program can be composed.

### Transformations and passes

MLIR provides a set of transformations that can optimize the IR. Such as `canonicalize`.

Check out <https://mlir.llvm.org/docs/Canonicalization/> and <https://mlir.llvm.org/docs/Passes/>.

### Translating

In our case, llvm is our target, so we end up translating all dialects down to the LLVM dialect, which then gets converted to LLVM IR.
