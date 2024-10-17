# Implementing Libfuncs

🚧 WIP

## A libfunc implementation.

A libfunc usually works with a `type`, such as `felt252`. The compiler
needs to have information on this type, such as its layout and size.
This is defined in `src/types.rs` and `src/types/{typename}.rs`.

On each `src/types/{typename}.rs` such as `src/types/felt252.rs` you will
find a `build` function, this has all the necessary arguments to generate
the proper type and return a MLIR type, such as
`IntegerType::new(context, 252)` (a 252 bit integer).

In `src/types.rs` we need to declare the type layout, for example the
`felt252` would have the layout returned by `get_integer_layout(252)`.
A type that doesn't have size would be `Layout::new::<()>()`, or if the
type is a pointer like box: `Layout::new::<*mut ()>()`

When adding a type, we also need to add the **serialization** and
**deserialization** functionality to convert the value to a memory representation that works with cairo-native.

You can find this functionality under `src/values.rs`.

There is a `Value` enum with all the possible values that can be passed as input/output.

This enum has a `impl` block with 2 important methods `to_ptr` and `from_ptr` which convert the Value into and from said
memory representation, held behind a pointer.

When passing the values as inputs, there is one more required step, that is to pass the bytes of those values in the
target platform ABI compatible way, this is done with the `AbiArgument` trait and the `to_bytes` method.

This trait, located in `src/arch.rs` is implemented currently for aarch64 and x86_64 (depending on the host platform) for some basic types, such as u64, u128, pointers, etc. Most importantly, it is also implemented for `ValueWithInfoWrapper` which allows to convert the Value using it's `to_ptr` method and correctly passing it as an argument in the given `buffer: &mut Vec<u8>`.

In `types.rs` we should also declare whether the type is complex under
`is_complex`, whether its a builtin in `is_builtin`, a zst in `is_zst` and define it's layout in the `TypeBuilder` trait implementation.

> Complex types are always passed by pointer (both as params and return
> values) and require a stack allocation. Examples of complex values include
> structs and enums, but not felts since LLVM considers them integers.

### Deserializing a type
When **deserializing** (a.k.a converting the inputs so the runner
accepts them), you are passed a bump allocator arena from `Bumpalo`, the
general idea is to get the layout and size of the type, allocate it under
the arena, get a pointer, and return it (Some types require additional data on the heap, such as arrays, such allocations should be done with libc's malloc.). Which will later be passed to the runner. It is important the pointers passed are allocated by the
arena and not Rust itself.

This is done in the `to_ptr` method.

### Serializing a type
When **serializing** a type, you will get a `ptr: NonNull<()>` (non null
pointer), which you will have to cast, dereference and then deserialize.

For a simple type to learn how it works, we recommend checking
`src/values.rs`, in the `from_ptr` method, look the u8 type in the match, for more complex types, check the felt252 type.
The hardest types to understand are the enums, dictionaries and arrays,
since they are complex types.

### Implementing the library function
Libfuncs are implemented under `src/libfuncs.rs` and
`src/libfuncs/{libfunc_name}.rs`. Just like types.

Using the `src/libfuncs/felt252.rs` libfuncs as a aid:

```rust,ignore
/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Felt252Concrete,
) -> Result<()> {
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

You can see it also defines a build function, in this case the last method
is a selector, this means in this case we have a group of related libfuncs,
which we will implement in this same file. This is where calls to melior
(MLIR) and most MLIR code is located, where one gets their hands dirty.

After implementing the libfuncs, we need to hookup the `build` method in
the `src/libfuncs.rs` match statement.

### Example libfunc implementation: u8_to_felt252
An example libfunc, converting a u8 to a felt252, extensively commented:

```rust,ignore
/// Generate MLIR operations for the `u8_to_felt252` libfunc.
pub fn build_to_felt252<'ctx, 'this>(
    // The Context from MLIR, this is like the heart of the MLIR API, its required to create most stuff like types.
    context: &'ctx Context,
    // This is the sierra program registry, it aids us at finding types, functions, etc.
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
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
) -> Result<()> {
    // We retrieve the felt252 type from the registry and call the "build" method to create the MLIR type.
    // We could also just call get_type() to hold on to the sierra type, and then `.layout(registry)` to get the type layout,
    // which is needed in some libfuncs doing more complex stuff.
    let felt252_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;

    // Retrieve the first argument passed to this library function, in this case its the u8 value we need to convert.
    let value: Value = entry.argument(0)?.into();

    // We create a "extui" operation from the "arith" dialect, which basically
    // zero extends the value to have the same bits as the given type.
    let result = entry.append_op_result(arith::extui(value, felt252_ty, location))?;

    // Using the helper argument, append the branching operation to the next statement, passing result as our output variable.
    entry.append_operation(helper.br(0, &[result], location));

    Ok(())
}
```

More info on the `extui` operation: <https://mlir.llvm.org/docs/Dialects/ArithOps/#arithextui-arithextuiop>
