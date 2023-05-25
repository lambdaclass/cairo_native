# Melior-next

[![GitHub Action](https://img.shields.io/github/actions/workflow/status/edg-l/melior-next/test.yaml?branch=main&style=flat-square)](https://github.com/edg-l/melior-next/actions?query=workflow%3Atest)
[![Crate](https://img.shields.io/crates/v/melior-next.svg?style=flat-square)](https://crates.io/crates/melior-next)
[![License](https://img.shields.io/github/license/edg-l/melior-next.svg?style=flat-square)](LICENSE)

The rustic MLIR bindings for Rust. Continued

This crate is a wrapper of [the MLIR C API](https://mlir.llvm.org/docs/CAPI/).

## Examples

### Building a function to add integers and executing it using the JIT engine.

```rust
use melior_next::{
    Context,
    dialect,
    ir::*,
    pass,
    utility::*,
    ExecutionEngine
};

let registry = dialect::Registry::new();
register_all_dialects(&registry);

let context = Context::new();
context.append_dialect_registry(&registry);
context.get_or_load_dialect("func");
register_all_llvm_translations(&context);

let location = Location::unknown(&context);
let mut module = Module::new(location);

let integer_type = Type::integer(&context, 64);

let function = {
    let region = Region::new();
    let block = Block::new(&[(integer_type, location), (integer_type, location)]);

    let sum = block.append_operation(
        operation::Builder::new("arith.addi", location)
            .add_operands(&[block.argument(0).unwrap().into(), block.argument(1).unwrap().into()])
            .add_results(&[integer_type])
            .build(),
    );

    block.append_operation(
        operation::Builder::new("func.return", Location::unknown(&context))
            .add_operands(&[sum.result(0).unwrap().into()])
            .build(),
    );

    region.append_block(block);

    operation::Builder::new("func.func", Location::unknown(&context))
        .add_attributes(
             &NamedAttribute::new_parsed_vec(&context, &[
                 ("function_type", "(i64, i64) -> i64"),
                 ("sym_name", "\"add\""),
                 ("llvm.emit_c_interface", "unit"),
             ]).unwrap()
         )
        .add_regions(vec![region])
        .build()
};

module.body().append_operation(function);

assert!(module.as_operation().verify());

let pass_manager = pass::Manager::new(&context);
register_all_passes();
pass_manager.add_pass(pass::conversion::convert_scf_to_cf());
pass_manager.add_pass(pass::conversion::convert_cf_to_llvm());
pass_manager.add_pass(pass::conversion::convert_func_to_llvm());
pass_manager.add_pass(pass::conversion::convert_arithmetic_to_llvm());
pass_manager.enable_verifier(true);
pass_manager.run(&mut module).unwrap();

let engine = ExecutionEngine::new(&module, 2, &[], false);

let mut argument1: i64 = 2;
let mut argument2: i64 = 4;
let mut result: i64 = -1;

unsafe {
    engine
        .invoke_packed(
                "add",
                &mut [
                    &mut argument1 as *mut i64 as *mut (),
                    &mut argument2 as *mut i64 as *mut (),
                    &mut result as *mut i64 as *mut ()
                ])
        .unwrap();
};

assert_eq!(result, 6);
```

## Goals

Melior aims to provide a simple, safe, and complete API for MLIR with a reasonably sane ownership model represented by the type system in Rust.

## Install

```sh
cargo add melior-next
```

### Dependencies

[LLVM/MLIR 16](https://llvm.org/) needs to be installed on your system. On Linux and macOS, you can install it via [Homebrew](https://brew.sh).

```sh
brew install llvm@16
```

## Documentation

On [GitHub Pages](https://edg-l.github.io/melior-next/melior_next).

## Contribution

Contribution is welcome! But, Melior is still in the alpha stage as well as the MLIR C API. Note that the API is unstable and can have breaking changes in the future.

### Technical notes

- We always use `&T` for MLIR objects instead of `&mut T` to mitigate the intricacy of representing a loose ownership model of the MLIR C API in Rust.
- Only UTF-8 is supported as string encoding.
  - Most string conversion between Rust and C is cached internally.

### Naming conventions

- `Mlir<X>` objects are named `<X>` if they have no destructor. Otherwise, they are named `<X>` for owned objects and `<X>Ref` for borrowed references.
- `mlir<X>Create` functions are renamed as `<X>::new`.
- `mlir<X>Get<Y>` functions are renamed as follows:
  - If the resulting objects refer to `&self`, they are named `<X>::as_<Y>`.
  - Otherwise, they are named just `<X>::<Y>` and may have arguments, such as position indices.

## References

- The overall design is inspired by [TheDan64/inkwell](https://github.com/TheDan64/inkwell).

## License

[Apache 2.0](LICENSE)

This is a fork of <https://github.com/raviqqe/melior>
