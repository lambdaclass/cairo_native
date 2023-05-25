//! Melior-next is the rustic MLIR bindings for Rust. It aims to provide a simple,
//! safe, and complete API for MLIR with a reasonably sane ownership model
//! represented by the type system in Rust.
//!
//! This crate is a wrapper of [the MLIR C API](https://mlir.llvm.org/docs/CAPI/).
//!
//! # Dependencies
//!
//! [LLVM/MLIR 16](https://llvm.org/) needs to be installed on your system.
//!
//! # Safety
//!
//! Although Melior aims to be completely safe, some part of the current API is
//! not.
//!
//! - Access to operations, types, or attributes that belong to dialects not
//!   loaded in contexts can lead to runtime errors or segmentation faults in
//!   the worst case.
//!   - Fix plan: Load all dialects by default on creation of contexts, and
//!     provide unsafe constructors of contexts for advanced users.
//! - IR object references returned from functions that move ownership of
//!   arguments might get invalidated later.
//!   - This is because we need to borrow `&self` rather than `&mut self` to
//!     return such references.
//!   - e.g. `Region::append_block()`
//!   - Fix plan: Use dynamic check, such as `RefCell`, for the objects.
//!
//! # Examples
//!
//! ## Building a function to add integers and executing it using the JIT engine.
//!
//! ```rust
//! use melior_next::{
//!     Context,
//!     dialect::{self, arith},
//!     ir::*,
//!     pass,
//!     utility::*,
//!     ExecutionEngine
//! };
//!
//! let registry = dialect::Registry::new();
//! register_all_dialects(&registry);
//!
//! let context = Context::new();
//! context.append_dialect_registry(&registry);
//! context.get_or_load_dialect("func");
//! context.get_or_load_dialect("arith");
//! register_all_llvm_translations(&context);
//!
//! let location = Location::unknown(&context);
//! let mut module = Module::new(location);
//!
//! let integer_type = Type::integer(&context, 64);
//!
//! let function = {
//!     let region = Region::new();
//!     let block = Block::new(&[(integer_type, location), (integer_type, location)]);
//!     let arg1 = block.argument(0).unwrap().into();
//!     let arg2 = block.argument(1).unwrap().into();
//!
//!     let sum = block.append_operation(
//!         arith::addi(arg1, arg2, integer_type, location)
//!     );
//!
//!     block.append_operation(
//!         operation::Builder::new("func.return", Location::unknown(&context))
//!             .add_operands(&[sum.result(0).unwrap().into()])
//!             .build(),
//!     );
//!
//!     region.append_block(block);
//!
//!     operation::Builder::new("func.func", Location::unknown(&context))
//!         .add_attributes(
//!             &NamedAttribute::new_parsed_vec(&context, &[
//!                 ("function_type", "(i64, i64) -> i64"),
//!                 ("sym_name", "\"add\""),
//!                 ("llvm.emit_c_interface", "unit"),
//!             ]).unwrap()
//!         )
//!         .add_regions(vec![region])
//!         .build()
//! };
//!
//! module.body().append_operation(function);
//!
//! assert!(module.as_operation().verify());
//!
//! let pass_manager = pass::Manager::new(&context);
//! register_all_passes();
//! pass_manager.add_pass(pass::conversion::convert_scf_to_cf());
//! pass_manager.add_pass(pass::conversion::convert_cf_to_llvm());
//! pass_manager.add_pass(pass::conversion::convert_func_to_llvm());
//! pass_manager.add_pass(pass::conversion::convert_arithmetic_to_llvm());
//! pass_manager.enable_verifier(true);
//! pass_manager.run(&mut module).unwrap();
//!
//! let engine = ExecutionEngine::new(&module, 2, &[], false);
//!
//! let mut argument1: i64 = 2;
//! let mut argument2: i64 = 4;
//! let mut result: i64 = -1;
//!
//! unsafe {
//!     engine
//!         .invoke_packed(
//!                 "add",
//!                 &mut [
//!                     &mut argument1 as *mut i64 as *mut (),
//!                     &mut argument2 as *mut i64 as *mut (),
//!                     &mut result as *mut i64 as *mut ()
//!                 ])
//!         .unwrap();
//! };
//!
//! assert_eq!(result, 6);
//! ```

#![deny(clippy::nursery)]
//#![deny(clippy::pedantic)]
#![deny(clippy::all)]

mod context;
pub mod diagnostics;
pub mod dialect;
mod error;
mod execution_engine;
pub mod ir;
mod logical_result;
mod mlir_sys;
pub mod pass;
mod string_ref;
pub mod utility;

pub use self::{
    context::{Context, ContextRef},
    error::Error,
    execution_engine::ExecutionEngine,
    logical_result::LogicalResult,
    string_ref::StringRef,
};

#[cfg(test)]
mod tests {
    use crate::{
        context::Context,
        dialect,
        ir::{operation, Block, Location, Module, NamedAttribute, Region, Type},
        utility::register_all_dialects,
    };

    #[test]
    fn build_module() {
        let context = Context::new();
        let module = Module::new(Location::unknown(&context));

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }

    #[test]
    fn build_module_with_dialect() {
        let registry = dialect::Registry::new();
        let context = Context::new();
        context.append_dialect_registry(&registry);
        let module = Module::new(Location::unknown(&context));

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }

    #[test]
    fn build_add() {
        let registry = dialect::Registry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.get_or_load_dialect("func");

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let integer_type = Type::integer(&context, 64);

        let function = {
            let region = Region::new();
            let block = Block::new(&[(integer_type, location), (integer_type, location)]);

            let sum = block.append_operation(
                operation::Builder::new("arith.addi", location)
                    .add_operands(&[
                        block.argument(0).unwrap().into(),
                        block.argument(1).unwrap().into(),
                    ])
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
                    &NamedAttribute::new_parsed_vec(
                        &context,
                        &[("function_type", "(i64, i64) -> i64"), ("sym_name", "\"add\"")],
                    )
                    .unwrap(),
                )
                .add_regions(vec![region])
                .build()
        };

        module.body().append_operation(function);

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }

    #[test]
    fn build_sum() {
        let registry = dialect::Registry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.get_or_load_dialect("func");
        context.get_or_load_dialect("memref");
        context.get_or_load_dialect("scf");

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let memref_type = Type::parse(&context, "memref<?xf32>").unwrap();

        let function = {
            let function_region = Region::new();
            let function_block = Block::new(&[(memref_type, location), (memref_type, location)]);
            let index_type = Type::parse(&context, "index").unwrap();

            let zero = function_block.append_operation(
                operation::Builder::new("arith.constant", location)
                    .add_results(&[index_type])
                    .add_attributes(
                        &NamedAttribute::new_parsed_vec(&context, &[("value", "0 : index")])
                            .unwrap(),
                    )
                    .build(),
            );
            let dim = function_block.append_operation(
                operation::Builder::new("memref.dim", location)
                    .add_operands(&[
                        function_block.argument(0).unwrap().into(),
                        zero.result(0).unwrap().into(),
                    ])
                    .add_results(&[index_type])
                    .build(),
            );

            let loop_block = Block::new(&[]);
            loop_block.add_argument(index_type, location);

            let one = function_block.append_operation(
                operation::Builder::new("arith.constant", location)
                    .add_results(&[index_type])
                    .add_attributes(&[
                        NamedAttribute::new_parsed(&context, "value", "1 : index").unwrap()
                    ])
                    .build(),
            );

            {
                let f32_type = Type::parse(&context, "f32").unwrap();

                let lhs = loop_block.append_operation(
                    operation::Builder::new("memref.load", location)
                        .add_operands(&[
                            function_block.argument(0).unwrap().into(),
                            loop_block.argument(0).unwrap().into(),
                        ])
                        .add_results(&[f32_type])
                        .build(),
                );

                let rhs = loop_block.append_operation(
                    operation::Builder::new("memref.load", location)
                        .add_operands(&[
                            function_block.argument(1).unwrap().into(),
                            loop_block.argument(0).unwrap().into(),
                        ])
                        .add_results(&[f32_type])
                        .build(),
                );

                let add = loop_block.append_operation(
                    operation::Builder::new("arith.addf", location)
                        .add_operands(&[
                            lhs.result(0).unwrap().into(),
                            rhs.result(0).unwrap().into(),
                        ])
                        .add_results(&[f32_type])
                        .build(),
                );

                loop_block.append_operation(
                    operation::Builder::new("memref.store", location)
                        .add_operands(&[
                            add.result(0).unwrap().into(),
                            function_block.argument(0).unwrap().into(),
                            loop_block.argument(0).unwrap().into(),
                        ])
                        .build(),
                );

                loop_block.append_operation(operation::Builder::new("scf.yield", location).build());
            }

            function_block.append_operation(
                {
                    let loop_region = Region::new();

                    loop_region.append_block(loop_block);

                    operation::Builder::new("scf.for", location)
                        .add_operands(&[
                            zero.result(0).unwrap().into(),
                            dim.result(0).unwrap().into(),
                            one.result(0).unwrap().into(),
                        ])
                        .add_regions(vec![loop_region])
                }
                .build(),
            );

            function_block.append_operation(
                operation::Builder::new("func.return", Location::unknown(&context)).build(),
            );

            function_region.append_block(function_block);

            operation::Builder::new("func.func", Location::unknown(&context))
                .add_attributes(
                    &NamedAttribute::new_parsed_vec(
                        &context,
                        &[
                            ("function_type", "(memref<?xf32>, memref<?xf32>) -> ()"),
                            ("sym_name", "\"sum\""),
                        ],
                    )
                    .unwrap(),
                )
                .add_regions(vec![function_region])
                .build()
        };

        module.body().append_operation(function);

        assert!(module.as_operation().verify());
        insta::assert_display_snapshot!(module.as_operation());
    }
}
