//! # Box libfuncs
//!
//! A heap allocated value, which is internally a pointer that can't be null.

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::{realloc_bindings::ReallocBindingsMeta, MetadataStorage},
    types::TypeBuilder,
};
use cairo_lang_sierra::{
    extensions::{
        boxing::BoxConcreteLibfunc,
        core::{CoreLibfunc, CoreType},
        lib_func::{
            BranchSignature, LibfuncSignature, SignatureAndTypeConcreteLibfunc,
            SignatureOnlyConcreteLibfunc,
        },
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith,
        llvm::{self, r#type::pointer, LoadStoreOptions},
        ods,
    },
    ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, Location},
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &BoxConcreteLibfunc,
) -> Result<()> {
    match selector {
        BoxConcreteLibfunc::Into(info) => {
            build_into_box(context, registry, entry, location, helper, metadata, info)
        }
        BoxConcreteLibfunc::Unbox(info) => {
            build_unbox(context, registry, entry, location, helper, metadata, info)
        }
        BoxConcreteLibfunc::ForwardSnapshot(info) => {
            build_forward_snapshot(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `into_box` libfunc.
pub fn build_into_box<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }

    let inner_type = registry.get_type(&info.ty)?;
    let inner_layout = inner_type.layout(registry)?;

    let value_len = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(
                IntegerType::new(context, 64).into(),
                inner_layout.pad_to_align().size().try_into()?,
            )
            .into(),
            location,
        ))
        .result(0)?
        .into();

    let ptr = entry
        .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
        .result(0)?
        .into();
    let ptr = entry
        .append_operation(ReallocBindingsMeta::realloc(
            context, ptr, value_len, location,
        ))
        .result(0)?
        .into();

    entry.append_operation(llvm::store(
        context,
        entry.argument(0)?.into(),
        ptr,
        location,
        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
            IntegerType::new(context, 64).into(),
            inner_layout.align() as i64,
        ))),
    ));

    entry.append_operation(helper.br(0, &[ptr], location));
    Ok(())
}

/// Generate MLIR operations for the `unbox` libfunc.
pub fn build_unbox<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let inner_type = registry.get_type(&info.ty)?;
    let inner_ty = inner_type.build(context, helper, registry, metadata, &info.ty)?;
    let inner_layout = inner_type.layout(registry)?;

    // Load the boxed value from memory.
    let value = entry
        .append_operation(llvm::load(
            context,
            entry.argument(0)?.into(),
            inner_ty,
            location,
            LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                IntegerType::new(context, 64).into(),
                inner_layout.align() as i64,
            ))),
        ))
        .result(0)?
        .into();

    entry.append_operation(ReallocBindingsMeta::free(
        context,
        entry.argument(0)?.into(),
        location,
    ));

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

fn build_forward_snapshot<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    super::snapshot_take::build(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        &SignatureOnlyConcreteLibfunc {
            signature: LibfuncSignature {
                param_signatures: info.signature.param_signatures.clone(),
                branch_signatures: info
                    .signature
                    .branch_signatures
                    .iter()
                    .map(|x| BranchSignature {
                        vars: x.vars.clone(),
                        ap_change: x.ap_change.clone(),
                    })
                    .collect(),
                fallthrough: info.signature.fallthrough,
            },
        },
    )
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{load_cairo, run_program_assert_output},
        values::JitValue,
    };

    #[test]
    fn run_box_unbox() {
        let program = load_cairo! {
            use box::BoxTrait;
            use box::BoxImpl;

            fn run_test() -> u32 {
                let x: u32 = 2_u32;
                let box_x: Box<u32> = BoxTrait::new(x);
                box_x.unbox()
            }
        };

        run_program_assert_output(&program, "run_test", &[], JitValue::Uint32(2));
    }

    #[test]
    fn run_box() {
        let program = load_cairo! {
            use box::BoxTrait;
            use box::BoxImpl;

            fn run_test() -> Box<u32>  {
                let x: u32 = 2_u32;
                let box_x: Box<u32> = BoxTrait::new(x);
                box_x
            }
        };

        run_program_assert_output(&program, "run_test", &[], JitValue::Uint32(2));
    }

    #[test]
    fn box_unbox_stack_allocated_enum_single() {
        let program = load_cairo! {
            use core::box::BoxTrait;

            enum MyEnum {
                A: felt252,
            }

            fn run_test() -> MyEnum {
                let x = BoxTrait::new(MyEnum::A(1234));
                x.unbox()
            }
        };

        run_program_assert_output(
            &program,
            "run_test",
            &[],
            JitValue::Enum {
                tag: 0,
                value: Box::new(JitValue::Felt252(1234.into())),
                debug_name: None,
            },
        );
    }

    #[test]
    fn box_unbox_stack_allocated_enum_c() {
        let program = load_cairo! {
            use core::box::BoxTrait;

            enum MyEnum {
                A: (),
                B: (),
            }

            fn run_test() -> MyEnum {
                let x = BoxTrait::new(MyEnum::A);
                x.unbox()
            }
        };

        run_program_assert_output(
            &program,
            "run_test",
            &[],
            JitValue::Enum {
                tag: 0,
                value: Box::new(JitValue::Struct {
                    fields: Vec::new(),
                    debug_name: None,
                }),
                debug_name: None,
            },
        );
    }

    #[test]
    fn box_unbox_stack_allocated_enum_c2() {
        let program = load_cairo! {
            use core::box::BoxTrait;

            enum MyEnum {
                A: (),
                B: (),
            }

            fn run_test() -> MyEnum {
                let x = BoxTrait::new(MyEnum::B);
                x.unbox()
            }
        };

        run_program_assert_output(
            &program,
            "run_test",
            &[],
            JitValue::Enum {
                tag: 1,
                value: Box::new(JitValue::Struct {
                    fields: Vec::new(),
                    debug_name: None,
                }),
                debug_name: None,
            },
        );
    }

    #[test]
    fn box_unbox_stack_allocated_enum() {
        let program = load_cairo! {
            use core::box::BoxTrait;

            enum MyEnum {
                A: felt252,
                B: u128,
            }

            fn run_test() -> MyEnum {
                let x = BoxTrait::new(MyEnum::A(1234));
                x.unbox()
            }
        };

        run_program_assert_output(
            &program,
            "run_test",
            &[],
            JitValue::Enum {
                tag: 0,
                value: Box::new(JitValue::Felt252(1234.into())),
                debug_name: None,
            },
        );
    }
}
