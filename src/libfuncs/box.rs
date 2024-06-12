////! # Box libfuncs
//! # Box libfuncs
////!
//!
////! A heap allocated value, which is internally a pointer that can't be null.
//! A heap allocated value, which is internally a pointer that can't be null.
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{
use crate::{
//    error::Result,
    error::Result,
//    metadata::{realloc_bindings::ReallocBindingsMeta, MetadataStorage},
    metadata::{realloc_bindings::ReallocBindingsMeta, MetadataStorage},
//    types::TypeBuilder,
    types::TypeBuilder,
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        boxing::BoxConcreteLibfunc,
        boxing::BoxConcreteLibfunc,
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        lib_func::{
        lib_func::{
//            BranchSignature, LibfuncSignature, SignatureAndTypeConcreteLibfunc,
            BranchSignature, LibfuncSignature, SignatureAndTypeConcreteLibfunc,
//            SignatureOnlyConcreteLibfunc,
            SignatureOnlyConcreteLibfunc,
//        },
        },
//    },
    },
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    dialect::{
    dialect::{
//        arith,
        arith,
//        llvm::{self, r#type::pointer, LoadStoreOptions},
        llvm::{self, r#type::pointer, LoadStoreOptions},
//        ods,
        ods,
//    },
    },
//    ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, Location},
    ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, Location},
//    Context,
    Context,
//};
};
//

///// Select and call the correct libfunc builder function from the selector.
/// Select and call the correct libfunc builder function from the selector.
//pub fn build<'ctx, 'this>(
pub fn build<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    selector: &BoxConcreteLibfunc,
    selector: &BoxConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        BoxConcreteLibfunc::Into(info) => {
        BoxConcreteLibfunc::Into(info) => {
//            build_into_box(context, registry, entry, location, helper, metadata, info)
            build_into_box(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        BoxConcreteLibfunc::Unbox(info) => {
        BoxConcreteLibfunc::Unbox(info) => {
//            build_unbox(context, registry, entry, location, helper, metadata, info)
            build_unbox(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        BoxConcreteLibfunc::ForwardSnapshot(info) => {
        BoxConcreteLibfunc::ForwardSnapshot(info) => {
//            build_forward_snapshot(context, registry, entry, location, helper, metadata, info)
            build_forward_snapshot(context, registry, entry, location, helper, metadata, info)
//        }
        }
//    }
    }
//}
}
//

///// Generate MLIR operations for the `into_box` libfunc.
/// Generate MLIR operations for the `into_box` libfunc.
//pub fn build_into_box<'ctx, 'this>(
pub fn build_into_box<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: &SignatureAndTypeConcreteLibfunc,
    info: &SignatureAndTypeConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    if metadata.get::<ReallocBindingsMeta>().is_none() {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
//        metadata.insert(ReallocBindingsMeta::new(context, helper));
        metadata.insert(ReallocBindingsMeta::new(context, helper));
//    }
    }
//

//    let inner_type = registry.get_type(&info.ty)?;
    let inner_type = registry.get_type(&info.ty)?;
//    let inner_layout = inner_type.layout(registry)?;
    let inner_layout = inner_type.layout(registry)?;
//

//    let value_len = entry
    let value_len = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(
            IntegerAttribute::new(
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                inner_layout.pad_to_align().size().try_into()?,
                inner_layout.pad_to_align().size().try_into()?,
//            )
            )
//            .into(),
            .into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let ptr = entry
    let ptr = entry
//        .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
        .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let ptr = entry
    let ptr = entry
//        .append_operation(ReallocBindingsMeta::realloc(
        .append_operation(ReallocBindingsMeta::realloc(
//            context, ptr, value_len, location,
            context, ptr, value_len, location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(llvm::store(
    entry.append_operation(llvm::store(
//        context,
        context,
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        ptr,
        ptr,
//        location,
        location,
//        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
//            IntegerType::new(context, 64).into(),
            IntegerType::new(context, 64).into(),
//            inner_layout.align() as i64,
            inner_layout.align() as i64,
//        ))),
        ))),
//    ));
    ));
//

//    entry.append_operation(helper.br(0, &[ptr], location));
    entry.append_operation(helper.br(0, &[ptr], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `unbox` libfunc.
/// Generate MLIR operations for the `unbox` libfunc.
//pub fn build_unbox<'ctx, 'this>(
pub fn build_unbox<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: &SignatureAndTypeConcreteLibfunc,
    info: &SignatureAndTypeConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let inner_type = registry.get_type(&info.ty)?;
    let inner_type = registry.get_type(&info.ty)?;
//    let inner_ty = inner_type.build(context, helper, registry, metadata, &info.ty)?;
    let inner_ty = inner_type.build(context, helper, registry, metadata, &info.ty)?;
//    let inner_layout = inner_type.layout(registry)?;
    let inner_layout = inner_type.layout(registry)?;
//

//    // Load the boxed value from memory.
    // Load the boxed value from memory.
//    let value = entry
    let value = entry
//        .append_operation(llvm::load(
        .append_operation(llvm::load(
//            context,
            context,
//            entry.argument(0)?.into(),
            entry.argument(0)?.into(),
//            inner_ty,
            inner_ty,
//            location,
            location,
//            LoadStoreOptions::new().align(Some(IntegerAttribute::new(
            LoadStoreOptions::new().align(Some(IntegerAttribute::new(
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                inner_layout.align() as i64,
                inner_layout.align() as i64,
//            ))),
            ))),
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(ReallocBindingsMeta::free(
    entry.append_operation(ReallocBindingsMeta::free(
//        context,
        context,
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        location,
        location,
//    ));
    ));
//

//    entry.append_operation(helper.br(0, &[value], location));
    entry.append_operation(helper.br(0, &[value], location));
//    Ok(())
    Ok(())
//}
}
//

//fn build_forward_snapshot<'ctx, 'this>(
fn build_forward_snapshot<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: &SignatureAndTypeConcreteLibfunc,
    info: &SignatureAndTypeConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    super::snapshot_take::build(
    super::snapshot_take::build(
//        context,
        context,
//        registry,
        registry,
//        entry,
        entry,
//        location,
        location,
//        helper,
        helper,
//        metadata,
        metadata,
//        &SignatureOnlyConcreteLibfunc {
        &SignatureOnlyConcreteLibfunc {
//            signature: LibfuncSignature {
            signature: LibfuncSignature {
//                param_signatures: info.signature.param_signatures.clone(),
                param_signatures: info.signature.param_signatures.clone(),
//                branch_signatures: info
                branch_signatures: info
//                    .signature
                    .signature
//                    .branch_signatures
                    .branch_signatures
//                    .iter()
                    .iter()
//                    .map(|x| BranchSignature {
                    .map(|x| BranchSignature {
//                        vars: x.vars.clone(),
                        vars: x.vars.clone(),
//                        ap_change: x.ap_change.clone(),
                        ap_change: x.ap_change.clone(),
//                    })
                    })
//                    .collect(),
                    .collect(),
//                fallthrough: info.signature.fallthrough,
                fallthrough: info.signature.fallthrough,
//            },
            },
//        },
        },
//    )
    )
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod test {
mod test {
//    use crate::{
    use crate::{
//        utils::test::{load_cairo, run_program_assert_output},
        utils::test::{load_cairo, run_program_assert_output},
//        values::JitValue,
        values::JitValue,
//    };
    };
//

//    #[test]
    #[test]
//    fn run_box_unbox() {
    fn run_box_unbox() {
//        let program = load_cairo! {
        let program = load_cairo! {
//            use box::BoxTrait;
            use box::BoxTrait;
//            use box::BoxImpl;
            use box::BoxImpl;
//

//            fn run_test() -> u32 {
            fn run_test() -> u32 {
//                let x: u32 = 2_u32;
                let x: u32 = 2_u32;
//                let box_x: Box<u32> = BoxTrait::new(x);
                let box_x: Box<u32> = BoxTrait::new(x);
//                box_x.unbox()
                box_x.unbox()
//            }
            }
//        };
        };
//

//        run_program_assert_output(&program, "run_test", &[], JitValue::Uint32(2));
        run_program_assert_output(&program, "run_test", &[], JitValue::Uint32(2));
//    }
    }
//

//    #[test]
    #[test]
//    fn run_box() {
    fn run_box() {
//        let program = load_cairo! {
        let program = load_cairo! {
//            use box::BoxTrait;
            use box::BoxTrait;
//            use box::BoxImpl;
            use box::BoxImpl;
//

//            fn run_test() -> Box<u32>  {
            fn run_test() -> Box<u32>  {
//                let x: u32 = 2_u32;
                let x: u32 = 2_u32;
//                let box_x: Box<u32> = BoxTrait::new(x);
                let box_x: Box<u32> = BoxTrait::new(x);
//                box_x
                box_x
//            }
            }
//        };
        };
//

//        run_program_assert_output(&program, "run_test", &[], JitValue::Uint32(2));
        run_program_assert_output(&program, "run_test", &[], JitValue::Uint32(2));
//    }
    }
//

//    #[test]
    #[test]
//    fn box_unbox_stack_allocated_enum_single() {
    fn box_unbox_stack_allocated_enum_single() {
//        let program = load_cairo! {
        let program = load_cairo! {
//            use core::box::BoxTrait;
            use core::box::BoxTrait;
//

//            enum MyEnum {
            enum MyEnum {
//                A: felt252,
                A: felt252,
//            }
            }
//

//            fn run_test() -> MyEnum {
            fn run_test() -> MyEnum {
//                let x = BoxTrait::new(MyEnum::A(1234));
                let x = BoxTrait::new(MyEnum::A(1234));
//                x.unbox()
                x.unbox()
//            }
            }
//        };
        };
//

//        run_program_assert_output(
        run_program_assert_output(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[],
            &[],
//            JitValue::Enum {
            JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Felt252(1234.into())),
                value: Box::new(JitValue::Felt252(1234.into())),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn box_unbox_stack_allocated_enum_c() {
    fn box_unbox_stack_allocated_enum_c() {
//        let program = load_cairo! {
        let program = load_cairo! {
//            use core::box::BoxTrait;
            use core::box::BoxTrait;
//

//            enum MyEnum {
            enum MyEnum {
//                A: (),
                A: (),
//                B: (),
                B: (),
//            }
            }
//

//            fn run_test() -> MyEnum {
            fn run_test() -> MyEnum {
//                let x = BoxTrait::new(MyEnum::A);
                let x = BoxTrait::new(MyEnum::A);
//                x.unbox()
                x.unbox()
//            }
            }
//        };
        };
//

//        run_program_assert_output(
        run_program_assert_output(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[],
            &[],
//            JitValue::Enum {
            JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Struct {
                value: Box::new(JitValue::Struct {
//                    fields: Vec::new(),
                    fields: Vec::new(),
//                    debug_name: None,
                    debug_name: None,
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn box_unbox_stack_allocated_enum_c2() {
    fn box_unbox_stack_allocated_enum_c2() {
//        let program = load_cairo! {
        let program = load_cairo! {
//            use core::box::BoxTrait;
            use core::box::BoxTrait;
//

//            enum MyEnum {
            enum MyEnum {
//                A: (),
                A: (),
//                B: (),
                B: (),
//            }
            }
//

//            fn run_test() -> MyEnum {
            fn run_test() -> MyEnum {
//                let x = BoxTrait::new(MyEnum::B);
                let x = BoxTrait::new(MyEnum::B);
//                x.unbox()
                x.unbox()
//            }
            }
//        };
        };
//

//        run_program_assert_output(
        run_program_assert_output(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[],
            &[],
//            JitValue::Enum {
            JitValue::Enum {
//                tag: 1,
                tag: 1,
//                value: Box::new(JitValue::Struct {
                value: Box::new(JitValue::Struct {
//                    fields: Vec::new(),
                    fields: Vec::new(),
//                    debug_name: None,
                    debug_name: None,
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn box_unbox_stack_allocated_enum() {
    fn box_unbox_stack_allocated_enum() {
//        let program = load_cairo! {
        let program = load_cairo! {
//            use core::box::BoxTrait;
            use core::box::BoxTrait;
//

//            enum MyEnum {
            enum MyEnum {
//                A: felt252,
                A: felt252,
//                B: u128,
                B: u128,
//            }
            }
//

//            fn run_test() -> MyEnum {
            fn run_test() -> MyEnum {
//                let x = BoxTrait::new(MyEnum::A(1234));
                let x = BoxTrait::new(MyEnum::A(1234));
//                x.unbox()
                x.unbox()
//            }
            }
//        };
        };
//

//        run_program_assert_output(
        run_program_assert_output(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[],
            &[],
//            JitValue::Enum {
            JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Felt252(1234.into())),
                value: Box::new(JitValue::Felt252(1234.into())),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        );
        );
//    }
    }
//}
}
