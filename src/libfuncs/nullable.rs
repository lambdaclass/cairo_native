////! # Nullable libfuncs
//! # Nullable libfuncs
////!
//!
////! Like a Box but it can be null.
//! Like a Box but it can be null.
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{error::Result, metadata::MetadataStorage};
use crate::{error::Result, metadata::MetadataStorage};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
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
//        nullable::NullableConcreteLibfunc,
        nullable::NullableConcreteLibfunc,
//    },
    },
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    dialect::{cf, llvm::r#type::pointer, ods},
    dialect::{cf, llvm::r#type::pointer, ods},
//    ir::{
    ir::{
//        attribute::IntegerAttribute, operation::OperationBuilder, r#type::IntegerType, Block,
        attribute::IntegerAttribute, operation::OperationBuilder, r#type::IntegerType, Block,
//        Identifier, Location,
        Identifier, Location,
//    },
    },
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
//    selector: &NullableConcreteLibfunc,
    selector: &NullableConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        NullableConcreteLibfunc::Null(info) => {
        NullableConcreteLibfunc::Null(info) => {
//            build_null(context, registry, entry, location, helper, metadata, info)
            build_null(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        NullableConcreteLibfunc::NullableFromBox(info) => {
        NullableConcreteLibfunc::NullableFromBox(info) => {
//            build_nullable_from_box(context, registry, entry, location, helper, metadata, info)
            build_nullable_from_box(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        NullableConcreteLibfunc::MatchNullable(info) => {
        NullableConcreteLibfunc::MatchNullable(info) => {
//            build_match_nullable(context, registry, entry, location, helper, metadata, info)
            build_match_nullable(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        NullableConcreteLibfunc::ForwardSnapshot(info) => {
        NullableConcreteLibfunc::ForwardSnapshot(info) => {
//            build_forward_snapshot(context, registry, entry, location, helper, metadata, info)
            build_forward_snapshot(context, registry, entry, location, helper, metadata, info)
//        }
        }
//    }
    }
//}
}
//

///// Generate MLIR operations for the `null` libfunc.
/// Generate MLIR operations for the `null` libfunc.
//#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
//fn build_null<'ctx, 'this>(
fn build_null<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    _metadata: &mut MetadataStorage,
    _metadata: &mut MetadataStorage,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let op =
    let op =
//        entry.append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into());
        entry.append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into());
//

//    entry.append_operation(helper.br(0, &[op.result(0)?.into()], location));
    entry.append_operation(helper.br(0, &[op.result(0)?.into()], location));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `nullable_from_box` libfunc.
/// Generate MLIR operations for the `nullable_from_box` libfunc.
//#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
//fn build_nullable_from_box<'ctx, 'this>(
fn build_nullable_from_box<'ctx, 'this>(
//    _context: &'ctx Context,
    _context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    _metadata: &mut MetadataStorage,
    _metadata: &mut MetadataStorage,
//    _info: &SignatureAndTypeConcreteLibfunc,
    _info: &SignatureAndTypeConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));
    entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `match_nullable` libfunc.
/// Generate MLIR operations for the `match_nullable` libfunc.
//#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
//fn build_match_nullable<'ctx, 'this>(
fn build_match_nullable<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    _metadata: &mut MetadataStorage,
    _metadata: &mut MetadataStorage,
//    _info: &SignatureAndTypeConcreteLibfunc,
    _info: &SignatureAndTypeConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let arg = entry.argument(0)?.into();
    let arg = entry.argument(0)?.into();
//

//    let op =
    let op =
//        entry.append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into());
        entry.append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into());
//    let nullptr = op.result(0)?.into();
    let nullptr = op.result(0)?.into();
//

//    let op = entry.append_operation(
    let op = entry.append_operation(
//        OperationBuilder::new("llvm.icmp", location)
        OperationBuilder::new("llvm.icmp", location)
//            .add_operands(&[arg, nullptr])
            .add_operands(&[arg, nullptr])
//            .add_attributes(&[(
            .add_attributes(&[(
//                Identifier::new(context, "predicate"),
                Identifier::new(context, "predicate"),
//                IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
                IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
//            )])
            )])
//            .add_results(&[IntegerType::new(context, 1).into()])
            .add_results(&[IntegerType::new(context, 1).into()])
//            .build()?,
            .build()?,
//    );
    );
//

//    let is_null_ptr = op.result(0)?.into();
    let is_null_ptr = op.result(0)?.into();
//

//    let block_is_null = helper.append_block(Block::new(&[]));
    let block_is_null = helper.append_block(Block::new(&[]));
//    let block_is_not_null = helper.append_block(Block::new(&[]));
    let block_is_not_null = helper.append_block(Block::new(&[]));
//

//    entry.append_operation(cf::cond_br(
    entry.append_operation(cf::cond_br(
//        context,
        context,
//        is_null_ptr,
        is_null_ptr,
//        block_is_null,
        block_is_null,
//        block_is_not_null,
        block_is_not_null,
//        &[],
        &[],
//        &[],
        &[],
//        location,
        location,
//    ));
    ));
//

//    block_is_null.append_operation(helper.br(0, &[], location));
    block_is_null.append_operation(helper.br(0, &[], location));
//

//    block_is_not_null.append_operation(helper.br(1, &[arg], location));
    block_is_not_null.append_operation(helper.br(1, &[arg], location));
//

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
//        utils::test::{jit_enum, jit_struct, load_cairo, run_program_assert_output},
        utils::test::{jit_enum, jit_struct, load_cairo, run_program_assert_output},
//        values::JitValue,
        values::JitValue,
//    };
    };
//

//    #[test]
    #[test]
//    fn run_null() {
    fn run_null() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use nullable::null;
            use nullable::null;
//            use nullable::match_nullable;
            use nullable::match_nullable;
//            use nullable::FromNullableResult;
            use nullable::FromNullableResult;
//            use nullable::nullable_from_box;
            use nullable::nullable_from_box;
//            use box::BoxTrait;
            use box::BoxTrait;
//

//            fn run_test() {
            fn run_test() {
//                let _a: Nullable<u8> = null();
                let _a: Nullable<u8> = null();
//            }
            }
//        );
        );
//

//        run_program_assert_output(&program, "run_test", &[], jit_struct!());
        run_program_assert_output(&program, "run_test", &[], jit_struct!());
//    }
    }
//

//    #[test]
    #[test]
//    fn run_null_jit() {
    fn run_null_jit() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use nullable::null;
            use nullable::null;
//            use nullable::match_nullable;
            use nullable::match_nullable;
//            use nullable::FromNullableResult;
            use nullable::FromNullableResult;
//            use nullable::nullable_from_box;
            use nullable::nullable_from_box;
//            use box::BoxTrait;
            use box::BoxTrait;
//

//            fn run_test() -> Nullable<u8> {
            fn run_test() -> Nullable<u8> {
//                let a: Nullable<u8> = null();
                let a: Nullable<u8> = null();
//                a
                a
//            }
            }
//        );
        );
//

//        run_program_assert_output(&program, "run_test", &[], JitValue::Null);
        run_program_assert_output(&program, "run_test", &[], JitValue::Null);
//    }
    }
//

//    #[test]
    #[test]
//    fn run_not_null() {
    fn run_not_null() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use nullable::null;
            use nullable::null;
//            use nullable::match_nullable;
            use nullable::match_nullable;
//            use nullable::FromNullableResult;
            use nullable::FromNullableResult;
//            use nullable::nullable_from_box;
            use nullable::nullable_from_box;
//            use box::BoxTrait;
            use box::BoxTrait;
//

//            fn run_test(x: u8) -> u8 {
            fn run_test(x: u8) -> u8 {
//                let b: Box<u8> = BoxTrait::new(x);
                let b: Box<u8> = BoxTrait::new(x);
//                let c = if x == 0 {
                let c = if x == 0 {
//                    null()
                    null()
//                } else {
                } else {
//                    nullable_from_box(b)
                    nullable_from_box(b)
//                };
                };
//                let d = match match_nullable(c) {
                let d = match match_nullable(c) {
//                    FromNullableResult::Null(_) => 99_u8,
                    FromNullableResult::Null(_) => 99_u8,
//                    FromNullableResult::NotNull(value) => value.unbox()
                    FromNullableResult::NotNull(value) => value.unbox()
//                };
                };
//                d
                d
//            }
            }
//        );
        );
//

//        run_program_assert_output(&program, "run_test", &[4u8.into()], 4u8.into());
        run_program_assert_output(&program, "run_test", &[4u8.into()], 4u8.into());
//        run_program_assert_output(&program, "run_test", &[0u8.into()], 99u8.into());
        run_program_assert_output(&program, "run_test", &[0u8.into()], 99u8.into());
//    }
    }
//

//    #[test]
    #[test]
//    fn match_snapshot_nullable_clone_bug() {
    fn match_snapshot_nullable_clone_bug() {
//        let program = load_cairo! {
        let program = load_cairo! {
//            use core::{NullableTrait, match_nullable, null, nullable::FromNullableResult};
            use core::{NullableTrait, match_nullable, null, nullable::FromNullableResult};
//

//            fn run_test(x: Option<u8>) -> Option<u8> {
            fn run_test(x: Option<u8>) -> Option<u8> {
//                let a = match x {
                let a = match x {
//                    Option::Some(x) => @NullableTrait::new(x),
                    Option::Some(x) => @NullableTrait::new(x),
//                    Option::None(_) => @null::<u8>(),
                    Option::None(_) => @null::<u8>(),
//                };
                };
//                let b = *a;
                let b = *a;
//                match match_nullable(b) {
                match match_nullable(b) {
//                    FromNullableResult::Null(_) => Option::None(()),
                    FromNullableResult::Null(_) => Option::None(()),
//                    FromNullableResult::NotNull(x) => Option::Some(x.unbox()),
                    FromNullableResult::NotNull(x) => Option::Some(x.unbox()),
//                }
                }
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
//            &[jit_enum!(0, 42u8.into())],
            &[jit_enum!(0, 42u8.into())],
//            jit_enum!(0, 42u8.into()),
            jit_enum!(0, 42u8.into()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[jit_enum!(
            &[jit_enum!(
//                1,
                1,
//                JitValue::Struct {
                JitValue::Struct {
//                    fields: Vec::new(),
                    fields: Vec::new(),
//                    debug_name: None
                    debug_name: None
//                }
                }
//            )],
            )],
//            jit_enum!(
            jit_enum!(
//                1,
                1,
//                JitValue::Struct {
                JitValue::Struct {
//                    fields: Vec::new(),
                    fields: Vec::new(),
//                    debug_name: None
                    debug_name: None
//                }
                }
//            ),
            ),
//        );
        );
//    }
    }
//}
}
