////! # `u64`-related libfuncs
//! # `u64`-related libfuncs
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{
use crate::{
//    block_ext::BlockExt,
    block_ext::BlockExt,
//    error::{Error, Result},
    error::{Error, Result},
//    metadata::MetadataStorage,
    metadata::MetadataStorage,
//    utils::ProgramRegistryExt,
    utils::ProgramRegistryExt,
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        int::{
        int::{
//            unsigned::{Uint64Concrete, Uint64Traits, UintConcrete},
            unsigned::{Uint64Concrete, Uint64Traits, UintConcrete},
//            IntConstConcreteLibfunc, IntOperationConcreteLibfunc, IntOperator,
            IntConstConcreteLibfunc, IntOperationConcreteLibfunc, IntOperator,
//        },
        },
//        lib_func::SignatureOnlyConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
//        ConcreteLibfunc,
        ConcreteLibfunc,
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
//        arith::{self, CmpiPredicate},
        arith::{self, CmpiPredicate},
//        cf, llvm, ods, scf,
        cf, llvm, ods, scf,
//    },
    },
//    ir::{
    ir::{
//        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
//        operation::OperationBuilder,
        operation::OperationBuilder,
//        r#type::IntegerType,
        r#type::IntegerType,
//        Attribute, Block, Location, Region, Value, ValueLike,
        Attribute, Block, Location, Region, Value, ValueLike,
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
//    selector: &Uint64Concrete,
    selector: &Uint64Concrete,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        UintConcrete::Const(info) => {
        UintConcrete::Const(info) => {
//            build_const(context, registry, entry, location, helper, metadata, info)
            build_const(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        UintConcrete::Operation(info) => {
        UintConcrete::Operation(info) => {
//            build_operation(context, registry, entry, location, helper, info)
            build_operation(context, registry, entry, location, helper, info)
//        }
        }
//        UintConcrete::SquareRoot(info) => {
        UintConcrete::SquareRoot(info) => {
//            build_square_root(context, registry, entry, location, helper, metadata, info)
            build_square_root(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        UintConcrete::Equal(info) => build_equal(context, registry, entry, location, helper, info),
        UintConcrete::Equal(info) => build_equal(context, registry, entry, location, helper, info),
//        UintConcrete::ToFelt252(info) => {
        UintConcrete::ToFelt252(info) => {
//            build_to_felt252(context, registry, entry, location, helper, metadata, info)
            build_to_felt252(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        UintConcrete::FromFelt252(info) => {
        UintConcrete::FromFelt252(info) => {
//            build_from_felt252(context, registry, entry, location, helper, metadata, info)
            build_from_felt252(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        UintConcrete::IsZero(info) => {
        UintConcrete::IsZero(info) => {
//            build_is_zero(context, registry, entry, location, helper, info)
            build_is_zero(context, registry, entry, location, helper, info)
//        }
        }
//        UintConcrete::Divmod(info) => {
        UintConcrete::Divmod(info) => {
//            build_divmod(context, registry, entry, location, helper, info)
            build_divmod(context, registry, entry, location, helper, info)
//        }
        }
//        UintConcrete::WideMul(info) => {
        UintConcrete::WideMul(info) => {
//            build_widemul(context, registry, entry, location, helper, metadata, info)
            build_widemul(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        UintConcrete::Bitwise(info) => {
        UintConcrete::Bitwise(info) => {
//            super::bitwise::build(context, registry, entry, location, helper, metadata, info)
            super::bitwise::build(context, registry, entry, location, helper, metadata, info)
//        }
        }
//    }
    }
//}
}
//

///// Generate MLIR operations for the `u64_const` libfunc.
/// Generate MLIR operations for the `u64_const` libfunc.
//pub fn build_const<'ctx, 'this>(
pub fn build_const<'ctx, 'this>(
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
//    info: &IntConstConcreteLibfunc<Uint64Traits>,
    info: &IntConstConcreteLibfunc<Uint64Traits>,
//) -> Result<()> {
) -> Result<()> {
//    let value = info.c;
    let value = info.c;
//    let value_ty = registry.build_type(
    let value_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.signature.branch_signatures[0].vars[0].ty,
        &info.signature.branch_signatures[0].vars[0].ty,
//    )?;
    )?;
//

//    let value = entry.const_int_from_type(context, location, value, value_ty)?;
    let value = entry.const_int_from_type(context, location, value, value_ty)?;
//

//    entry.append_operation(helper.br(0, &[value], location));
    entry.append_operation(helper.br(0, &[value], location));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the u64 operation libfunc.
/// Generate MLIR operations for the u64 operation libfunc.
//pub fn build_operation<'ctx, 'this>(
pub fn build_operation<'ctx, 'this>(
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
//    info: &IntOperationConcreteLibfunc,
    info: &IntOperationConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let range_check: Value =
    let range_check: Value =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//

//    let lhs: Value = entry.argument(1)?.into();
    let lhs: Value = entry.argument(1)?.into();
//    let rhs: Value = entry.argument(2)?.into();
    let rhs: Value = entry.argument(2)?.into();
//

//    let op_name = match info.operator {
    let op_name = match info.operator {
//        IntOperator::OverflowingAdd => "llvm.intr.uadd.with.overflow",
        IntOperator::OverflowingAdd => "llvm.intr.uadd.with.overflow",
//        IntOperator::OverflowingSub => "llvm.intr.usub.with.overflow",
        IntOperator::OverflowingSub => "llvm.intr.usub.with.overflow",
//    };
    };
//

//    let values_type = lhs.r#type();
    let values_type = lhs.r#type();
//

//    let result_type = llvm::r#type::r#struct(
    let result_type = llvm::r#type::r#struct(
//        context,
        context,
//        &[values_type, IntegerType::new(context, 1).into()],
        &[values_type, IntegerType::new(context, 1).into()],
//        false,
        false,
//    );
    );
//

//    let op = entry.append_operation(
    let op = entry.append_operation(
//        OperationBuilder::new(op_name, location)
        OperationBuilder::new(op_name, location)
//            .add_operands(&[lhs, rhs])
            .add_operands(&[lhs, rhs])
//            .add_results(&[result_type])
            .add_results(&[result_type])
//            .build()?,
            .build()?,
//    );
    );
//    let result = op.result(0)?.into();
    let result = op.result(0)?.into();
//

//    let op = entry.append_operation(llvm::extract_value(
    let op = entry.append_operation(llvm::extract_value(
//        context,
        context,
//        result,
        result,
//        DenseI64ArrayAttribute::new(context, &[0]),
        DenseI64ArrayAttribute::new(context, &[0]),
//        values_type,
        values_type,
//        location,
        location,
//    ));
    ));
//    let op_result = op.result(0)?.into();
    let op_result = op.result(0)?.into();
//

//    let op = entry.append_operation(llvm::extract_value(
    let op = entry.append_operation(llvm::extract_value(
//        context,
        context,
//        result,
        result,
//        DenseI64ArrayAttribute::new(context, &[1]),
        DenseI64ArrayAttribute::new(context, &[1]),
//        IntegerType::new(context, 1).into(),
        IntegerType::new(context, 1).into(),
//        location,
        location,
//    ));
    ));
//    let op_overflow = op.result(0)?.into();
    let op_overflow = op.result(0)?.into();
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        op_overflow,
        op_overflow,
//        [1, 0],
        [1, 0],
//        [&[range_check, op_result], &[range_check, op_result]],
        [&[range_check, op_result], &[range_check, op_result]],
//        location,
        location,
//    ));
    ));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `u64_eq` libfunc.
/// Generate MLIR operations for the `u64_eq` libfunc.
//pub fn build_equal<'ctx, 'this>(
pub fn build_equal<'ctx, 'this>(
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
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let arg0: Value = entry.argument(0)?.into();
    let arg0: Value = entry.argument(0)?.into();
//    let arg1: Value = entry.argument(1)?.into();
    let arg1: Value = entry.argument(1)?.into();
//

//    let op0 = entry.append_operation(arith::cmpi(
    let op0 = entry.append_operation(arith::cmpi(
//        context,
        context,
//        CmpiPredicate::Eq,
        CmpiPredicate::Eq,
//        arg0,
        arg0,
//        arg1,
        arg1,
//        location,
        location,
//    ));
    ));
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        op0.result(0)?.into(),
        op0.result(0)?.into(),
//        [1, 0],
        [1, 0],
//        [&[]; 2],
        [&[]; 2],
//        location,
        location,
//    ));
    ));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `u64_is_zero` libfunc.
/// Generate MLIR operations for the `u64_is_zero` libfunc.
//pub fn build_is_zero<'ctx, 'this>(
pub fn build_is_zero<'ctx, 'this>(
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
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let arg0: Value = entry.argument(0)?.into();
    let arg0: Value = entry.argument(0)?.into();
//

//    let op = entry.append_operation(arith::constant(
    let op = entry.append_operation(arith::constant(
//        context,
        context,
//        IntegerAttribute::new(arg0.r#type(), 0).into(),
        IntegerAttribute::new(arg0.r#type(), 0).into(),
//        location,
        location,
//    ));
    ));
//    let const_0 = op.result(0)?.into();
    let const_0 = op.result(0)?.into();
//

//    let op = entry.append_operation(arith::cmpi(
    let op = entry.append_operation(arith::cmpi(
//        context,
        context,
//        CmpiPredicate::Eq,
        CmpiPredicate::Eq,
//        arg0,
        arg0,
//        const_0,
        const_0,
//        location,
        location,
//    ));
    ));
//    let condition = op.result(0)?.into();
    let condition = op.result(0)?.into();
//

//    entry.append_operation(helper.cond_br(context, condition, [0, 1], [&[], &[arg0]], location));
    entry.append_operation(helper.cond_br(context, condition, [0, 1], [&[], &[arg0]], location));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `u64_safe_divmod` libfunc.
/// Generate MLIR operations for the `u64_safe_divmod` libfunc.
//pub fn build_divmod<'ctx, 'this>(
pub fn build_divmod<'ctx, 'this>(
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
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let range_check =
    let range_check =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//

//    let lhs: Value = entry.argument(1)?.into();
    let lhs: Value = entry.argument(1)?.into();
//    let rhs: Value = entry.argument(2)?.into();
    let rhs: Value = entry.argument(2)?.into();
//

//    let op = entry.append_operation(arith::divui(lhs, rhs, location));
    let op = entry.append_operation(arith::divui(lhs, rhs, location));
//

//    let result_div = op.result(0)?.into();
    let result_div = op.result(0)?.into();
//    let op = entry.append_operation(arith::remui(lhs, rhs, location));
    let op = entry.append_operation(arith::remui(lhs, rhs, location));
//    let result_rem = op.result(0)?.into();
    let result_rem = op.result(0)?.into();
//

//    entry.append_operation(helper.br(0, &[range_check, result_div, result_rem], location));
    entry.append_operation(helper.br(0, &[range_check, result_div, result_rem], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `u64_widemul` libfunc.
/// Generate MLIR operations for the `u64_widemul` libfunc.
//pub fn build_widemul<'ctx, 'this>(
pub fn build_widemul<'ctx, 'this>(
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
//    info: &SignatureOnlyConcreteLibfunc,
    info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let target_type = registry.build_type(
    let target_type = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.output_types()[0][0],
        &info.output_types()[0][0],
//    )?;
    )?;
//    let lhs: Value = entry.argument(0)?.into();
    let lhs: Value = entry.argument(0)?.into();
//    let rhs: Value = entry.argument(1)?.into();
    let rhs: Value = entry.argument(1)?.into();
//

//    let op = entry.append_operation(arith::extui(lhs, target_type, location));
    let op = entry.append_operation(arith::extui(lhs, target_type, location));
//    let lhs = op.result(0)?.into();
    let lhs = op.result(0)?.into();
//

//    let op = entry.append_operation(arith::extui(rhs, target_type, location));
    let op = entry.append_operation(arith::extui(rhs, target_type, location));
//    let rhs = op.result(0)?.into();
    let rhs = op.result(0)?.into();
//

//    let op = entry.append_operation(arith::muli(lhs, rhs, location));
    let op = entry.append_operation(arith::muli(lhs, rhs, location));
//    let result = op.result(0)?.into();
    let result = op.result(0)?.into();
//

//    entry.append_operation(helper.br(0, &[result], location));
    entry.append_operation(helper.br(0, &[result], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `u64_to_felt252` libfunc.
/// Generate MLIR operations for the `u64_to_felt252` libfunc.
//pub fn build_to_felt252<'ctx, 'this>(
pub fn build_to_felt252<'ctx, 'this>(
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
//    info: &SignatureOnlyConcreteLibfunc,
    info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let felt252_ty = registry.build_type(
    let felt252_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.branch_signatures()[0].vars[0].ty,
        &info.branch_signatures()[0].vars[0].ty,
//    )?;
    )?;
//    let value: Value = entry.argument(0)?.into();
    let value: Value = entry.argument(0)?.into();
//

//    let op = entry.append_operation(arith::extui(value, felt252_ty, location));
    let op = entry.append_operation(arith::extui(value, felt252_ty, location));
//

//    let result = op.result(0)?.into();
    let result = op.result(0)?.into();
//

//    entry.append_operation(helper.br(0, &[result], location));
    entry.append_operation(helper.br(0, &[result], location));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `u64_sqrt` libfunc.
/// Generate MLIR operations for the `u64_sqrt` libfunc.
//pub fn build_square_root<'ctx, 'this>(
pub fn build_square_root<'ctx, 'this>(
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
//    let range_check =
    let range_check =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//

//    let i32_ty = IntegerType::new(context, 32).into();
    let i32_ty = IntegerType::new(context, 32).into();
//    let i64_ty = IntegerType::new(context, 64).into();
    let i64_ty = IntegerType::new(context, 64).into();
//

//    let k1 = entry
    let k1 = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(i64_ty, 1).into(),
            IntegerAttribute::new(i64_ty, 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let is_small = entry
    let is_small = entry
//        .append_operation(arith::cmpi(
        .append_operation(arith::cmpi(
//            context,
            context,
//            CmpiPredicate::Ule,
            CmpiPredicate::Ule,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            k1,
            k1,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let result = entry
    let result = entry
//        .append_operation(scf::r#if(
        .append_operation(scf::r#if(
//            is_small,
            is_small,
//            &[i64_ty],
            &[i64_ty],
//            {
            {
//                let region = Region::new();
                let region = Region::new();
//                let block = region.append_block(Block::new(&[]));
                let block = region.append_block(Block::new(&[]));
//

//                block.append_operation(scf::r#yield(&[entry.argument(1)?.into()], location));
                block.append_operation(scf::r#yield(&[entry.argument(1)?.into()], location));
//

//                region
                region
//            },
            },
//            {
            {
//                let region = Region::new();
                let region = Region::new();
//                let block = region.append_block(Block::new(&[]));
                let block = region.append_block(Block::new(&[]));
//

//                let k64 = entry
                let k64 = entry
//                    .append_operation(arith::constant(
                    .append_operation(arith::constant(
//                        context,
                        context,
//                        IntegerAttribute::new(i64_ty, 64).into(),
                        IntegerAttribute::new(i64_ty, 64).into(),
//                        location,
                        location,
//                    ))
                    ))
//                    .result(0)?
                    .result(0)?
//                    .into();
                    .into();
//

//                let leading_zeros = block
                let leading_zeros = block
//                    .append_operation(
                    .append_operation(
//                        ods::llvm::intr_ctlz(
                        ods::llvm::intr_ctlz(
//                            context,
                            context,
//                            i64_ty,
                            i64_ty,
//                            entry.argument(1)?.into(),
                            entry.argument(1)?.into(),
//                            IntegerAttribute::new(IntegerType::new(context, 1).into(), 1),
                            IntegerAttribute::new(IntegerType::new(context, 1).into(), 1),
//                            location,
                            location,
//                        )
                        )
//                        .into(),
                        .into(),
//                    )
                    )
//                    .result(0)?
                    .result(0)?
//                    .into();
                    .into();
//

//                let num_bits = block
                let num_bits = block
//                    .append_operation(arith::subi(k64, leading_zeros, location))
                    .append_operation(arith::subi(k64, leading_zeros, location))
//                    .result(0)?
                    .result(0)?
//                    .into();
                    .into();
//

//                let shift_amount = block
                let shift_amount = block
//                    .append_operation(arith::addi(num_bits, k1, location))
                    .append_operation(arith::addi(num_bits, k1, location))
//                    .result(0)?
                    .result(0)?
//                    .into();
                    .into();
//

//                let parity_mask = block
                let parity_mask = block
//                    .append_operation(arith::constant(
                    .append_operation(arith::constant(
//                        context,
                        context,
//                        IntegerAttribute::new(i64_ty, -2).into(),
                        IntegerAttribute::new(i64_ty, -2).into(),
//                        location,
                        location,
//                    ))
                    ))
//                    .result(0)?
                    .result(0)?
//                    .into();
                    .into();
//                let shift_amount = block
                let shift_amount = block
//                    .append_operation(arith::andi(shift_amount, parity_mask, location))
                    .append_operation(arith::andi(shift_amount, parity_mask, location))
//                    .result(0)?
                    .result(0)?
//                    .into();
                    .into();
//

//                let k0 = block
                let k0 = block
//                    .append_operation(arith::constant(
                    .append_operation(arith::constant(
//                        context,
                        context,
//                        IntegerAttribute::new(i64_ty, 0).into(),
                        IntegerAttribute::new(i64_ty, 0).into(),
//                        location,
                        location,
//                    ))
                    ))
//                    .result(0)?
                    .result(0)?
//                    .into();
                    .into();
//                let result = block
                let result = block
//                    .append_operation(scf::r#while(
                    .append_operation(scf::r#while(
//                        &[k0, shift_amount],
                        &[k0, shift_amount],
//                        &[i64_ty, i64_ty],
                        &[i64_ty, i64_ty],
//                        {
                        {
//                            let region = Region::new();
                            let region = Region::new();
//                            let block = region.append_block(Block::new(&[
                            let block = region.append_block(Block::new(&[
//                                (i64_ty, location),
                                (i64_ty, location),
//                                (i64_ty, location),
                                (i64_ty, location),
//                            ]));
                            ]));
//

//                            let result = block
                            let result = block
//                                .append_operation(arith::shli(
                                .append_operation(arith::shli(
//                                    block.argument(0)?.into(),
                                    block.argument(0)?.into(),
//                                    k1,
                                    k1,
//                                    location,
                                    location,
//                                ))
                                ))
//                                .result(0)?
                                .result(0)?
//                                .into();
                                .into();
//                            let large_candidate = block
                            let large_candidate = block
//                                .append_operation(arith::xori(result, k1, location))
                                .append_operation(arith::xori(result, k1, location))
//                                .result(0)?
                                .result(0)?
//                                .into();
                                .into();
//

//                            let large_candidate_squared = block
                            let large_candidate_squared = block
//                                .append_operation(arith::muli(
                                .append_operation(arith::muli(
//                                    large_candidate,
                                    large_candidate,
//                                    large_candidate,
                                    large_candidate,
//                                    location,
                                    location,
//                                ))
                                ))
//                                .result(0)?
                                .result(0)?
//                                .into();
                                .into();
//

//                            let threshold = block
                            let threshold = block
//                                .append_operation(arith::shrui(
                                .append_operation(arith::shrui(
//                                    entry.argument(1)?.into(),
                                    entry.argument(1)?.into(),
//                                    block.argument(1)?.into(),
                                    block.argument(1)?.into(),
//                                    location,
                                    location,
//                                ))
                                ))
//                                .result(0)?
                                .result(0)?
//                                .into();
                                .into();
//                            let threshold_is_poison = block
                            let threshold_is_poison = block
//                                .append_operation(arith::cmpi(
                                .append_operation(arith::cmpi(
//                                    context,
                                    context,
//                                    CmpiPredicate::Eq,
                                    CmpiPredicate::Eq,
//                                    block.argument(1)?.into(),
                                    block.argument(1)?.into(),
//                                    k64,
                                    k64,
//                                    location,
                                    location,
//                                ))
                                ))
//                                .result(0)?
                                .result(0)?
//                                .into();
                                .into();
//                            let threshold = block
                            let threshold = block
//                                .append_operation(
                                .append_operation(
//                                    OperationBuilder::new("arith.select", location)
                                    OperationBuilder::new("arith.select", location)
//                                        .add_operands(&[threshold_is_poison, k0, threshold])
                                        .add_operands(&[threshold_is_poison, k0, threshold])
//                                        .add_results(&[i64_ty])
                                        .add_results(&[i64_ty])
//                                        .build()?,
                                        .build()?,
//                                )
                                )
//                                .result(0)?
                                .result(0)?
//                                .into();
                                .into();
//

//                            let is_in_range = block
                            let is_in_range = block
//                                .append_operation(arith::cmpi(
                                .append_operation(arith::cmpi(
//                                    context,
                                    context,
//                                    CmpiPredicate::Ule,
                                    CmpiPredicate::Ule,
//                                    large_candidate_squared,
                                    large_candidate_squared,
//                                    threshold,
                                    threshold,
//                                    location,
                                    location,
//                                ))
                                ))
//                                .result(0)?
                                .result(0)?
//                                .into();
                                .into();
//

//                            let result = block
                            let result = block
//                                .append_operation(
                                .append_operation(
//                                    OperationBuilder::new("arith.select", location)
                                    OperationBuilder::new("arith.select", location)
//                                        .add_operands(&[is_in_range, large_candidate, result])
                                        .add_operands(&[is_in_range, large_candidate, result])
//                                        .add_results(&[i64_ty])
                                        .add_results(&[i64_ty])
//                                        .build()?,
                                        .build()?,
//                                )
                                )
//                                .result(0)?
                                .result(0)?
//                                .into();
                                .into();
//

//                            let k2 = block
                            let k2 = block
//                                .append_operation(arith::constant(
                                .append_operation(arith::constant(
//                                    context,
                                    context,
//                                    IntegerAttribute::new(i64_ty, 2).into(),
                                    IntegerAttribute::new(i64_ty, 2).into(),
//                                    location,
                                    location,
//                                ))
                                ))
//                                .result(0)?
                                .result(0)?
//                                .into();
                                .into();
//

//                            let shift_amount = block
                            let shift_amount = block
//                                .append_operation(arith::subi(
                                .append_operation(arith::subi(
//                                    block.argument(1)?.into(),
                                    block.argument(1)?.into(),
//                                    k2,
                                    k2,
//                                    location,
                                    location,
//                                ))
                                ))
//                                .result(0)?
                                .result(0)?
//                                .into();
                                .into();
//

//                            let should_continue = block
                            let should_continue = block
//                                .append_operation(arith::cmpi(
                                .append_operation(arith::cmpi(
//                                    context,
                                    context,
//                                    CmpiPredicate::Sge,
                                    CmpiPredicate::Sge,
//                                    shift_amount,
                                    shift_amount,
//                                    k0,
                                    k0,
//                                    location,
                                    location,
//                                ))
                                ))
//                                .result(0)?
                                .result(0)?
//                                .into();
                                .into();
//                            block.append_operation(scf::condition(
                            block.append_operation(scf::condition(
//                                should_continue,
                                should_continue,
//                                &[result, shift_amount],
                                &[result, shift_amount],
//                                location,
                                location,
//                            ));
                            ));
//

//                            region
                            region
//                        },
                        },
//                        {
                        {
//                            let region = Region::new();
                            let region = Region::new();
//                            let block = region.append_block(Block::new(&[
                            let block = region.append_block(Block::new(&[
//                                (i64_ty, location),
                                (i64_ty, location),
//                                (i64_ty, location),
                                (i64_ty, location),
//                            ]));
                            ]));
//

//                            block.append_operation(scf::r#yield(
                            block.append_operation(scf::r#yield(
//                                &[block.argument(0)?.into(), block.argument(1)?.into()],
                                &[block.argument(0)?.into(), block.argument(1)?.into()],
//                                location,
                                location,
//                            ));
                            ));
//

//                            region
                            region
//                        },
                        },
//                        location,
                        location,
//                    ))
                    ))
//                    .result(0)?
                    .result(0)?
//                    .into();
                    .into();
//

//                block.append_operation(scf::r#yield(&[result], location));
                block.append_operation(scf::r#yield(&[result], location));
//

//                region
                region
//            },
            },
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let result = entry
    let result = entry
//        .append_operation(arith::trunci(result, i32_ty, location))
        .append_operation(arith::trunci(result, i32_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.br(0, &[range_check, result], location));
    entry.append_operation(helper.br(0, &[range_check, result], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `u64_from_felt252` libfunc.
/// Generate MLIR operations for the `u64_from_felt252` libfunc.
//pub fn build_from_felt252<'ctx, 'this>(
pub fn build_from_felt252<'ctx, 'this>(
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
//    info: &SignatureOnlyConcreteLibfunc,
    info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let range_check: Value =
    let range_check: Value =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//

//    let value: Value = entry.argument(1)?.into();
    let value: Value = entry.argument(1)?.into();
//

//    let felt252_ty = registry.build_type(
    let felt252_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.param_signatures()[1].ty,
        &info.param_signatures()[1].ty,
//    )?;
    )?;
//    let result_ty = registry.build_type(
    let result_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.branch_signatures()[0].vars[1].ty,
        &info.branch_signatures()[0].vars[1].ty,
//    )?;
    )?;
//

//    let op = entry.append_operation(arith::constant(
    let op = entry.append_operation(arith::constant(
//        context,
        context,
//        Attribute::parse(context, &format!("{} : {}", u64::MAX, felt252_ty))
        Attribute::parse(context, &format!("{} : {}", u64::MAX, felt252_ty))
//            .ok_or(Error::ParseAttributeError)?,
            .ok_or(Error::ParseAttributeError)?,
//        location,
        location,
//    ));
    ));
//    let const_max = op.result(0)?.into();
    let const_max = op.result(0)?.into();
//

//    let op = entry.append_operation(arith::cmpi(
    let op = entry.append_operation(arith::cmpi(
//        context,
        context,
//        CmpiPredicate::Ule,
        CmpiPredicate::Ule,
//        value,
        value,
//        const_max,
        const_max,
//        location,
        location,
//    ));
    ));
//    let is_ule = op.result(0)?.into();
    let is_ule = op.result(0)?.into();
//

//    let block_success = helper.append_block(Block::new(&[]));
    let block_success = helper.append_block(Block::new(&[]));
//    let block_failure = helper.append_block(Block::new(&[]));
    let block_failure = helper.append_block(Block::new(&[]));
//

//    entry.append_operation(cf::cond_br(
    entry.append_operation(cf::cond_br(
//        context,
        context,
//        is_ule,
        is_ule,
//        block_success,
        block_success,
//        block_failure,
        block_failure,
//        &[],
        &[],
//        &[],
        &[],
//        location,
        location,
//    ));
    ));
//

//    let op = block_success.append_operation(arith::trunci(value, result_ty, location));
    let op = block_success.append_operation(arith::trunci(value, result_ty, location));
//    let value = op.result(0)?.into();
    let value = op.result(0)?.into();
//

//    block_success.append_operation(helper.br(0, &[range_check, value], location));
    block_success.append_operation(helper.br(0, &[range_check, value], location));
//    block_failure.append_operation(helper.br(1, &[range_check], location));
    block_failure.append_operation(helper.br(1, &[range_check], location));
//

//    Ok(())
    Ok(())
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod test {
mod test {
//    use crate::{
    use crate::{
//        utils::test::{jit_enum, jit_panic, jit_struct, load_cairo},
        utils::test::{jit_enum, jit_panic, jit_struct, load_cairo},
//        values::JitValue,
        values::JitValue,
//    };
    };
//    use cairo_lang_sierra::program::Program;
    use cairo_lang_sierra::program::Program;
//    use lazy_static::lazy_static;
    use lazy_static::lazy_static;
//    use num_bigint::BigUint;
    use num_bigint::BigUint;
//    use starknet_types_core::felt::Felt;
    use starknet_types_core::felt::Felt;
//

//    lazy_static! {
    lazy_static! {
//        static ref U64_OVERFLOWING_ADD: (String, Program) = load_cairo! {
        static ref U64_OVERFLOWING_ADD: (String, Program) = load_cairo! {
//            fn run_test(lhs: u64, rhs: u64) -> u64 {
            fn run_test(lhs: u64, rhs: u64) -> u64 {
//                lhs + rhs
                lhs + rhs
//            }
            }
//        };
        };
//        static ref U64_OVERFLOWING_SUB: (String, Program) = load_cairo! {
        static ref U64_OVERFLOWING_SUB: (String, Program) = load_cairo! {
//            fn run_test(lhs: u64, rhs: u64) -> u64 {
            fn run_test(lhs: u64, rhs: u64) -> u64 {
//                lhs - rhs
                lhs - rhs
//            }
            }
//        };
        };
//        static ref U64_SAFE_DIVMOD: (String, Program) = load_cairo! {
        static ref U64_SAFE_DIVMOD: (String, Program) = load_cairo! {
//            fn run_test(lhs: u64, rhs: u64) -> (u64, u64) {
            fn run_test(lhs: u64, rhs: u64) -> (u64, u64) {
//                let q = lhs / rhs;
                let q = lhs / rhs;
//                let r = lhs % rhs;
                let r = lhs % rhs;
//

//                (q, r)
                (q, r)
//            }
            }
//        };
        };
//        static ref U64_EQUAL: (String, Program) = load_cairo! {
        static ref U64_EQUAL: (String, Program) = load_cairo! {
//            fn run_test(lhs: u64, rhs: u64) -> bool {
            fn run_test(lhs: u64, rhs: u64) -> bool {
//                lhs == rhs
                lhs == rhs
//            }
            }
//        };
        };
//        static ref U64_IS_ZERO: (String, Program) = load_cairo! {
        static ref U64_IS_ZERO: (String, Program) = load_cairo! {
//            use zeroable::IsZeroResult;
            use zeroable::IsZeroResult;
//

//            extern fn u64_is_zero(a: u64) -> IsZeroResult<u64> implicits() nopanic;
            extern fn u64_is_zero(a: u64) -> IsZeroResult<u64> implicits() nopanic;
//

//            fn run_test(value: u64) -> bool {
            fn run_test(value: u64) -> bool {
//                match u64_is_zero(value) {
                match u64_is_zero(value) {
//                    IsZeroResult::Zero(_) => true,
                    IsZeroResult::Zero(_) => true,
//                    IsZeroResult::NonZero(_) => false,
                    IsZeroResult::NonZero(_) => false,
//                }
                }
//            }
            }
//        };
        };
//        static ref U64_SQRT: (String, Program) = load_cairo! {
        static ref U64_SQRT: (String, Program) = load_cairo! {
//            use core::integer::u64_sqrt;
            use core::integer::u64_sqrt;
//

//            fn run_test(value: u64) -> u32 {
            fn run_test(value: u64) -> u32 {
//                u64_sqrt(value)
                u64_sqrt(value)
//            }
            }
//        };
        };
//        static ref U64_WIDEMUL: (String, Program) = load_cairo! {
        static ref U64_WIDEMUL: (String, Program) = load_cairo! {
//            use integer::u64_wide_mul;
            use integer::u64_wide_mul;
//            fn run_test(lhs: u64, rhs: u64) -> u128 {
            fn run_test(lhs: u64, rhs: u64) -> u128 {
//                u64_wide_mul(lhs, rhs)
                u64_wide_mul(lhs, rhs)
//            }
            }
//        };
        };
//    }
    }
//

//    use crate::utils::test::run_program_assert_output;
    use crate::utils::test::run_program_assert_output;
//

//    #[test]
    #[test]
//    fn u64_const_min() {
    fn u64_const_min() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            fn run_test() -> u64 {
            fn run_test() -> u64 {
//                0_u64
                0_u64
//            }
            }
//        );
        );
//

//        run_program_assert_output(&program, "run_test", &[], 0u64.into());
        run_program_assert_output(&program, "run_test", &[], 0u64.into());
//    }
    }
//

//    #[test]
    #[test]
//    fn u64_const_max() {
    fn u64_const_max() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            fn run_test() -> u64 {
            fn run_test() -> u64 {
//                18446744073709551615_u64
                18446744073709551615_u64
//            }
            }
//        );
        );
//

//        run_program_assert_output(&program, "run_test", &[], u64::MAX.into());
        run_program_assert_output(&program, "run_test", &[], u64::MAX.into());
//    }
    }
//

//    #[test]
    #[test]
//    fn u64_to_felt252() {
    fn u64_to_felt252() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use traits::Into;
            use traits::Into;
//

//            fn run_test() -> felt252 {
            fn run_test() -> felt252 {
//                2_u64.into()
                2_u64.into()
//            }
            }
//        );
        );
//

//        run_program_assert_output(&program, "run_test", &[], Felt::from(2).into());
        run_program_assert_output(&program, "run_test", &[], Felt::from(2).into());
//    }
    }
//

//    #[test]
    #[test]
//    fn u64_from_felt252() {
    fn u64_from_felt252() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use traits::TryInto;
            use traits::TryInto;
//

//            fn run_test() -> (Option<u64>, Option<u64>) {
            fn run_test() -> (Option<u64>, Option<u64>) {
//                (
                (
//                    18446744073709551615.try_into(),
                    18446744073709551615.try_into(),
//                    18446744073709551616.try_into(),
                    18446744073709551616.try_into(),
//                )
                )
//            }
            }
//        );
        );
//

//        run_program_assert_output(
        run_program_assert_output(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[],
            &[],
//            jit_struct!(
            jit_struct!(
//                jit_enum!(0, 18446744073709551615u64.into()),
                jit_enum!(0, 18446744073709551615u64.into()),
//                jit_enum!(1, jit_struct!()),
                jit_enum!(1, jit_struct!()),
//            ),
            ),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn u64_overflowing_add() {
    fn u64_overflowing_add() {
//        #[track_caller]
        #[track_caller]
//        fn run(lhs: u64, rhs: u64) {
        fn run(lhs: u64, rhs: u64) {
//            let program = &U64_OVERFLOWING_ADD;
            let program = &U64_OVERFLOWING_ADD;
//            let error = Felt::from_bytes_be_slice(b"u64_add Overflow");
            let error = Felt::from_bytes_be_slice(b"u64_add Overflow");
//

//            let add = lhs.checked_add(rhs);
            let add = lhs.checked_add(rhs);
//

//            match add {
            match add {
//                Some(result) => {
                Some(result) => {
//                    run_program_assert_output(
                    run_program_assert_output(
//                        program,
                        program,
//                        "run_test",
                        "run_test",
//                        &[lhs.into(), rhs.into()],
                        &[lhs.into(), rhs.into()],
//                        jit_enum!(0, jit_struct!(result.into())),
                        jit_enum!(0, jit_struct!(result.into())),
//                    );
                    );
//                }
                }
//                None => {
                None => {
//                    run_program_assert_output(
                    run_program_assert_output(
//                        program,
                        program,
//                        "run_test",
                        "run_test",
//                        &[lhs.into(), rhs.into()],
                        &[lhs.into(), rhs.into()],
//                        jit_panic!(JitValue::Felt252(error)),
                        jit_panic!(JitValue::Felt252(error)),
//                    );
                    );
//                }
                }
//            }
            }
//        }
        }
//

//        const MAX: u64 = u64::MAX;
        const MAX: u64 = u64::MAX;
//

//        run(0, 0);
        run(0, 0);
//        run(0, 1);
        run(0, 1);
//        run(0, MAX - 1);
        run(0, MAX - 1);
//        run(0, MAX);
        run(0, MAX);
//

//        run(1, 0);
        run(1, 0);
//        run(1, 1);
        run(1, 1);
//        run(1, MAX - 1);
        run(1, MAX - 1);
//        run(1, MAX);
        run(1, MAX);
//

//        run(MAX - 1, 0);
        run(MAX - 1, 0);
//        run(MAX - 1, 1);
        run(MAX - 1, 1);
//        run(MAX - 1, MAX - 1);
        run(MAX - 1, MAX - 1);
//        run(MAX - 1, MAX);
        run(MAX - 1, MAX);
//

//        run(MAX, 0);
        run(MAX, 0);
//        run(MAX, 1);
        run(MAX, 1);
//        run(MAX, MAX - 1);
        run(MAX, MAX - 1);
//        run(MAX, MAX);
        run(MAX, MAX);
//    }
    }
//

//    #[test]
    #[test]
//    fn u64_overflowing_sub() {
    fn u64_overflowing_sub() {
//        #[track_caller]
        #[track_caller]
//        fn run(lhs: u64, rhs: u64) {
        fn run(lhs: u64, rhs: u64) {
//            let program = &U64_OVERFLOWING_SUB;
            let program = &U64_OVERFLOWING_SUB;
//            let error = Felt::from_bytes_be_slice(b"u64_sub Overflow");
            let error = Felt::from_bytes_be_slice(b"u64_sub Overflow");
//

//            let add = lhs.checked_sub(rhs);
            let add = lhs.checked_sub(rhs);
//

//            match add {
            match add {
//                Some(result) => {
                Some(result) => {
//                    run_program_assert_output(
                    run_program_assert_output(
//                        program,
                        program,
//                        "run_test",
                        "run_test",
//                        &[lhs.into(), rhs.into()],
                        &[lhs.into(), rhs.into()],
//                        jit_enum!(0, jit_struct!(result.into())),
                        jit_enum!(0, jit_struct!(result.into())),
//                    );
                    );
//                }
                }
//                None => {
                None => {
//                    run_program_assert_output(
                    run_program_assert_output(
//                        program,
                        program,
//                        "run_test",
                        "run_test",
//                        &[lhs.into(), rhs.into()],
                        &[lhs.into(), rhs.into()],
//                        jit_panic!(JitValue::Felt252(error)),
                        jit_panic!(JitValue::Felt252(error)),
//                    );
                    );
//                }
                }
//            }
            }
//        }
        }
//

//        const MAX: u64 = u64::MAX;
        const MAX: u64 = u64::MAX;
//

//        run(0, 0);
        run(0, 0);
//        run(0, 1);
        run(0, 1);
//        run(0, MAX - 1);
        run(0, MAX - 1);
//        run(0, MAX);
        run(0, MAX);
//

//        run(1, 0);
        run(1, 0);
//        run(1, 1);
        run(1, 1);
//        run(1, MAX - 1);
        run(1, MAX - 1);
//        run(1, MAX);
        run(1, MAX);
//

//        run(MAX - 1, 0);
        run(MAX - 1, 0);
//        run(MAX - 1, 1);
        run(MAX - 1, 1);
//        run(MAX - 1, MAX - 1);
        run(MAX - 1, MAX - 1);
//        run(MAX - 1, MAX);
        run(MAX - 1, MAX);
//

//        run(MAX, 0);
        run(MAX, 0);
//        run(MAX, 1);
        run(MAX, 1);
//        run(MAX, MAX - 1);
        run(MAX, MAX - 1);
//        run(MAX, MAX);
        run(MAX, MAX);
//    }
    }
//

//    #[test]
    #[test]
//    fn u64_equal() {
    fn u64_equal() {
//        let program = &U64_EQUAL;
        let program = &U64_EQUAL;
//

//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0u64.into(), 0u64.into()],
            &[0u64.into(), 0u64.into()],
//            jit_enum!(1, jit_struct!()),
            jit_enum!(1, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[1u64.into(), 0u64.into()],
            &[1u64.into(), 0u64.into()],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0u64.into(), 1u64.into()],
            &[0u64.into(), 1u64.into()],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[1u64.into(), 1u64.into()],
            &[1u64.into(), 1u64.into()],
//            jit_enum!(1, jit_struct!()),
            jit_enum!(1, jit_struct!()),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn u64_is_zero() {
    fn u64_is_zero() {
//        let program = &U64_IS_ZERO;
        let program = &U64_IS_ZERO;
//

//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0u64.into()],
            &[0u64.into()],
//            jit_enum!(1, jit_struct!()),
            jit_enum!(1, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[1u64.into()],
            &[1u64.into()],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn u64_safe_divmod() {
    fn u64_safe_divmod() {
//        let program = &U64_IS_ZERO;
        let program = &U64_IS_ZERO;
//

//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0u64.into(), 0u64.into()],
            &[0u64.into(), 0u64.into()],
//            jit_enum!(1, jit_struct!()),
            jit_enum!(1, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0u64.into(), 1u64.into()],
            &[0u64.into(), 1u64.into()],
//            jit_enum!(1, jit_struct!()),
            jit_enum!(1, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0u64.into(), 0xFFFFFFFFFFFFFFFFu64.into()],
            &[0u64.into(), 0xFFFFFFFFFFFFFFFFu64.into()],
//            jit_enum!(1, jit_struct!()),
            jit_enum!(1, jit_struct!()),
//        );
        );
//

//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[1u64.into(), 0u64.into()],
            &[1u64.into(), 0u64.into()],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[1u64.into(), 1u64.into()],
            &[1u64.into(), 1u64.into()],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[1u64.into(), 0xFFFFFFFFFFFFFFFFu64.into()],
            &[1u64.into(), 0xFFFFFFFFFFFFFFFFu64.into()],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//

//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0xFFFFFFFFFFFFFFFFu64.into(), 0u64.into()],
            &[0xFFFFFFFFFFFFFFFFu64.into(), 0u64.into()],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0xFFFFFFFFFFFFFFFFu64.into(), 1u64.into()],
            &[0xFFFFFFFFFFFFFFFFu64.into(), 1u64.into()],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0xFFFFFFFFFFFFFFFFu64.into(), 0xFFFFFFFFFFFFFFFFu64.into()],
            &[0xFFFFFFFFFFFFFFFFu64.into(), 0xFFFFFFFFFFFFFFFFu64.into()],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn u64_sqrt() {
    fn u64_sqrt() {
//        let program = &U64_SQRT;
        let program = &U64_SQRT;
//

//        run_program_assert_output(program, "run_test", &[0u64.into()], 0u32.into());
        run_program_assert_output(program, "run_test", &[0u64.into()], 0u32.into());
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[u64::MAX.into()],
            &[u64::MAX.into()],
//            0xFFFFFFFFu32.into(),
            0xFFFFFFFFu32.into(),
//        );
        );
//

//        for i in 0..u64::BITS {
        for i in 0..u64::BITS {
//            let x = 1u64 << i;
            let x = 1u64 << i;
//            let y: u32 = BigUint::from(x)
            let y: u32 = BigUint::from(x)
//                .sqrt()
                .sqrt()
//                .try_into()
                .try_into()
//                .expect("should always fit int oa u64");
                .expect("should always fit int oa u64");
//

//            run_program_assert_output(program, "run_test", &[x.into()], y.into());
            run_program_assert_output(program, "run_test", &[x.into()], y.into());
//        }
        }
//    }
    }
//

//    #[test]
    #[test]
//    fn u64_widemul() {
    fn u64_widemul() {
//        let program = &U64_WIDEMUL;
        let program = &U64_WIDEMUL;
//

//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0u64.into(), 0u64.into()],
            &[0u64.into(), 0u64.into()],
//            0u128.into(),
            0u128.into(),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0u64.into(), 1u64.into()],
            &[0u64.into(), 1u64.into()],
//            0u128.into(),
            0u128.into(),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[1u64.into(), 0u64.into()],
            &[1u64.into(), 0u64.into()],
//            0u128.into(),
            0u128.into(),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[1u64.into(), 1u64.into()],
            &[1u64.into(), 1u64.into()],
//            1u128.into(),
            1u128.into(),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[u64::MAX.into(), u64::MAX.into()],
            &[u64::MAX.into(), u64::MAX.into()],
//            (u64::MAX as u128 * u64::MAX as u128).into(),
            (u64::MAX as u128 * u64::MAX as u128).into(),
//        );
        );
//    }
    }
//}
}
