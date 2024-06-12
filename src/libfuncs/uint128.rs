////! # `u128`-related libfuncs
//! # `u128`-related libfuncs
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{
use crate::{
//    block_ext::BlockExt, error::Result, metadata::MetadataStorage, utils::ProgramRegistryExt,
    block_ext::BlockExt, error::Result, metadata::MetadataStorage, utils::ProgramRegistryExt,
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
//            unsigned128::{Uint128Concrete, Uint128Traits},
            unsigned128::{Uint128Concrete, Uint128Traits},
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
//        llvm, ods, scf,
        llvm, ods, scf,
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
//        Block, Location, Region, Value, ValueLike,
        Block, Location, Region, Value, ValueLike,
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
//    selector: &Uint128Concrete,
    selector: &Uint128Concrete,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        Uint128Concrete::ByteReverse(info) => {
        Uint128Concrete::ByteReverse(info) => {
//            build_byte_reverse(context, registry, entry, location, helper, metadata, info)
            build_byte_reverse(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Uint128Concrete::Const(info) => {
        Uint128Concrete::Const(info) => {
//            build_const(context, registry, entry, location, helper, metadata, info)
            build_const(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Uint128Concrete::Divmod(info) => {
        Uint128Concrete::Divmod(info) => {
//            build_divmod(context, registry, entry, location, helper, metadata, info)
            build_divmod(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Uint128Concrete::Equal(info) => {
        Uint128Concrete::Equal(info) => {
//            build_equal(context, registry, entry, location, helper, metadata, info)
            build_equal(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Uint128Concrete::FromFelt252(info) => {
        Uint128Concrete::FromFelt252(info) => {
//            build_from_felt252(context, registry, entry, location, helper, metadata, info)
            build_from_felt252(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Uint128Concrete::GuaranteeMul(info) => {
        Uint128Concrete::GuaranteeMul(info) => {
//            build_guarantee_mul(context, registry, entry, location, helper, metadata, info)
            build_guarantee_mul(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Uint128Concrete::IsZero(info) => {
        Uint128Concrete::IsZero(info) => {
//            build_is_zero(context, registry, entry, location, helper, metadata, info)
            build_is_zero(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Uint128Concrete::MulGuaranteeVerify(info) => {
        Uint128Concrete::MulGuaranteeVerify(info) => {
//            build_guarantee_verify(context, registry, entry, location, helper, metadata, info)
            build_guarantee_verify(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Uint128Concrete::Operation(info) => {
        Uint128Concrete::Operation(info) => {
//            build_operation(context, registry, entry, location, helper, metadata, info)
            build_operation(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Uint128Concrete::SquareRoot(info) => {
        Uint128Concrete::SquareRoot(info) => {
//            build_square_root(context, registry, entry, location, helper, metadata, info)
            build_square_root(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Uint128Concrete::ToFelt252(info) => {
        Uint128Concrete::ToFelt252(info) => {
//            build_to_felt252(context, registry, entry, location, helper, metadata, info)
            build_to_felt252(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Uint128Concrete::Bitwise(info) => {
        Uint128Concrete::Bitwise(info) => {
//            super::bitwise::build(context, registry, entry, location, helper, metadata, info)
            super::bitwise::build(context, registry, entry, location, helper, metadata, info)
//        }
        }
//    }
    }
//}
}
//

///// Generate MLIR operations for the `u128_byte_reverse` libfunc.
/// Generate MLIR operations for the `u128_byte_reverse` libfunc.
//pub fn build_byte_reverse<'ctx, 'this>(
pub fn build_byte_reverse<'ctx, 'this>(
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
//    let bitwise =
    let bitwise =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//

//    let arg1 = entry.argument(1)?.into();
    let arg1 = entry.argument(1)?.into();
//

//    let res = entry
    let res = entry
//        .append_operation(ods::llvm::intr_bswap(context, arg1, location).into())
        .append_operation(ods::llvm::intr_bswap(context, arg1, location).into())
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.br(0, &[bitwise, res], location));
    entry.append_operation(helper.br(0, &[bitwise, res], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `u128_const` libfunc.
/// Generate MLIR operations for the `u128_const` libfunc.
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
//    info: &IntConstConcreteLibfunc<Uint128Traits>,
    info: &IntConstConcreteLibfunc<Uint128Traits>,
//) -> Result<()> {
) -> Result<()> {
//    let value = info.c;
    let value = info.c;
//

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
//        &info.branch_signatures()[0].vars[0].ty,
        &info.branch_signatures()[0].vars[0].ty,
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

///// Generate MLIR operations for the `u128_safe_divmod` libfunc.
/// Generate MLIR operations for the `u128_safe_divmod` libfunc.
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

//    let lhs: Value = entry.argument(1)?.into();
    let lhs: Value = entry.argument(1)?.into();
//    let rhs: Value = entry.argument(2)?.into();
    let rhs: Value = entry.argument(2)?.into();
//

//    let op = entry.append_operation(arith::divui(lhs, rhs, location));
    let op = entry.append_operation(arith::divui(lhs, rhs, location));
//    let result_div = op.result(0)?.into();
    let result_div = op.result(0)?.into();
//

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

///// Generate MLIR operations for the `u128_equal` libfunc.
/// Generate MLIR operations for the `u128_equal` libfunc.
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
//    _metadata: &mut MetadataStorage,
    _metadata: &mut MetadataStorage,
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

///// Generate MLIR operations for the `u128s_from_felt252` libfunc.
/// Generate MLIR operations for the `u128s_from_felt252` libfunc.
//pub fn build_from_felt252<'ctx, 'this>(
pub fn build_from_felt252<'ctx, 'this>(
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

//    let arg1 = entry.argument(1)?.into();
    let arg1 = entry.argument(1)?.into();
//

//    let k1 = entry
    let k1 = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(IntegerType::new(context, 252).into(), 1).into(),
            IntegerAttribute::new(IntegerType::new(context, 252).into(), 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let k128 = entry
    let k128 = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(IntegerType::new(context, 252).into(), 128).into(),
            IntegerAttribute::new(IntegerType::new(context, 252).into(), 128).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let min_wide_val = entry
    let min_wide_val = entry
//        .append_operation(arith::shli(k1, k128, location))
        .append_operation(arith::shli(k1, k128, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let is_wide = entry
    let is_wide = entry
//        .append_operation(arith::cmpi(
        .append_operation(arith::cmpi(
//            context,
            context,
//            CmpiPredicate::Uge,
            CmpiPredicate::Uge,
//            arg1,
            arg1,
//            min_wide_val,
            min_wide_val,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let lsb_bits = entry
    let lsb_bits = entry
//        .append_operation(arith::trunci(
        .append_operation(arith::trunci(
//            arg1,
            arg1,
//            IntegerType::new(context, 128).into(),
            IntegerType::new(context, 128).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let msb_bits = entry
    let msb_bits = entry
//        .append_operation(arith::shrui(arg1, k128, location))
        .append_operation(arith::shrui(arg1, k128, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let msb_bits = entry
    let msb_bits = entry
//        .append_operation(arith::trunci(
        .append_operation(arith::trunci(
//            msb_bits,
            msb_bits,
//            IntegerType::new(context, 128).into(),
            IntegerType::new(context, 128).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        is_wide,
        is_wide,
//        [1, 0],
        [1, 0],
//        [&[range_check, msb_bits, lsb_bits], &[range_check, lsb_bits]],
        [&[range_check, msb_bits, lsb_bits], &[range_check, lsb_bits]],
//        location,
        location,
//    ));
    ));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `u128_is_zero` libfunc.
/// Generate MLIR operations for the `u128_is_zero` libfunc.
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
//    _metadata: &mut MetadataStorage,
    _metadata: &mut MetadataStorage,
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
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `u128_add` and `u128_sub` libfuncs.
/// Generate MLIR operations for the `u128_add` and `u128_sub` libfuncs.
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
//    _metadata: &mut MetadataStorage,
    _metadata: &mut MetadataStorage,
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

//    let result_struct: Value = entry
    let result_struct: Value = entry
//        .append_operation(
        .append_operation(
//            OperationBuilder::new(op_name, location)
            OperationBuilder::new(op_name, location)
//                .add_operands(&[lhs, rhs])
                .add_operands(&[lhs, rhs])
//                .add_results(&[result_type])
                .add_results(&[result_type])
//                .build()?,
                .build()?,
//        )
        )
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let result = entry
    let result = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            result_struct,
            result_struct,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            values_type,
            values_type,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let overflow = entry
    let overflow = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            result_struct,
            result_struct,
//            DenseI64ArrayAttribute::new(context, &[1]),
            DenseI64ArrayAttribute::new(context, &[1]),
//            IntegerType::new(context, 1).into(),
            IntegerType::new(context, 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        overflow,
        overflow,
//        [1, 0],
        [1, 0],
//        [&[range_check, result], &[range_check, result]],
        [&[range_check, result], &[range_check, result]],
//        location,
        location,
//    ));
    ));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `u128_sqrt` libfunc.
/// Generate MLIR operations for the `u128_sqrt` libfunc.
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

//    let i64_ty = IntegerType::new(context, 64).into();
    let i64_ty = IntegerType::new(context, 64).into();
//    let i128_ty = IntegerType::new(context, 128).into();
    let i128_ty = IntegerType::new(context, 128).into();
//

//    let k1 = entry
    let k1 = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(i128_ty, 1).into(),
            IntegerAttribute::new(i128_ty, 1).into(),
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
//            &[i128_ty],
            &[i128_ty],
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

//                let k128 = entry
                let k128 = entry
//                    .append_operation(arith::constant(
                    .append_operation(arith::constant(
//                        context,
                        context,
//                        IntegerAttribute::new(i128_ty, 128).into(),
                        IntegerAttribute::new(i128_ty, 128).into(),
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
//                            i128_ty,
                            i128_ty,
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
//                    .append_operation(arith::subi(k128, leading_zeros, location))
                    .append_operation(arith::subi(k128, leading_zeros, location))
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
//                        IntegerAttribute::new(i128_ty, -2).into(),
                        IntegerAttribute::new(i128_ty, -2).into(),
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
//                        IntegerAttribute::new(i128_ty, 0).into(),
                        IntegerAttribute::new(i128_ty, 0).into(),
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
//                        &[i128_ty, i128_ty],
                        &[i128_ty, i128_ty],
//                        {
                        {
//                            let region = Region::new();
                            let region = Region::new();
//                            let block = region.append_block(Block::new(&[
                            let block = region.append_block(Block::new(&[
//                                (i128_ty, location),
                                (i128_ty, location),
//                                (i128_ty, location),
                                (i128_ty, location),
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
//                                    k128,
                                    k128,
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
//                                        .add_results(&[i128_ty])
                                        .add_results(&[i128_ty])
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
//                                        .add_results(&[i128_ty])
                                        .add_results(&[i128_ty])
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
//                                    IntegerAttribute::new(i128_ty, 2).into(),
                                    IntegerAttribute::new(i128_ty, 2).into(),
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
//                                (i128_ty, location),
                                (i128_ty, location),
//                                (i128_ty, location),
                                (i128_ty, location),
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
//        .append_operation(arith::trunci(result, i64_ty, location))
        .append_operation(arith::trunci(result, i64_ty, location))
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

///// Generate MLIR operations for the `u128_to_felt252` libfunc.
/// Generate MLIR operations for the `u128_to_felt252` libfunc.
//pub fn build_to_felt252<'ctx, 'this>(
pub fn build_to_felt252<'ctx, 'this>(
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
//    let op = entry.append_operation(arith::extui(
    let op = entry.append_operation(arith::extui(
//        entry.argument(0)?.into(),
        entry.argument(0)?.into(),
//        IntegerType::new(context, 252).into(),
        IntegerType::new(context, 252).into(),
//        location,
        location,
//    ));
    ));
//

//    entry.append_operation(helper.br(0, &[op.result(0)?.into()], location));
    entry.append_operation(helper.br(0, &[op.result(0)?.into()], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `u128_guarantee_mul` libfunc.
/// Generate MLIR operations for the `u128_guarantee_mul` libfunc.
//pub fn build_guarantee_mul<'ctx, 'this>(
pub fn build_guarantee_mul<'ctx, 'this>(
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
//    let lhs: Value = entry.argument(0)?.into();
    let lhs: Value = entry.argument(0)?.into();
//    let rhs: Value = entry.argument(1)?.into();
    let rhs: Value = entry.argument(1)?.into();
//

//    let origin_type = lhs.r#type();
    let origin_type = lhs.r#type();
//

//    let target_type = IntegerType::new(context, 256).into();
    let target_type = IntegerType::new(context, 256).into();
//    let guarantee_type = registry.build_type(
    let guarantee_type = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.output_types()[0][2],
        &info.output_types()[0][2],
//    )?;
    )?;
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

//    let op = entry.append_operation(arith::trunci(result, origin_type, location));
    let op = entry.append_operation(arith::trunci(result, origin_type, location));
//    let result_lo = op.result(0)?.into();
    let result_lo = op.result(0)?.into();
//

//    let op = entry.append_operation(arith::constant(
    let op = entry.append_operation(arith::constant(
//        context,
        context,
//        IntegerAttribute::new(target_type, 128).into(),
        IntegerAttribute::new(target_type, 128).into(),
//        location,
        location,
//    ));
    ));
//    let const_128 = op.result(0)?.into();
    let const_128 = op.result(0)?.into();
//

//    let op = entry.append_operation(arith::shrui(result, const_128, location));
    let op = entry.append_operation(arith::shrui(result, const_128, location));
//    let result_hi = op.result(0)?.into();
    let result_hi = op.result(0)?.into();
//    let op = entry.append_operation(arith::trunci(result_hi, origin_type, location));
    let op = entry.append_operation(arith::trunci(result_hi, origin_type, location));
//    let result_hi = op.result(0)?.into();
    let result_hi = op.result(0)?.into();
//

//    let op = entry.append_operation(llvm::undef(guarantee_type, location));
    let op = entry.append_operation(llvm::undef(guarantee_type, location));
//    let guarantee = op.result(0)?.into();
    let guarantee = op.result(0)?.into();
//

//    entry.append_operation(helper.br(0, &[result_hi, result_lo, guarantee], location));
    entry.append_operation(helper.br(0, &[result_hi, result_lo, guarantee], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `u128_guarantee_verify` libfunc.
/// Generate MLIR operations for the `u128_guarantee_verify` libfunc.
//pub fn build_guarantee_verify<'ctx, 'this>(
pub fn build_guarantee_verify<'ctx, 'this>(
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

//    entry.append_operation(helper.br(0, &[range_check], location));
    entry.append_operation(helper.br(0, &[range_check], location));
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
//        utils::test::{jit_enum, jit_panic, jit_struct, load_cairo, run_program_assert_output},
        utils::test::{jit_enum, jit_panic, jit_struct, load_cairo, run_program_assert_output},
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
//

//    use starknet_types_core::felt::Felt;
    use starknet_types_core::felt::Felt;
//

//    lazy_static! {
    lazy_static! {
//        static ref U128_BYTE_REVERSE: (String, Program) = load_cairo! {
        static ref U128_BYTE_REVERSE: (String, Program) = load_cairo! {
//            extern fn u128_byte_reverse(input: u128) -> u128 implicits(Bitwise) nopanic;
            extern fn u128_byte_reverse(input: u128) -> u128 implicits(Bitwise) nopanic;
//

//            fn run_test(value: u128) -> u128 {
            fn run_test(value: u128) -> u128 {
//                u128_byte_reverse(value)
                u128_byte_reverse(value)
//            }
            }
//        };
        };
//        static ref U128_CONST: (String, Program) = load_cairo! {
        static ref U128_CONST: (String, Program) = load_cairo! {
//            fn run_test() -> u128 {
            fn run_test() -> u128 {
//                1234567890
                1234567890
//            }
            }
//        };
        };
//        static ref U128_SAFE_DIVMOD: (String, Program) = load_cairo! {
        static ref U128_SAFE_DIVMOD: (String, Program) = load_cairo! {
//            fn run_test(lhs: u128, rhs: u128) -> (u128, u128) {
            fn run_test(lhs: u128, rhs: u128) -> (u128, u128) {
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
//        static ref U128_EQUAL: (String, Program) = load_cairo! {
        static ref U128_EQUAL: (String, Program) = load_cairo! {
//            fn run_test(lhs: u128, rhs: u128) -> bool {
            fn run_test(lhs: u128, rhs: u128) -> bool {
//                lhs == rhs
                lhs == rhs
//            }
            }
//        };
        };
//        static ref U128_FROM_FELT252: (String, Program) = load_cairo! {
        static ref U128_FROM_FELT252: (String, Program) = load_cairo! {
//            enum U128sFromFelt252Result {
            enum U128sFromFelt252Result {
//                Narrow: u128,
                Narrow: u128,
//                Wide: (u128, u128),
                Wide: (u128, u128),
//            }
            }
//

//            extern fn u128s_from_felt252(a: felt252) -> U128sFromFelt252Result implicits(RangeCheck) nopanic;
            extern fn u128s_from_felt252(a: felt252) -> U128sFromFelt252Result implicits(RangeCheck) nopanic;
//

//            fn run_test(value: felt252) -> U128sFromFelt252Result {
            fn run_test(value: felt252) -> U128sFromFelt252Result {
//                u128s_from_felt252(value)
                u128s_from_felt252(value)
//            }
            }
//        };
        };
//        static ref U128_IS_ZERO: (String, Program) = load_cairo! {
        static ref U128_IS_ZERO: (String, Program) = load_cairo! {
//            use zeroable::IsZeroResult;
            use zeroable::IsZeroResult;
//

//            extern fn u128_is_zero(a: u128) -> IsZeroResult<u128> implicits() nopanic;
            extern fn u128_is_zero(a: u128) -> IsZeroResult<u128> implicits() nopanic;
//

//            fn run_test(value: u128) -> bool {
            fn run_test(value: u128) -> bool {
//                match u128_is_zero(value) {
                match u128_is_zero(value) {
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
//        static ref U128_ADD: (String, Program) = load_cairo! {
        static ref U128_ADD: (String, Program) = load_cairo! {
//            fn run_test(lhs: u128, rhs: u128) -> u128 {
            fn run_test(lhs: u128, rhs: u128) -> u128 {
//                lhs + rhs
                lhs + rhs
//            }
            }
//        };
        };
//        static ref U128_SUB: (String, Program) = load_cairo! {
        static ref U128_SUB: (String, Program) = load_cairo! {
//            fn run_test(lhs: u128, rhs: u128) -> u128 {
            fn run_test(lhs: u128, rhs: u128) -> u128 {
//                lhs - rhs
                lhs - rhs
//            }
            }
//        };
        };
//        static ref U128_WIDEMUL: (String, Program) = load_cairo! {
        static ref U128_WIDEMUL: (String, Program) = load_cairo! {
//            use integer::u128_wide_mul;
            use integer::u128_wide_mul;
//            fn run_test(lhs: u128, rhs: u128) -> (u128, u128) {
            fn run_test(lhs: u128, rhs: u128) -> (u128, u128) {
//                u128_wide_mul(lhs, rhs)
                u128_wide_mul(lhs, rhs)
//            }
            }
//        };
        };
//        static ref U128_TO_FELT252: (String, Program) = load_cairo! {
        static ref U128_TO_FELT252: (String, Program) = load_cairo! {
//            extern fn u128_to_felt252(a: u128) -> felt252 nopanic;
            extern fn u128_to_felt252(a: u128) -> felt252 nopanic;
//

//            fn run_test(value: u128) -> felt252 {
            fn run_test(value: u128) -> felt252 {
//                u128_to_felt252(value)
                u128_to_felt252(value)
//            }
            }
//        };
        };
//        static ref U128_SQRT: (String, Program) = load_cairo! {
        static ref U128_SQRT: (String, Program) = load_cairo! {
//            use core::integer::u128_sqrt;
            use core::integer::u128_sqrt;
//

//            fn run_test(value: u128) -> u64 {
            fn run_test(value: u128) -> u64 {
//                u128_sqrt(value)
                u128_sqrt(value)
//            }
            }
//        };
        };
//    }
    }
//

//    #[test]
    #[test]
//    fn u128_byte_reverse() {
    fn u128_byte_reverse() {
//        run_program_assert_output(
        run_program_assert_output(
//            &U128_BYTE_REVERSE,
            &U128_BYTE_REVERSE,
//            "run_test",
            "run_test",
//            &[0x00000000_00000000_00000000_00000000u128.into()],
            &[0x00000000_00000000_00000000_00000000u128.into()],
//            0x00000000_00000000_00000000_00000000u128.into(),
            0x00000000_00000000_00000000_00000000u128.into(),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            &U128_BYTE_REVERSE,
            &U128_BYTE_REVERSE,
//            "run_test",
            "run_test",
//            &[0x00000000_00000000_00000000_00000001u128.into()],
            &[0x00000000_00000000_00000000_00000001u128.into()],
//            0x01000000_00000000_00000000_00000000u128.into(),
            0x01000000_00000000_00000000_00000000u128.into(),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            &U128_BYTE_REVERSE,
            &U128_BYTE_REVERSE,
//            "run_test",
            "run_test",
//            &[0x12345678_90ABCDEF_12345678_90ABCDEFu128.into()],
            &[0x12345678_90ABCDEF_12345678_90ABCDEFu128.into()],
//            0xEFCDAB90_78563412_EFCDAB90_78563412u128.into(),
            0xEFCDAB90_78563412_EFCDAB90_78563412u128.into(),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn u128_const() {
    fn u128_const() {
//        run_program_assert_output(&U128_CONST, "run_test", &[], 1234567890_u128.into());
        run_program_assert_output(&U128_CONST, "run_test", &[], 1234567890_u128.into());
//    }
    }
//

//    #[test]
    #[test]
//    fn u128_safe_divmod() {
    fn u128_safe_divmod() {
//        let program = &U128_SAFE_DIVMOD;
        let program = &U128_SAFE_DIVMOD;
//        let max_value = 0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128;
        let max_value = 0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128;
//        let error = JitValue::Felt252(Felt::from_bytes_be_slice(b"Division by 0"));
        let error = JitValue::Felt252(Felt::from_bytes_be_slice(b"Division by 0"));
//

//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0u128.into(), 0u128.into()],
            &[0u128.into(), 0u128.into()],
//            jit_panic!(error.clone()),
            jit_panic!(error.clone()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0u128.into(), 1u128.into()],
            &[0u128.into(), 1u128.into()],
//            jit_enum!(0, jit_struct!(jit_struct!(0u128.into(), 0u128.into()))),
            jit_enum!(0, jit_struct!(jit_struct!(0u128.into(), 0u128.into()))),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0u128.into(), max_value.into()],
            &[0u128.into(), max_value.into()],
//            jit_enum!(0, jit_struct!(jit_struct!(0u128.into(), 0u128.into()))),
            jit_enum!(0, jit_struct!(jit_struct!(0u128.into(), 0u128.into()))),
//        );
        );
//

//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[1u128.into(), 0u128.into()],
            &[1u128.into(), 0u128.into()],
//            jit_panic!(error.clone()),
            jit_panic!(error.clone()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[1u128.into(), 1u128.into()],
            &[1u128.into(), 1u128.into()],
//            jit_enum!(0, jit_struct!(jit_struct!(1u128.into(), 0u128.into()))),
            jit_enum!(0, jit_struct!(jit_struct!(1u128.into(), 0u128.into()))),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[1u128.into(), max_value.into()],
            &[1u128.into(), max_value.into()],
//            jit_enum!(0, jit_struct!(jit_struct!(0u128.into(), 1u128.into()))),
            jit_enum!(0, jit_struct!(jit_struct!(0u128.into(), 1u128.into()))),
//        );
        );
//

//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[max_value.into(), 0u128.into()],
            &[max_value.into(), 0u128.into()],
//            jit_panic!(error),
            jit_panic!(error),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[max_value.into(), 1u128.into()],
            &[max_value.into(), 1u128.into()],
//            jit_enum!(0, jit_struct!(jit_struct!(u128::MAX.into(), 0u128.into()))),
            jit_enum!(0, jit_struct!(jit_struct!(u128::MAX.into(), 0u128.into()))),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[max_value.into(), max_value.into()],
            &[max_value.into(), max_value.into()],
//            jit_enum!(0, jit_struct!(jit_struct!(1u128.into(), 0u128.into()))),
            jit_enum!(0, jit_struct!(jit_struct!(1u128.into(), 0u128.into()))),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn u128_equal() {
    fn u128_equal() {
//        let program = &U128_EQUAL;
        let program = &U128_EQUAL;
//

//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0u128.into(), 0u128.into()],
            &[0u128.into(), 0u128.into()],
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
//            &[1u128.into(), 0u128.into()],
            &[1u128.into(), 0u128.into()],
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
//            &[0u128.into(), 1u128.into()],
            &[0u128.into(), 1u128.into()],
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
//            &[1u128.into(), 1u128.into()],
            &[1u128.into(), 1u128.into()],
//            jit_enum!(1, jit_struct!()),
            jit_enum!(1, jit_struct!()),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn u128_from_felt252() {
    fn u128_from_felt252() {
//        run_program_assert_output(
        run_program_assert_output(
//            &U128_FROM_FELT252,
            &U128_FROM_FELT252,
//            "run_test",
            "run_test",
//            &[Felt::ZERO.into()],
            &[Felt::ZERO.into()],
//            jit_enum!(0, 0u128.into()),
            jit_enum!(0, 0u128.into()),
//        );
        );
//

//        run_program_assert_output(
        run_program_assert_output(
//            &U128_FROM_FELT252,
            &U128_FROM_FELT252,
//            "run_test",
            "run_test",
//            &[Felt::ONE.into()],
            &[Felt::ONE.into()],
//            jit_enum!(0, 1u128.into()),
            jit_enum!(0, 1u128.into()),
//        );
        );
//

//        run_program_assert_output(
        run_program_assert_output(
//            &U128_FROM_FELT252,
            &U128_FROM_FELT252,
//            "run_test",
            "run_test",
//            &[Felt::from(u128::MAX).into()],
            &[Felt::from(u128::MAX).into()],
//            jit_enum!(0, u128::MAX.into()),
            jit_enum!(0, u128::MAX.into()),
//        );
        );
//

//        run_program_assert_output(
        run_program_assert_output(
//            &U128_FROM_FELT252,
            &U128_FROM_FELT252,
//            "run_test",
            "run_test",
//            &[
            &[
//                Felt::from_dec_str("340282366920938463463374607431768211456")
                Felt::from_dec_str("340282366920938463463374607431768211456")
//                    .unwrap()
                    .unwrap()
//                    .into(),
                    .into(),
//            ],
            ],
//            jit_enum!(1, jit_struct!(1u128.into(), 0u128.into())),
            jit_enum!(1, jit_struct!(1u128.into(), 0u128.into())),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn u128_is_zero() {
    fn u128_is_zero() {
//        run_program_assert_output(
        run_program_assert_output(
//            &U128_IS_ZERO,
            &U128_IS_ZERO,
//            "run_test",
            "run_test",
//            &[0u128.into()],
            &[0u128.into()],
//            jit_enum!(1, jit_struct!()),
            jit_enum!(1, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            &U128_IS_ZERO,
            &U128_IS_ZERO,
//            "run_test",
            "run_test",
//            &[1u128.into()],
            &[1u128.into()],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn u128_add() {
    fn u128_add() {
//        #[track_caller]
        #[track_caller]
//        fn run(lhs: u128, rhs: u128) {
        fn run(lhs: u128, rhs: u128) {
//            let program = &U128_ADD;
            let program = &U128_ADD;
//            let error = Felt::from_bytes_be_slice(b"u128_add Overflow");
            let error = Felt::from_bytes_be_slice(b"u128_add Overflow");
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

//        const MAX: u128 = u128::MAX;
        const MAX: u128 = u128::MAX;
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
//    fn u128_sub() {
    fn u128_sub() {
//        #[track_caller]
        #[track_caller]
//        fn run(lhs: u128, rhs: u128) {
        fn run(lhs: u128, rhs: u128) {
//            let program = &U128_SUB;
            let program = &U128_SUB;
//            let error = Felt::from_bytes_be_slice(b"u128_sub Overflow");
            let error = Felt::from_bytes_be_slice(b"u128_sub Overflow");
//

//            let res = lhs.checked_sub(rhs);
            let res = lhs.checked_sub(rhs);
//

//            match res {
            match res {
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

//        const MAX: u128 = u128::MAX;
        const MAX: u128 = u128::MAX;
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
//    fn u128_to_felt252() {
    fn u128_to_felt252() {
//        let program = &U128_TO_FELT252;
        let program = &U128_TO_FELT252;
//

//        run_program_assert_output(program, "run_test", &[0u128.into()], Felt::ZERO.into());
        run_program_assert_output(program, "run_test", &[0u128.into()], Felt::ZERO.into());
//        run_program_assert_output(program, "run_test", &[1u128.into()], Felt::ONE.into());
        run_program_assert_output(program, "run_test", &[1u128.into()], Felt::ONE.into());
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[u128::MAX.into()],
            &[u128::MAX.into()],
//            Felt::from(u128::MAX).into(),
            Felt::from(u128::MAX).into(),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn u128_sqrt() {
    fn u128_sqrt() {
//        let program = &U128_SQRT;
        let program = &U128_SQRT;
//

//        run_program_assert_output(program, "run_test", &[0u128.into()], 0u64.into());
        run_program_assert_output(program, "run_test", &[0u128.into()], 0u64.into());
//        run_program_assert_output(program, "run_test", &[u128::MAX.into()], u64::MAX.into());
        run_program_assert_output(program, "run_test", &[u128::MAX.into()], u64::MAX.into());
//

//        for i in 0..u128::BITS {
        for i in 0..u128::BITS {
//            let x = 1u128 << i;
            let x = 1u128 << i;
//            let y: u64 = BigUint::from(x)
            let y: u64 = BigUint::from(x)
//                .sqrt()
                .sqrt()
//                .try_into()
                .try_into()
//                .expect("should always fit into a u128");
                .expect("should always fit into a u128");
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
//    fn u128_widemul() {
    fn u128_widemul() {
//        let program = &U128_WIDEMUL;
        let program = &U128_WIDEMUL;
//

//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0u128.into(), 0u128.into()],
            &[0u128.into(), 0u128.into()],
//            jit_struct!(0u128.into(), 0u128.into()),
            jit_struct!(0u128.into(), 0u128.into()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0u128.into(), 1u128.into()],
            &[0u128.into(), 1u128.into()],
//            jit_struct!(0u128.into(), 0u128.into()),
            jit_struct!(0u128.into(), 0u128.into()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[1u128.into(), 0u128.into()],
            &[1u128.into(), 0u128.into()],
//            jit_struct!(0u128.into(), 0u128.into()),
            jit_struct!(0u128.into(), 0u128.into()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[1u128.into(), 1u128.into()],
            &[1u128.into(), 1u128.into()],
//            jit_struct!(0u128.into(), 1u128.into()),
            jit_struct!(0u128.into(), 1u128.into()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[u128::MAX.into(), u128::MAX.into()],
            &[u128::MAX.into(), u128::MAX.into()],
//            jit_struct!((u128::MAX - 1).into(), 1u128.into()),
            jit_struct!((u128::MAX - 1).into(), 1u128.into()),
//        );
        );
//    }
    }
//}
}
