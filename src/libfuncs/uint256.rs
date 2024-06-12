////! # `u256`-related libfuncs
//! # `u256`-related libfuncs
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{error::Result, metadata::MetadataStorage, utils::ProgramRegistryExt};
use crate::{error::Result, metadata::MetadataStorage, utils::ProgramRegistryExt};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        int::unsigned256::Uint256Concrete,
        int::unsigned256::Uint256Concrete,
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
//        Block, Location, Region, Value,
        Block, Location, Region, Value,
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
//    selector: &Uint256Concrete,
    selector: &Uint256Concrete,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        Uint256Concrete::Divmod(info) => {
        Uint256Concrete::Divmod(info) => {
//            build_divmod(context, registry, entry, location, helper, metadata, info)
            build_divmod(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Uint256Concrete::IsZero(info) => {
        Uint256Concrete::IsZero(info) => {
//            build_is_zero(context, registry, entry, location, helper, metadata, info)
            build_is_zero(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Uint256Concrete::SquareRoot(info) => {
        Uint256Concrete::SquareRoot(info) => {
//            build_square_root(context, registry, entry, location, helper, metadata, info)
            build_square_root(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Uint256Concrete::InvModN(info) => build_u256_guarantee_inv_mod_n(
        Uint256Concrete::InvModN(info) => build_u256_guarantee_inv_mod_n(
//            context, registry, entry, location, helper, metadata, info,
            context, registry, entry, location, helper, metadata, info,
//        ),
        ),
//    }
    }
//}
}
//

///// Generate MLIR operations for the `u256_safe_divmod` libfunc.
/// Generate MLIR operations for the `u256_safe_divmod` libfunc.
//pub fn build_divmod<'ctx, 'this>(
pub fn build_divmod<'ctx, 'this>(
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
//    let range_check =
    let range_check =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//

//    let i128_ty = IntegerType::new(context, 128).into();
    let i128_ty = IntegerType::new(context, 128).into();
//    let i256_ty = IntegerType::new(context, 256).into();
    let i256_ty = IntegerType::new(context, 256).into();
//

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
//        &info.output_types()[0][3],
        &info.output_types()[0][3],
//    )?;
    )?;
//

//    let lhs_struct: Value = entry.argument(1)?.into();
    let lhs_struct: Value = entry.argument(1)?.into();
//    let rhs_struct: Value = entry.argument(2)?.into();
    let rhs_struct: Value = entry.argument(2)?.into();
//

//    let lhs_lo = entry
    let lhs_lo = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            lhs_struct,
            lhs_struct,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            i128_ty,
            i128_ty,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let lhs_hi = entry
    let lhs_hi = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            lhs_struct,
            lhs_struct,
//            DenseI64ArrayAttribute::new(context, &[1]),
            DenseI64ArrayAttribute::new(context, &[1]),
//            i128_ty,
            i128_ty,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let rhs_lo = entry
    let rhs_lo = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            rhs_struct,
            rhs_struct,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            i128_ty,
            i128_ty,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let rhs_hi = entry
    let rhs_hi = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            rhs_struct,
            rhs_struct,
//            DenseI64ArrayAttribute::new(context, &[1]),
            DenseI64ArrayAttribute::new(context, &[1]),
//            i128_ty,
            i128_ty,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let lhs_lo = entry
    let lhs_lo = entry
//        .append_operation(arith::extui(lhs_lo, i256_ty, location))
        .append_operation(arith::extui(lhs_lo, i256_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let lhs_hi = entry
    let lhs_hi = entry
//        .append_operation(arith::extui(lhs_hi, i256_ty, location))
        .append_operation(arith::extui(lhs_hi, i256_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let rhs_lo = entry
    let rhs_lo = entry
//        .append_operation(arith::extui(rhs_lo, i256_ty, location))
        .append_operation(arith::extui(rhs_lo, i256_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let rhs_hi = entry
    let rhs_hi = entry
//        .append_operation(arith::extui(rhs_hi, i256_ty, location))
        .append_operation(arith::extui(rhs_hi, i256_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let k128 = entry
    let k128 = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(i256_ty, 128).into(),
            IntegerAttribute::new(i256_ty, 128).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let lhs_hi = entry
    let lhs_hi = entry
//        .append_operation(arith::shli(lhs_hi, k128, location))
        .append_operation(arith::shli(lhs_hi, k128, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let rhs_hi = entry
    let rhs_hi = entry
//        .append_operation(arith::shli(rhs_hi, k128, location))
        .append_operation(arith::shli(rhs_hi, k128, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let lhs = entry
    let lhs = entry
//        .append_operation(arith::ori(lhs_hi, lhs_lo, location))
        .append_operation(arith::ori(lhs_hi, lhs_lo, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let rhs = entry
    let rhs = entry
//        .append_operation(arith::ori(rhs_hi, rhs_lo, location))
        .append_operation(arith::ori(rhs_hi, rhs_lo, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let result_div = entry
    let result_div = entry
//        .append_operation(arith::divui(lhs, rhs, location))
        .append_operation(arith::divui(lhs, rhs, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_rem = entry
    let result_rem = entry
//        .append_operation(arith::remui(lhs, rhs, location))
        .append_operation(arith::remui(lhs, rhs, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let result_div_lo = entry
    let result_div_lo = entry
//        .append_operation(arith::trunci(result_div, i128_ty, location))
        .append_operation(arith::trunci(result_div, i128_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_div_hi = entry
    let result_div_hi = entry
//        .append_operation(arith::shrui(result_div, k128, location))
        .append_operation(arith::shrui(result_div, k128, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_div_hi = entry
    let result_div_hi = entry
//        .append_operation(arith::trunci(result_div_hi, i128_ty, location))
        .append_operation(arith::trunci(result_div_hi, i128_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let result_rem_lo = entry
    let result_rem_lo = entry
//        .append_operation(arith::trunci(result_rem, i128_ty, location))
        .append_operation(arith::trunci(result_rem, i128_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_rem_hi = entry
    let result_rem_hi = entry
//        .append_operation(arith::shrui(result_rem, k128, location))
        .append_operation(arith::shrui(result_rem, k128, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_rem_hi = entry
    let result_rem_hi = entry
//        .append_operation(arith::trunci(result_rem_hi, i128_ty, location))
        .append_operation(arith::trunci(result_rem_hi, i128_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let result_div = entry
    let result_div = entry
//        .append_operation(llvm::undef(
        .append_operation(llvm::undef(
//            llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
            llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_div = entry
    let result_div = entry
//        .append_operation(llvm::insert_value(
        .append_operation(llvm::insert_value(
//            context,
            context,
//            result_div,
            result_div,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            result_div_lo,
            result_div_lo,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_div = entry
    let result_div = entry
//        .append_operation(llvm::insert_value(
        .append_operation(llvm::insert_value(
//            context,
            context,
//            result_div,
            result_div,
//            DenseI64ArrayAttribute::new(context, &[1]),
            DenseI64ArrayAttribute::new(context, &[1]),
//            result_div_hi,
            result_div_hi,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let result_rem = entry
    let result_rem = entry
//        .append_operation(llvm::undef(
        .append_operation(llvm::undef(
//            llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
            llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_rem = entry
    let result_rem = entry
//        .append_operation(llvm::insert_value(
        .append_operation(llvm::insert_value(
//            context,
            context,
//            result_rem,
            result_rem,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            result_rem_lo,
            result_rem_lo,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_rem = entry
    let result_rem = entry
//        .append_operation(llvm::insert_value(
        .append_operation(llvm::insert_value(
//            context,
            context,
//            result_rem,
            result_rem,
//            DenseI64ArrayAttribute::new(context, &[1]),
            DenseI64ArrayAttribute::new(context, &[1]),
//            result_rem_hi,
            result_rem_hi,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let op = entry.append_operation(llvm::undef(guarantee_type, location));
    let op = entry.append_operation(llvm::undef(guarantee_type, location));
//    let guarantee = op.result(0)?.into();
    let guarantee = op.result(0)?.into();
//

//    entry.append_operation(helper.br(
    entry.append_operation(helper.br(
//        0,
        0,
//        &[range_check, result_div, result_rem, guarantee],
        &[range_check, result_div, result_rem, guarantee],
//        location,
        location,
//    ));
    ));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `u256_is_zero` libfunc.
/// Generate MLIR operations for the `u256_is_zero` libfunc.
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
//    let i128_ty = IntegerType::new(context, 128).into();
    let i128_ty = IntegerType::new(context, 128).into();
//

//    let val_struct = entry.argument(0)?.into();
    let val_struct = entry.argument(0)?.into();
//    let val_lo = entry
    let val_lo = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            val_struct,
            val_struct,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            i128_ty,
            i128_ty,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let val_hi = entry
    let val_hi = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            val_struct,
            val_struct,
//            DenseI64ArrayAttribute::new(context, &[1]),
            DenseI64ArrayAttribute::new(context, &[1]),
//            i128_ty,
            i128_ty,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let k0 = entry
    let k0 = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(i128_ty, 0).into(),
            IntegerAttribute::new(i128_ty, 0).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let val_lo_is_zero = entry
    let val_lo_is_zero = entry
//        .append_operation(arith::cmpi(
        .append_operation(arith::cmpi(
//            context,
            context,
//            CmpiPredicate::Eq,
            CmpiPredicate::Eq,
//            val_lo,
            val_lo,
//            k0,
            k0,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let val_hi_is_zero = entry
    let val_hi_is_zero = entry
//        .append_operation(arith::cmpi(
        .append_operation(arith::cmpi(
//            context,
            context,
//            CmpiPredicate::Eq,
            CmpiPredicate::Eq,
//            val_hi,
            val_hi,
//            k0,
            k0,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let val_is_zero = entry
    let val_is_zero = entry
//        .append_operation(arith::andi(val_lo_is_zero, val_hi_is_zero, location))
        .append_operation(arith::andi(val_lo_is_zero, val_hi_is_zero, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        val_is_zero,
        val_is_zero,
//        [0, 1],
        [0, 1],
//        [&[], &[val_struct]],
        [&[], &[val_struct]],
//        location,
        location,
//    ));
    ));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `u256_sqrt` libfunc.
/// Generate MLIR operations for the `u256_sqrt` libfunc.
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

//    let i128_ty = IntegerType::new(context, 128).into();
    let i128_ty = IntegerType::new(context, 128).into();
//    let i256_ty = IntegerType::new(context, 256).into();
    let i256_ty = IntegerType::new(context, 256).into();
//

//    let arg_struct = entry.argument(1)?.into();
    let arg_struct = entry.argument(1)?.into();
//    let arg_lo = entry
    let arg_lo = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            arg_struct,
            arg_struct,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            i128_ty,
            i128_ty,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let arg_hi = entry
    let arg_hi = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            arg_struct,
            arg_struct,
//            DenseI64ArrayAttribute::new(context, &[1]),
            DenseI64ArrayAttribute::new(context, &[1]),
//            i128_ty,
            i128_ty,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let arg_lo = entry
    let arg_lo = entry
//        .append_operation(arith::extui(arg_lo, i256_ty, location))
        .append_operation(arith::extui(arg_lo, i256_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let arg_hi = entry
    let arg_hi = entry
//        .append_operation(arith::extui(arg_hi, i256_ty, location))
        .append_operation(arith::extui(arg_hi, i256_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let k128 = entry
    let k128 = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(i256_ty, 128).into(),
            IntegerAttribute::new(i256_ty, 128).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let arg_hi = entry
    let arg_hi = entry
//        .append_operation(arith::shli(arg_hi, k128, location))
        .append_operation(arith::shli(arg_hi, k128, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let arg_value = entry
    let arg_value = entry
//        .append_operation(arith::ori(arg_hi, arg_lo, location))
        .append_operation(arith::ori(arg_hi, arg_lo, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let k1 = entry
    let k1 = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(i256_ty, 1).into(),
            IntegerAttribute::new(i256_ty, 1).into(),
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
//            arg_value,
            arg_value,
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
//            &[i256_ty],
            &[i256_ty],
//            {
            {
//                let region = Region::new();
                let region = Region::new();
//                let block = region.append_block(Block::new(&[]));
                let block = region.append_block(Block::new(&[]));
//

//                block.append_operation(scf::r#yield(&[arg_value], location));
                block.append_operation(scf::r#yield(&[arg_value], location));
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
//                        IntegerAttribute::new(i256_ty, 256).into(),
                        IntegerAttribute::new(i256_ty, 256).into(),
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
//                            i256_ty,
                            i256_ty,
//                            arg_value,
                            arg_value,
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
//                        IntegerAttribute::new(i256_ty, -2).into(),
                        IntegerAttribute::new(i256_ty, -2).into(),
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
//                        IntegerAttribute::new(i256_ty, 0).into(),
                        IntegerAttribute::new(i256_ty, 0).into(),
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
//                        &[i256_ty, i256_ty],
                        &[i256_ty, i256_ty],
//                        {
                        {
//                            let region = Region::new();
                            let region = Region::new();
//                            let block = region.append_block(Block::new(&[
                            let block = region.append_block(Block::new(&[
//                                (i256_ty, location),
                                (i256_ty, location),
//                                (i256_ty, location),
                                (i256_ty, location),
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
//                                    arg_value,
                                    arg_value,
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
//                                        .add_results(&[i256_ty])
                                        .add_results(&[i256_ty])
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
//                                        .add_results(&[i256_ty])
                                        .add_results(&[i256_ty])
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
//                                    IntegerAttribute::new(i256_ty, 2).into(),
                                    IntegerAttribute::new(i256_ty, 2).into(),
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
//                                (i256_ty, location),
                                (i256_ty, location),
//                                (i256_ty, location),
                                (i256_ty, location),
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
//        .append_operation(arith::trunci(result, i128_ty, location))
        .append_operation(arith::trunci(result, i128_ty, location))
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

///// Generate MLIR operations for the `u256_guarantee_inv_mod_n` libfunc.
/// Generate MLIR operations for the `u256_guarantee_inv_mod_n` libfunc.
//pub fn build_u256_guarantee_inv_mod_n<'ctx, 'this>(
pub fn build_u256_guarantee_inv_mod_n<'ctx, 'this>(
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
//    let i128_ty = IntegerType::new(context, 128).into();
    let i128_ty = IntegerType::new(context, 128).into();
//    let i256_ty = IntegerType::new(context, 256).into();
    let i256_ty = IntegerType::new(context, 256).into();
//

//    let lhs_struct = entry.argument(1)?.into();
    let lhs_struct = entry.argument(1)?.into();
//    let lhs_lo = entry
    let lhs_lo = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            lhs_struct,
            lhs_struct,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            i128_ty,
            i128_ty,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let lhs_hi = entry
    let lhs_hi = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            lhs_struct,
            lhs_struct,
//            DenseI64ArrayAttribute::new(context, &[1]),
            DenseI64ArrayAttribute::new(context, &[1]),
//            i128_ty,
            i128_ty,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let rhs_struct = entry.argument(2)?.into();
    let rhs_struct = entry.argument(2)?.into();
//    let rhs_lo = entry
    let rhs_lo = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            rhs_struct,
            rhs_struct,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            i128_ty,
            i128_ty,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let rhs_hi = entry
    let rhs_hi = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            rhs_struct,
            rhs_struct,
//            DenseI64ArrayAttribute::new(context, &[1]),
            DenseI64ArrayAttribute::new(context, &[1]),
//            i128_ty,
            i128_ty,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let lhs_lo = entry
    let lhs_lo = entry
//        .append_operation(arith::extui(lhs_lo, i256_ty, location))
        .append_operation(arith::extui(lhs_lo, i256_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let lhs_hi = entry
    let lhs_hi = entry
//        .append_operation(arith::extui(lhs_hi, i256_ty, location))
        .append_operation(arith::extui(lhs_hi, i256_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let rhs_lo = entry
    let rhs_lo = entry
//        .append_operation(arith::extui(rhs_lo, i256_ty, location))
        .append_operation(arith::extui(rhs_lo, i256_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let rhs_hi = entry
    let rhs_hi = entry
//        .append_operation(arith::extui(rhs_hi, i256_ty, location))
        .append_operation(arith::extui(rhs_hi, i256_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let k128 = entry
    let k128 = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(i256_ty, 128).into(),
            IntegerAttribute::new(i256_ty, 128).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let lhs_hi = entry
    let lhs_hi = entry
//        .append_operation(arith::shli(lhs_hi, k128, location))
        .append_operation(arith::shli(lhs_hi, k128, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let rhs_hi = entry
    let rhs_hi = entry
//        .append_operation(arith::shli(rhs_hi, k128, location))
        .append_operation(arith::shli(rhs_hi, k128, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let lhs = entry
    let lhs = entry
//        .append_operation(arith::ori(lhs_hi, lhs_lo, location))
        .append_operation(arith::ori(lhs_hi, lhs_lo, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let rhs = entry
    let rhs = entry
//        .append_operation(arith::ori(rhs_hi, rhs_lo, location))
        .append_operation(arith::ori(rhs_hi, rhs_lo, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let k0 = entry
    let k0 = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(i256_ty, 0).into(),
            IntegerAttribute::new(i256_ty, 0).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let k1 = entry
    let k1 = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(i256_ty, 1).into(),
            IntegerAttribute::new(i256_ty, 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let result = entry.append_operation(scf::r#while(
    let result = entry.append_operation(scf::r#while(
//        &[lhs, rhs, k1, k0],
        &[lhs, rhs, k1, k0],
//        &[i256_ty, i256_ty, i256_ty, i256_ty],
        &[i256_ty, i256_ty, i256_ty, i256_ty],
//        {
        {
//            let region = Region::new();
            let region = Region::new();
//            let block = region.append_block(Block::new(&[
            let block = region.append_block(Block::new(&[
//                (i256_ty, location),
                (i256_ty, location),
//                (i256_ty, location),
                (i256_ty, location),
//                (i256_ty, location),
                (i256_ty, location),
//                (i256_ty, location),
                (i256_ty, location),
//            ]));
            ]));
//

//            let q = block
            let q = block
//                .append_operation(arith::divui(
                .append_operation(arith::divui(
//                    block.argument(1)?.into(),
                    block.argument(1)?.into(),
//                    block.argument(0)?.into(),
                    block.argument(0)?.into(),
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//

//            let q_c = block
            let q_c = block
//                .append_operation(arith::muli(q, block.argument(0)?.into(), location))
                .append_operation(arith::muli(q, block.argument(0)?.into(), location))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            let c = block
            let c = block
//                .append_operation(arith::subi(block.argument(1)?.into(), q_c, location))
                .append_operation(arith::subi(block.argument(1)?.into(), q_c, location))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//

//            let q_uc = block
            let q_uc = block
//                .append_operation(arith::muli(q, block.argument(2)?.into(), location))
                .append_operation(arith::muli(q, block.argument(2)?.into(), location))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            let u_c = block
            let u_c = block
//                .append_operation(arith::subi(block.argument(3)?.into(), q_uc, location))
                .append_operation(arith::subi(block.argument(3)?.into(), q_uc, location))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//

//            let should_continue = block
            let should_continue = block
//                .append_operation(arith::cmpi(context, CmpiPredicate::Ne, c, k0, location))
                .append_operation(arith::cmpi(context, CmpiPredicate::Ne, c, k0, location))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            block.append_operation(scf::condition(
            block.append_operation(scf::condition(
//                should_continue,
                should_continue,
//                &[c, block.argument(0)?.into(), u_c, block.argument(2)?.into()],
                &[c, block.argument(0)?.into(), u_c, block.argument(2)?.into()],
//                location,
                location,
//            ));
            ));
//

//            region
            region
//        },
        },
//        {
        {
//            let region = Region::new();
            let region = Region::new();
//            let block = region.append_block(Block::new(&[
            let block = region.append_block(Block::new(&[
//                (i256_ty, location),
                (i256_ty, location),
//                (i256_ty, location),
                (i256_ty, location),
//                (i256_ty, location),
                (i256_ty, location),
//                (i256_ty, location),
                (i256_ty, location),
//            ]));
            ]));
//

//            block.append_operation(scf::r#yield(
            block.append_operation(scf::r#yield(
//                &[
                &[
//                    block.argument(0)?.into(),
                    block.argument(0)?.into(),
//                    block.argument(1)?.into(),
                    block.argument(1)?.into(),
//                    block.argument(2)?.into(),
                    block.argument(2)?.into(),
//                    block.argument(3)?.into(),
                    block.argument(3)?.into(),
//                ],
                ],
//                location,
                location,
//            ));
            ));
//

//            region
            region
//        },
        },
//        location,
        location,
//    ));
    ));
//

//    let inv = result.result(3)?.into();
    let inv = result.result(3)?.into();
//

//    let inv = entry
    let inv = entry
//        .append_operation(scf::r#if(
        .append_operation(scf::r#if(
//            entry
            entry
//                .append_operation(arith::cmpi(context, CmpiPredicate::Slt, inv, k0, location))
                .append_operation(arith::cmpi(context, CmpiPredicate::Slt, inv, k0, location))
//                .result(0)?
                .result(0)?
//                .into(),
                .into(),
//            &[i256_ty],
            &[i256_ty],
//            {
            {
//                let region = Region::new();
                let region = Region::new();
//                let block = region.append_block(Block::new(&[]));
                let block = region.append_block(Block::new(&[]));
//

//                block.append_operation(scf::r#yield(
                block.append_operation(scf::r#yield(
//                    &[entry
                    &[entry
//                        .append_operation(arith::addi(inv, rhs, location))
                        .append_operation(arith::addi(inv, rhs, location))
//                        .result(0)?
                        .result(0)?
//                        .into()],
                        .into()],
//                    location,
                    location,
//                ));
                ));
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

//                block.append_operation(scf::r#yield(
                block.append_operation(scf::r#yield(
//                    &[entry
                    &[entry
//                        .append_operation(arith::remui(inv, rhs, location))
                        .append_operation(arith::remui(inv, rhs, location))
//                        .result(0)?
                        .result(0)?
//                        .into()],
                        .into()],
//                    location,
                    location,
//                ));
                ));
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

//    let inv_lo = entry
    let inv_lo = entry
//        .append_operation(arith::trunci(inv, i128_ty, location))
        .append_operation(arith::trunci(inv, i128_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let inv_hi = entry
    let inv_hi = entry
//        .append_operation(arith::shrui(inv, k128, location))
        .append_operation(arith::shrui(inv, k128, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let inv_hi = entry
    let inv_hi = entry
//        .append_operation(arith::trunci(inv_hi, i128_ty, location))
        .append_operation(arith::trunci(inv_hi, i128_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let return_ty = registry.build_type(
    let return_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.output_types()[0][1],
        &info.output_types()[0][1],
//    )?;
    )?;
//    let result_inv = entry
    let result_inv = entry
//        .append_operation(llvm::undef(return_ty, location))
        .append_operation(llvm::undef(return_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_inv = entry
    let result_inv = entry
//        .append_operation(llvm::insert_value(
        .append_operation(llvm::insert_value(
//            context,
            context,
//            result_inv,
            result_inv,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            inv_lo,
            inv_lo,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let result_inv = entry
    let result_inv = entry
//        .append_operation(llvm::insert_value(
        .append_operation(llvm::insert_value(
//            context,
            context,
//            result_inv,
            result_inv,
//            DenseI64ArrayAttribute::new(context, &[1]),
            DenseI64ArrayAttribute::new(context, &[1]),
//            inv_hi,
            inv_hi,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let lhs_is_invertible = entry
    let lhs_is_invertible = entry
//        .append_operation(arith::cmpi(
        .append_operation(arith::cmpi(
//            context,
            context,
//            CmpiPredicate::Eq,
            CmpiPredicate::Eq,
//            result.result(1)?.into(),
            result.result(1)?.into(),
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
//    let inv_not_zero = entry
    let inv_not_zero = entry
//        .append_operation(arith::cmpi(context, CmpiPredicate::Ne, inv, k0, location))
        .append_operation(arith::cmpi(context, CmpiPredicate::Ne, inv, k0, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let condition = entry
    let condition = entry
//        .append_operation(arith::andi(lhs_is_invertible, inv_not_zero, location))
        .append_operation(arith::andi(lhs_is_invertible, inv_not_zero, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

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
//    let op = entry.append_operation(llvm::undef(guarantee_type, location));
    let op = entry.append_operation(llvm::undef(guarantee_type, location));
//    let guarantee = op.result(0)?.into();
    let guarantee = op.result(0)?.into();
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        condition,
        condition,
//        [0, 1],
        [0, 1],
//        [
        [
//            &[
            &[
//                entry.argument(0)?.into(),
                entry.argument(0)?.into(),
//                result_inv,
                result_inv,
//                guarantee,
                guarantee,
//                guarantee,
                guarantee,
//                guarantee,
                guarantee,
//                guarantee,
                guarantee,
//                guarantee,
                guarantee,
//                guarantee,
                guarantee,
//                guarantee,
                guarantee,
//                guarantee,
                guarantee,
//            ],
            ],
//            &[entry.argument(0)?.into(), guarantee, guarantee],
            &[entry.argument(0)?.into(), guarantee, guarantee],
//        ],
        ],
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
//    use num_traits::One;
    use num_traits::One;
//    use starknet_types_core::felt::Felt;
    use starknet_types_core::felt::Felt;
//    use std::ops::Shl;
    use std::ops::Shl;
//

//    lazy_static! {
    lazy_static! {
//        static ref U256_IS_ZERO: (String, Program) = load_cairo! {
        static ref U256_IS_ZERO: (String, Program) = load_cairo! {
//            use zeroable::IsZeroResult;
            use zeroable::IsZeroResult;
//

//            extern fn u256_is_zero(a: u256) -> IsZeroResult<u256> implicits() nopanic;
            extern fn u256_is_zero(a: u256) -> IsZeroResult<u256> implicits() nopanic;
//

//            fn run_test(value: u256) -> bool {
            fn run_test(value: u256) -> bool {
//                match u256_is_zero(value) {
                match u256_is_zero(value) {
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
//        static ref U256_SAFE_DIVMOD: (String, Program) = load_cairo! {
        static ref U256_SAFE_DIVMOD: (String, Program) = load_cairo! {
//            fn run_test(lhs: u256, rhs: u256) -> (u256, u256) {
            fn run_test(lhs: u256, rhs: u256) -> (u256, u256) {
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
//        static ref U256_SQRT: (String, Program) = load_cairo! {
        static ref U256_SQRT: (String, Program) = load_cairo! {
//            use core::integer::u256_sqrt;
            use core::integer::u256_sqrt;
//

//            fn run_test(value: u256) -> u128 {
            fn run_test(value: u256) -> u128 {
//                u256_sqrt(value)
                u256_sqrt(value)
//            }
            }
//        };
        };
//        static ref U256_INV_MOD_N: (String, Program) = load_cairo! {
        static ref U256_INV_MOD_N: (String, Program) = load_cairo! {
//            use core::math::u256_inv_mod;
            use core::math::u256_inv_mod;
//

//            fn run_test(a: u256, n: NonZero<u256>) -> Option<NonZero<u256>> {
            fn run_test(a: u256, n: NonZero<u256>) -> Option<NonZero<u256>> {
//                u256_inv_mod(a, n)
                u256_inv_mod(a, n)
//            }
            }
//        };
        };
//    }
    }
//

//    fn u256(value: BigUint) -> JitValue {
    fn u256(value: BigUint) -> JitValue {
//        assert!(value.bits() <= 256);
        assert!(value.bits() <= 256);
//        jit_struct!(
        jit_struct!(
//            JitValue::Uint128((&value & &u128::MAX.into()).try_into().unwrap()),
            JitValue::Uint128((&value & &u128::MAX.into()).try_into().unwrap()),
//            JitValue::Uint128(((&value >> 128u32) & &u128::MAX.into()).try_into().unwrap()),
            JitValue::Uint128(((&value >> 128u32) & &u128::MAX.into()).try_into().unwrap()),
//        )
        )
//    }
    }
//

//    #[test]
    #[test]
//    fn u256_is_zero() {
    fn u256_is_zero() {
//        run_program_assert_output(
        run_program_assert_output(
//            &U256_IS_ZERO,
            &U256_IS_ZERO,
//            "run_test",
            "run_test",
//            &[u256(0u32.into())],
            &[u256(0u32.into())],
//            jit_enum!(1, jit_struct!()),
            jit_enum!(1, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            &U256_IS_ZERO,
            &U256_IS_ZERO,
//            "run_test",
            "run_test",
//            &[u256(1u32.into())],
            &[u256(1u32.into())],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            &U256_IS_ZERO,
            &U256_IS_ZERO,
//            "run_test",
            "run_test",
//            &[u256(BigUint::one() << 128u32)],
            &[u256(BigUint::one() << 128u32)],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            &U256_IS_ZERO,
            &U256_IS_ZERO,
//            "run_test",
            "run_test",
//            &[u256((BigUint::one() << 128u32) + 1u32)],
            &[u256((BigUint::one() << 128u32) + 1u32)],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn u256_safe_divmod() {
    fn u256_safe_divmod() {
//        #[track_caller]
        #[track_caller]
//        fn run(lhs: (u128, u128), rhs: (u128, u128), result: JitValue) {
        fn run(lhs: (u128, u128), rhs: (u128, u128), result: JitValue) {
//            run_program_assert_output(
            run_program_assert_output(
//                &U256_SAFE_DIVMOD,
                &U256_SAFE_DIVMOD,
//                "run_test",
                "run_test",
//                &[
                &[
//                    jit_struct!(lhs.1.into(), lhs.0.into()),
                    jit_struct!(lhs.1.into(), lhs.0.into()),
//                    jit_struct!(rhs.1.into(), rhs.0.into()),
                    jit_struct!(rhs.1.into(), rhs.0.into()),
//                ],
                ],
//                result,
                result,
//            )
            )
//        }
        }
//

//        let u256_is_zero = Felt::from_bytes_be_slice(b"Division by 0");
        let u256_is_zero = Felt::from_bytes_be_slice(b"Division by 0");
//        let max_value = 0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128;
        let max_value = 0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128;
//

//        run((0, 0), (0, 0), jit_panic!(u256_is_zero));
        run((0, 0), (0, 0), jit_panic!(u256_is_zero));
//        run(
        run(
//            (0, 0),
            (0, 0),
//            (0, 1),
            (0, 1),
//            jit_enum!(
            jit_enum!(
//                0,
                0,
//                jit_struct!(jit_struct!(
                jit_struct!(jit_struct!(
//                    jit_struct!(0u128.into(), 0u128.into()),
                    jit_struct!(0u128.into(), 0u128.into()),
//                    jit_struct!(0u128.into(), 0u128.into()),
                    jit_struct!(0u128.into(), 0u128.into()),
//                ))
                ))
//            ),
            ),
//        );
        );
//        run(
        run(
//            (0, 0),
            (0, 0),
//            (max_value, max_value),
            (max_value, max_value),
//            jit_enum!(
            jit_enum!(
//                0,
                0,
//                jit_struct!(jit_struct!(
                jit_struct!(jit_struct!(
//                    jit_struct!(0u128.into(), 0u128.into()),
                    jit_struct!(0u128.into(), 0u128.into()),
//                    jit_struct!(0u128.into(), 0u128.into()),
                    jit_struct!(0u128.into(), 0u128.into()),
//                ))
                ))
//            ),
            ),
//        );
        );
//

//        run((0, 1), (0, 0), jit_panic!(u256_is_zero));
        run((0, 1), (0, 0), jit_panic!(u256_is_zero));
//        run(
        run(
//            (0, 1),
            (0, 1),
//            (0, 1),
            (0, 1),
//            jit_enum!(
            jit_enum!(
//                0,
                0,
//                jit_struct!(jit_struct!(
                jit_struct!(jit_struct!(
//                    jit_struct!(1u128.into(), 0u128.into()),
                    jit_struct!(1u128.into(), 0u128.into()),
//                    jit_struct!(0u128.into(), 0u128.into()),
                    jit_struct!(0u128.into(), 0u128.into()),
//                ))
                ))
//            ),
            ),
//        );
        );
//        run(
        run(
//            (0, 1),
            (0, 1),
//            (max_value, max_value),
            (max_value, max_value),
//            jit_enum!(
            jit_enum!(
//                0,
                0,
//                jit_struct!(jit_struct!(
                jit_struct!(jit_struct!(
//                    jit_struct!(0u128.into(), 0u128.into()),
                    jit_struct!(0u128.into(), 0u128.into()),
//                    jit_struct!(1u128.into(), 0u128.into()),
                    jit_struct!(1u128.into(), 0u128.into()),
//                ))
                ))
//            ),
            ),
//        );
        );
//        run((max_value, max_value), (0, 0), jit_panic!(u256_is_zero));
        run((max_value, max_value), (0, 0), jit_panic!(u256_is_zero));
//

//        run(
        run(
//            (max_value, max_value),
            (max_value, max_value),
//            (0, 1),
            (0, 1),
//            jit_enum!(
            jit_enum!(
//                0,
                0,
//                jit_struct!(jit_struct!(
                jit_struct!(jit_struct!(
//                    jit_struct!(max_value.into(), max_value.into()),
                    jit_struct!(max_value.into(), max_value.into()),
//                    jit_struct!(0u128.into(), 0u128.into()),
                    jit_struct!(0u128.into(), 0u128.into()),
//                ))
                ))
//            ),
            ),
//        );
        );
//        run(
        run(
//            (max_value, max_value),
            (max_value, max_value),
//            (max_value, max_value),
            (max_value, max_value),
//            jit_enum!(
            jit_enum!(
//                0,
                0,
//                jit_struct!(jit_struct!(
                jit_struct!(jit_struct!(
//                    jit_struct!(1u128.into(), 0u128.into()),
                    jit_struct!(1u128.into(), 0u128.into()),
//                    jit_struct!(0u128.into(), 0u128.into()),
                    jit_struct!(0u128.into(), 0u128.into()),
//                ))
                ))
//            ),
            ),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn u256_sqrt() {
    fn u256_sqrt() {
//        #[track_caller]
        #[track_caller]
//        fn run(value: (u128, u128), result: JitValue) {
        fn run(value: (u128, u128), result: JitValue) {
//            run_program_assert_output(
            run_program_assert_output(
//                &U256_SQRT,
                &U256_SQRT,
//                "run_test",
                "run_test",
//                &[jit_struct!(value.1.into(), value.0.into())],
                &[jit_struct!(value.1.into(), value.0.into())],
//                result,
                result,
//            )
            )
//        }
        }
//

//        run((0u128, 0u128), 0u128.into());
        run((0u128, 0u128), 0u128.into());
//        run((0u128, 1u128), 1u128.into());
        run((0u128, 1u128), 1u128.into());
//        run((u128::MAX, u128::MAX), u128::MAX.into());
        run((u128::MAX, u128::MAX), u128::MAX.into());
//

//        for i in 0..u128::BITS {
        for i in 0..u128::BITS {
//            let x = 1u128 << i;
            let x = 1u128 << i;
//            let y: u128 = BigUint::from(x)
            let y: u128 = BigUint::from(x)
//                .sqrt()
                .sqrt()
//                .try_into()
                .try_into()
//                .expect("should always fit into a u128");
                .expect("should always fit into a u128");
//

//            run((0, x), y.into());
            run((0, x), y.into());
//        }
        }
//

//        for i in 0..u128::BITS {
        for i in 0..u128::BITS {
//            let x = 1u128 << i;
            let x = 1u128 << i;
//            let y: u128 = BigUint::from(x)
            let y: u128 = BigUint::from(x)
//                .shl(128usize)
                .shl(128usize)
//                .sqrt()
                .sqrt()
//                .try_into()
                .try_into()
//                .expect("should always fit into a u128");
                .expect("should always fit into a u128");
//

//            run((x, 0), y.into());
            run((x, 0), y.into());
//        }
        }
//    }
    }
//

//    #[test]
    #[test]
//    fn u256_inv_mod_n() {
    fn u256_inv_mod_n() {
//        #[track_caller]
        #[track_caller]
//        fn run(a: (u128, u128), n: (u128, u128), result: JitValue) {
        fn run(a: (u128, u128), n: (u128, u128), result: JitValue) {
//            run_program_assert_output(
            run_program_assert_output(
//                &U256_INV_MOD_N,
                &U256_INV_MOD_N,
//                "run_test",
                "run_test",
//                &[
                &[
//                    jit_struct!(a.0.into(), a.1.into()),
                    jit_struct!(a.0.into(), a.1.into()),
//                    jit_struct!(n.0.into(), n.1.into()),
                    jit_struct!(n.0.into(), n.1.into()),
//                ],
                ],
//                result,
                result,
//            )
            )
//        }
        }
//

//        let none = jit_enum!(1, jit_struct!());
        let none = jit_enum!(1, jit_struct!());
//

//        // Not invertible.
        // Not invertible.
//        run((0, 0), (0, 0), none.clone());
        run((0, 0), (0, 0), none.clone());
//        run((1, 0), (1, 0), none.clone());
        run((1, 0), (1, 0), none.clone());
//        run((0, 0), (1, 0), none.clone());
        run((0, 0), (1, 0), none.clone());
//        run((0, 0), (7, 0), none.clone());
        run((0, 0), (7, 0), none.clone());
//        run((3, 0), (6, 0), none.clone());
        run((3, 0), (6, 0), none.clone());
//        run((4, 0), (6, 0), none.clone());
        run((4, 0), (6, 0), none.clone());
//        run((8, 0), (4, 0), none.clone());
        run((8, 0), (4, 0), none.clone());
//        run((8, 0), (24, 0), none.clone());
        run((8, 0), (24, 0), none.clone());
//        run(
        run(
//            (
            (
//                112713230461650448610759614893138283713,
                112713230461650448610759614893138283713,
//                311795268193434200766998031144865279193,
                311795268193434200766998031144865279193,
//            ),
            ),
//            (
            (
//                214442144331145623175443765631916854552,
                214442144331145623175443765631916854552,
//                85683151001472364977354294776284843870,
                85683151001472364977354294776284843870,
//            ),
            ),
//            none.clone(),
            none.clone(),
//        );
        );
//        run(
        run(
//            (
            (
//                138560372230216185616572678448146427468,
                138560372230216185616572678448146427468,
//                178030013799389090502578959553486954963,
                178030013799389090502578959553486954963,
//            ),
            ),
//            (
            (
//                299456334380503763038201670272353657683,
                299456334380503763038201670272353657683,
//                285941620966047830312853638602560712796,
                285941620966047830312853638602560712796,
//            ),
            ),
//            none,
            none,
//        );
        );
//

//        // Invertible.
        // Invertible.
//        run(
        run(
//            (5, 0),
            (5, 0),
//            (24, 0),
            (24, 0),
//            jit_enum!(0, jit_struct!(5u128.into(), 0u128.into())),
            jit_enum!(0, jit_struct!(5u128.into(), 0u128.into())),
//        );
        );
//        run(
        run(
//            (29, 0),
            (29, 0),
//            (24, 0),
            (24, 0),
//            jit_enum!(0, jit_struct!(5u128.into(), 0u128.into())),
            jit_enum!(0, jit_struct!(5u128.into(), 0u128.into())),
//        );
        );
//        run(
        run(
//            (1, 0),
            (1, 0),
//            (24, 0),
            (24, 0),
//            jit_enum!(0, jit_struct!(1u128.into(), 0u128.into())),
            jit_enum!(0, jit_struct!(1u128.into(), 0u128.into())),
//        );
        );
//        run(
        run(
//            (1, 0),
            (1, 0),
//            (5, 0),
            (5, 0),
//            jit_enum!(0, jit_struct!(1u128.into(), 0u128.into())),
            jit_enum!(0, jit_struct!(1u128.into(), 0u128.into())),
//        );
        );
//        run(
        run(
//            (2, 0),
            (2, 0),
//            (5, 0),
            (5, 0),
//            jit_enum!(0, jit_struct!(3u128.into(), 0u128.into())),
            jit_enum!(0, jit_struct!(3u128.into(), 0u128.into())),
//        );
        );
//    }
    }
//}
}
