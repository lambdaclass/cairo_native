//! # `u256`-related libfuncs
//!
//! TODO

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::MetadataStorage,
    types::TypeBuilder,
};
use cairo_lang_sierra::{
    extensions::{
        int::unsigned256::Uint256Concrete, lib_func::SignatureOnlyConcreteLibfunc, GenericLibfunc,
        GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        llvm,
    },
    ir::{
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        r#type::IntegerType,
        Block, Location, Value,
    },
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Uint256Concrete,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        Uint256Concrete::Divmod(info) => {
            build_divmod(context, registry, entry, location, helper, metadata, info)
        }
        Uint256Concrete::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, metadata, info)
        }
        Uint256Concrete::SquareRoot(info) => {
            build_square_root(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `u256_safe_divmod` libfunc.
pub fn build_divmod<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let i128_ty = IntegerType::new(context, 128).into();
    let i256_ty = IntegerType::new(context, 256).into();

    let lhs_struct: Value = entry.argument(1)?.into();
    let rhs_struct: Value = entry.argument(2)?.into();

    let lhs_lo = entry
        .append_operation(llvm::extract_value(
            context,
            lhs_struct,
            DenseI64ArrayAttribute::new(context, &[0]),
            i128_ty,
            location,
        ))
        .result(0)?
        .into();
    let lhs_hi = entry
        .append_operation(llvm::extract_value(
            context,
            lhs_struct,
            DenseI64ArrayAttribute::new(context, &[1]),
            i128_ty,
            location,
        ))
        .result(0)?
        .into();
    let rhs_lo = entry
        .append_operation(llvm::extract_value(
            context,
            rhs_struct,
            DenseI64ArrayAttribute::new(context, &[0]),
            i128_ty,
            location,
        ))
        .result(0)?
        .into();
    let rhs_hi = entry
        .append_operation(llvm::extract_value(
            context,
            rhs_struct,
            DenseI64ArrayAttribute::new(context, &[1]),
            i128_ty,
            location,
        ))
        .result(0)?
        .into();

    let lhs_lo = entry
        .append_operation(arith::extui(lhs_lo, i256_ty, location))
        .result(0)?
        .into();
    let lhs_hi = entry
        .append_operation(arith::extui(lhs_hi, i256_ty, location))
        .result(0)?
        .into();
    let rhs_lo = entry
        .append_operation(arith::extui(rhs_lo, i256_ty, location))
        .result(0)?
        .into();
    let rhs_hi = entry
        .append_operation(arith::extui(rhs_hi, i256_ty, location))
        .result(0)?
        .into();

    let k128 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(128, i256_ty).into(),
            location,
        ))
        .result(0)?
        .into();
    let lhs_hi = entry
        .append_operation(arith::shli(lhs_hi, k128, location))
        .result(0)?
        .into();
    let rhs_hi = entry
        .append_operation(arith::shli(rhs_hi, k128, location))
        .result(0)?
        .into();

    let lhs = entry
        .append_operation(arith::ori(lhs_hi, lhs_lo, location))
        .result(0)?
        .into();
    let rhs = entry
        .append_operation(arith::ori(rhs_hi, rhs_lo, location))
        .result(0)?
        .into();

    let result_div = entry
        .append_operation(arith::divui(lhs, rhs, location))
        .result(0)?
        .into();
    let result_rem = entry
        .append_operation(arith::remui(lhs, rhs, location))
        .result(0)?
        .into();

    let result_div_lo = entry
        .append_operation(arith::trunci(result_div, i128_ty, location))
        .result(0)?
        .into();
    let result_div_hi = entry
        .append_operation(arith::shrui(result_div, k128, location))
        .result(0)?
        .into();
    let result_div_hi = entry
        .append_operation(arith::trunci(result_div_hi, i128_ty, location))
        .result(0)?
        .into();

    let result_rem_lo = entry
        .append_operation(arith::trunci(result_rem, i128_ty, location))
        .result(0)?
        .into();
    let result_rem_hi = entry
        .append_operation(arith::shrui(result_rem, k128, location))
        .result(0)?
        .into();
    let result_rem_hi = entry
        .append_operation(arith::trunci(result_rem_hi, i128_ty, location))
        .result(0)?
        .into();

    let result_div = entry
        .append_operation(llvm::undef(
            llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
            location,
        ))
        .result(0)?
        .into();
    let result_div = entry
        .append_operation(llvm::insert_value(
            context,
            result_div,
            DenseI64ArrayAttribute::new(context, &[0]),
            result_div_lo,
            location,
        ))
        .result(0)?
        .into();
    let result_div = entry
        .append_operation(llvm::insert_value(
            context,
            result_div,
            DenseI64ArrayAttribute::new(context, &[1]),
            result_div_hi,
            location,
        ))
        .result(0)?
        .into();

    let result_rem = entry
        .append_operation(llvm::undef(
            llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
            location,
        ))
        .result(0)?
        .into();
    let result_rem = entry
        .append_operation(llvm::insert_value(
            context,
            result_rem,
            DenseI64ArrayAttribute::new(context, &[0]),
            result_rem_lo,
            location,
        ))
        .result(0)?
        .into();
    let result_rem = entry
        .append_operation(llvm::insert_value(
            context,
            result_rem,
            DenseI64ArrayAttribute::new(context, &[1]),
            result_rem_hi,
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.br(
        0,
        &[entry.argument(0)?.into(), result_div, result_rem],
        location,
    ));
    Ok(())
}

/// Generate MLIR operations for the `u256_is_zero` libfunc.
pub fn build_is_zero<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let i128_ty = IntegerType::new(context, 128).into();

    let val_struct = entry.argument(0)?.into();
    let val_lo = entry
        .append_operation(llvm::extract_value(
            context,
            val_struct,
            DenseI64ArrayAttribute::new(context, &[0]),
            i128_ty,
            location,
        ))
        .result(0)?
        .into();
    let val_hi = entry
        .append_operation(llvm::extract_value(
            context,
            val_struct,
            DenseI64ArrayAttribute::new(context, &[1]),
            i128_ty,
            location,
        ))
        .result(0)?
        .into();

    let k0 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(0, i128_ty).into(),
            location,
        ))
        .result(0)?
        .into();
    let val_lo_is_zero = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Eq,
            val_lo,
            k0,
            location,
        ))
        .result(0)?
        .into();
    let val_hi_is_zero = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Eq,
            val_hi,
            k0,
            location,
        ))
        .result(0)?
        .into();

    let val_is_zero = entry
        .append_operation(arith::andi(val_lo_is_zero, val_hi_is_zero, location))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(val_is_zero, [0, 1], [&[], &[val_struct]], location));
    Ok(())
}

/// Generate MLIR operations for the `u256_sqrt` libfunc.
pub fn build_square_root<'ctx, 'this, TType, TLibfunc>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _entry: &'this Block<'ctx>,
    _location: Location<'ctx>,
    _helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    todo!()
}

#[cfg(test)]
mod test {
    use crate::utils::test::{felt, load_cairo, run_program};
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;
    use serde_json::json;

    lazy_static! {
        static ref U256_IS_ZERO: (String, Program) = load_cairo! {
            use zeroable::IsZeroResult;

            extern fn u256_is_zero(a: u256) -> IsZeroResult<u256> implicits() nopanic;

            fn run_test(value: u256) -> bool {
                match u256_is_zero(value) {
                    IsZeroResult::Zero(_) => true,
                    IsZeroResult::NonZero(_) => false,
                }
            }
        };
        static ref U256_SAFE_DIVMOD: (String, Program) = load_cairo! {
            fn run_test(lhs: u256, rhs: u256) -> (u256, u256) {
                let q = lhs / rhs;
                let r = lhs % rhs;

                (q, r)
            }
        };
    }

    #[test]
    fn u256_is_zero() {
        let r = |(value_hi, value_lo)| {
            run_program(&U256_IS_ZERO, "run_test", json!([[value_lo, value_hi]]))
        };

        assert_eq!(r((0u128, 0u128)), json!([[1, []]]));
        assert_eq!(r((0u128, 1u128)), json!([[0, []]]));
        assert_eq!(r((1u128, 0u128)), json!([[0, []]]));
        assert_eq!(r((1u128, 1u128)), json!([[0, []]]));
    }

    #[test]
    fn u256_safe_divmod() {
        let r = |(lhs_hi, lhs_lo), (rhs_hi, rhs_lo)| {
            run_program(
                &U256_SAFE_DIVMOD,
                "run_test",
                json!([(), [lhs_lo, lhs_hi], [rhs_lo, rhs_hi]]),
            )
        };

        let u256_is_zero = json!([felt("2161886914012515606576")]);
        let max_value = 0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128;

        assert_eq!(r((0, 0), (0, 0)), json!([(), [1, [[], u256_is_zero]]]));
        assert_eq!(
            r((0, 0), (0, 1)),
            json!([(), [0, [[[0u128, 0u128], [0u128, 0u128]]]]])
        );
        assert_eq!(
            r((0, 0), (max_value, max_value)),
            json!([(), [0, [[[0u128, 0u128], [0u128, 0u128]]]]])
        );

        assert_eq!(r((0, 1), (0, 0)), json!([(), [1, [[], u256_is_zero]]]));
        assert_eq!(
            r((0, 1), (0, 1)),
            json!([(), [0, [[[1u128, 0u128], [0u128, 0u128]]]]])
        );
        assert_eq!(
            r((0, 1), (max_value, max_value)),
            json!([(), [0, [[[0u128, 0u128], [1u128, 0u128]]]]])
        );

        assert_eq!(
            r((max_value, max_value), (0, 0)),
            json!([(), [1, [[], u256_is_zero]]])
        );
        assert_eq!(
            r((max_value, max_value), (0, 1)),
            json!([(), [0, [[[max_value, max_value], [0u128, 0u128]]]]])
        );
        assert_eq!(
            r((max_value, max_value), (max_value, max_value)),
            json!([(), [0, [[[1u128, 0u128], [0u128, 0u128]]]]])
        );
    }
}
