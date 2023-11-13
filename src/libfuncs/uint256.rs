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
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        int::unsigned256::Uint256Concrete, lib_func::SignatureOnlyConcreteLibfunc, ConcreteLibfunc,
        GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        llvm, scf,
    },
    ir::{
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        operation::OperationBuilder,
        r#type::IntegerType,
        Block, Identifier, Location, Region, Value,
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
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let i128_ty = IntegerType::new(context, 128).into();
    let i256_ty = IntegerType::new(context, 256).into();

    let guarantee_type = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.output_types()[0][3],
    )?;

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

    let op = entry.append_operation(llvm::undef(guarantee_type, location));
    let guarantee = op.result(0)?.into();

    entry.append_operation(helper.br(
        0,
        &[entry.argument(0)?.into(), result_div, result_rem, guarantee],
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

    let arg_struct = entry.argument(1)?.into();
    let arg_lo = entry
        .append_operation(llvm::extract_value(
            context,
            arg_struct,
            DenseI64ArrayAttribute::new(context, &[0]),
            i128_ty,
            location,
        ))
        .result(0)?
        .into();
    let arg_hi = entry
        .append_operation(llvm::extract_value(
            context,
            arg_struct,
            DenseI64ArrayAttribute::new(context, &[1]),
            i128_ty,
            location,
        ))
        .result(0)?
        .into();

    let arg_lo = entry
        .append_operation(arith::extui(arg_lo, i256_ty, location))
        .result(0)?
        .into();
    let arg_hi = entry
        .append_operation(arith::extui(arg_hi, i256_ty, location))
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
    let arg_hi = entry
        .append_operation(arith::shli(arg_hi, k128, location))
        .result(0)?
        .into();

    let arg_value = entry
        .append_operation(arith::ori(arg_hi, arg_lo, location))
        .result(0)?
        .into();

    let k1 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(1, i256_ty).into(),
            location,
        ))
        .result(0)?
        .into();

    let is_small = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Ule,
            arg_value,
            k1,
            location,
        ))
        .result(0)?
        .into();

    let result = entry
        .append_operation(scf::r#if(
            is_small,
            &[i256_ty],
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                block.append_operation(scf::r#yield(&[arg_value], location));

                region
            },
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                let k128 = entry
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(256, i256_ty).into(),
                        location,
                    ))
                    .result(0)?
                    .into();

                let leading_zeros = block
                    .append_operation(
                        OperationBuilder::new("llvm.intr.ctlz", location)
                            .add_attributes(&[(
                                Identifier::new(context, "is_zero_poison"),
                                IntegerAttribute::new(1, IntegerType::new(context, 1).into())
                                    .into(),
                            )])
                            .add_operands(&[arg_value])
                            .add_results(&[i256_ty])
                            .build(),
                    )
                    .result(0)?
                    .into();

                let num_bits = block
                    .append_operation(arith::subi(k128, leading_zeros, location))
                    .result(0)?
                    .into();

                let shift_amount = block
                    .append_operation(arith::addi(num_bits, k1, location))
                    .result(0)?
                    .into();

                let parity_mask = block
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(-2, i256_ty).into(),
                        location,
                    ))
                    .result(0)?
                    .into();
                let shift_amount = block
                    .append_operation(arith::andi(shift_amount, parity_mask, location))
                    .result(0)?
                    .into();

                let k0 = block
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(0, i256_ty).into(),
                        location,
                    ))
                    .result(0)?
                    .into();
                let result = block
                    .append_operation(scf::r#while(
                        &[k0, shift_amount],
                        &[i256_ty, i256_ty],
                        {
                            let region = Region::new();
                            let block = region.append_block(Block::new(&[
                                (i256_ty, location),
                                (i256_ty, location),
                            ]));

                            let result = block
                                .append_operation(arith::shli(
                                    block.argument(0)?.into(),
                                    k1,
                                    location,
                                ))
                                .result(0)?
                                .into();
                            let large_candidate = block
                                .append_operation(arith::xori(result, k1, location))
                                .result(0)?
                                .into();

                            let large_candidate_squared = block
                                .append_operation(arith::muli(
                                    large_candidate,
                                    large_candidate,
                                    location,
                                ))
                                .result(0)?
                                .into();

                            let threshold = block
                                .append_operation(arith::shrui(
                                    arg_value,
                                    block.argument(1)?.into(),
                                    location,
                                ))
                                .result(0)?
                                .into();
                            let threshold_is_poison = block
                                .append_operation(arith::cmpi(
                                    context,
                                    CmpiPredicate::Eq,
                                    block.argument(1)?.into(),
                                    k128,
                                    location,
                                ))
                                .result(0)?
                                .into();
                            let threshold = block
                                .append_operation(
                                    OperationBuilder::new("arith.select", location)
                                        .add_operands(&[threshold_is_poison, k0, threshold])
                                        .add_results(&[i256_ty])
                                        .build(),
                                )
                                .result(0)?
                                .into();

                            let is_in_range = block
                                .append_operation(arith::cmpi(
                                    context,
                                    CmpiPredicate::Ule,
                                    large_candidate_squared,
                                    threshold,
                                    location,
                                ))
                                .result(0)?
                                .into();

                            let result = block
                                .append_operation(
                                    OperationBuilder::new("arith.select", location)
                                        .add_operands(&[is_in_range, large_candidate, result])
                                        .add_results(&[i256_ty])
                                        .build(),
                                )
                                .result(0)?
                                .into();

                            let k2 = block
                                .append_operation(arith::constant(
                                    context,
                                    IntegerAttribute::new(2, i256_ty).into(),
                                    location,
                                ))
                                .result(0)?
                                .into();

                            let shift_amount = block
                                .append_operation(arith::subi(
                                    block.argument(1)?.into(),
                                    k2,
                                    location,
                                ))
                                .result(0)?
                                .into();

                            let should_continue = block
                                .append_operation(arith::cmpi(
                                    context,
                                    CmpiPredicate::Sge,
                                    shift_amount,
                                    k0,
                                    location,
                                ))
                                .result(0)?
                                .into();
                            block.append_operation(scf::condition(
                                should_continue,
                                &[result, shift_amount],
                                location,
                            ));

                            region
                        },
                        {
                            let region = Region::new();
                            let block = region.append_block(Block::new(&[
                                (i256_ty, location),
                                (i256_ty, location),
                            ]));

                            block.append_operation(scf::r#yield(
                                &[block.argument(0)?.into(), block.argument(1)?.into()],
                                location,
                            ));

                            region
                        },
                        location,
                    ))
                    .result(0)?
                    .into();

                block.append_operation(scf::r#yield(&[result], location));

                region
            },
            location,
        ))
        .result(0)?
        .into();

    let result = entry
        .append_operation(arith::trunci(result, i128_ty, location))
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[entry.argument(0)?.into(), result], location));
    Ok(())
}

#[cfg(test)]
mod test {
    /* TODO: fix tests
    use crate::utils::test::{felt, load_cairo, run_program};
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;
    use num_bigint::ToBigUint;
    use serde_json::json;
    use std::ops::Shl;

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
        static ref U256_SQRT: (String, Program) = load_cairo! {
            use core::integer::u256_sqrt;

            fn run_test(value: u256) -> u128 {
                u256_sqrt(value)
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

    #[test]
    fn u256_sqrt() {
        let r = |hi, lo| run_program(&U256_SQRT, "run_test", json!([(), [lo, hi]]));

        assert_eq!(r(0u128, 0u128), json!([(), 0u64]));
        assert_eq!(r(u128::MAX, u128::MAX), json!([(), u128::MAX]));

        for i in 0..u128::BITS {
            let x = 1u128 << i;
            let y: u64 = x.to_biguint().unwrap().sqrt().try_into().unwrap();

            assert_eq!(r(0u128, x), json!([(), y]));
        }
        for i in 0..u128::BITS {
            let x = 1u128 << i;
            let y: u128 = x
                .to_biguint()
                .unwrap()
                .shl(128usize)
                .sqrt()
                .try_into()
                .unwrap();

            assert_eq!(r(x, 0u128), json!([(), y]));
        }
    }

    */
}
