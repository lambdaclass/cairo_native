//! # `u64`-related libfuncs

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
        int::{
            unsigned::{
                Uint64Concrete, Uint64Traits, UintConcrete, UintConstConcreteLibfunc,
                UintOperationConcreteLibfunc,
            },
            IntOperator,
        },
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc, GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        cf, llvm, scf,
    },
    ir::{
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        operation::OperationBuilder,
        r#type::IntegerType,
        Attribute, Block, Location, Region, Value, ValueLike,
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
    selector: &Uint64Concrete,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        UintConcrete::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::Operation(info) => {
            build_operation(context, registry, entry, location, helper, info)
        }
        UintConcrete::SquareRoot(info) => {
            build_square_root(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::Equal(info) => build_equal(context, registry, entry, location, helper, info),
        UintConcrete::ToFelt252(info) => {
            build_to_felt252(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::FromFelt252(info) => {
            build_from_felt252(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, info)
        }
        UintConcrete::Divmod(info) => {
            build_divmod(context, registry, entry, location, helper, info)
        }
        UintConcrete::WideMul(_) => todo!(),
    }
}

/// Generate MLIR operations for the `u64_const` libfunc.
pub fn build_const<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &UintConstConcreteLibfunc<Uint64Traits>,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let value = info.c;
    let value_ty = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)?
        .build(context, helper, registry, metadata)?;

    let op0 = entry.append_operation(arith::constant(
        context,
        Attribute::parse(context, &format!("{value} : {value_ty}")).unwrap(),
        location,
    ));
    entry.append_operation(helper.br(0, &[op0.result(0)?.into()], location));

    Ok(())
}

/// Generate MLIR operations for the u64 operation libfunc.
pub fn build_operation<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    info: &UintOperationConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let range_check: Value = entry.argument(0)?.into();
    let lhs: Value = entry.argument(1)?.into();
    let rhs: Value = entry.argument(2)?.into();

    let op_name = match info.operator {
        IntOperator::OverflowingAdd => "llvm.intr.uadd.with.overflow",
        IntOperator::OverflowingSub => "llvm.intr.usub.with.overflow",
    };

    let values_type = lhs.r#type();

    let result_type = llvm::r#type::r#struct(
        context,
        &[values_type, IntegerType::new(context, 1).into()],
        false,
    );

    let op = entry.append_operation(
        OperationBuilder::new(op_name, location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_type])
            .build(),
    );
    let result = op.result(0)?.into();

    let op = entry.append_operation(llvm::extract_value(
        context,
        result,
        DenseI64ArrayAttribute::new(context, &[0]),
        values_type,
        location,
    ));
    let op_result = op.result(0)?.into();

    let op = entry.append_operation(llvm::extract_value(
        context,
        result,
        DenseI64ArrayAttribute::new(context, &[1]),
        IntegerType::new(context, 1).into(),
        location,
    ));
    let op_overflow = op.result(0)?.into();

    entry.append_operation(helper.cond_br(
        op_overflow,
        [1, 0],
        [&[range_check, op_result], &[range_check, op_result]],
        location,
    ));
    Ok(())
}

/// Generate MLIR operations for the `u64_eq` libfunc.
pub fn build_equal<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let arg0: Value = entry.argument(0)?.into();
    let arg1: Value = entry.argument(1)?.into();

    let op0 = entry.append_operation(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        arg0,
        arg1,
        location,
    ));

    entry.append_operation(helper.cond_br(op0.result(0)?.into(), [1, 0], [&[]; 2], location));

    Ok(())
}

/// Generate MLIR operations for the `u64_is_zero` libfunc.
pub fn build_is_zero<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let arg0: Value = entry.argument(0)?.into();

    let op = entry.append_operation(arith::constant(
        context,
        IntegerAttribute::new(0, arg0.r#type()).into(),
        location,
    ));
    let const_0 = op.result(0)?.into();

    let op = entry.append_operation(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        arg0,
        const_0,
        location,
    ));
    let condition = op.result(0)?.into();

    entry.append_operation(helper.cond_br(condition, [0, 1], [&[], &[arg0]], location));

    Ok(())
}

/// Generate MLIR operations for the `u64_safe_divmod` libfunc.
pub fn build_divmod<'ctx, 'this, TType, TLibfunc>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let lhs: Value = entry.argument(1)?.into();
    let rhs: Value = entry.argument(2)?.into();

    let op = entry.append_operation(arith::divui(lhs, rhs, location));

    let result_div = op.result(0)?.into();
    let op = entry.append_operation(arith::remui(lhs, rhs, location));
    let result_rem = op.result(0)?.into();

    entry.append_operation(helper.br(
        0,
        &[entry.argument(0)?.into(), result_div, result_rem],
        location,
    ));

    Ok(())
}

/// Generate MLIR operations for the `u64_to_felt252` libfunc.
pub fn build_to_felt252<'ctx, 'this, TType, TLibfunc>(
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
    let felt252_ty = registry
        .get_type(&info.branch_signatures()[0].vars[0].ty)?
        .build(context, helper, registry, metadata)?;
    let value: Value = entry.argument(0)?.into();

    let op = entry.append_operation(arith::extui(value, felt252_ty, location));

    let result = op.result(0)?.into();

    entry.append_operation(helper.br(0, &[result], location));

    Ok(())
}

/// Generate MLIR operations for the `u64_sqrt` libfunc.
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
    let i32_ty = IntegerType::new(context, 32).into();
    let i64_ty = IntegerType::new(context, 64).into();

    let k1 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(1, i64_ty).into(),
            location,
        ))
        .result(0)?
        .into();

    let is_small = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Ule,
            entry.argument(1)?.into(),
            k1,
            location,
        ))
        .result(0)?
        .into();

    let result = entry
        .append_operation(scf::r#if(
            is_small,
            &[i64_ty],
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                block.append_operation(scf::r#yield(&[entry.argument(1)?.into()], location));

                region
            },
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                let k64 = entry
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(64, i64_ty).into(),
                        location,
                    ))
                    .result(0)?
                    .into();
                let k1_i1 = entry
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(1, IntegerType::new(context, 1).into()).into(),
                        location,
                    ))
                    .result(0)?
                    .into();

                let leading_zeros = block
                    .append_operation(
                        OperationBuilder::new("llvm.intr.ctlz", location)
                            .add_operands(&[entry.argument(1)?.into(), k1_i1])
                            .add_results(&[i64_ty])
                            .build(),
                    )
                    .result(0)?
                    .into();

                let num_bits = block
                    .append_operation(arith::subi(k64, leading_zeros, location))
                    .result(0)?
                    .into();

                let shift_amount = block
                    .append_operation(arith::addi(num_bits, k1, location))
                    .result(0)?
                    .into();

                let parity_mask = block
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(-2, i64_ty).into(),
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
                        IntegerAttribute::new(0, i64_ty).into(),
                        location,
                    ))
                    .result(0)?
                    .into();
                let result = block
                    .append_operation(scf::r#while(
                        &[k0, shift_amount],
                        &[i64_ty, i64_ty],
                        {
                            let region = Region::new();
                            let block = region.append_block(Block::new(&[
                                (i64_ty, location),
                                (i64_ty, location),
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
                                    entry.argument(1)?.into(),
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
                                    k64,
                                    location,
                                ))
                                .result(0)?
                                .into();
                            let threshold = block
                                .append_operation(
                                    OperationBuilder::new("arith.select", location)
                                        .add_operands(&[threshold_is_poison, k0, threshold])
                                        .add_results(&[i64_ty])
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
                                        .add_results(&[i64_ty])
                                        .build(),
                                )
                                .result(0)?
                                .into();

                            let k2 = block
                                .append_operation(arith::constant(
                                    context,
                                    IntegerAttribute::new(2, i64_ty).into(),
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
                                (i64_ty, location),
                                (i64_ty, location),
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
        .append_operation(arith::trunci(result, i32_ty, location))
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[entry.argument(0)?.into(), result], location));
    Ok(())
}

/// Generate MLIR operations for the `u64_from_felt252` libfunc.
pub fn build_from_felt252<'ctx, 'this, TType, TLibfunc>(
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
    let range_check: Value = entry.argument(0)?.into();
    let value: Value = entry.argument(1)?.into();

    let felt252_ty = registry
        .get_type(&info.param_signatures()[1].ty)?
        .build(context, helper, registry, metadata)?;
    let result_ty = registry
        .get_type(&info.branch_signatures()[0].vars[1].ty)?
        .build(context, helper, registry, metadata)?;

    let op = entry.append_operation(arith::constant(
        context,
        Attribute::parse(context, &format!("{} : {}", u64::MAX, felt252_ty)).unwrap(),
        location,
    ));
    let const_max = op.result(0)?.into();

    let op = entry.append_operation(arith::cmpi(
        context,
        CmpiPredicate::Ule,
        value,
        const_max,
        location,
    ));
    let is_ule = op.result(0)?.into();

    let block_success = helper.append_block(Block::new(&[]));
    let block_failure = helper.append_block(Block::new(&[]));

    entry.append_operation(cf::cond_br(
        context,
        is_ule,
        block_success,
        block_failure,
        &[],
        &[],
        location,
    ));

    let op = block_success.append_operation(arith::trunci(value, result_ty, location));
    let value = op.result(0)?.into();
    block_success.append_operation(helper.br(0, &[range_check, value], location));

    block_failure.append_operation(helper.br(1, &[range_check], location));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        types::felt252::PRIME,
        utils::test::{load_cairo, run_program},
    };
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;
    use num_bigint::{BigInt, Sign, ToBigUint};
    use serde_json::json;
    use std::ops::Neg;

    lazy_static! {
        static ref U64_OVERFLOWING_ADD: (String, Program) = load_cairo! {
            fn run_test(lhs: u64, rhs: u64) -> u64 {
                lhs + rhs
            }
        };
        static ref U64_OVERFLOWING_SUB: (String, Program) = load_cairo! {
            fn run_test(lhs: u64, rhs: u64) -> u64 {
                lhs - rhs
            }
        };
        static ref U64_SAFE_DIVMOD: (String, Program) = load_cairo! {
            fn run_test(lhs: u64, rhs: u64) -> (u64, u64) {
                let q = lhs / rhs;
                let r = lhs % rhs;

                (q, r)
            }
        };
        static ref U64_EQUAL: (String, Program) = load_cairo! {
            fn run_test(lhs: u64, rhs: u64) -> bool {
                lhs == rhs
            }
        };
        static ref U64_IS_ZERO: (String, Program) = load_cairo! {
            use zeroable::IsZeroResult;

            extern fn u64_is_zero(a: u64) -> IsZeroResult<u64> implicits() nopanic;

            fn run_test(value: u64) -> bool {
                match u64_is_zero(value) {
                    IsZeroResult::Zero(_) => true,
                    IsZeroResult::NonZero(_) => false,
                }
            }
        };
        static ref U64_SQRT: (String, Program) = load_cairo! {
            use core::integer::u64_sqrt;

            fn run_test(value: u64) -> u32 {
                u64_sqrt(value)
            }
        };
    }

    // Parse numeric string into felt, wrapping negatives around the prime modulo.
    fn f(value: &str) -> [u32; 8] {
        let value = value.parse::<BigInt>().unwrap();
        let value = match value.sign() {
            Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
            _ => value.to_biguint().unwrap(),
        };

        let mut u32_digits = value.to_u32_digits();
        u32_digits.resize(8, 0);
        u32_digits.try_into().unwrap()
    }

    #[test]
    fn u64_const_min() {
        let program = load_cairo!(
            fn run_test() -> u64 {
                0_u64
            }
        );
        let result = run_program(&program, "run_test", json!([]));

        assert_eq!(result, json!([0]));
    }

    #[test]
    fn u64_const_max() {
        let program = load_cairo!(
            fn run_test() -> u64 {
                18446744073709551615_u64
            }
        );
        let result = run_program(&program, "run_test", json!([]));

        assert_eq!(result, json!([18446744073709551615u64]));
    }

    #[test]
    fn u64_to_felt252() {
        let program = load_cairo!(
            use traits::Into;

            fn run_test() -> felt252 {
                2_u64.into()
            }
        );
        let result = run_program(&program, "run_test", json!([]));

        assert_eq!(result, json!([[2, 0, 0, 0, 0, 0, 0, 0]]));
    }

    #[test]
    fn u64_from_felt252() {
        let program = load_cairo!(
            use traits::TryInto;

            fn run_test() -> (Option<u64>, Option<u64>) {
                (
                    18446744073709551615.try_into(),
                    18446744073709551616.try_into(),
                )
            }
        );
        let result = run_program(&program, "run_test", json!([null]));

        assert_eq!(
            result,
            json!([null, [[0, 18446744073709551615u64], [1, []]]])
        );
    }

    #[test]
    fn u64_overflowing_add() {
        fn run<const LHS: u64, const RHS: u64>() -> serde_json::Value {
            run_program(&U64_OVERFLOWING_ADD, "run_test", json!([(), LHS, RHS]))
        }

        let add_error = f("155801121779312277930962096923588980599");

        assert_eq!(run::<0, 0>(), json!([(), [0, [0]]]));
        assert_eq!(run::<0, 1>(), json!([(), [0, [1]]]));
        assert_eq!(
            run::<0, 18446744073709551614>(),
            json!([(), [0, [18446744073709551614u64]]])
        );
        assert_eq!(
            run::<0, 18446744073709551615>(),
            json!([(), [0, [18446744073709551615u64]]])
        );

        assert_eq!(run::<1, 0>(), json!([(), [0, [1]]]));
        assert_eq!(run::<1, 1>(), json!([(), [0, [2]]]));
        assert_eq!(
            run::<1, 18446744073709551614>(),
            json!([(), [0, [18446744073709551615u64]]])
        );
        assert_eq!(
            run::<1, 18446744073709551615>(),
            json!([(), [1, [[], [add_error]]]])
        );

        assert_eq!(
            run::<18446744073709551614, 0>(),
            json!([(), [0, [18446744073709551614u64]]])
        );
        assert_eq!(
            run::<18446744073709551614, 1>(),
            json!([(), [0, [18446744073709551615u64]]])
        );
        assert_eq!(
            run::<18446744073709551614, 18446744073709551614>(),
            json!([(), [1, [[], [add_error]]]])
        );
        assert_eq!(
            run::<18446744073709551614, 18446744073709551615>(),
            json!([(), [1, [[], [add_error]]]])
        );

        assert_eq!(
            run::<18446744073709551615, 0>(),
            json!([(), [0, [18446744073709551615u64]]])
        );
        assert_eq!(
            run::<18446744073709551615, 1>(),
            json!([(), [1, [[], [add_error]]]])
        );
        assert_eq!(
            run::<18446744073709551615, 18446744073709551614>(),
            json!([(), [1, [[], [add_error]]]])
        );
        assert_eq!(
            run::<18446744073709551615, 18446744073709551615>(),
            json!([(), [1, [[], [add_error]]]])
        );
    }

    #[test]
    fn u64_overflowing_sub() {
        fn run<const LHS: u64, const RHS: u64>() -> serde_json::Value {
            run_program(&U64_OVERFLOWING_SUB, "run_test", json!([(), LHS, RHS]))
        }

        let sub_error = f("155801121784903550401946791117314617207");

        assert_eq!(run::<0, 0>(), json!([(), [0, [0]]]));
        assert_eq!(run::<0, 1>(), json!([(), [1, [[], [sub_error]]]]));
        assert_eq!(
            run::<0, 18446744073709551614>(),
            json!([(), [1, [[], [sub_error]]]])
        );
        assert_eq!(
            run::<0, 18446744073709551615>(),
            json!([(), [1, [[], [sub_error]]]])
        );

        assert_eq!(run::<1, 0>(), json!([(), [0, [1]]]));
        assert_eq!(run::<1, 1>(), json!([(), [0, [0]]]));
        assert_eq!(
            run::<1, 18446744073709551614>(),
            json!([(), [1, [[], [sub_error]]]])
        );
        assert_eq!(
            run::<1, 18446744073709551615>(),
            json!([(), [1, [[], [sub_error]]]])
        );

        assert_eq!(
            run::<18446744073709551614, 0>(),
            json!([(), [0, [18446744073709551614u64]]])
        );
        assert_eq!(
            run::<18446744073709551614, 1>(),
            json!([(), [0, [18446744073709551613u64]]])
        );
        assert_eq!(
            run::<18446744073709551614, 18446744073709551614>(),
            json!([(), [0, [0]]])
        );
        assert_eq!(
            run::<18446744073709551614, 18446744073709551615>(),
            json!([(), [1, [[], [sub_error]]]])
        );

        assert_eq!(
            run::<18446744073709551615, 0>(),
            json!([(), [0, [18446744073709551615u64]]])
        );
        assert_eq!(
            run::<18446744073709551615, 1>(),
            json!([(), [0, [18446744073709551614u64]]])
        );
        assert_eq!(
            run::<18446744073709551615, 18446744073709551614>(),
            json!([(), [0, [1]]])
        );
        assert_eq!(
            run::<18446744073709551615, 18446744073709551615>(),
            json!([(), [0, [0]]])
        );
    }

    #[test]
    fn u64_equal() {
        let r = |lhs, rhs| run_program(&U64_EQUAL, "run_test", json!([lhs, rhs]));

        assert_eq!(r(0, 0), json!([[1, []]]));
        assert_eq!(r(0, 1), json!([[0, []]]));
        assert_eq!(r(1, 0), json!([[0, []]]));
        assert_eq!(r(1, 1), json!([[1, []]]));
    }

    #[test]
    fn u64_is_zero() {
        let r = |value| run_program(&U64_IS_ZERO, "run_test", json!([value]));

        assert_eq!(r(0), json!([[1, []]]));
        assert_eq!(r(1), json!([[0, []]]));
    }

    #[test]
    fn u64_safe_divmod() {
        let r = |lhs, rhs| run_program(&U64_SAFE_DIVMOD, "run_test", json!([(), lhs, rhs]));

        let u64_is_zero = json!([f("8445995464992694320")]);

        assert_eq!(r(0, 0), json!([(), [1, [[], u64_is_zero]]]));
        assert_eq!(r(0, 1), json!([(), [0, [[0u64, 0u64]]]]));
        assert_eq!(
            r(0, 0xFFFFFFFFFFFFFFFFu64),
            json!([(), [0, [[0u64, 0u64]]]])
        );

        assert_eq!(r(1, 0), json!([(), [1, [[], u64_is_zero]]]));
        assert_eq!(r(1, 1), json!([(), [0, [[1u64, 0u64]]]]));
        assert_eq!(
            r(1, 0xFFFFFFFFFFFFFFFFu64),
            json!([(), [0, [[0u64, 1u64]]]])
        );

        assert_eq!(
            r(0xFFFFFFFFFFFFFFFFu64, 0),
            json!([(), [1, [[], u64_is_zero]]])
        );
        assert_eq!(
            r(0xFFFFFFFFFFFFFFFFu64, 1),
            json!([(), [0, [[0xFFFFFFFFFFFFFFFFu64, 0u64]]]])
        );
        assert_eq!(
            r(0xFFFFFFFFFFFFFFFFu64, 0xFFFFFFFFFFFFFFFFu64),
            json!([(), [0, [[1u64, 0u64]]]])
        );
    }

    #[test]
    fn u64_sqrt() {
        let r = |value| run_program(&U64_SQRT, "run_test", json!([(), value]));

        assert_eq!(r(0u64), json!([(), 0u32]));
        assert_eq!(r(u64::MAX), json!([(), u32::MAX]));

        for i in 0..u64::BITS {
            let x = 1u64 << i;
            let y: u64 = x.to_biguint().unwrap().sqrt().try_into().unwrap();

            assert_eq!(r(x), json!([(), y]));
        }
    }
}
