//! # `i128`-related libfuncs
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
        int::{
            signed128::{Sint128Concrete, Sint128Traits},
            IntConstConcreteLibfunc, IntOperationConcreteLibfunc, IntOperator,
        },
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc, GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        cf, llvm,
    },
    ir::{
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        operation::OperationBuilder,
        r#type::IntegerType,
        Attribute, Block, Location, Value, ValueLike,
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
    selector: &Sint128Concrete,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        Sint128Concrete::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        Sint128Concrete::Operation(info) => {
            build_operation(context, registry, entry, location, helper, info)
        }
        Sint128Concrete::Equal(info) => {
            build_equal(context, registry, entry, location, helper, info)
        }
        Sint128Concrete::ToFelt252(info) => {
            build_to_felt252(context, registry, entry, location, helper, metadata, info)
        }
        Sint128Concrete::FromFelt252(info) => {
            build_from_felt252(context, registry, entry, location, helper, metadata, info)
        }
        Sint128Concrete::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, info)
        }
        Sint128Concrete::Diff(info) => build_diff(context, registry, entry, location, helper, info),
    }
}

/// Generate MLIR operations for the `i128_const` libfunc.
pub fn build_const<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &IntConstConcreteLibfunc<Sint128Traits>,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let value = info.c;
    let value_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.signature.branch_signatures[0].vars[0].ty,
    )?;

    let op0 = entry.append_operation(arith::constant(
        context,
        Attribute::parse(context, &format!("{value} : {value_ty}")).unwrap(),
        location,
    ));
    entry.append_operation(helper.br(0, &[op0.result(0)?.into()], location));

    Ok(())
}

/// Generate MLIR operations for the i128 operation libfunc.
pub fn build_operation<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    info: &IntOperationConcreteLibfunc,
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
        IntOperator::OverflowingAdd => "llvm.intr.sadd.with.overflow",
        IntOperator::OverflowingSub => "llvm.intr.ssub.with.overflow",
    };

    let values_type = lhs.r#type();

    let result_type = llvm::r#type::r#struct(
        context,
        &[values_type, IntegerType::new(context, 1).into()],
        false,
    );

    let result = entry
        .append_operation(
            OperationBuilder::new(op_name, location)
                .add_operands(&[lhs, rhs])
                .add_results(&[result_type])
                .build()?,
        )
        .result(0)?
        .into();

    let op_result = entry
        .append_operation(llvm::extract_value(
            context,
            result,
            DenseI64ArrayAttribute::new(context, &[0]),
            values_type,
            location,
        ))
        .result(0)?
        .into();

    // Create a const operation to get the 0 value to compare against
    let zero_const = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(0.into(), values_type).into(),
            location,
        ))
        .result(0)?
        .into();
    // Check if the result is positive
    let is_positive = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Sge,
            op_result,
            zero_const,
            location,
        ))
        .result(0)?
        .into();

    // Check overflow flag
    let op_overflow = entry
        .append_operation(llvm::extract_value(
            context,
            result,
            DenseI64ArrayAttribute::new(context, &[1]),
            IntegerType::new(context, 1).into(),
            location,
        ))
        .result(0)?
        .into();

    let block_not_overflow = helper.append_block(Block::new(&[]));
    let block_overflow = helper.append_block(Block::new(&[]));

    // The libfunc has three possible outputs: In Range, Overflow & Underflow
    entry.append_operation(cf::cond_br(
        context,
        op_overflow,
        block_overflow,
        block_not_overflow,
        &[],
        &[],
        location,
    ));
    // Check wether the result is positive to distinguish between undeflowing & overflowing results
    block_overflow.append_operation(helper.cond_br(
        context,
        is_positive,
        [1, 2],
        [&[range_check, op_result], &[range_check, op_result]],
        location,
    ));
    // No Oveflow/Underflow -> In range result
    block_not_overflow.append_operation(helper.br(0, &[range_check, op_result], location));
    Ok(())
}

/// Generate MLIR operations for the `i128_eq` libfunc.
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

    entry.append_operation(helper.cond_br(
        context,
        op0.result(0)?.into(),
        [1, 0],
        [&[]; 2],
        location,
    ));

    Ok(())
}

/// Generate MLIR operations for the `i128_is_zero` libfunc.
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

    let condition = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Eq,
            arg0,
            const_0,
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(context, condition, [0, 1], [&[], &[arg0]], location));

    Ok(())
}

/// Generate MLIR operations for the `i128_to_felt252` libfunc.
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
    let felt252_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;
    let value: Value = entry.argument(0)?.into();

    let result = entry
        .append_operation(arith::extui(value, felt252_ty, location))
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[result], location));

    Ok(())
}

/// Generate MLIR operations for the `i128_from_felt252` libfunc.
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

    let felt252_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.param_signatures()[1].ty,
    )?;
    let result_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[1].ty,
    )?;

    let const_max = entry
        .append_operation(arith::constant(
            context,
            Attribute::parse(context, &format!("{} : {}", i128::MAX, felt252_ty)).unwrap(),
            location,
        ))
        .result(0)?
        .into();

    let is_ule = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Ule,
            value,
            const_max,
            location,
        ))
        .result(0)?
        .into();

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

    let value = block_success
        .append_operation(arith::trunci(value, result_ty, location))
        .result(0)?
        .into();
    block_success.append_operation(helper.br(0, &[range_check, value], location));

    block_failure.append_operation(helper.br(1, &[range_check], location));

    Ok(())
}

/// Generate MLIR operations for the `i128_diff` libfunc.
pub fn build_diff<'ctx, 'this, TType, TLibfunc>(
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
    let range_check: Value = entry.argument(0)?.into();
    let lhs: Value = entry.argument(1)?.into();
    let rhs: Value = entry.argument(2)?.into();

    // Check if lhs >= rhs
    let is_ge = entry
        .append_operation(arith::cmpi(context, CmpiPredicate::Sge, lhs, rhs, location))
        .result(0)?
        .into();

    let result = entry
        .append_operation(arith::subi(lhs, rhs, location))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(
        context,
        is_ge,
        [0, 1],
        [&[range_check, result], &[range_check, result]],
        location,
    ));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{jit_enum, jit_panic, jit_struct, load_cairo, run_program_assert_output},
        values::JITValue,
    };
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;
    use starknet_types_core::felt::Felt;

    lazy_static! {
        static ref I128_OVERFLOWING_ADD: (String, Program) = load_cairo! {
            fn run_test(lhs: i128, rhs: i128) -> i128 {
                lhs + rhs
            }
        };
        static ref I128_OVERFLOWING_SUB: (String, Program) = load_cairo! {
            fn run_test(lhs: i128, rhs: i128) -> i128 {
                lhs - rhs
            }
        };
        static ref I128_EQUAL: (String, Program) = load_cairo! {
            fn run_test(lhs: i128, rhs: i128) -> bool {
                lhs == rhs
            }
        };
        static ref I128_IS_ZERO: (String, Program) = load_cairo! {
            use zeroable::IsZeroResult;

            extern fn i128_is_zero(a: i128) -> IsZeroResult<i128> implicits() nopanic;

            fn run_test(value: i128) -> bool {
                match i128_is_zero(value) {
                    IsZeroResult::Zero(_) => true,
                    IsZeroResult::NonZero(_) => false,
                }
            }
        };
    }

    #[test]
    fn i128_const_min() {
        let program = load_cairo!(
            fn run_test() -> i128 {
                -170141183460469231731687303715884105728_i128
            }
        );

        run_program_assert_output(&program, "run_test", &[], &[i128::MIN.into()]);
    }

    #[test]
    fn i128_const_max() {
        let program = load_cairo!(
            fn run_test() -> i128 {
                170141183460469231731687303715884105727_i128
            }
        );

        run_program_assert_output(&program, "run_test", &[], &[(i128::MAX).into()]);
    }

    #[test]
    fn i128_to_felt252() {
        let program = load_cairo!(
            use traits::Into;

            fn run_test() -> felt252 {
                2_i128.into()
            }
        );

        run_program_assert_output(&program, "run_test", &[], &[Felt::from(2).into()]);
    }

    #[test]
    fn i128_from_felt252() {
        let program = load_cairo!(
            use traits::TryInto;

            fn run_test() -> (Option<i128>, Option<i128>) {
                (
                    170141183460469231731687303715884105727.try_into(),
                    170141183460469231731687303715884105728.try_into(),
                )
            }
        );

        run_program_assert_output(
            &program,
            "run_test",
            &[],
            &[jit_struct!(
                jit_enum!(0, 170141183460469231731687303715884105727i128.into()),
                jit_enum!(1, jit_struct!()),
            )],
        );
    }

    #[test]
    fn i128_overflowing_add() {
        #[track_caller]
        fn run(lhs: i128, rhs: i128) {
            let program = &I128_OVERFLOWING_ADD;
            let error = Felt::from_bytes_be_slice(b"i128_add Overflow");

            let add = lhs.checked_add(rhs);

            match add {
                Some(result) => {
                    run_program_assert_output(
                        program,
                        "run_test",
                        &[lhs.into(), rhs.into()],
                        &[jit_enum!(0, jit_struct!(result.into()))],
                    );
                }
                None => {
                    run_program_assert_output(
                        program,
                        "run_test",
                        &[lhs.into(), rhs.into()],
                        &[jit_panic!(JITValue::Felt252(error))],
                    );
                }
            }
        }

        const MAX: i128 = i128::MAX;

        run(0, 0);
        run(0, 1);
        run(0, MAX - 1);
        run(0, MAX);

        run(1, 0);
        run(1, 1);
        run(1, MAX - 1);
        run(1, MAX);

        run(MAX - 1, 0);
        run(MAX - 1, 1);
        run(MAX - 1, MAX - 1);
        run(MAX - 1, MAX);

        run(MAX, 0);
        run(MAX, 1);
        run(MAX, MAX - 1);
        run(MAX, MAX);
    }

    #[test]
    fn i128_overflowing_sub() {
        #[track_caller]
        fn run(lhs: i128, rhs: i128) {
            let program = &I128_OVERFLOWING_SUB;
            let error = Felt::from_bytes_be_slice(b"i128_sub Overflow");

            let add = lhs.checked_sub(rhs);

            match add {
                Some(result) => {
                    run_program_assert_output(
                        program,
                        "run_test",
                        &[lhs.into(), rhs.into()],
                        &[jit_enum!(0, jit_struct!(result.into()))],
                    );
                }
                None => {
                    run_program_assert_output(
                        program,
                        "run_test",
                        &[lhs.into(), rhs.into()],
                        &[jit_panic!(JITValue::Felt252(error))],
                    );
                }
            }
        }

        const MAX: i128 = i128::MAX;

        run(0, 0);
        run(0, 1);
        run(0, MAX - 1);
        run(0, MAX);

        run(1, 0);
        run(1, 1);
        run(1, MAX - 1);
        run(1, MAX);

        run(MAX - 1, 0);
        run(MAX - 1, 1);
        run(MAX - 1, MAX - 1);
        run(MAX - 1, MAX);

        run(MAX, 0);
        run(MAX, 1);
        run(MAX, MAX - 1);
        run(MAX, MAX);
    }

    #[test]
    fn i128_equal() {
        let program = &I128_EQUAL;

        run_program_assert_output(
            program,
            "run_test",
            &[0i128.into(), 0i128.into()],
            &[jit_enum!(1, jit_struct!())],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1i128.into(), 0i128.into()],
            &[jit_enum!(0, jit_struct!())],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0i128.into(), 1i128.into()],
            &[jit_enum!(0, jit_struct!())],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1i128.into(), 1i128.into()],
            &[jit_enum!(1, jit_struct!())],
        );
    }

    #[test]
    fn i128_is_zero() {
        let program = &I128_IS_ZERO;

        run_program_assert_output(
            program,
            "run_test",
            &[0i128.into()],
            &[jit_enum!(1, jit_struct!())],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1i128.into()],
            &[jit_enum!(0, jit_struct!())],
        );
    }

    #[test]
    fn i128_safe_divmod() {
        let program = &I128_IS_ZERO;

        run_program_assert_output(
            program,
            "run_test",
            &[0i128.into(), 0i128.into()],
            &[jit_enum!(1, jit_struct!())],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0i128.into(), 1i128.into()],
            &[jit_enum!(1, jit_struct!())],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0i128.into(), i128::MAX.into()],
            &[jit_enum!(1, jit_struct!())],
        );

        run_program_assert_output(
            program,
            "run_test",
            &[1i128.into(), 0i128.into()],
            &[jit_enum!(0, jit_struct!())],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1i128.into(), 1i128.into()],
            &[jit_enum!(0, jit_struct!())],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1i128.into(), i128::MAX.into()],
            &[jit_enum!(0, jit_struct!())],
        );

        run_program_assert_output(
            program,
            "run_test",
            &[i128::MAX.into(), 0i128.into()],
            &[jit_enum!(0, jit_struct!())],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[i128::MAX.into(), 1i128.into()],
            &[jit_enum!(0, jit_struct!())],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[i128::MAX.into(), i128::MAX.into()],
            &[jit_enum!(0, jit_struct!())],
        );
    }
}
