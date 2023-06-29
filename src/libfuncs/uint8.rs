//! # `u8`-related libfuncs

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{metadata::MetadataStorage, types::TypeBuilder};
use cairo_lang_sierra::{
    extensions::{
        int::{
            unsigned::{
                Uint8Concrete, Uint8Traits, UintConcrete, UintConstConcreteLibfunc,
                UintOperationConcreteLibfunc,
            },
            IntOperator,
        },
        lib_func::SignatureOnlyConcreteLibfunc,
        GenericLibfunc, GenericType,
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
    selector: &Uint8Concrete,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    match selector {
        UintConcrete::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::Operation(info) => {
            build_operation(context, registry, entry, location, helper, info)
        }
        UintConcrete::SquareRoot(_) => todo!(),
        UintConcrete::Equal(info) => build_equal(context, registry, entry, location, helper, info),
        UintConcrete::ToFelt252(_) => todo!(),
        UintConcrete::FromFelt252(_) => todo!(),
        UintConcrete::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, info)
        }
        UintConcrete::Divmod(info) => {
            build_divmod(context, registry, entry, location, helper, info)
        }
        UintConcrete::WideMul(_) => todo!(),
    }
}

/// Generate MLIR operations for the `u8_const` libfunc.
pub fn build_const<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &UintConstConcreteLibfunc<Uint8Traits>,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let value = info.c;
    let value_ty = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)
        .unwrap()
        .build(context, helper, registry, metadata)
        .unwrap();

    let op0 = entry.append_operation(arith::constant(
        context,
        Attribute::parse(context, &format!("{value} : {value_ty}")).unwrap(),
        location,
    ));
    entry.append_operation(helper.br(0, &[op0.result(0).unwrap().into()], location));

    Ok(())
}

/// Generate MLIR operations for the u8 operation libfunc.
pub fn build_operation<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    info: &UintOperationConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let range_check: Value = entry.argument(0).unwrap().into();
    let lhs: Value = entry.argument(1).unwrap().into();
    let rhs: Value = entry.argument(2).unwrap().into();

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
    let result = op.result(0).unwrap().into();

    let op = entry.append_operation(llvm::extract_value(
        context,
        result,
        DenseI64ArrayAttribute::new(context, &[0]),
        values_type,
        location,
    ));
    let op_result = op.result(0).unwrap().into();

    let op = entry.append_operation(llvm::extract_value(
        context,
        result,
        DenseI64ArrayAttribute::new(context, &[1]),
        IntegerType::new(context, 1).into(),
        location,
    ));
    let op_overflow = op.result(0).unwrap().into();

    entry.append_operation(helper.cond_br(
        op_overflow,
        [1, 0],
        [&[range_check, op_result], &[range_check, op_result]],
        location,
    ));
    Ok(())
}

/// Generate MLIR operations for the `u8_eq` libfunc.
pub fn build_equal<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let arg0: Value = entry.argument(0).unwrap().into();
    let arg1: Value = entry.argument(1).unwrap().into();

    let op0 = entry.append_operation(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        arg0,
        arg1,
        location,
    ));

    entry.append_operation(helper.cond_br(
        op0.result(0).unwrap().into(),
        [0, 1],
        [&[]; 2],
        location,
    ));

    Ok(())
}

/// Generate MLIR operations for the `u8_is_zero` libfunc.
pub fn build_is_zero<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let arg0: Value = entry.argument(0).unwrap().into();

    let op = entry.append_operation(arith::constant(
        context,
        IntegerAttribute::new(0, arg0.r#type()).into(),
        location,
    ));
    let const_0 = op.result(0).unwrap().into();

    let op = entry.append_operation(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        arg0,
        const_0,
        location,
    ));
    let condition = op.result(0).unwrap().into();

    entry.append_operation(helper.cond_br(condition, [0, 1], [&[], &[arg0]], location));

    Ok(())
}

/// Generate MLIR operations for the `u8_safe_divmod` libfunc.
pub fn build_divmod<'ctx, 'this, TType, TLibfunc>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let lhs: Value = entry.argument(0).unwrap().into();
    let rhs: Value = entry.argument(1).unwrap().into();

    let op = entry.append_operation(arith::divui(lhs, rhs, location));

    let result_div = op.result(0).unwrap().into();
    let op = entry.append_operation(arith::remui(lhs, rhs, location));
    let result_rem = op.result(0).unwrap().into();

    entry.append_operation(helper.br(0, &[result_div, result_rem], location));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::run_cairo;
    use serde_json::json;

    fn error_value(n: u128) -> serde_json::Value {
        json!([(), [1, [[], [n]]]])
    }

    #[test]
    fn run_const_min() {
        let result = run_cairo! { run_test() in mod {
            fn run_test() -> u8 {
                0_u8
            }
        }};

        assert_eq!(result, json!([0]));
    }

    #[test]
    fn run_const_max() {
        let result = run_cairo! { run_test() in mod {
            fn run_test() -> u8 {
                255_u8
            }
        }};

        assert_eq!(result, json!([255]));
    }

    #[test]
    fn run_add() {
        fn run<const LHS: u8, const RHS: u8>() -> serde_json::Value {
            run_cairo! { run_test((), LHS, RHS) in mod {
                fn run_test(lhs: u8, rhs: u8) -> u8 {
                    lhs + rhs
                }
            }}
        }

        assert_eq!(run::<0, 0>(), json!([(), [0, [0]]]));
        assert_eq!(run::<0, 1>(), json!([(), [0, [1]]]));
        assert_eq!(run::<0, 254>(), json!([(), [0, [254]]]));
        assert_eq!(run::<0, 255>(), json!([(), [0, [255]]]));

        assert_eq!(run::<1, 0>(), json!([(), [0, [1]]]));
        assert_eq!(run::<1, 1>(), json!([(), [0, [2]]]));
        assert_eq!(run::<1, 254>(), json!([(), [0, [255]]]));
        assert_eq!(
            run::<1, 255>(),
            error_value(608642104203229548495787928534675319)
        );

        assert_eq!(run::<254, 0>(), json!([(), [0, [254]]]));
        assert_eq!(run::<254, 1>(), json!([(), [0, [255]]]));
        assert_eq!(
            run::<254, 254>(),
            error_value(608642104203229548495787928534675319)
        );
        assert_eq!(
            run::<254, 255>(),
            error_value(608642104203229548495787928534675319)
        );

        assert_eq!(run::<255, 0>(), json!([(), [0, [255]]]));
        assert_eq!(
            run::<255, 1>(),
            error_value(608642104203229548495787928534675319)
        );
        assert_eq!(
            run::<255, 254>(),
            error_value(608642104203229548495787928534675319)
        );
        assert_eq!(
            run::<255, 255>(),
            error_value(608642104203229548495787928534675319)
        );
    }

    #[test]
    fn run_sub() {
        fn run<const LHS: u8, const RHS: u8>() -> serde_json::Value {
            run_cairo! { run_test((), LHS, RHS) in mod {
                fn run_test(lhs: u8, rhs: u8) -> u8 {
                    lhs - rhs
                }
            }}
        }

        assert_eq!(run::<0, 0>(), json!([(), [0, [0]]]));
        assert_eq!(
            run::<0, 1>(),
            error_value(608642109794502019480482122260311927)
        );
        assert_eq!(
            run::<0, 254>(),
            error_value(608642109794502019480482122260311927)
        );
        assert_eq!(
            run::<0, 255>(),
            error_value(608642109794502019480482122260311927)
        );

        assert_eq!(run::<1, 0>(), json!([(), [0, [1]]]));
        assert_eq!(run::<1, 1>(), json!([(), [0, [0]]]));
        assert_eq!(
            run::<1, 254>(),
            error_value(608642109794502019480482122260311927)
        );
        assert_eq!(
            run::<1, 255>(),
            error_value(608642109794502019480482122260311927)
        );

        assert_eq!(run::<254, 0>(), json!([(), [0, [254]]]));
        assert_eq!(run::<254, 1>(), json!([(), [0, [253]]]));
        assert_eq!(run::<254, 254>(), json!([(), [0, [0]]]));
        assert_eq!(
            run::<254, 255>(),
            error_value(608642109794502019480482122260311927)
        );

        assert_eq!(run::<255, 0>(), json!([(), [0, [255]]]));
        assert_eq!(run::<255, 1>(), json!([(), [0, [254]]]));
        assert_eq!(run::<255, 254>(), json!([(), [0, [1]]]));
        assert_eq!(run::<255, 255>(), json!([(), [0, [0]]]));
    }
}
