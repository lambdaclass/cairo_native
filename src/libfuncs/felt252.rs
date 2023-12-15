//! # `Felt`-related libfuncs

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::{prime_modulo::PrimeModuloMeta, MetadataStorage},
    types::{felt252::Felt252, TypeBuilder},
    utils::{mlir_asm, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        felt252::{
            Felt252BinaryOperationConcrete, Felt252BinaryOperator, Felt252Concrete,
            Felt252ConstConcreteLibfunc,
        },
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc, GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::arith::{self, CmpiPredicate},
    ir::{
        attribute::IntegerAttribute, r#type::IntegerType, Attribute, Block, Location, Value,
        ValueLike,
    },
    Context,
};
use num_bigint::{Sign, ToBigInt};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Felt252Concrete,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        Felt252Concrete::BinaryOperation(info) => {
            build_binary_operation(context, registry, entry, location, helper, metadata, info)
        }
        Felt252Concrete::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        Felt252Concrete::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the following libfuncs:
///   - `felt252_add` and `felt252_add_const`.
///   - `felt252_sub` and `felt252_sub_const`.
///   - `felt252_mul` and `felt252_mul_const`.
///   - `felt252_div` and `felt252_div_const`.
pub fn build_binary_operation<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &Felt252BinaryOperationConcrete,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let bool_ty = IntegerType::new(context, 1).into();
    let felt252_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;
    let i256 = IntegerType::new(context, 256).into();
    let i512 = IntegerType::new(context, 512).into();

    let attr_prime_i256 = Attribute::parse(
        context,
        &format!(
            "{} : {i256}",
            metadata.get::<PrimeModuloMeta<Felt252>>().unwrap().prime()
        ),
    )
    .unwrap();
    let attr_prime_i512 = Attribute::parse(
        context,
        &format!(
            "{} : {i512}",
            metadata.get::<PrimeModuloMeta<Felt252>>().unwrap().prime()
        ),
    )
    .unwrap();

    let attr_cmp_uge = IntegerAttribute::new(
        CmpiPredicate::Uge as i64,
        IntegerType::new(context, 64).into(),
    )
    .into();
    let attr_cmp_ult = IntegerAttribute::new(
        CmpiPredicate::Ult as i64,
        IntegerType::new(context, 64).into(),
    )
    .into();

    let (op, lhs, rhs) = match info {
        Felt252BinaryOperationConcrete::WithVar(operation) => (
            operation.operator,
            entry.argument(0)?.into(),
            entry.argument(1)?.into(),
        ),
        Felt252BinaryOperationConcrete::WithConst(operation) => {
            let value = match operation.c.sign() {
                Sign::Minus => {
                    let prime = metadata.get::<PrimeModuloMeta<Felt252>>().unwrap().prime();
                    (&operation.c + prime.to_bigint().unwrap())
                        .to_biguint()
                        .unwrap()
                }
                _ => operation.c.to_biguint().unwrap(),
            };

            let attr_c = Attribute::parse(context, &format!("{value} : {felt252_ty}")).unwrap();

            // TODO: Ensure that the constant is on the right side of the operation.
            mlir_asm! { context, entry, location =>
                ; rhs = "arith.constant"() { "value" = attr_c } : () -> felt252_ty
            }

            (operation.operator, entry.argument(0)?.into(), rhs)
        }
    };

    let result = match op {
        Felt252BinaryOperator::Add => {
            mlir_asm! { context, entry, location =>
                ; lhs = "arith.extui"(lhs) : (felt252_ty) -> i256
                ; rhs = "arith.extui"(rhs) : (felt252_ty) -> i256
                ; result = "arith.addi"(lhs, rhs) : (i256, i256) -> i256

                ; prime = "arith.constant"() { "value" = attr_prime_i256 } : () -> i256
                ; result_mod = "arith.subi"(result, prime) : (i256, i256) -> i256
                ; is_out_of_range = "arith.cmpi"(result, prime) { "predicate" = attr_cmp_uge } : (i256, i256) -> bool_ty

                ; result = "arith.select"(is_out_of_range, result_mod, result) : (bool_ty, i256, i256) -> i256
                ; result = "arith.trunci"(result) : (i256) -> felt252_ty
            };

            result
        }
        Felt252BinaryOperator::Sub => {
            mlir_asm! { context, entry, location =>
                ; lhs = "arith.extui"(lhs) : (felt252_ty) -> i256
                ; rhs = "arith.extui"(rhs) : (felt252_ty) -> i256
                ; result = "arith.subi"(lhs, rhs) : (i256, i256) -> i256

                ; prime = "arith.constant"() { "value" = attr_prime_i256 } : () -> i256
                ; result_mod = "arith.addi"(result, prime) : (i256, i256) -> i256
                ; is_out_of_range = "arith.cmpi"(lhs, rhs) { "predicate" = attr_cmp_ult } : (i256, i256) -> bool_ty

                ; result = "arith.select"(is_out_of_range, result_mod, result) : (bool_ty, i256, i256) -> i256
                ; result = "arith.trunci"(result) : (i256) -> felt252_ty
            }

            result
        }
        Felt252BinaryOperator::Mul => {
            mlir_asm! { context, entry, location =>
                ; lhs = "arith.extui"(lhs) : (felt252_ty) -> i512
                ; rhs = "arith.extui"(rhs) : (felt252_ty) -> i512
                ; result = "arith.muli"(lhs, rhs) : (i512, i512) -> i512

                ; prime = "arith.constant"() { "value" = attr_prime_i512 } : () -> i512
                ; result_mod = "arith.remui"(result, prime) : (i512, i512) -> i512
                ; is_out_of_range = "arith.cmpi"(result, prime) { "predicate" = attr_cmp_uge } : (i512, i512) -> bool_ty

                ; result = "arith.select"(is_out_of_range, result_mod, result) : (bool_ty, i512, i512) -> i512
                ; result = "arith.trunci"(result) : (i512) -> felt252_ty
            }

            result
        }
        Felt252BinaryOperator::Div => {
            // TODO: Implement `felt252_div` and `felt252_div_const`.
            todo!("Implement `felt252_div` and `felt252_div_const`")
        }
    };

    entry.append_operation(helper.br(0, &[result], location));

    Ok(())
}

/// Generate MLIR operations for the `felt252_const` libfunc.
pub fn build_const<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &Felt252ConstConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let value = match info.c.sign() {
        Sign::Minus => {
            let prime = metadata.get::<PrimeModuloMeta<Felt252>>().unwrap().prime();
            (&info.c + prime.to_bigint().unwrap()).to_biguint().unwrap()
        }
        _ => info.c.to_biguint().unwrap(),
    };

    let felt252_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;

    let attr_c = Attribute::parse(context, &format!("{value} : {felt252_ty}")).unwrap();

    mlir_asm! { context, entry, location =>
        ; k0 = "arith.constant"() { "value" = attr_c } : () -> felt252_ty
    }

    entry.append_operation(helper.br(0, &[k0], location));
    Ok(())
}

/// Generate MLIR operations for the `felt252_is_zero` libfunc.
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

    entry.append_operation(helper.cond_br(context, condition, [0, 1], [&[], &[arg0]], location));

    Ok(())
}

#[cfg(test)]
pub mod test {
    use crate::{
        utils::test::{load_cairo, run_program, run_program_assert_output},
        values::JITValue,
    };
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;

    lazy_static! {
        static ref FELT252_ADD: (String, Program) = load_cairo! {
            fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
                lhs + rhs
            }
        };

        static ref FELT252_SUB: (String, Program) = load_cairo! {
            fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
                lhs - rhs
            }
        };

        static ref FELT252_MUL: (String, Program) = load_cairo! {
            fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
                lhs * rhs
            }
        };

        // TODO: Add test program for `felt252_div`.

        // TODO: Add test program for `felt252_add_const`.
        // TODO: Add test program for `felt252_sub_const`.
        // TODO: Add test program for `felt252_mul_const`.
        // TODO: Add test program for `felt252_div_const`.

        static ref FELT252_CONST: (String, Program) = load_cairo! {
            fn run_test() -> (felt252, felt252, felt252, felt252) {
                (0, 1, -2, -1)
            }
        };

        static ref FELT252_IS_ZERO: (String, Program) = load_cairo! {
            fn run_test(x: felt252) -> felt252 {
                match x {
                    0 => 1,
                    _ => 0,
                }
            }
        };
    }

    #[test]
    fn felt252_add_new() {
        run_program_assert_output(
            &FELT252_ADD,
            "run_test",
            &[JITValue::felt_str("0"), JITValue::felt_str("0")],
            &[JITValue::felt_str("0")],
        );
        run_program_assert_output(
            &FELT252_ADD,
            "run_test",
            &[JITValue::felt_str("0"), JITValue::felt_str("1")],
            &[JITValue::felt_str("1")],
        );
        run_program_assert_output(
            &FELT252_ADD,
            "run_test",
            &[JITValue::felt_str("0"), JITValue::felt_str("0")],
            &[JITValue::felt_str("0")],
        );
    }

    #[test]
    fn felt252_add() {
        run_program_assert_output(
            &FELT252_ADD,
            "run_test",
            &[JITValue::felt_str("0"), JITValue::felt_str("0")],
            &[JITValue::felt_str("0")],
        );
        run_program_assert_output(
            &FELT252_ADD,
            "run_test",
            &[JITValue::felt_str("0"), JITValue::felt_str("1")],
            &[JITValue::felt_str("1")],
        );
        run_program_assert_output(
            &FELT252_ADD,
            "run_test",
            &[JITValue::felt_str("0"), JITValue::felt_str("0")],
            &[JITValue::felt_str("0")],
        );

        fn r(lhs: JITValue, rhs: JITValue) -> Vec<JITValue> {
            run_program(&FELT252_ADD, "run_test", &[lhs, rhs]).return_values
        }

        assert_eq!(
            r(JITValue::felt_str("0"), JITValue::felt_str("1")),
            vec![JITValue::felt_str("1")]
        );
        assert_eq!(
            r(JITValue::felt_str("0"), JITValue::felt_str("-2")),
            vec![JITValue::felt_str("-2")]
        );
        assert_eq!(
            r(JITValue::felt_str("0"), JITValue::felt_str("-1")),
            vec![JITValue::felt_str("-1")]
        );

        assert_eq!(
            r(JITValue::felt_str("1"), JITValue::felt_str("0")),
            vec![JITValue::felt_str("1")]
        );
        assert_eq!(
            r(JITValue::felt_str("1"), JITValue::felt_str("1")),
            vec![JITValue::felt_str("2")]
        );
        assert_eq!(
            r(JITValue::felt_str("1"), JITValue::felt_str("-2")),
            vec![JITValue::felt_str("-1")]
        );
        assert_eq!(
            r(JITValue::felt_str("1"), JITValue::felt_str("-1")),
            vec![JITValue::felt_str("0")]
        );

        assert_eq!(
            r(JITValue::felt_str("-2"), JITValue::felt_str("0")),
            vec![JITValue::felt_str("-2")]
        );
        assert_eq!(
            r(JITValue::felt_str("-2"), JITValue::felt_str("1")),
            vec![JITValue::felt_str("-1")]
        );
        assert_eq!(
            r(JITValue::felt_str("-2"), JITValue::felt_str("-2")),
            vec![JITValue::felt_str("-4")]
        );
        assert_eq!(
            r(JITValue::felt_str("-2"), JITValue::felt_str("-1")),
            vec![JITValue::felt_str("-3")]
        );

        assert_eq!(
            r(JITValue::felt_str("-1"), JITValue::felt_str("0")),
            vec![JITValue::felt_str("-1")]
        );
        assert_eq!(
            r(JITValue::felt_str("-1"), JITValue::felt_str("1")),
            vec![JITValue::felt_str("0")]
        );
        assert_eq!(
            r(JITValue::felt_str("-1"), JITValue::felt_str("-2")),
            vec![JITValue::felt_str("-3")]
        );
        assert_eq!(
            r(JITValue::felt_str("-1"), JITValue::felt_str("-1")),
            vec![JITValue::felt_str("-2")]
        );
    }

    #[test]
    fn felt252_sub() {
        let r = |lhs, rhs| run_program(&FELT252_SUB, "run_test", &[lhs, rhs]).return_values;

        assert_eq!(
            r(JITValue::felt_str("0"), JITValue::felt_str("0")),
            vec![JITValue::felt_str("0")]
        );
        assert_eq!(
            r(JITValue::felt_str("0"), JITValue::felt_str("1")),
            vec![JITValue::felt_str("-1")]
        );
        assert_eq!(
            r(JITValue::felt_str("0"), JITValue::felt_str("-2")),
            vec![JITValue::felt_str("2")]
        );
        assert_eq!(
            r(JITValue::felt_str("0"), JITValue::felt_str("-1")),
            vec![JITValue::felt_str("1")]
        );

        assert_eq!(
            r(JITValue::felt_str("1"), JITValue::felt_str("0")),
            vec![JITValue::felt_str("1")]
        );
        assert_eq!(
            r(JITValue::felt_str("1"), JITValue::felt_str("1")),
            vec![JITValue::felt_str("0")]
        );
        assert_eq!(
            r(JITValue::felt_str("1"), JITValue::felt_str("-2")),
            vec![JITValue::felt_str("3")]
        );
        assert_eq!(
            r(JITValue::felt_str("1"), JITValue::felt_str("-1")),
            vec![JITValue::felt_str("2")]
        );

        assert_eq!(
            r(JITValue::felt_str("-2"), JITValue::felt_str("0")),
            vec![JITValue::felt_str("-2")]
        );
        assert_eq!(
            r(JITValue::felt_str("-2"), JITValue::felt_str("1")),
            vec![JITValue::felt_str("-3")]
        );
        assert_eq!(
            r(JITValue::felt_str("-2"), JITValue::felt_str("-2")),
            vec![JITValue::felt_str("0")]
        );
        assert_eq!(
            r(JITValue::felt_str("-2"), JITValue::felt_str("-1")),
            vec![JITValue::felt_str("-1")]
        );

        assert_eq!(
            r(JITValue::felt_str("-1"), JITValue::felt_str("0")),
            vec![JITValue::felt_str("-1")]
        );
        assert_eq!(
            r(JITValue::felt_str("-1"), JITValue::felt_str("1")),
            vec![JITValue::felt_str("-2")]
        );
        assert_eq!(
            r(JITValue::felt_str("-1"), JITValue::felt_str("-2")),
            vec![JITValue::felt_str("1")]
        );
        assert_eq!(
            r(JITValue::felt_str("-1"), JITValue::felt_str("-1")),
            vec![JITValue::felt_str("0")]
        );
    }

    #[test]
    fn felt252_mul() {
        let r = |lhs, rhs| run_program(&FELT252_MUL, "run_test", &[lhs, rhs]).return_values;

        assert_eq!(
            r(JITValue::felt_str("0"), JITValue::felt_str("0")),
            vec![JITValue::felt_str("0")]
        );
        assert_eq!(
            r(JITValue::felt_str("0"), JITValue::felt_str("1")),
            vec![JITValue::felt_str("0")]
        );
        assert_eq!(
            r(JITValue::felt_str("0"), JITValue::felt_str("-2")),
            vec![JITValue::felt_str("0")]
        );
        assert_eq!(
            r(JITValue::felt_str("0"), JITValue::felt_str("-1")),
            vec![JITValue::felt_str("0")]
        );

        assert_eq!(
            r(JITValue::felt_str("1"), JITValue::felt_str("0")),
            vec![JITValue::felt_str("0")]
        );
        assert_eq!(
            r(JITValue::felt_str("1"), JITValue::felt_str("1")),
            vec![JITValue::felt_str("1")]
        );
        assert_eq!(
            r(JITValue::felt_str("1"), JITValue::felt_str("-2")),
            vec![JITValue::felt_str("-2")]
        );
        assert_eq!(
            r(JITValue::felt_str("1"), JITValue::felt_str("-1")),
            vec![JITValue::felt_str("-1")]
        );

        assert_eq!(
            r(JITValue::felt_str("-2"), JITValue::felt_str("0")),
            vec![JITValue::felt_str("0")]
        );
        assert_eq!(
            r(JITValue::felt_str("-2"), JITValue::felt_str("1")),
            vec![JITValue::felt_str("-2")]
        );
        assert_eq!(
            r(JITValue::felt_str("-2"), JITValue::felt_str("-2")),
            vec![JITValue::felt_str("4")]
        );
        assert_eq!(
            r(JITValue::felt_str("-2"), JITValue::felt_str("-1")),
            vec![JITValue::felt_str("2")]
        );

        assert_eq!(
            r(JITValue::felt_str("-1"), JITValue::felt_str("0")),
            vec![JITValue::felt_str("0")]
        );
        assert_eq!(
            r(JITValue::felt_str("-1"), JITValue::felt_str("1")),
            vec![JITValue::felt_str("-1")]
        );
        assert_eq!(
            r(JITValue::felt_str("-1"), JITValue::felt_str("-2")),
            vec![JITValue::felt_str("2")]
        );
        assert_eq!(
            r(JITValue::felt_str("-1"), JITValue::felt_str("-1")),
            vec![JITValue::felt_str("1")]
        );
    }

    #[test]
    fn felt252_const() {
        assert_eq!(
            run_program(&FELT252_CONST, "run_test", &[]).return_values,
            vec![JITValue::Struct {
                fields: vec![
                    JITValue::felt_str("0"),
                    JITValue::felt_str("1"),
                    JITValue::felt_str("-2"),
                    JITValue::felt_str("-1")
                ],
                debug_name: None
            }]
        );
    }

    #[test]
    fn felt252_is_zero() {
        let r = |x| run_program(&FELT252_IS_ZERO, "run_test", &[x]).return_values;

        assert_eq!(r(JITValue::felt_str("0")), vec![JITValue::felt_str("1")]);
        assert_eq!(r(JITValue::felt_str("1")), vec![JITValue::felt_str("0")]);
        assert_eq!(r(JITValue::felt_str("-2")), vec![JITValue::felt_str("0")]);
        assert_eq!(r(JITValue::felt_str("-1")), vec![JITValue::felt_str("0")]);
    }
}
