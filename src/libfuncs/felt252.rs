//! # `Felt`-related libfuncs

use super::LibfuncHelper;
use crate::{
    error::{Error, Result},
    metadata::{prime_modulo::PrimeModuloMeta, MetadataStorage},
    utils::{mlir_asm, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        felt252::{
            Felt252BinaryOperationConcrete, Felt252BinaryOperator, Felt252Concrete,
            Felt252ConstConcreteLibfunc,
        },
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        cf,
    },
    ir::{
        attribute::IntegerAttribute, r#type::IntegerType, Attribute, Block, Location, Value,
        ValueLike,
    },
    Context,
};
use num_bigint::{Sign, ToBigInt};
use starknet_types_core::felt::Felt;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Felt252Concrete,
) -> Result<()> {
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
pub fn build_binary_operation<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &Felt252BinaryOperationConcrete,
) -> Result<()> {
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
            metadata
                .get::<PrimeModuloMeta<Felt>>()
                .ok_or(Error::MissingMetadata)?
                .prime()
        ),
    )
    .ok_or(Error::ParseAttributeError)?;
    let attr_prime_i512 = Attribute::parse(
        context,
        &format!(
            "{} : {i512}",
            metadata
                .get::<PrimeModuloMeta<Felt>>()
                .ok_or(Error::MissingMetadata)?
                .prime()
        ),
    )
    .ok_or(Error::ParseAttributeError)?;

    let attr_cmp_uge = IntegerAttribute::new(
        IntegerType::new(context, 64).into(),
        CmpiPredicate::Uge as i64,
    )
    .into();
    let attr_cmp_ult = IntegerAttribute::new(
        IntegerType::new(context, 64).into(),
        CmpiPredicate::Ult as i64,
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
                    let prime = metadata
                        .get::<PrimeModuloMeta<Felt>>()
                        .ok_or(Error::MissingMetadata)?
                        .prime();
                    (&operation.c + prime.to_bigint().expect("always is Some"))
                        .to_biguint()
                        .expect("always positive")
                }
                _ => operation.c.to_biguint().expect("sign already checked"),
            };

            let attr_c = Attribute::parse(context, &format!("{value} : {felt252_ty}"))
                .ok_or(Error::MissingMetadata)?;

            // TODO: Ensure that the constant is on the correct side of the operation.
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
            // The extended euclidean algorithm calculates the greatest common divisor of two integers,
            // as well as the bezout coefficients x and y such that for inputs a and b, ax+by=gcd(a,b)
            // We use this in felt division to find the modular inverse of a given number
            // If a is the number we're trying to find the inverse of, we can do
            // ax+y*PRIME=gcd(a,PRIME)=1 => ax = 1 (mod PRIME)
            // Hence for input a, we return x
            // The input MUST be non-zero
            // See https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
            let start_block = helper.append_block(Block::new(&[(i512, location)]));
            let loop_block = helper.append_block(Block::new(&[
                (i512, location),
                (i512, location),
                (i512, location),
                (i512, location),
            ]));
            let negative_check_block = helper.append_block(Block::new(&[]));
            // Block containing final result
            let inverse_result_block = helper.append_block(Block::new(&[(i512, location)]));
            // Egcd works by calculating a series of remainders, each the remainder of dividing the previous two
            // For the initial setup, r0 = PRIME, r1 = a
            // This order is chosen because if we reverse them, then the first iteration will just swap them
            let prev_remainder = start_block
                .append_operation(arith::constant(context, attr_prime_i512, location))
                .result(0)?
                .into();
            let remainder = start_block.argument(0)?.into();
            // Similarly we'll calculate another series which starts 0,1,... and from which we will retrieve the modular inverse of a
            let prev_inverse = start_block
                .append_operation(arith::constant(
                    context,
                    IntegerAttribute::new(i512, 0).into(),
                    location,
                ))
                .result(0)?
                .into();
            let inverse = start_block
                .append_operation(arith::constant(
                    context,
                    IntegerAttribute::new(i512, 1).into(),
                    location,
                ))
                .result(0)?
                .into();
            start_block.append_operation(cf::br(
                loop_block,
                &[prev_remainder, remainder, prev_inverse, inverse],
                location,
            ));

            //---Loop body---
            // Arguments are rem_(i-1), rem, inv_(i-1), inv
            let prev_remainder = loop_block.argument(0)?.into();
            let remainder = loop_block.argument(1)?.into();
            let prev_inverse = loop_block.argument(2)?.into();
            let inverse = loop_block.argument(3)?.into();

            // First calculate q = rem_(i-1)/rem_i, rounded down
            let quotient = loop_block
                .append_operation(arith::divui(prev_remainder, remainder, location))
                .result(0)?
                .into();
            // Then r_(i+1) = r_(i-1) - q * r_i, and inv_(i+1) = inv_(i-1) - q * inv_i
            let rem_times_quo = loop_block
                .append_operation(arith::muli(remainder, quotient, location))
                .result(0)?
                .into();
            let inv_times_quo = loop_block
                .append_operation(arith::muli(inverse, quotient, location))
                .result(0)?
                .into();
            let next_remainder = loop_block
                .append_operation(arith::subi(prev_remainder, rem_times_quo, location))
                .result(0)?
                .into();
            let next_inverse = loop_block
                .append_operation(arith::subi(prev_inverse, inv_times_quo, location))
                .result(0)?
                .into();

            // If r_(i+1) is 0, then inv_i is the inverse
            let zero = loop_block
                .append_operation(arith::constant(
                    context,
                    IntegerAttribute::new(i512, 0).into(),
                    location,
                ))
                .result(0)?
                .into();
            let next_remainder_eq_zero = loop_block
                .append_operation(arith::cmpi(
                    context,
                    CmpiPredicate::Eq,
                    next_remainder,
                    zero,
                    location,
                ))
                .result(0)?
                .into();
            loop_block.append_operation(cf::cond_br(
                context,
                next_remainder_eq_zero,
                negative_check_block,
                loop_block,
                &[],
                &[remainder, next_remainder, inverse, next_inverse],
                location,
            ));

            // egcd sometimes returns a negative number for the inverse,
            // in such cases we must simply wrap it around back into [0, PRIME)
            // this suffices because |inv_i| <= divfloor(PRIME,2)
            let zero = negative_check_block
                .append_operation(arith::constant(
                    context,
                    IntegerAttribute::new(i512, 0).into(),
                    location,
                ))
                .result(0)?
                .into();

            let is_negative = negative_check_block
                .append_operation(arith::cmpi(
                    context,
                    CmpiPredicate::Slt,
                    inverse,
                    zero,
                    location,
                ))
                .result(0)?
                .into();
            // if the inverse is < 0, add PRIME
            let prime = negative_check_block
                .append_operation(arith::constant(context, attr_prime_i512, location))
                .result(0)?
                .into();
            let wrapped_inverse = negative_check_block
                .append_operation(arith::addi(inverse, prime, location))
                .result(0)?
                .into();
            let inverse = negative_check_block
                .append_operation(arith::select(
                    is_negative,
                    wrapped_inverse,
                    inverse,
                    location,
                ))
                .result(0)?
                .into();
            negative_check_block.append_operation(cf::br(
                inverse_result_block,
                &[inverse],
                location,
            ));

            // Div Logic Start
            // Fetch operands
            let lhs = entry
                .append_operation(arith::extui(lhs, i512, location))
                .result(0)?
                .into();
            let rhs = entry
                .append_operation(arith::extui(rhs, i512, location))
                .result(0)?
                .into();
            // Calculate inverse of rhs, callling the inverse implementation's starting block
            entry.append_operation(cf::br(start_block, &[rhs], location));
            // Fetch the inverse result from the result block
            let inverse = inverse_result_block.argument(0)?.into();
            // Peform lhs * (1/ rhs)
            let result = inverse_result_block
                .append_operation(arith::muli(lhs, inverse, location))
                .result(0)?
                .into();
            // Apply modulo and convert result to felt252
            mlir_asm! { context, inverse_result_block, location =>
                ; result_mod = "arith.remui"(result, prime) : (i512, i512) -> i512
                ; is_out_of_range = "arith.cmpi"(result, prime) { "predicate" = attr_cmp_uge } : (i512, i512) -> bool_ty

                ; result = "arith.select"(is_out_of_range, result_mod, result) : (bool_ty, i512, i512) -> i512
                ; result = "arith.trunci"(result) : (i512) -> felt252_ty
            }
            inverse_result_block.append_operation(helper.br(0, &[result], location));
            return Ok(());
        }
    };

    entry.append_operation(helper.br(0, &[result], location));

    Ok(())
}

/// Generate MLIR operations for the `felt252_const` libfunc.
pub fn build_const<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &Felt252ConstConcreteLibfunc,
) -> Result<()> {
    let value = match info.c.sign() {
        Sign::Minus => {
            let prime = metadata
                .get::<PrimeModuloMeta<Felt>>()
                .ok_or(Error::MissingMetadata)?
                .prime();
            (&info.c + prime.to_bigint().expect("always is Some"))
                .to_biguint()
                .expect("always is positive")
        }
        _ => info.c.to_biguint().expect("sign already checked"),
    };

    let felt252_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;

    let attr_c = Attribute::parse(context, &format!("{value} : {felt252_ty}"))
        .ok_or(Error::ParseAttributeError)?;

    mlir_asm! { context, entry, location =>
        ; k0 = "arith.constant"() { "value" = attr_c } : () -> felt252_ty
    }

    entry.append_operation(helper.br(0, &[k0], location));
    Ok(())
}

/// Generate MLIR operations for the `felt252_is_zero` libfunc.
pub fn build_is_zero<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let arg0: Value = entry.argument(0)?.into();

    let op = entry.append_operation(arith::constant(
        context,
        IntegerAttribute::new(arg0.r#type(), 0).into(),
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
        values::JitValue,
    };
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;

    lazy_static! {
        static ref FELT252_ADD: (String, Program) = load_cairo! {
            use core::debug::PrintTrait;
            fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
                lhs.print();
                rhs.print();
                let result = lhs + rhs;

    result.print();

    result
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
    fn felt252_add() {
        run_program_assert_output(
            &FELT252_ADD,
            "run_test",
            &[JitValue::felt_str("0"), JitValue::felt_str("0")],
            JitValue::felt_str("0"),
        );
        run_program_assert_output(
            &FELT252_ADD,
            "run_test",
            &[JitValue::felt_str("1"), JitValue::felt_str("2")],
            JitValue::felt_str("3"),
        );

        fn r(lhs: JitValue, rhs: JitValue) -> JitValue {
            run_program(&FELT252_ADD, "run_test", &[lhs, rhs]).return_value
        }

        assert_eq!(
            r(JitValue::felt_str("0"), JitValue::felt_str("1")),
            JitValue::felt_str("1")
        );
        assert_eq!(
            r(JitValue::felt_str("0"), JitValue::felt_str("-2")),
            JitValue::felt_str("-2")
        );
        assert_eq!(
            r(JitValue::felt_str("0"), JitValue::felt_str("-1")),
            JitValue::felt_str("-1")
        );

        assert_eq!(
            r(JitValue::felt_str("1"), JitValue::felt_str("0")),
            JitValue::felt_str("1")
        );
        assert_eq!(
            r(JitValue::felt_str("1"), JitValue::felt_str("1")),
            JitValue::felt_str("2")
        );
        assert_eq!(
            r(JitValue::felt_str("1"), JitValue::felt_str("-2")),
            JitValue::felt_str("-1")
        );
        assert_eq!(
            r(JitValue::felt_str("1"), JitValue::felt_str("-1")),
            JitValue::felt_str("0")
        );

        assert_eq!(
            r(JitValue::felt_str("-2"), JitValue::felt_str("0")),
            JitValue::felt_str("-2")
        );
        assert_eq!(
            r(JitValue::felt_str("-2"), JitValue::felt_str("1")),
            JitValue::felt_str("-1")
        );
        assert_eq!(
            r(JitValue::felt_str("-2"), JitValue::felt_str("-2")),
            JitValue::felt_str("-4")
        );
        assert_eq!(
            r(JitValue::felt_str("-2"), JitValue::felt_str("-1")),
            JitValue::felt_str("-3")
        );

        assert_eq!(
            r(JitValue::felt_str("-1"), JitValue::felt_str("0")),
            JitValue::felt_str("-1")
        );
        assert_eq!(
            r(JitValue::felt_str("-1"), JitValue::felt_str("1")),
            JitValue::felt_str("0")
        );
        assert_eq!(
            r(JitValue::felt_str("-1"), JitValue::felt_str("-2")),
            JitValue::felt_str("-3")
        );
        assert_eq!(
            r(JitValue::felt_str("-1"), JitValue::felt_str("-1")),
            JitValue::felt_str("-2")
        );
    }

    #[test]
    fn felt252_sub() {
        let r = |lhs, rhs| run_program(&FELT252_SUB, "run_test", &[lhs, rhs]).return_value;

        assert_eq!(
            r(JitValue::felt_str("0"), JitValue::felt_str("0")),
            JitValue::felt_str("0")
        );
        assert_eq!(
            r(JitValue::felt_str("0"), JitValue::felt_str("1")),
            JitValue::felt_str("-1")
        );
        assert_eq!(
            r(JitValue::felt_str("0"), JitValue::felt_str("-2")),
            JitValue::felt_str("2")
        );
        assert_eq!(
            r(JitValue::felt_str("0"), JitValue::felt_str("-1")),
            JitValue::felt_str("1")
        );

        assert_eq!(
            r(JitValue::felt_str("1"), JitValue::felt_str("0")),
            JitValue::felt_str("1")
        );
        assert_eq!(
            r(JitValue::felt_str("1"), JitValue::felt_str("1")),
            JitValue::felt_str("0")
        );
        assert_eq!(
            r(JitValue::felt_str("1"), JitValue::felt_str("-2")),
            JitValue::felt_str("3")
        );
        assert_eq!(
            r(JitValue::felt_str("1"), JitValue::felt_str("-1")),
            JitValue::felt_str("2")
        );

        assert_eq!(
            r(JitValue::felt_str("-2"), JitValue::felt_str("0")),
            JitValue::felt_str("-2")
        );
        assert_eq!(
            r(JitValue::felt_str("-2"), JitValue::felt_str("1")),
            JitValue::felt_str("-3")
        );
        assert_eq!(
            r(JitValue::felt_str("-2"), JitValue::felt_str("-2")),
            JitValue::felt_str("0")
        );
        assert_eq!(
            r(JitValue::felt_str("-2"), JitValue::felt_str("-1")),
            JitValue::felt_str("-1")
        );

        assert_eq!(
            r(JitValue::felt_str("-1"), JitValue::felt_str("0")),
            JitValue::felt_str("-1")
        );
        assert_eq!(
            r(JitValue::felt_str("-1"), JitValue::felt_str("1")),
            JitValue::felt_str("-2")
        );
        assert_eq!(
            r(JitValue::felt_str("-1"), JitValue::felt_str("-2")),
            JitValue::felt_str("1")
        );
        assert_eq!(
            r(JitValue::felt_str("-1"), JitValue::felt_str("-1")),
            JitValue::felt_str("0")
        );
    }

    #[test]
    fn felt252_mul() {
        let r = |lhs, rhs| run_program(&FELT252_MUL, "run_test", &[lhs, rhs]).return_value;

        assert_eq!(
            r(JitValue::felt_str("0"), JitValue::felt_str("0")),
            JitValue::felt_str("0")
        );
        assert_eq!(
            r(JitValue::felt_str("0"), JitValue::felt_str("1")),
            JitValue::felt_str("0")
        );
        assert_eq!(
            r(JitValue::felt_str("0"), JitValue::felt_str("-2")),
            JitValue::felt_str("0")
        );
        assert_eq!(
            r(JitValue::felt_str("0"), JitValue::felt_str("-1")),
            JitValue::felt_str("0")
        );

        assert_eq!(
            r(JitValue::felt_str("1"), JitValue::felt_str("0")),
            JitValue::felt_str("0")
        );
        assert_eq!(
            r(JitValue::felt_str("1"), JitValue::felt_str("1")),
            JitValue::felt_str("1")
        );
        assert_eq!(
            r(JitValue::felt_str("1"), JitValue::felt_str("-2")),
            JitValue::felt_str("-2")
        );
        assert_eq!(
            r(JitValue::felt_str("1"), JitValue::felt_str("-1")),
            JitValue::felt_str("-1")
        );

        assert_eq!(
            r(JitValue::felt_str("-2"), JitValue::felt_str("0")),
            JitValue::felt_str("0")
        );
        assert_eq!(
            r(JitValue::felt_str("-2"), JitValue::felt_str("1")),
            JitValue::felt_str("-2")
        );
        assert_eq!(
            r(JitValue::felt_str("-2"), JitValue::felt_str("-2")),
            JitValue::felt_str("4")
        );
        assert_eq!(
            r(JitValue::felt_str("-2"), JitValue::felt_str("-1")),
            JitValue::felt_str("2")
        );

        assert_eq!(
            r(JitValue::felt_str("-1"), JitValue::felt_str("0")),
            JitValue::felt_str("0")
        );
        assert_eq!(
            r(JitValue::felt_str("-1"), JitValue::felt_str("1")),
            JitValue::felt_str("-1")
        );
        assert_eq!(
            r(JitValue::felt_str("-1"), JitValue::felt_str("-2")),
            JitValue::felt_str("2")
        );
        assert_eq!(
            r(JitValue::felt_str("-1"), JitValue::felt_str("-1")),
            JitValue::felt_str("1")
        );
    }

    #[test]
    fn felt252_const() {
        assert_eq!(
            run_program(&FELT252_CONST, "run_test", &[]).return_value,
            JitValue::Struct {
                fields: vec![
                    JitValue::felt_str("0"),
                    JitValue::felt_str("1"),
                    JitValue::felt_str("-2"),
                    JitValue::felt_str("-1")
                ],
                debug_name: None
            }
        );
    }

    #[test]
    fn felt252_is_zero() {
        let r = |x| run_program(&FELT252_IS_ZERO, "run_test", &[x]).return_value;

        assert_eq!(r(JitValue::felt_str("0")), JitValue::felt_str("1"));
        assert_eq!(r(JitValue::felt_str("1")), JitValue::felt_str("0"));
        assert_eq!(r(JitValue::felt_str("-2")), JitValue::felt_str("0"));
        assert_eq!(r(JitValue::felt_str("-1")), JitValue::felt_str("0"));
    }
}
