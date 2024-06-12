////! # `Felt`-related libfuncs
//! # `Felt`-related libfuncs
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{
use crate::{
//    error::{Error, Result},
    error::{Error, Result},
//    metadata::{prime_modulo::PrimeModuloMeta, MetadataStorage},
    metadata::{prime_modulo::PrimeModuloMeta, MetadataStorage},
//    utils::{mlir_asm, ProgramRegistryExt},
    utils::{mlir_asm, ProgramRegistryExt},
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        felt252::{
        felt252::{
//            Felt252BinaryOperationConcrete, Felt252BinaryOperator, Felt252Concrete,
            Felt252BinaryOperationConcrete, Felt252BinaryOperator, Felt252Concrete,
//            Felt252ConstConcreteLibfunc,
            Felt252ConstConcreteLibfunc,
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
//        cf,
        cf,
//    },
    },
//    ir::{
    ir::{
//        attribute::IntegerAttribute, r#type::IntegerType, Attribute, Block, Location, Value,
        attribute::IntegerAttribute, r#type::IntegerType, Attribute, Block, Location, Value,
//        ValueLike,
        ValueLike,
//    },
    },
//    Context,
    Context,
//};
};
//use num_bigint::{Sign, ToBigInt};
use num_bigint::{Sign, ToBigInt};
//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
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
//    selector: &Felt252Concrete,
    selector: &Felt252Concrete,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        Felt252Concrete::BinaryOperation(info) => {
        Felt252Concrete::BinaryOperation(info) => {
//            build_binary_operation(context, registry, entry, location, helper, metadata, info)
            build_binary_operation(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Felt252Concrete::Const(info) => {
        Felt252Concrete::Const(info) => {
//            build_const(context, registry, entry, location, helper, metadata, info)
            build_const(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Felt252Concrete::IsZero(info) => {
        Felt252Concrete::IsZero(info) => {
//            build_is_zero(context, registry, entry, location, helper, metadata, info)
            build_is_zero(context, registry, entry, location, helper, metadata, info)
//        }
        }
//    }
    }
//}
}
//

///// Generate MLIR operations for the following libfuncs:
/// Generate MLIR operations for the following libfuncs:
/////   - `felt252_add` and `felt252_add_const`.
///   - `felt252_add` and `felt252_add_const`.
/////   - `felt252_sub` and `felt252_sub_const`.
///   - `felt252_sub` and `felt252_sub_const`.
/////   - `felt252_mul` and `felt252_mul_const`.
///   - `felt252_mul` and `felt252_mul_const`.
/////   - `felt252_div` and `felt252_div_const`.
///   - `felt252_div` and `felt252_div_const`.
//pub fn build_binary_operation<'ctx, 'this>(
pub fn build_binary_operation<'ctx, 'this>(
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
//    info: &Felt252BinaryOperationConcrete,
    info: &Felt252BinaryOperationConcrete,
//) -> Result<()> {
) -> Result<()> {
//    let bool_ty = IntegerType::new(context, 1).into();
    let bool_ty = IntegerType::new(context, 1).into();
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
//    let i256 = IntegerType::new(context, 256).into();
    let i256 = IntegerType::new(context, 256).into();
//    let i512 = IntegerType::new(context, 512).into();
    let i512 = IntegerType::new(context, 512).into();
//

//    let attr_prime_i256 = Attribute::parse(
    let attr_prime_i256 = Attribute::parse(
//        context,
        context,
//        &format!(
        &format!(
//            "{} : {i256}",
            "{} : {i256}",
//            metadata
            metadata
//                .get::<PrimeModuloMeta<Felt>>()
                .get::<PrimeModuloMeta<Felt>>()
//                .ok_or(Error::MissingMetadata)?
                .ok_or(Error::MissingMetadata)?
//                .prime()
                .prime()
//        ),
        ),
//    )
    )
//    .ok_or(Error::ParseAttributeError)?;
    .ok_or(Error::ParseAttributeError)?;
//    let attr_prime_i512 = Attribute::parse(
    let attr_prime_i512 = Attribute::parse(
//        context,
        context,
//        &format!(
        &format!(
//            "{} : {i512}",
            "{} : {i512}",
//            metadata
            metadata
//                .get::<PrimeModuloMeta<Felt>>()
                .get::<PrimeModuloMeta<Felt>>()
//                .ok_or(Error::MissingMetadata)?
                .ok_or(Error::MissingMetadata)?
//                .prime()
                .prime()
//        ),
        ),
//    )
    )
//    .ok_or(Error::ParseAttributeError)?;
    .ok_or(Error::ParseAttributeError)?;
//

//    let attr_cmp_uge = IntegerAttribute::new(
    let attr_cmp_uge = IntegerAttribute::new(
//        IntegerType::new(context, 64).into(),
        IntegerType::new(context, 64).into(),
//        CmpiPredicate::Uge as i64,
        CmpiPredicate::Uge as i64,
//    )
    )
//    .into();
    .into();
//    let attr_cmp_ult = IntegerAttribute::new(
    let attr_cmp_ult = IntegerAttribute::new(
//        IntegerType::new(context, 64).into(),
        IntegerType::new(context, 64).into(),
//        CmpiPredicate::Ult as i64,
        CmpiPredicate::Ult as i64,
//    )
    )
//    .into();
    .into();
//

//    let (op, lhs, rhs) = match info {
    let (op, lhs, rhs) = match info {
//        Felt252BinaryOperationConcrete::WithVar(operation) => (
        Felt252BinaryOperationConcrete::WithVar(operation) => (
//            operation.operator,
            operation.operator,
//            entry.argument(0)?.into(),
            entry.argument(0)?.into(),
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//        ),
        ),
//        Felt252BinaryOperationConcrete::WithConst(operation) => {
        Felt252BinaryOperationConcrete::WithConst(operation) => {
//            let value = match operation.c.sign() {
            let value = match operation.c.sign() {
//                Sign::Minus => {
                Sign::Minus => {
//                    let prime = metadata
                    let prime = metadata
//                        .get::<PrimeModuloMeta<Felt>>()
                        .get::<PrimeModuloMeta<Felt>>()
//                        .ok_or(Error::MissingMetadata)?
                        .ok_or(Error::MissingMetadata)?
//                        .prime();
                        .prime();
//                    (&operation.c + prime.to_bigint().expect("always is Some"))
                    (&operation.c + prime.to_bigint().expect("always is Some"))
//                        .to_biguint()
                        .to_biguint()
//                        .expect("always positive")
                        .expect("always positive")
//                }
                }
//                _ => operation.c.to_biguint().expect("sign already checked"),
                _ => operation.c.to_biguint().expect("sign already checked"),
//            };
            };
//

//            let attr_c = Attribute::parse(context, &format!("{value} : {felt252_ty}"))
            let attr_c = Attribute::parse(context, &format!("{value} : {felt252_ty}"))
//                .ok_or(Error::MissingMetadata)?;
                .ok_or(Error::MissingMetadata)?;
//

//            // TODO: Ensure that the constant is on the correct side of the operation.
            // TODO: Ensure that the constant is on the correct side of the operation.
//            mlir_asm! { context, entry, location =>
            mlir_asm! { context, entry, location =>
//                ; rhs = "arith.constant"() { "value" = attr_c } : () -> felt252_ty
                ; rhs = "arith.constant"() { "value" = attr_c } : () -> felt252_ty
//            }
            }
//

//            (operation.operator, entry.argument(0)?.into(), rhs)
            (operation.operator, entry.argument(0)?.into(), rhs)
//        }
        }
//    };
    };
//

//    let result = match op {
    let result = match op {
//        Felt252BinaryOperator::Add => {
        Felt252BinaryOperator::Add => {
//            mlir_asm! { context, entry, location =>
            mlir_asm! { context, entry, location =>
//                ; lhs = "arith.extui"(lhs) : (felt252_ty) -> i256
                ; lhs = "arith.extui"(lhs) : (felt252_ty) -> i256
//                ; rhs = "arith.extui"(rhs) : (felt252_ty) -> i256
                ; rhs = "arith.extui"(rhs) : (felt252_ty) -> i256
//                ; result = "arith.addi"(lhs, rhs) : (i256, i256) -> i256
                ; result = "arith.addi"(lhs, rhs) : (i256, i256) -> i256
//

//                ; prime = "arith.constant"() { "value" = attr_prime_i256 } : () -> i256
                ; prime = "arith.constant"() { "value" = attr_prime_i256 } : () -> i256
//                ; result_mod = "arith.subi"(result, prime) : (i256, i256) -> i256
                ; result_mod = "arith.subi"(result, prime) : (i256, i256) -> i256
//                ; is_out_of_range = "arith.cmpi"(result, prime) { "predicate" = attr_cmp_uge } : (i256, i256) -> bool_ty
                ; is_out_of_range = "arith.cmpi"(result, prime) { "predicate" = attr_cmp_uge } : (i256, i256) -> bool_ty
//

//                ; result = "arith.select"(is_out_of_range, result_mod, result) : (bool_ty, i256, i256) -> i256
                ; result = "arith.select"(is_out_of_range, result_mod, result) : (bool_ty, i256, i256) -> i256
//                ; result = "arith.trunci"(result) : (i256) -> felt252_ty
                ; result = "arith.trunci"(result) : (i256) -> felt252_ty
//            };
            };
//

//            result
            result
//        }
        }
//        Felt252BinaryOperator::Sub => {
        Felt252BinaryOperator::Sub => {
//            mlir_asm! { context, entry, location =>
            mlir_asm! { context, entry, location =>
//                ; lhs = "arith.extui"(lhs) : (felt252_ty) -> i256
                ; lhs = "arith.extui"(lhs) : (felt252_ty) -> i256
//                ; rhs = "arith.extui"(rhs) : (felt252_ty) -> i256
                ; rhs = "arith.extui"(rhs) : (felt252_ty) -> i256
//                ; result = "arith.subi"(lhs, rhs) : (i256, i256) -> i256
                ; result = "arith.subi"(lhs, rhs) : (i256, i256) -> i256
//

//                ; prime = "arith.constant"() { "value" = attr_prime_i256 } : () -> i256
                ; prime = "arith.constant"() { "value" = attr_prime_i256 } : () -> i256
//                ; result_mod = "arith.addi"(result, prime) : (i256, i256) -> i256
                ; result_mod = "arith.addi"(result, prime) : (i256, i256) -> i256
//                ; is_out_of_range = "arith.cmpi"(lhs, rhs) { "predicate" = attr_cmp_ult } : (i256, i256) -> bool_ty
                ; is_out_of_range = "arith.cmpi"(lhs, rhs) { "predicate" = attr_cmp_ult } : (i256, i256) -> bool_ty
//

//                ; result = "arith.select"(is_out_of_range, result_mod, result) : (bool_ty, i256, i256) -> i256
                ; result = "arith.select"(is_out_of_range, result_mod, result) : (bool_ty, i256, i256) -> i256
//                ; result = "arith.trunci"(result) : (i256) -> felt252_ty
                ; result = "arith.trunci"(result) : (i256) -> felt252_ty
//            }
            }
//

//            result
            result
//        }
        }
//        Felt252BinaryOperator::Mul => {
        Felt252BinaryOperator::Mul => {
//            mlir_asm! { context, entry, location =>
            mlir_asm! { context, entry, location =>
//                ; lhs = "arith.extui"(lhs) : (felt252_ty) -> i512
                ; lhs = "arith.extui"(lhs) : (felt252_ty) -> i512
//                ; rhs = "arith.extui"(rhs) : (felt252_ty) -> i512
                ; rhs = "arith.extui"(rhs) : (felt252_ty) -> i512
//                ; result = "arith.muli"(lhs, rhs) : (i512, i512) -> i512
                ; result = "arith.muli"(lhs, rhs) : (i512, i512) -> i512
//

//                ; prime = "arith.constant"() { "value" = attr_prime_i512 } : () -> i512
                ; prime = "arith.constant"() { "value" = attr_prime_i512 } : () -> i512
//                ; result_mod = "arith.remui"(result, prime) : (i512, i512) -> i512
                ; result_mod = "arith.remui"(result, prime) : (i512, i512) -> i512
//                ; is_out_of_range = "arith.cmpi"(result, prime) { "predicate" = attr_cmp_uge } : (i512, i512) -> bool_ty
                ; is_out_of_range = "arith.cmpi"(result, prime) { "predicate" = attr_cmp_uge } : (i512, i512) -> bool_ty
//

//                ; result = "arith.select"(is_out_of_range, result_mod, result) : (bool_ty, i512, i512) -> i512
                ; result = "arith.select"(is_out_of_range, result_mod, result) : (bool_ty, i512, i512) -> i512
//                ; result = "arith.trunci"(result) : (i512) -> felt252_ty
                ; result = "arith.trunci"(result) : (i512) -> felt252_ty
//            }
            }
//

//            result
            result
//        }
        }
//        Felt252BinaryOperator::Div => {
        Felt252BinaryOperator::Div => {
//            // The extended euclidean algorithm calculates the greatest common divisor of two integers,
            // The extended euclidean algorithm calculates the greatest common divisor of two integers,
//            // as well as the bezout coefficients x and y such that for inputs a and b, ax+by=gcd(a,b)
            // as well as the bezout coefficients x and y such that for inputs a and b, ax+by=gcd(a,b)
//            // We use this in felt division to find the modular inverse of a given number
            // We use this in felt division to find the modular inverse of a given number
//            // If a is the number we're trying to find the inverse of, we can do
            // If a is the number we're trying to find the inverse of, we can do
//            // ax+y*PRIME=gcd(a,PRIME)=1 => ax = 1 (mod PRIME)
            // ax+y*PRIME=gcd(a,PRIME)=1 => ax = 1 (mod PRIME)
//            // Hence for input a, we return x
            // Hence for input a, we return x
//            // The input MUST be non-zero
            // The input MUST be non-zero
//            // See https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
            // See https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
//            let start_block = helper.append_block(Block::new(&[(i512, location)]));
            let start_block = helper.append_block(Block::new(&[(i512, location)]));
//            let loop_block = helper.append_block(Block::new(&[
            let loop_block = helper.append_block(Block::new(&[
//                (i512, location),
                (i512, location),
//                (i512, location),
                (i512, location),
//                (i512, location),
                (i512, location),
//                (i512, location),
                (i512, location),
//            ]));
            ]));
//            let negative_check_block = helper.append_block(Block::new(&[]));
            let negative_check_block = helper.append_block(Block::new(&[]));
//            // Block containing final result
            // Block containing final result
//            let inverse_result_block = helper.append_block(Block::new(&[(i512, location)]));
            let inverse_result_block = helper.append_block(Block::new(&[(i512, location)]));
//            // Egcd works by calculating a series of remainders, each the remainder of dividing the previous two
            // Egcd works by calculating a series of remainders, each the remainder of dividing the previous two
//            // For the initial setup, r0 = PRIME, r1 = a
            // For the initial setup, r0 = PRIME, r1 = a
//            // This order is chosen because if we reverse them, then the first iteration will just swap them
            // This order is chosen because if we reverse them, then the first iteration will just swap them
//            let prev_remainder = start_block
            let prev_remainder = start_block
//                .append_operation(arith::constant(context, attr_prime_i512, location))
                .append_operation(arith::constant(context, attr_prime_i512, location))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            let remainder = start_block.argument(0)?.into();
            let remainder = start_block.argument(0)?.into();
//            // Similarly we'll calculate another series which starts 0,1,... and from which we will retrieve the modular inverse of a
            // Similarly we'll calculate another series which starts 0,1,... and from which we will retrieve the modular inverse of a
//            let prev_inverse = start_block
            let prev_inverse = start_block
//                .append_operation(arith::constant(
                .append_operation(arith::constant(
//                    context,
                    context,
//                    IntegerAttribute::new(i512, 0).into(),
                    IntegerAttribute::new(i512, 0).into(),
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            let inverse = start_block
            let inverse = start_block
//                .append_operation(arith::constant(
                .append_operation(arith::constant(
//                    context,
                    context,
//                    IntegerAttribute::new(i512, 1).into(),
                    IntegerAttribute::new(i512, 1).into(),
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            start_block.append_operation(cf::br(
            start_block.append_operation(cf::br(
//                loop_block,
                loop_block,
//                &[prev_remainder, remainder, prev_inverse, inverse],
                &[prev_remainder, remainder, prev_inverse, inverse],
//                location,
                location,
//            ));
            ));
//

//            //---Loop body---
            //---Loop body---
//            // Arguments are rem_(i-1), rem, inv_(i-1), inv
            // Arguments are rem_(i-1), rem, inv_(i-1), inv
//            let prev_remainder = loop_block.argument(0)?.into();
            let prev_remainder = loop_block.argument(0)?.into();
//            let remainder = loop_block.argument(1)?.into();
            let remainder = loop_block.argument(1)?.into();
//            let prev_inverse = loop_block.argument(2)?.into();
            let prev_inverse = loop_block.argument(2)?.into();
//            let inverse = loop_block.argument(3)?.into();
            let inverse = loop_block.argument(3)?.into();
//

//            // First calculate q = rem_(i-1)/rem_i, rounded down
            // First calculate q = rem_(i-1)/rem_i, rounded down
//            let quotient = loop_block
            let quotient = loop_block
//                .append_operation(arith::divui(prev_remainder, remainder, location))
                .append_operation(arith::divui(prev_remainder, remainder, location))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            // Then r_(i+1) = r_(i-1) - q * r_i, and inv_(i+1) = inv_(i-1) - q * inv_i
            // Then r_(i+1) = r_(i-1) - q * r_i, and inv_(i+1) = inv_(i-1) - q * inv_i
//            let rem_times_quo = loop_block
            let rem_times_quo = loop_block
//                .append_operation(arith::muli(remainder, quotient, location))
                .append_operation(arith::muli(remainder, quotient, location))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            let inv_times_quo = loop_block
            let inv_times_quo = loop_block
//                .append_operation(arith::muli(inverse, quotient, location))
                .append_operation(arith::muli(inverse, quotient, location))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            let next_remainder = loop_block
            let next_remainder = loop_block
//                .append_operation(arith::subi(prev_remainder, rem_times_quo, location))
                .append_operation(arith::subi(prev_remainder, rem_times_quo, location))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            let next_inverse = loop_block
            let next_inverse = loop_block
//                .append_operation(arith::subi(prev_inverse, inv_times_quo, location))
                .append_operation(arith::subi(prev_inverse, inv_times_quo, location))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//

//            // If r_(i+1) is 0, then inv_i is the inverse
            // If r_(i+1) is 0, then inv_i is the inverse
//            let zero = loop_block
            let zero = loop_block
//                .append_operation(arith::constant(
                .append_operation(arith::constant(
//                    context,
                    context,
//                    IntegerAttribute::new(i512, 0).into(),
                    IntegerAttribute::new(i512, 0).into(),
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            let next_remainder_eq_zero = loop_block
            let next_remainder_eq_zero = loop_block
//                .append_operation(arith::cmpi(
                .append_operation(arith::cmpi(
//                    context,
                    context,
//                    CmpiPredicate::Eq,
                    CmpiPredicate::Eq,
//                    next_remainder,
                    next_remainder,
//                    zero,
                    zero,
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            loop_block.append_operation(cf::cond_br(
            loop_block.append_operation(cf::cond_br(
//                context,
                context,
//                next_remainder_eq_zero,
                next_remainder_eq_zero,
//                negative_check_block,
                negative_check_block,
//                loop_block,
                loop_block,
//                &[],
                &[],
//                &[remainder, next_remainder, inverse, next_inverse],
                &[remainder, next_remainder, inverse, next_inverse],
//                location,
                location,
//            ));
            ));
//

//            // egcd sometimes returns a negative number for the inverse,
            // egcd sometimes returns a negative number for the inverse,
//            // in such cases we must simply wrap it around back into [0, PRIME)
            // in such cases we must simply wrap it around back into [0, PRIME)
//            // this suffices because |inv_i| <= divfloor(PRIME,2)
            // this suffices because |inv_i| <= divfloor(PRIME,2)
//            let zero = negative_check_block
            let zero = negative_check_block
//                .append_operation(arith::constant(
                .append_operation(arith::constant(
//                    context,
                    context,
//                    IntegerAttribute::new(i512, 0).into(),
                    IntegerAttribute::new(i512, 0).into(),
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//

//            let is_negative = negative_check_block
            let is_negative = negative_check_block
//                .append_operation(arith::cmpi(
                .append_operation(arith::cmpi(
//                    context,
                    context,
//                    CmpiPredicate::Slt,
                    CmpiPredicate::Slt,
//                    inverse,
                    inverse,
//                    zero,
                    zero,
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            // if the inverse is < 0, add PRIME
            // if the inverse is < 0, add PRIME
//            let prime = negative_check_block
            let prime = negative_check_block
//                .append_operation(arith::constant(context, attr_prime_i512, location))
                .append_operation(arith::constant(context, attr_prime_i512, location))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            let wrapped_inverse = negative_check_block
            let wrapped_inverse = negative_check_block
//                .append_operation(arith::addi(inverse, prime, location))
                .append_operation(arith::addi(inverse, prime, location))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            let inverse = negative_check_block
            let inverse = negative_check_block
//                .append_operation(arith::select(
                .append_operation(arith::select(
//                    is_negative,
                    is_negative,
//                    wrapped_inverse,
                    wrapped_inverse,
//                    inverse,
                    inverse,
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            negative_check_block.append_operation(cf::br(
            negative_check_block.append_operation(cf::br(
//                inverse_result_block,
                inverse_result_block,
//                &[inverse],
                &[inverse],
//                location,
                location,
//            ));
            ));
//

//            // Div Logic Start
            // Div Logic Start
//            // Fetch operands
            // Fetch operands
//            let lhs = entry
            let lhs = entry
//                .append_operation(arith::extui(lhs, i512, location))
                .append_operation(arith::extui(lhs, i512, location))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            let rhs = entry
            let rhs = entry
//                .append_operation(arith::extui(rhs, i512, location))
                .append_operation(arith::extui(rhs, i512, location))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            // Calculate inverse of rhs, callling the inverse implementation's starting block
            // Calculate inverse of rhs, callling the inverse implementation's starting block
//            entry.append_operation(cf::br(start_block, &[rhs], location));
            entry.append_operation(cf::br(start_block, &[rhs], location));
//            // Fetch the inverse result from the result block
            // Fetch the inverse result from the result block
//            let inverse = inverse_result_block.argument(0)?.into();
            let inverse = inverse_result_block.argument(0)?.into();
//            // Peform lhs * (1/ rhs)
            // Peform lhs * (1/ rhs)
//            let result = inverse_result_block
            let result = inverse_result_block
//                .append_operation(arith::muli(lhs, inverse, location))
                .append_operation(arith::muli(lhs, inverse, location))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//            // Apply modulo and convert result to felt252
            // Apply modulo and convert result to felt252
//            mlir_asm! { context, inverse_result_block, location =>
            mlir_asm! { context, inverse_result_block, location =>
//                ; result_mod = "arith.remui"(result, prime) : (i512, i512) -> i512
                ; result_mod = "arith.remui"(result, prime) : (i512, i512) -> i512
//                ; is_out_of_range = "arith.cmpi"(result, prime) { "predicate" = attr_cmp_uge } : (i512, i512) -> bool_ty
                ; is_out_of_range = "arith.cmpi"(result, prime) { "predicate" = attr_cmp_uge } : (i512, i512) -> bool_ty
//

//                ; result = "arith.select"(is_out_of_range, result_mod, result) : (bool_ty, i512, i512) -> i512
                ; result = "arith.select"(is_out_of_range, result_mod, result) : (bool_ty, i512, i512) -> i512
//                ; result = "arith.trunci"(result) : (i512) -> felt252_ty
                ; result = "arith.trunci"(result) : (i512) -> felt252_ty
//            }
            }
//            inverse_result_block.append_operation(helper.br(0, &[result], location));
            inverse_result_block.append_operation(helper.br(0, &[result], location));
//            return Ok(());
            return Ok(());
//        }
        }
//    };
    };
//

//    entry.append_operation(helper.br(0, &[result], location));
    entry.append_operation(helper.br(0, &[result], location));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `felt252_const` libfunc.
/// Generate MLIR operations for the `felt252_const` libfunc.
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
//    info: &Felt252ConstConcreteLibfunc,
    info: &Felt252ConstConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let value = match info.c.sign() {
    let value = match info.c.sign() {
//        Sign::Minus => {
        Sign::Minus => {
//            let prime = metadata
            let prime = metadata
//                .get::<PrimeModuloMeta<Felt>>()
                .get::<PrimeModuloMeta<Felt>>()
//                .ok_or(Error::MissingMetadata)?
                .ok_or(Error::MissingMetadata)?
//                .prime();
                .prime();
//            (&info.c + prime.to_bigint().expect("always is Some"))
            (&info.c + prime.to_bigint().expect("always is Some"))
//                .to_biguint()
                .to_biguint()
//                .expect("always is positive")
                .expect("always is positive")
//        }
        }
//        _ => info.c.to_biguint().expect("sign already checked"),
        _ => info.c.to_biguint().expect("sign already checked"),
//    };
    };
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
//        &info.branch_signatures()[0].vars[0].ty,
        &info.branch_signatures()[0].vars[0].ty,
//    )?;
    )?;
//

//    let attr_c = Attribute::parse(context, &format!("{value} : {felt252_ty}"))
    let attr_c = Attribute::parse(context, &format!("{value} : {felt252_ty}"))
//        .ok_or(Error::ParseAttributeError)?;
        .ok_or(Error::ParseAttributeError)?;
//

//    mlir_asm! { context, entry, location =>
    mlir_asm! { context, entry, location =>
//        ; k0 = "arith.constant"() { "value" = attr_c } : () -> felt252_ty
        ; k0 = "arith.constant"() { "value" = attr_c } : () -> felt252_ty
//    }
    }
//

//    entry.append_operation(helper.br(0, &[k0], location));
    entry.append_operation(helper.br(0, &[k0], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `felt252_is_zero` libfunc.
/// Generate MLIR operations for the `felt252_is_zero` libfunc.
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
//

//    Ok(())
    Ok(())
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//pub mod test {
pub mod test {
//    use crate::{
    use crate::{
//        utils::test::{load_cairo, run_program, run_program_assert_output},
        utils::test::{load_cairo, run_program, run_program_assert_output},
//        values::JitValue,
        values::JitValue,
//    };
    };
//    use cairo_lang_sierra::program::Program;
    use cairo_lang_sierra::program::Program;
//    use lazy_static::lazy_static;
    use lazy_static::lazy_static;
//

//    lazy_static! {
    lazy_static! {
//        static ref FELT252_ADD: (String, Program) = load_cairo! {
        static ref FELT252_ADD: (String, Program) = load_cairo! {
//            use core::debug::PrintTrait;
            use core::debug::PrintTrait;
//            fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
            fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
//                lhs.print();
                lhs.print();
//                rhs.print();
                rhs.print();
//                let result = lhs + rhs;
                let result = lhs + rhs;
//

//    result.print();
    result.print();
//

//    result
    result
//            }
            }
//        };
        };
//

//        static ref FELT252_SUB: (String, Program) = load_cairo! {
        static ref FELT252_SUB: (String, Program) = load_cairo! {
//            fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
            fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
//                lhs - rhs
                lhs - rhs
//            }
            }
//        };
        };
//

//        static ref FELT252_MUL: (String, Program) = load_cairo! {
        static ref FELT252_MUL: (String, Program) = load_cairo! {
//            fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
            fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
//                lhs * rhs
                lhs * rhs
//            }
            }
//        };
        };
//

//        // TODO: Add test program for `felt252_div`.
        // TODO: Add test program for `felt252_div`.
//

//        // TODO: Add test program for `felt252_add_const`.
        // TODO: Add test program for `felt252_add_const`.
//        // TODO: Add test program for `felt252_sub_const`.
        // TODO: Add test program for `felt252_sub_const`.
//        // TODO: Add test program for `felt252_mul_const`.
        // TODO: Add test program for `felt252_mul_const`.
//        // TODO: Add test program for `felt252_div_const`.
        // TODO: Add test program for `felt252_div_const`.
//

//        static ref FELT252_CONST: (String, Program) = load_cairo! {
        static ref FELT252_CONST: (String, Program) = load_cairo! {
//            fn run_test() -> (felt252, felt252, felt252, felt252) {
            fn run_test() -> (felt252, felt252, felt252, felt252) {
//                (0, 1, -2, -1)
                (0, 1, -2, -1)
//            }
            }
//        };
        };
//

//        static ref FELT252_IS_ZERO: (String, Program) = load_cairo! {
        static ref FELT252_IS_ZERO: (String, Program) = load_cairo! {
//            fn run_test(x: felt252) -> felt252 {
            fn run_test(x: felt252) -> felt252 {
//                match x {
                match x {
//                    0 => 1,
                    0 => 1,
//                    _ => 0,
                    _ => 0,
//                }
                }
//            }
            }
//        };
        };
//    }
    }
//

//    #[test]
    #[test]
//    fn felt252_add() {
    fn felt252_add() {
//        run_program_assert_output(
        run_program_assert_output(
//            &FELT252_ADD,
            &FELT252_ADD,
//            "run_test",
            "run_test",
//            &[JitValue::felt_str("0"), JitValue::felt_str("0")],
            &[JitValue::felt_str("0"), JitValue::felt_str("0")],
//            JitValue::felt_str("0"),
            JitValue::felt_str("0"),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            &FELT252_ADD,
            &FELT252_ADD,
//            "run_test",
            "run_test",
//            &[JitValue::felt_str("1"), JitValue::felt_str("2")],
            &[JitValue::felt_str("1"), JitValue::felt_str("2")],
//            JitValue::felt_str("3"),
            JitValue::felt_str("3"),
//        );
        );
//

//        fn r(lhs: JitValue, rhs: JitValue) -> JitValue {
        fn r(lhs: JitValue, rhs: JitValue) -> JitValue {
//            run_program(&FELT252_ADD, "run_test", &[lhs, rhs]).return_value
            run_program(&FELT252_ADD, "run_test", &[lhs, rhs]).return_value
//        }
        }
//

//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("0"), JitValue::felt_str("1")),
            r(JitValue::felt_str("0"), JitValue::felt_str("1")),
//            JitValue::felt_str("1")
            JitValue::felt_str("1")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("0"), JitValue::felt_str("-2")),
            r(JitValue::felt_str("0"), JitValue::felt_str("-2")),
//            JitValue::felt_str("-2")
            JitValue::felt_str("-2")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("0"), JitValue::felt_str("-1")),
            r(JitValue::felt_str("0"), JitValue::felt_str("-1")),
//            JitValue::felt_str("-1")
            JitValue::felt_str("-1")
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("1"), JitValue::felt_str("0")),
            r(JitValue::felt_str("1"), JitValue::felt_str("0")),
//            JitValue::felt_str("1")
            JitValue::felt_str("1")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("1"), JitValue::felt_str("1")),
            r(JitValue::felt_str("1"), JitValue::felt_str("1")),
//            JitValue::felt_str("2")
            JitValue::felt_str("2")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("1"), JitValue::felt_str("-2")),
            r(JitValue::felt_str("1"), JitValue::felt_str("-2")),
//            JitValue::felt_str("-1")
            JitValue::felt_str("-1")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("1"), JitValue::felt_str("-1")),
            r(JitValue::felt_str("1"), JitValue::felt_str("-1")),
//            JitValue::felt_str("0")
            JitValue::felt_str("0")
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-2"), JitValue::felt_str("0")),
            r(JitValue::felt_str("-2"), JitValue::felt_str("0")),
//            JitValue::felt_str("-2")
            JitValue::felt_str("-2")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-2"), JitValue::felt_str("1")),
            r(JitValue::felt_str("-2"), JitValue::felt_str("1")),
//            JitValue::felt_str("-1")
            JitValue::felt_str("-1")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-2"), JitValue::felt_str("-2")),
            r(JitValue::felt_str("-2"), JitValue::felt_str("-2")),
//            JitValue::felt_str("-4")
            JitValue::felt_str("-4")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-2"), JitValue::felt_str("-1")),
            r(JitValue::felt_str("-2"), JitValue::felt_str("-1")),
//            JitValue::felt_str("-3")
            JitValue::felt_str("-3")
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-1"), JitValue::felt_str("0")),
            r(JitValue::felt_str("-1"), JitValue::felt_str("0")),
//            JitValue::felt_str("-1")
            JitValue::felt_str("-1")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-1"), JitValue::felt_str("1")),
            r(JitValue::felt_str("-1"), JitValue::felt_str("1")),
//            JitValue::felt_str("0")
            JitValue::felt_str("0")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-1"), JitValue::felt_str("-2")),
            r(JitValue::felt_str("-1"), JitValue::felt_str("-2")),
//            JitValue::felt_str("-3")
            JitValue::felt_str("-3")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-1"), JitValue::felt_str("-1")),
            r(JitValue::felt_str("-1"), JitValue::felt_str("-1")),
//            JitValue::felt_str("-2")
            JitValue::felt_str("-2")
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn felt252_sub() {
    fn felt252_sub() {
//        let r = |lhs, rhs| run_program(&FELT252_SUB, "run_test", &[lhs, rhs]).return_value;
        let r = |lhs, rhs| run_program(&FELT252_SUB, "run_test", &[lhs, rhs]).return_value;
//

//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("0"), JitValue::felt_str("0")),
            r(JitValue::felt_str("0"), JitValue::felt_str("0")),
//            JitValue::felt_str("0")
            JitValue::felt_str("0")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("0"), JitValue::felt_str("1")),
            r(JitValue::felt_str("0"), JitValue::felt_str("1")),
//            JitValue::felt_str("-1")
            JitValue::felt_str("-1")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("0"), JitValue::felt_str("-2")),
            r(JitValue::felt_str("0"), JitValue::felt_str("-2")),
//            JitValue::felt_str("2")
            JitValue::felt_str("2")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("0"), JitValue::felt_str("-1")),
            r(JitValue::felt_str("0"), JitValue::felt_str("-1")),
//            JitValue::felt_str("1")
            JitValue::felt_str("1")
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("1"), JitValue::felt_str("0")),
            r(JitValue::felt_str("1"), JitValue::felt_str("0")),
//            JitValue::felt_str("1")
            JitValue::felt_str("1")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("1"), JitValue::felt_str("1")),
            r(JitValue::felt_str("1"), JitValue::felt_str("1")),
//            JitValue::felt_str("0")
            JitValue::felt_str("0")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("1"), JitValue::felt_str("-2")),
            r(JitValue::felt_str("1"), JitValue::felt_str("-2")),
//            JitValue::felt_str("3")
            JitValue::felt_str("3")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("1"), JitValue::felt_str("-1")),
            r(JitValue::felt_str("1"), JitValue::felt_str("-1")),
//            JitValue::felt_str("2")
            JitValue::felt_str("2")
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-2"), JitValue::felt_str("0")),
            r(JitValue::felt_str("-2"), JitValue::felt_str("0")),
//            JitValue::felt_str("-2")
            JitValue::felt_str("-2")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-2"), JitValue::felt_str("1")),
            r(JitValue::felt_str("-2"), JitValue::felt_str("1")),
//            JitValue::felt_str("-3")
            JitValue::felt_str("-3")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-2"), JitValue::felt_str("-2")),
            r(JitValue::felt_str("-2"), JitValue::felt_str("-2")),
//            JitValue::felt_str("0")
            JitValue::felt_str("0")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-2"), JitValue::felt_str("-1")),
            r(JitValue::felt_str("-2"), JitValue::felt_str("-1")),
//            JitValue::felt_str("-1")
            JitValue::felt_str("-1")
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-1"), JitValue::felt_str("0")),
            r(JitValue::felt_str("-1"), JitValue::felt_str("0")),
//            JitValue::felt_str("-1")
            JitValue::felt_str("-1")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-1"), JitValue::felt_str("1")),
            r(JitValue::felt_str("-1"), JitValue::felt_str("1")),
//            JitValue::felt_str("-2")
            JitValue::felt_str("-2")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-1"), JitValue::felt_str("-2")),
            r(JitValue::felt_str("-1"), JitValue::felt_str("-2")),
//            JitValue::felt_str("1")
            JitValue::felt_str("1")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-1"), JitValue::felt_str("-1")),
            r(JitValue::felt_str("-1"), JitValue::felt_str("-1")),
//            JitValue::felt_str("0")
            JitValue::felt_str("0")
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn felt252_mul() {
    fn felt252_mul() {
//        let r = |lhs, rhs| run_program(&FELT252_MUL, "run_test", &[lhs, rhs]).return_value;
        let r = |lhs, rhs| run_program(&FELT252_MUL, "run_test", &[lhs, rhs]).return_value;
//

//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("0"), JitValue::felt_str("0")),
            r(JitValue::felt_str("0"), JitValue::felt_str("0")),
//            JitValue::felt_str("0")
            JitValue::felt_str("0")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("0"), JitValue::felt_str("1")),
            r(JitValue::felt_str("0"), JitValue::felt_str("1")),
//            JitValue::felt_str("0")
            JitValue::felt_str("0")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("0"), JitValue::felt_str("-2")),
            r(JitValue::felt_str("0"), JitValue::felt_str("-2")),
//            JitValue::felt_str("0")
            JitValue::felt_str("0")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("0"), JitValue::felt_str("-1")),
            r(JitValue::felt_str("0"), JitValue::felt_str("-1")),
//            JitValue::felt_str("0")
            JitValue::felt_str("0")
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("1"), JitValue::felt_str("0")),
            r(JitValue::felt_str("1"), JitValue::felt_str("0")),
//            JitValue::felt_str("0")
            JitValue::felt_str("0")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("1"), JitValue::felt_str("1")),
            r(JitValue::felt_str("1"), JitValue::felt_str("1")),
//            JitValue::felt_str("1")
            JitValue::felt_str("1")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("1"), JitValue::felt_str("-2")),
            r(JitValue::felt_str("1"), JitValue::felt_str("-2")),
//            JitValue::felt_str("-2")
            JitValue::felt_str("-2")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("1"), JitValue::felt_str("-1")),
            r(JitValue::felt_str("1"), JitValue::felt_str("-1")),
//            JitValue::felt_str("-1")
            JitValue::felt_str("-1")
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-2"), JitValue::felt_str("0")),
            r(JitValue::felt_str("-2"), JitValue::felt_str("0")),
//            JitValue::felt_str("0")
            JitValue::felt_str("0")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-2"), JitValue::felt_str("1")),
            r(JitValue::felt_str("-2"), JitValue::felt_str("1")),
//            JitValue::felt_str("-2")
            JitValue::felt_str("-2")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-2"), JitValue::felt_str("-2")),
            r(JitValue::felt_str("-2"), JitValue::felt_str("-2")),
//            JitValue::felt_str("4")
            JitValue::felt_str("4")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-2"), JitValue::felt_str("-1")),
            r(JitValue::felt_str("-2"), JitValue::felt_str("-1")),
//            JitValue::felt_str("2")
            JitValue::felt_str("2")
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-1"), JitValue::felt_str("0")),
            r(JitValue::felt_str("-1"), JitValue::felt_str("0")),
//            JitValue::felt_str("0")
            JitValue::felt_str("0")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-1"), JitValue::felt_str("1")),
            r(JitValue::felt_str("-1"), JitValue::felt_str("1")),
//            JitValue::felt_str("-1")
            JitValue::felt_str("-1")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-1"), JitValue::felt_str("-2")),
            r(JitValue::felt_str("-1"), JitValue::felt_str("-2")),
//            JitValue::felt_str("2")
            JitValue::felt_str("2")
//        );
        );
//        assert_eq!(
        assert_eq!(
//            r(JitValue::felt_str("-1"), JitValue::felt_str("-1")),
            r(JitValue::felt_str("-1"), JitValue::felt_str("-1")),
//            JitValue::felt_str("1")
            JitValue::felt_str("1")
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn felt252_const() {
    fn felt252_const() {
//        assert_eq!(
        assert_eq!(
//            run_program(&FELT252_CONST, "run_test", &[]).return_value,
            run_program(&FELT252_CONST, "run_test", &[]).return_value,
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![
                fields: vec![
//                    JitValue::felt_str("0"),
                    JitValue::felt_str("0"),
//                    JitValue::felt_str("1"),
                    JitValue::felt_str("1"),
//                    JitValue::felt_str("-2"),
                    JitValue::felt_str("-2"),
//                    JitValue::felt_str("-1")
                    JitValue::felt_str("-1")
//                ],
                ],
//                debug_name: None
                debug_name: None
//            }
            }
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn felt252_is_zero() {
    fn felt252_is_zero() {
//        let r = |x| run_program(&FELT252_IS_ZERO, "run_test", &[x]).return_value;
        let r = |x| run_program(&FELT252_IS_ZERO, "run_test", &[x]).return_value;
//

//        assert_eq!(r(JitValue::felt_str("0")), JitValue::felt_str("1"));
        assert_eq!(r(JitValue::felt_str("0")), JitValue::felt_str("1"));
//        assert_eq!(r(JitValue::felt_str("1")), JitValue::felt_str("0"));
        assert_eq!(r(JitValue::felt_str("1")), JitValue::felt_str("0"));
//        assert_eq!(r(JitValue::felt_str("-2")), JitValue::felt_str("0"));
        assert_eq!(r(JitValue::felt_str("-2")), JitValue::felt_str("0"));
//        assert_eq!(r(JitValue::felt_str("-1")), JitValue::felt_str("0"));
        assert_eq!(r(JitValue::felt_str("-1")), JitValue::felt_str("0"));
//    }
    }
//}
}
