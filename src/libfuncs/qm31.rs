use crate::{
    error::{Error, Result},
    metadata::runtime_bindings::RuntimeBindingsMeta,
    utils::get_integer_layout,
};
use crate::{libfuncs::LibfuncHelper, metadata::MetadataStorage};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        lib_func::SignatureOnlyConcreteLibfunc,
        qm31::{QM31BinaryOpConcreteLibfunc, QM31Concrete, QM31ConstConcreteLibfunc},
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        cf, llvm,
    },
    helpers::{ArithBlockExt, BuiltinBlockExt, LlvmBlockExt},
    ir::{r#type::IntegerType, Block, BlockLike, Location},
    Context,
};

const M31_PRIME: u32 = 0x7fffffff;

pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &QM31Concrete,
) -> Result<()> {
    match selector {
        QM31Concrete::BinaryOperation(info) => {
            build_binary_op(context, registry, entry, location, helper, metadata, info)
        }
        QM31Concrete::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        QM31Concrete::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, metadata, info)
        }
        QM31Concrete::Pack(info) => {
            build_pack(context, registry, entry, location, helper, metadata, info)
        }
        QM31Concrete::Unpack(info) => {
            build_unpack(context, registry, entry, location, helper, metadata, info)
        }
        QM31Concrete::FromM31(info) => {
            build_from_m31(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `qm31_const` libfunc.
///
/// Receives 4 const m31 and returns a qm31.
///
/// # Constraints
///
/// m31 are always between 0 and 2**31 - 2 (inclusive)
///
/// # Cairo Signature
///
/// ```cairo
/// fn qm31_const<
///     const W0: m31, const W1: m31, const W2: m31, const W3: m31,
/// >() -> qm31 nopanic;
/// ```
pub fn build_const<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &QM31ConstConcreteLibfunc,
) -> Result<()> {
    let m31_ty = IntegerType::new(context, 31).into();
    let qm31_ty = llvm::r#type::array(m31_ty, 4);

    let m31_0 = entry.const_int_from_type(context, location, info.w0, m31_ty)?;
    let m31_1 = entry.const_int_from_type(context, location, info.w1, m31_ty)?;
    let m31_2 = entry.const_int_from_type(context, location, info.w2, m31_ty)?;
    let m31_3 = entry.const_int_from_type(context, location, info.w3, m31_ty)?;

    let qm31 = entry.append_op_result(llvm::undef(qm31_ty, location))?;
    let qm31 = entry.insert_values(context, location, qm31, &[m31_0, m31_1, m31_2, m31_3])?;

    helper.br(entry, 0, &[qm31], location)
}

/// Generate MLIR operations for the `qm31_is_zero` libfunc.
///
/// Receives a qm31 and returns a Some(qm31) if the argument is not 0,
/// otherwise a None.
///
/// # Cairo Signature
///
/// ```cairo
/// fn qm31_is_zero(a: qm31) -> core::internal::OptionRev<NonZero<qm31>> nopanic;
/// ```
pub fn build_is_zero<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let qm31 = entry.arg(0)?;
    let m31_ty = IntegerType::new(context, 31).into();

    let m31_0 = entry.extract_value(context, location, qm31, m31_ty, 0)?;
    let m31_1 = entry.extract_value(context, location, qm31, m31_ty, 1)?;
    let m31_2 = entry.extract_value(context, location, qm31, m31_ty, 2)?;
    let m31_3 = entry.extract_value(context, location, qm31, m31_ty, 3)?;

    // Check that every limb is equal to 0:
    //      (m31_0 | m31_1 | m31_2 | m31_3) == 0
    let cond = entry.append_op_result(arith::ori(m31_0, m31_1, location))?;
    let cond = entry.append_op_result(arith::ori(cond, m31_2, location))?;
    let cond = entry.append_op_result(arith::ori(cond, m31_3, location))?;
    let k0 = entry.const_int_from_type(context, location, 0, m31_ty)?;
    let cond =
        entry.append_op_result(arith::cmpi(context, CmpiPredicate::Eq, k0, cond, location))?;

    helper.cond_br(
        context,
        entry,
        cond,
        [0, 1],
        [&[], &[entry.arg(0)?]],
        location,
    )
}

/// Generate MLIR operations for the `qm31_pack` libfunc.
///
/// Receives four m31 and packs them into a qm31.
///
/// # Constraints
///
/// m31 are always between 0 and 2**31 - 2 (inclusive)
///
/// # Cairo Signature
///
/// ```cairo
/// fn qm31_pack(w0: m31, w1: m31, w2: m31, w3: m31) -> qm31 nopanic;
/// ```
pub fn build_pack<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let m31_0 = entry.arg(0)?;
    let m31_1 = entry.arg(1)?;
    let m31_2 = entry.arg(2)?;
    let m31_3 = entry.arg(3)?;

    let m31_ty = IntegerType::new(context, 31).into();
    let qm31_ty = llvm::r#type::array(m31_ty, 4);

    let qm31 = entry.append_op_result(llvm::undef(qm31_ty, location))?;
    let qm31 = entry.insert_values(context, location, qm31, &[m31_0, m31_1, m31_2, m31_3])?;

    helper.br(entry, 0, &[qm31], location)
}

/// Generate MLIR operations for the `qm31_unpack` libfunc.
///
/// Receives a qm31 and unpacks it, returning an array with the
/// four m31.
///
/// # Constraints
///
/// m31 are always between 0 and 2**31 - 2 (inclusive)
///
/// # Cairo Signature
///
/// ```cairo
/// fn qm31_unpack(a: qm31) -> (m31, m31, m31, m31) implicits(crate::RangeCheck) nopanic;
/// ```
pub fn build_unpack<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check =
        super::increment_builtin_counter_by(context, entry, location, entry.arg(0)?, 5)?;
    let m31_ty = IntegerType::new(context, 31);
    let qm31 = entry.arg(1)?;

    let m31_0 = entry.extract_value(context, location, qm31, m31_ty.into(), 0)?;
    let m31_1 = entry.extract_value(context, location, qm31, m31_ty.into(), 1)?;
    let m31_2 = entry.extract_value(context, location, qm31, m31_ty.into(), 2)?;
    let m31_3 = entry.extract_value(context, location, qm31, m31_ty.into(), 3)?;

    helper.br(
        entry,
        0,
        &[range_check, m31_0, m31_1, m31_2, m31_3],
        location,
    )
}

/// Generate MLIR operations for the `qm31_from_m31` libfunc.
///
/// Receives a m31 and returns a qm31 in which its first coeffiecient
/// has the value of the input and the other ones are 0.
///
/// # Constraints
///
/// m31 are always between 0 and 2**31 - 2 (inclusive)
///
/// # Cairo Signature
///
/// ```cairo
/// fn qm31_unpack(a: qm31) -> (m31, m31, m31, m31) implicits(crate::RangeCheck) nopanic;
/// ```
pub fn build_from_m31<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let m31 = entry.arg(0)?;

    let m31_ty = IntegerType::new(context, 31).into();
    let qm31_ty = llvm::r#type::array(m31_ty, 4);
    let k0 = entry.const_int_from_type(context, location, 0, m31_ty)?;

    let qm31 = entry.append_op_result(llvm::undef(qm31_ty, location))?;
    let qm31 = entry.insert_values(context, location, qm31, &[m31, k0, k0, k0])?;

    helper.br(entry, 0, &[qm31], location)
}

fn m31_add<'ctx, 'this>(
    context: &'ctx Context,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
) -> Result<()> {
    let lhs_value = entry.arg(0)?;
    let rhs_value = entry.arg(1)?;

    let lhs_value = entry.extui(lhs_value, IntegerType::new(context, 32).into(), location)?;
    let rhs_value = entry.extui(rhs_value, IntegerType::new(context, 32).into(), location)?;

    let res = entry.append_op_result(arith::addi(lhs_value, rhs_value, location))?;
    let prime = entry.const_int(context, location, M31_PRIME, 32)?;
    let res_mod = entry.append_op_result(arith::subi(res, prime, location))?;
    let is_out_of_range = entry.cmpi(context, CmpiPredicate::Uge, res, prime, location)?;

    let res = entry.append_op_result(arith::select(is_out_of_range, res_mod, res, location))?;

    let res = entry.trunci(res, IntegerType::new(context, 31).into(), location)?;

    helper.br(entry, 0, &[res], location)
}

fn m31_sub<'ctx, 'this>(
    context: &'ctx Context,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
) -> Result<()> {
    let lhs_value = entry.arg(0)?;
    let rhs_value = entry.arg(1)?;

    let res = entry.append_op_result(arith::subi(lhs_value, rhs_value, location))?;
    let prime = entry.const_int(context, location, M31_PRIME, 31)?;
    let res_mod = entry.append_op_result(arith::addi(res, prime, location))?;
    let is_out_of_range =
        entry.cmpi(context, CmpiPredicate::Ult, lhs_value, rhs_value, location)?;

    let res = entry.append_op_result(arith::select(is_out_of_range, res_mod, res, location))?;
    helper.br(entry, 0, &[res], location)
}

fn m31_mul<'ctx, 'this>(
    context: &'ctx Context,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
) -> Result<()> {
    let lhs_value = entry.arg(0)?;
    let rhs_value = entry.arg(1)?;

    let lhs_value = entry.extui(lhs_value, IntegerType::new(context, 64).into(), location)?;
    let rhs_value = entry.extui(rhs_value, IntegerType::new(context, 64).into(), location)?;
    let res = entry.muli(lhs_value, rhs_value, location)?;

    let prime = entry.const_int(context, location, M31_PRIME, 64)?;
    let res_mod = entry.append_op_result(arith::remui(res, prime, location))?;
    let is_out_of_range = entry.cmpi(context, CmpiPredicate::Uge, res, prime, location)?;

    let res = entry.append_op_result(arith::select(is_out_of_range, res_mod, res, location))?;
    let res = entry.trunci(res, IntegerType::new(context, 31).into(), location)?;

    helper.br(entry, 0, &[res], location)
}

fn m31_div<'ctx, 'this>(
    context: &'ctx Context,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
) -> Result<()> {
    let lhs_value = entry.arg(0)?;
    let rhs_value = entry.arg(1)?;

    let i31 = IntegerType::new(context, 31).into();
    let i64 = IntegerType::new(context, 64).into();

    let start_block = helper.append_block(Block::new(&[(i31, location)]));
    let loop_block = helper.append_block(Block::new(&[
        (i31, location),
        (i31, location),
        (i31, location),
        (i31, location),
    ]));
    let negative_check_block = helper.append_block(Block::new(&[]));
    // Block containing final result
    let inverse_result_block = helper.append_block(Block::new(&[(i31, location)]));
    // Egcd works by calculating a series of remainders, each the remainder of dividing the previous two
    // For the initial setup, r0 = PRIME, r1 = a
    // This order is chosen because if we reverse them, then the first iteration will just swap them
    let prev_remainder = start_block.const_int_from_type(context, location, M31_PRIME, i31)?;
    let remainder = start_block.arg(0)?;
    // Similarly we'll calculate another series which starts 0,1,... and from which we will retrieve the modular inverse of a
    let prev_inverse = start_block.const_int_from_type(context, location, 0, i31)?;
    let inverse = start_block.const_int_from_type(context, location, 1, i31)?;
    start_block.append_operation(cf::br(
        loop_block,
        &[prev_remainder, remainder, prev_inverse, inverse],
        location,
    ));

    //---Loop body---
    // Arguments are rem_(i-1), rem, inv_(i-1), inv
    let prev_remainder = loop_block.arg(0)?;
    let remainder = loop_block.arg(1)?;
    let prev_inverse = loop_block.arg(2)?;
    let inverse = loop_block.arg(3)?;

    // First calculate q = rem_(i-1)/rem_i, rounded down
    let quotient =
        loop_block.append_op_result(arith::divui(prev_remainder, remainder, location))?;
    // Then r_(i+1) = r_(i-1) - q * r_i, and inv_(i+1) = inv_(i-1) - q * inv_i
    let rem_times_quo = loop_block.muli(remainder, quotient, location)?;
    let inv_times_quo = loop_block.muli(inverse, quotient, location)?;
    let next_remainder =
        loop_block.append_op_result(arith::subi(prev_remainder, rem_times_quo, location))?;
    let next_inverse =
        loop_block.append_op_result(arith::subi(prev_inverse, inv_times_quo, location))?;

    // If r_(i+1) is 0, then inv_i is the inverse
    let zero = loop_block.const_int_from_type(context, location, 0, i31)?;
    let next_remainder_eq_zero =
        loop_block.cmpi(context, CmpiPredicate::Eq, next_remainder, zero, location)?;
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
    let zero = negative_check_block.const_int_from_type(context, location, 0, i31)?;
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
    let prime = negative_check_block.const_int_from_type(context, location, M31_PRIME, i31)?;
    let wrapped_inverse = negative_check_block.addi(inverse, prime, location)?;
    let inverse = negative_check_block.append_op_result(arith::select(
        is_negative,
        wrapped_inverse,
        inverse,
        location,
    ))?;
    negative_check_block.append_operation(cf::br(inverse_result_block, &[inverse], location));

    // Div Logic Start
    // Fetch operands
    let lhs_value = entry.extui(lhs_value, i64, location)?;
    // Calculate inverse of rhs, callling the inverse implementation's starting block
    entry.append_operation(cf::br(start_block, &[rhs_value], location));
    // Fetch the inverse result from the result block
    let inverse = inverse_result_block.arg(0)?;
    let inverse = inverse_result_block.extui(inverse, i64, location)?;
    // Peform lhs * (1/ rhs)
    let result = inverse_result_block.muli(lhs_value, inverse, location)?;
    // Apply modulo and convert result to m31
    let prime = inverse_result_block.extui(prime, i64, location)?;
    let result_mod =
        inverse_result_block.append_op_result(arith::remui(result, prime, location))?;
    let is_out_of_range =
        inverse_result_block.cmpi(context, CmpiPredicate::Uge, result, prime, location)?;

    let result = inverse_result_block.append_op_result(arith::select(
        is_out_of_range,
        result_mod,
        result,
        location,
    ))?;
    let result = inverse_result_block.trunci(result, i31, location)?;

    helper.br(inverse_result_block, 0, &[result], location)
}

/// Generate MLIR operations for the QM31 and M31 binary operations libfuncs.
///
/// Depending on the type of the parameters, it chooses which type of representation
/// it will manage (QM31 or M31). It either receives 2 qm31 or 2 m31 (which are represented
/// as bounded ints)
///
/// # Cairo Signature
/// ```cairo
/// // qm31
/// fn qm31_add(a: qm31, b: qm31) -> qm31 nopanic;
/// fn qm31_sub(a: qm31, b: qm31) -> qm31 nopanic;
/// fn qm31_mul(a: qm31, b: qm31) -> qm31 nopanic;
/// fn qm31_div(a: qm31, b: NonZero<qm31>) -> qm31 nopanic;
///
/// // m31
/// extern fn m31_add(a: m31, b: m31) -> m31 nopanic;
/// extern fn m31_sub(a: m31, b: m31) -> m31 nopanic;
/// extern fn m31_mul(a: m31, b: m31) -> m31 nopanic;
/// extern fn m31_div(a: m31, b: NonZero<m31>) -> m31 nopanic;
/// ```
pub fn build_binary_op<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &QM31BinaryOpConcreteLibfunc,
) -> Result<()> {
    // If the parameter is a bounded int, then we need to generate the operations
    // for the m31
    let type_concrete = registry.get_type(&info.param_signatures()[0].ty)?;
    if let CoreTypeConcrete::BoundedInt(_) = type_concrete {
        match info.operator {
            cairo_lang_sierra::extensions::qm31::QM31BinaryOperator::Add => {
                return m31_add(context, entry, location, helper);
            }
            cairo_lang_sierra::extensions::qm31::QM31BinaryOperator::Sub => {
                return m31_sub(context, entry, location, helper);
            }
            cairo_lang_sierra::extensions::qm31::QM31BinaryOperator::Mul => {
                return m31_mul(context, entry, location, helper);
            }
            cairo_lang_sierra::extensions::qm31::QM31BinaryOperator::Div => {
                return m31_div(context, entry, location, helper);
            }
        }
    }
    let m31_ty = IntegerType::new(context, 31).into();
    let qm31_ty = llvm::r#type::array(m31_ty, 4);

    let lhs = entry.arg(0)?;
    let rhs = entry.arg(1)?;

    let lhs_ptr = entry.alloca1(context, location, qm31_ty, get_integer_layout(31).align())?;
    let rhs_ptr = entry.alloca1(context, location, qm31_ty, get_integer_layout(31).align())?;

    entry.store(context, location, lhs_ptr, lhs)?;
    entry.store(context, location, rhs_ptr, rhs)?;

    let result = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?
        .libfunc_qm31_bin_op(
            context,
            helper,
            entry,
            lhs_ptr,
            rhs_ptr,
            info.operator,
            location,
        )?;

    helper.br(entry, 0, &[result], location)
}

#[cfg(test)]
mod test {
    use ark_ff::Zero;
    use cairo_lang_sierra::extensions::utils::Range;
    use cairo_vm::Felt252;
    use num_bigint::BigInt;

    use crate::{
        jit_enum, jit_struct, libfuncs::qm31::M31_PRIME, load_cairo,
        runtime::to_representative_coefficients, utils::testing::run_program, Value,
    };

    impl From<&starknet_types_core::qm31::QM31> for Value {
        fn from(qm31: &starknet_types_core::qm31::QM31) -> Self {
            let coefficients = to_representative_coefficients(qm31.clone());
            Value::QM31(
                coefficients[0],
                coefficients[1],
                coefficients[2],
                coefficients[3],
            )
        }
    }

    #[test]
    fn run_unpack() {
        let program = load_cairo! {
            use core::qm31::{QM31Trait, m31, qm31};

            fn run_test_1() -> [m31;4] {
                let qm31 = QM31Trait::new(1, 2, 3, 4);
                let unpacked_qm31 = qm31.unpack();

                unpacked_qm31
            }

            fn run_test_2() -> [m31;4] {
                let qm31 = QM31Trait::new(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2);
                let unpacked_qm31 = qm31.unpack();

                unpacked_qm31
            }
        };

        let result = run_program(&program, "run_test_1", &[]).return_value;
        let m31_range = Range::closed(0, BigInt::from(2147483646));
        let Value::Struct { fields, .. } = result else {
            panic!("Expected a Value::Struct()");
        };
        assert_eq!(
            fields,
            vec![
                Value::BoundedInt {
                    value: Felt252::from(1),
                    range: m31_range.clone()
                },
                Value::BoundedInt {
                    value: Felt252::from(2),
                    range: m31_range.clone()
                },
                Value::BoundedInt {
                    value: Felt252::from(3),
                    range: m31_range.clone()
                },
                Value::BoundedInt {
                    value: Felt252::from(4),
                    range: m31_range.clone()
                },
            ]
        );

        let result = run_program(&program, "run_test_2", &[]).return_value;
        let Value::Struct { fields, .. } = result else {
            panic!("Expected a Value::Struct()");
        };
        assert_eq!(
            fields,
            vec![
                Value::BoundedInt {
                    value: Felt252::from(0x544b2fba),
                    range: m31_range.clone()
                },
                Value::BoundedInt {
                    value: Felt252::from(0x673cff77),
                    range: m31_range.clone()
                },
                Value::BoundedInt {
                    value: Felt252::from(0x60713d44),
                    range: m31_range.clone()
                },
                Value::BoundedInt {
                    value: Felt252::from(0x499602d2),
                    range: m31_range.clone()
                },
            ]
        );
    }

    #[test]
    fn run_pack() {
        let program = load_cairo! {
            use core::qm31::{QM31Trait, qm31};

            fn run_test() -> qm31 {
                let qm31 = QM31Trait::new(1, 2, 3, 4);
                qm31
            }

            fn run_test_large_coefficients() -> qm31 {
                let qm31 = QM31Trait::new(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2);
                qm31
            }
        };
        // With small coefficients
        let result = run_program(&program, "run_test", &[]).return_value;
        assert_eq!(result, Value::QM31(1, 2, 3, 4));

        // With big coefficients
        let result = run_program(&program, "run_test_large_coefficients", &[]).return_value;
        let qm31_expected = starknet_types_core::qm31::QM31::from_coefficients(
            0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2,
        );
        let expected_coefficients = qm31_expected.to_coefficients();
        assert_eq!(
            result,
            Value::QM31(
                expected_coefficients.0,
                expected_coefficients.1,
                expected_coefficients.2,
                expected_coefficients.3
            )
        );
    }

    #[test]
    fn run_const() {
        let program = load_cairo! {
            use core::qm31::{qm31_const, qm31};

            fn run_test() -> qm31 {
                let qm31 = qm31_const::<1, 2, 3, 4>();
                qm31
            }

            fn run_test_large_coefficients() -> qm31 {
                let qm31 = qm31_const::<0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2>();
                qm31
            }
        };

        let result = run_program(&program, "run_test", &[]).return_value;
        assert_eq!(result, Value::QM31(1, 2, 3, 4));

        let result = run_program(&program, "run_test_large_coefficients", &[]).return_value;
        assert_eq!(
            result,
            Value::QM31(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2)
        );
    }

    #[test]
    fn run_is_zero() {
        let program = load_cairo! {
            use core::qm31::{QM31Trait, qm31, qm31_is_zero};
            use core::internal::OptionRev;

            fn run_test(input: qm31) -> OptionRev<NonZero<qm31>> {
                qm31_is_zero(input)
            }

            fn run_test_edge_case() -> OptionRev<NonZero<qm31>> {
                let lhs = QM31Trait::new(0x7ffffffe, 0x7ffffffe, 0x7ffffffe, 0x7ffffffe);
                let rhs = QM31Trait::new(1, 1, 1, 1);
                let qm31 = lhs + rhs;
                qm31_is_zero(qm31)
            }
        };

        let result = run_program(&program, "run_test", &[Value::QM31(0, 0, 0, 0)]).return_value;
        assert_eq!(result, jit_enum!(0, jit_struct!()));

        let result = run_program(&program, "run_test", &[Value::QM31(0, 0, 1, 0)]).return_value;
        assert_eq!(result, jit_enum!(1, Value::QM31(0, 0, 1, 0)));

        let result = run_program(
            &program,
            "run_test",
            &[Value::QM31(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2)],
        )
        .return_value;
        assert_eq!(
            result,
            jit_enum!(
                1,
                Value::QM31(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2)
            )
        );

        let result = run_program(&program, "run_test_edge_case", &[]).return_value;
        assert_eq!(result, jit_enum!(0, jit_struct!()));
    }

    #[test]
    fn run_qm31_add() {
        let program = load_cairo! {
            use core::qm31::{QM31Trait, qm31};

            fn run_test(lhs: qm31, rhs: qm31) -> qm31 {
                lhs + rhs
            }
        };

        let a = starknet_types_core::qm31::QM31::from_coefficients(
            0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2,
        );
        let b = starknet_types_core::qm31::QM31::from_coefficients(
            0x499602d2, 0x544b2fba, 0x673cff77, 0x60713d44,
        );
        let c = starknet_types_core::qm31::QM31::from_coefficients(
            0x1de1328d, 0x3b882f32, 0x47ae3cbc, 0x2a074017,
        );
        let d = starknet_types_core::qm31::QM31::from_coefficients(
            0x7ffffffe, 0x7ffffffe, 0x7ffffffe, 0x7ffffffe,
        );
        let e = starknet_types_core::qm31::QM31::from_coefficients(1, 1, 1, 1);

        let result =
            run_program(&program, "run_test", &[Value::from(&a), Value::from(&b)]).return_value;
        let expected_qm31 = a.clone() + b.clone();
        assert_eq!(result, Value::from(&expected_qm31));

        let result =
            run_program(&program, "run_test", &[Value::from(&b), Value::from(&c)]).return_value;
        let expected_qm31 = b + c.clone();
        assert_eq!(result, Value::from(&expected_qm31));

        let result =
            run_program(&program, "run_test", &[Value::from(&a), Value::from(&c)]).return_value;
        let expected_qm31 = a + c;
        assert_eq!(result, Value::from(&expected_qm31));

        let result =
            run_program(&program, "run_test", &[Value::from(&d), Value::from(&e)]).return_value;
        assert_eq!(result, Value::QM31(0, 0, 0, 0));
    }

    #[test]
    fn run_qm31_sub() {
        let program = load_cairo! {
            use core::qm31::{QM31Trait, qm31};

            fn run_test(lhs: qm31, rhs: qm31) -> qm31 {
                lhs - rhs
            }
        };

        let a = starknet_types_core::qm31::QM31::from_coefficients(
            0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2,
        );
        let b = starknet_types_core::qm31::QM31::from_coefficients(
            0x499602d2, 0x544b2fba, 0x673cff77, 0x60713d44,
        );
        let c = starknet_types_core::qm31::QM31::from_coefficients(
            0x1de1328d, 0x3b882f32, 0x47ae3cbc, 0x2a074017,
        );

        let result =
            run_program(&program, "run_test", &[Value::from(&c), Value::from(&a)]).return_value;
        let expected_qm31 = c.clone() - a.clone();
        assert_eq!(result, Value::from(&expected_qm31));

        let result =
            run_program(&program, "run_test", &[Value::from(&a), Value::from(&b)]).return_value;
        let expected_qm31 = a - b.clone();
        assert_eq!(result, Value::from(&expected_qm31));

        let result =
            run_program(&program, "run_test", &[Value::from(&b), Value::from(&c)]).return_value;
        let expected_qm31 = b - c;
        assert_eq!(result, Value::from(&expected_qm31));
    }

    #[test]
    fn run_qm31_mul() {
        let program = load_cairo! {
            use core::qm31::{QM31Trait, qm31, m31};

            fn run_test(lhs: qm31, rhs: qm31) -> qm31 {
                lhs * rhs
            }
        };

        let a = starknet_types_core::qm31::QM31::from_coefficients(
            0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2,
        );
        let b = starknet_types_core::qm31::QM31::from_coefficients(
            0x499602d2, 0x544b2fba, 0x673cff77, 0x60713d44,
        );
        let c = starknet_types_core::qm31::QM31::from_coefficients(
            0x1de1328d, 0x3b882f32, 0x47ae3cbc, 0x2a074017,
        );
        let d = starknet_types_core::qm31::QM31::zero();

        let result =
            run_program(&program, "run_test", &[Value::from(&a), Value::from(&b)]).return_value;
        let expected_qm31 = a.clone() * b.clone();
        assert_eq!(result, Value::from(&expected_qm31));

        let result =
            run_program(&program, "run_test", &[Value::from(&a), Value::from(&c)]).return_value;
        let expected_qm31 = a.clone() * c.clone();
        assert_eq!(result, Value::from(&expected_qm31));

        let result =
            run_program(&program, "run_test", &[Value::from(&b), Value::from(&c)]).return_value;
        let expected_qm31 = b.clone() * c;
        assert_eq!(result, Value::from(&expected_qm31));

        let result =
            run_program(&program, "run_test", &[Value::from(&d), Value::from(&b)]).return_value;
        let expected_qm31 = d * b;
        assert_eq!(result, Value::from(&expected_qm31));
    }

    #[test]
    fn run_qm31_div() {
        let program = load_cairo! {
            use core::qm31::{QM31Trait, qm31, m31};

            fn run_test(lhs: qm31, rhs: qm31) -> qm31 {
                lhs / rhs
            }
        };

        let a = starknet_types_core::qm31::QM31::from_coefficients(
            0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2,
        );
        let b = starknet_types_core::qm31::QM31::from_coefficients(
            0x499602d2, 0x544b2fba, 0x673cff77, 0x60713d44,
        );
        let c = starknet_types_core::qm31::QM31::from_coefficients(
            0x1de1328d, 0x3b882f32, 0x47ae3cbc, 0x2a074017,
        );
        let d = starknet_types_core::qm31::QM31::from_coefficients(0, 0, 0, 0);

        let result =
            run_program(&program, "run_test", &[Value::from(&c), Value::from(&a)]).return_value;
        let expected_qm31 = (c.clone() / a.clone()).unwrap();
        assert_eq!(
            result,
            jit_enum!(0, jit_struct!(Value::from(&expected_qm31)))
        );

        let result =
            run_program(&program, "run_test", &[Value::from(&a), Value::from(&b)]).return_value;
        let expected_qm31 = (a.clone() / b.clone()).unwrap();
        assert_eq!(
            result,
            jit_enum!(0, jit_struct!(Value::from(&expected_qm31)))
        );

        let result =
            run_program(&program, "run_test", &[Value::from(&b), Value::from(&c)]).return_value;
        let expected_qm31 = (b.clone() / c).unwrap();
        assert_eq!(
            result,
            jit_enum!(0, jit_struct!(Value::from(&expected_qm31)))
        );

        let result =
            run_program(&program, "run_test", &[Value::from(&b), Value::from(&d)]).return_value;
        if let Value::Enum { tag, .. } = result {
            assert_eq!(tag, 1);
        } else {
            panic!("Expected a Value::Enum()");
        }
    }

    #[test]
    fn run_from_m31() {
        let program = load_cairo! {
            use core::qm31::{QM31Trait, qm31, m31, qm31_from_m31};

            fn run_test_with_0() -> qm31 {
                qm31_from_m31(0)
            }

            fn run_test_with_1() -> qm31 {
                qm31_from_m31(1)
            }

            fn run_test_with_big_coefficient() -> qm31 {
                qm31_from_m31(0x60713d44)
            }
        };

        let result = run_program(&program, "run_test_with_0", &[]).return_value;
        assert_eq!(result, Value::QM31(0, 0, 0, 0));

        let result = run_program(&program, "run_test_with_1", &[]).return_value;
        assert_eq!(result, Value::QM31(1, 0, 0, 0));

        let result = run_program(&program, "run_test_with_big_coefficient", &[]).return_value;
        assert_eq!(result, Value::QM31(0x60713d44, 0, 0, 0));
    }

    #[test]
    fn run_m31_add() {
        // TODO: Refactor cairo functions to receive m31 as parameters so we don't need different ones
        // to test different cases and we can unify them into one. This can be done when issue #1217 gets closed.
        let program = load_cairo! {
            use core::qm31::m31_ops;
            use core::qm31::m31;

            fn run_test_1() -> m31 {
                m31_ops::add(1, 1)
            }

            fn run_test_2() -> m31 {
                m31_ops::add(0x567effa3, 0x5ffeb970)
            }

            fn run_test_3() -> m31 {
                m31_ops::add(0x7ffffffe, 1)
            }
        };
        let expected_range = Range {
            lower: 0.into(),
            upper: M31_PRIME.into(),
        };
        let result = run_program(&program, "run_test_1", &[]).return_value;
        assert_eq!(
            result,
            Value::BoundedInt {
                value: Felt252::from(2),
                range: expected_range.clone()
            }
        );

        let result = run_program(&program, "run_test_2", &[]).return_value;
        assert_eq!(
            result,
            Value::BoundedInt {
                value: Felt252::from(0x367db914),
                range: expected_range.clone()
            }
        );

        let result = run_program(&program, "run_test_3", &[]).return_value;
        assert_eq!(
            result,
            Value::BoundedInt {
                value: Felt252::from(0),
                range: expected_range.clone()
            }
        );
    }

    #[test]
    fn run_m31_sub() {
        // TODO: Refactor cairo functions to receive m31 as parameters so we don't need different ones
        // to test different cases and we can unify them into one. This can be done when issue #1217 gets closed.
        let program = load_cairo! {
            use core::qm31::m31_ops;
            use core::qm31::m31;

            fn run_test_1() -> m31 {
                m31_ops::sub(2, 1)
            }

            fn run_test_2() -> m31 {
                m31_ops::sub(0x567effa3, 0x567effa9)
            }

            fn run_test_3() -> m31 {
                m31_ops::sub(0, 1)
            }
        };
        let expected_range = Range {
            lower: 0.into(),
            upper: M31_PRIME.into(),
        };
        let result = run_program(&program, "run_test_1", &[]).return_value;
        assert_eq!(
            result,
            Value::BoundedInt {
                value: Felt252::from(1),
                range: expected_range.clone()
            }
        );

        let result = run_program(&program, "run_test_2", &[]).return_value;
        assert_eq!(
            result,
            Value::BoundedInt {
                value: Felt252::from(0x7ffffff9),
                range: expected_range.clone()
            }
        );

        let result = run_program(&program, "run_test_3", &[]).return_value;
        assert_eq!(
            result,
            Value::BoundedInt {
                value: Felt252::from(0x7ffffffe),
                range: expected_range.clone()
            }
        );
    }

    #[test]
    fn run_m31_mul() {
        // TODO: Refactor cairo functions to receive m31 as parameters so we don't need different ones
        // to test different cases and we can unify them into one. This can be done when issue #1217 gets closed.
        let program = load_cairo! {
            use core::qm31::m31_ops;
            use core::qm31::m31;

            fn run_test_1() -> m31 {
                m31_ops::mul(5, 5)
            }

            fn run_test_2() -> m31 {
                m31_ops::mul(0x567effa3, 0x567effa9)
            }

            fn run_test_3() -> m31 {
                m31_ops::mul(0x7ffffffe, 2)
            }
        };
        let expected_range = Range {
            lower: 0.into(),
            upper: M31_PRIME.into(),
        };
        let result = run_program(&program, "run_test_1", &[]).return_value;
        assert_eq!(
            result,
            Value::BoundedInt {
                value: Felt252::from(25),
                range: expected_range.clone()
            }
        );

        let result = run_program(&program, "run_test_2", &[]).return_value;
        assert_eq!(
            result,
            Value::BoundedInt {
                value: Felt252::from(0x69274523),
                range: expected_range.clone()
            }
        );

        let result = run_program(&program, "run_test_3", &[]).return_value;
        assert_eq!(
            result,
            Value::BoundedInt {
                value: Felt252::from(0x7ffffffd),
                range: expected_range.clone()
            }
        );
    }

    #[test]
    fn run_m31_div() {
        // TODO: Refactor cairo functions to receive m31 as parameters so we don't need different ones
        // to test different cases and we can unify them into one. This can be done when issue #1217 gets closed.
        let program = load_cairo! {
            use core::qm31::m31_ops;
            use core::qm31::m31;

            fn run_test_1() -> m31 {
                m31_ops::div(25, 5)
            }

            fn run_test_2() -> m31 {
                m31_ops::div(0x567effa3, 0x567effa9)
            }
        };
        let expected_range = Range {
            lower: 0.into(),
            upper: M31_PRIME.into(),
        };
        let result = run_program(&program, "run_test_1", &[]).return_value;
        assert_eq!(
            result,
            Value::BoundedInt {
                value: Felt252::from(5),
                range: expected_range.clone()
            }
        );

        let result = run_program(&program, "run_test_2", &[]).return_value;
        assert_eq!(
            result,
            Value::BoundedInt {
                value: Felt252::from(0x5138acb),
                range: expected_range.clone()
            }
        );
    }
}
