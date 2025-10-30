use crate::{
    error::{Error, Result},
    metadata::runtime_bindings::RuntimeBindingsMeta,
    utils::get_integer_layout,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        qm31::{QM31BinaryOpConcreteLibfunc, QM31Concrete, QM31ConstConcreteLibfunc},
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        llvm,
    },
    helpers::{ArithBlockExt, BuiltinBlockExt, LlvmBlockExt},
    ir::{r#type::IntegerType, Block, Location},
    Context,
};

use crate::{libfuncs::LibfuncHelper, metadata::MetadataStorage};

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

/// Generate MLIR operations for the QM31 binary operations libfuncs.
///
/// Receives two qm31 and performs an operation (add, sub, mul or div) with both.
/// Returns the result of the operation.
///
/// # Cairo Signature
/// ```cairo
/// fn qm31_add(a: qm31, b: qm31) -> qm31 nopanic;
/// fn qm31_sub(a: qm31, b: qm31) -> qm31 nopanic;
/// fn qm31_mul(a: qm31, b: qm31) -> qm31 nopanic;
/// fn qm31_mul(a: qm31, b: qm31) -> qm31 nopanic;
/// ```
pub fn build_binary_op<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &QM31BinaryOpConcreteLibfunc,
) -> Result<()> {
    let m31_ty = IntegerType::new(context, 31).into();
    let qm31_ty = llvm::r#type::array(m31_ty, 4);

    let lhs = entry.arg(0)?;
    let rhs = entry.arg(1)?;

    let lhs_ptr = entry.alloca1(context, location, qm31_ty, get_integer_layout(31).align())?;
    let rhs_ptr = entry.alloca1(context, location, qm31_ty, get_integer_layout(31).align())?;
    let res_ptr = entry.alloca1(context, location, qm31_ty, get_integer_layout(31).align())?;

    entry.store(context, location, lhs_ptr, lhs)?;
    entry.store(context, location, rhs_ptr, rhs)?;

    metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?
        .libfunc_qm31_bin_op(
            context,
            helper,
            entry,
            lhs_ptr,
            rhs_ptr,
            res_ptr,
            info.operator,
            location,
        )?;

    let result = entry.load(context, location, res_ptr, qm31_ty)?;

    helper.br(entry, 0, &[result], location)
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

#[cfg(test)]
mod test {
    use ark_ff::{One, Zero};
    use cairo_lang_sierra::extensions::utils::Range;
    use cairo_vm::Felt252;
    use num_bigint::BigInt;

    use crate::{
        jit_enum, jit_struct, load_cairo, runtime::to_representative_coefficients,
        utils::testing::run_program, Value,
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

            fn run_test_1() -> OptionRev<NonZero<qm31>> {
                let qm31 = QM31Trait::new(0, 0, 0, 0);
                qm31_is_zero(qm31)
            }

            fn run_test_2() -> OptionRev<NonZero<qm31>> {
                let qm31 = QM31Trait::new(0, 0, 1, 0);
                qm31_is_zero(qm31)
            }

            fn run_test_3() -> OptionRev<NonZero<qm31>> {
                let qm31 = QM31Trait::new(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2);
                qm31_is_zero(qm31)
            }

            fn run_test_4() -> OptionRev<NonZero<qm31>> {
                let lhs = QM31Trait::new(0x7ffffffe, 0x7ffffffe, 0x7ffffffe, 0x7ffffffe);
                let rhs = QM31Trait::new(1, 1, 1, 1);
                let qm31 = lhs + rhs;
                qm31_is_zero(qm31)
            }
        };

        let result = run_program(&program, "run_test_1", &[]).return_value;
        assert_eq!(result, jit_enum!(0, jit_struct!()));

        let result = run_program(&program, "run_test_2", &[]).return_value;
        assert_eq!(result, jit_enum!(1, Value::QM31(0, 0, 1, 0)));

        let result = run_program(&program, "run_test_3", &[]).return_value;
        assert_eq!(
            result,
            jit_enum!(
                1,
                Value::QM31(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2)
            )
        );

        let result = run_program(&program, "run_test_4", &[]).return_value;
        assert_eq!(result, jit_enum!(0, jit_struct!()));
    }

    #[test]
    fn run_add() {
        let program = load_cairo! {
            use core::qm31::{QM31Trait, qm31};

            fn run_a_plus_b() -> qm31 {
                let a = QM31Trait::new(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2);
                let b = QM31Trait::new(0x499602d2, 0x544b2fba, 0x673cff77, 0x60713d44);
                a + b
            }

            fn run_b_plus_c() -> qm31 {
                let b = QM31Trait::new(0x499602d2, 0x544b2fba, 0x673cff77, 0x60713d44);
                let c = QM31Trait::new(0x1de1328d, 0x3b882f32, 0x47ae3cbc, 0x2a074017);
                b + c
            }

            fn run_a_plus_c() -> qm31 {
                let a = QM31Trait::new(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2);
                let c = QM31Trait::new(0x1de1328d, 0x3b882f32, 0x47ae3cbc, 0x2a074017);
                a + c
            }

            fn run_d_plus_e() -> qm31 {
                let d = QM31Trait::new(0x7ffffffe, 0x7ffffffe, 0x7ffffffe, 0x7ffffffe);
                let e = QM31Trait::new(1, 1, 1, 1);

                d + e
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

        let result = run_program(&program, "run_a_plus_b", &[]).return_value;
        let expected_qm31 = a.clone() + b.clone();
        assert_eq!(result, Value::from(&expected_qm31));

        let result = run_program(&program, "run_b_plus_c", &[]).return_value;
        let expected_qm31 = b + c.clone();
        assert_eq!(result, Value::from(&expected_qm31));

        let result = run_program(&program, "run_a_plus_c", &[]).return_value;
        let expected_qm31 = a + c;
        assert_eq!(result, Value::from(&expected_qm31));

        let result = run_program(&program, "run_d_plus_e", &[]).return_value;
        assert_eq!(result, Value::QM31(0, 0, 0, 0));
    }

    #[test]
    fn run_sub() {
        let program = load_cairo! {
            use core::qm31::{QM31Trait, qm31, m31};

            fn run_c_minus_a() -> qm31 {
                let a = QM31Trait::new(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2);
                let c = QM31Trait::new(0x1de1328d, 0x3b882f32, 0x47ae3cbc, 0x2a074017);
                c - a
            }

            fn run_a_minus_b() -> qm31 {
                let a = QM31Trait::new(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2);
                let b = QM31Trait::new(0x499602d2, 0x544b2fba, 0x673cff77, 0x60713d44);
                a - b
            }

            fn run_b_minus_c() -> qm31 {
                let b = QM31Trait::new(0x499602d2, 0x544b2fba, 0x673cff77, 0x60713d44);
                let c = QM31Trait::new(0x1de1328d, 0x3b882f32, 0x47ae3cbc, 0x2a074017);
                b - c
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

        let result = run_program(&program, "run_c_minus_a", &[]).return_value;
        let expected_qm31 = c.clone() - a.clone();
        assert_eq!(result, Value::from(&expected_qm31));

        let result = run_program(&program, "run_a_minus_b", &[]).return_value;
        let expected_qm31 = a - b.clone();
        assert_eq!(result, Value::from(&expected_qm31));

        let result = run_program(&program, "run_b_minus_c", &[]).return_value;
        let expected_qm31 = b - c;
        assert_eq!(result, Value::from(&expected_qm31));
    }

    #[test]
    fn run_mul() {
        let program = load_cairo! {
            use core::qm31::{QM31Trait, qm31, m31};

            fn run_a_times_b() -> qm31 {
                let a = QM31Trait::new(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2);
                let b = QM31Trait::new(0x499602d2, 0x544b2fba, 0x673cff77, 0x60713d44);
                a * b
            }

            fn run_a_times_c() -> qm31 {
                let a = QM31Trait::new(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2);
                let c = QM31Trait::new(0x1de1328d, 0x3b882f32, 0x47ae3cbc, 0x2a074017);
                a * c
            }

            fn run_b_times_c() -> qm31 {
                let b = QM31Trait::new(0x499602d2, 0x544b2fba, 0x673cff77, 0x60713d44);
                let c = QM31Trait::new(0x1de1328d, 0x3b882f32, 0x47ae3cbc, 0x2a074017);
                b * c
            }

            fn run_b_times_d() -> qm31 {
                let b = QM31Trait::new(0x499602d2, 0x544b2fba, 0x673cff77, 0x60713d44);
                let d = QM31Trait::new(0, 0, 0, 0);
                b * d
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

        let result = run_program(&program, "run_a_times_b", &[]).return_value;
        let expected_qm31 = a.clone() * b.clone();
        assert_eq!(result, Value::from(&expected_qm31));

        let result = run_program(&program, "run_a_times_c", &[]).return_value;
        let expected_qm31 = a.clone() * c.clone();
        assert_eq!(result, Value::from(&expected_qm31));

        let result = run_program(&program, "run_b_times_c", &[]).return_value;
        let expected_qm31 = b.clone() * c;
        assert_eq!(result, Value::from(&expected_qm31));

        let result = run_program(&program, "run_b_times_d", &[]).return_value;
        let expected_qm31 = d * b;
        assert_eq!(result, Value::from(&expected_qm31));
    }

    #[test]
    fn run_div() {
        let program = load_cairo! {
            use core::qm31::{QM31Trait, qm31};

            fn run_a_divided_by_b() -> qm31 {
                let a = QM31Trait::new(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2);
                let b = QM31Trait::new(0x499602d2, 0x544b2fba, 0x673cff77, 0x60713d44);
                a / b
            }

            fn run_c_divided_by_a() -> qm31 {
                let a = QM31Trait::new(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2);
                let c = QM31Trait::new(0x1de1328d, 0x3b882f32, 0x47ae3cbc, 0x2a074017);
                c / a
            }

            fn run_b_divided_by_c() -> qm31 {
                let b = QM31Trait::new(0x499602d2, 0x544b2fba, 0x673cff77, 0x60713d44);
                let c = QM31Trait::new(0x1de1328d, 0x3b882f32, 0x47ae3cbc, 0x2a074017);
                b / c
            }

            fn run_b_divided_by_d() -> qm31 {
                let b = QM31Trait::new(0x499602d2, 0x544b2fba, 0x673cff77, 0x60713d44);
                let d = QM31Trait::new(0, 0, 0, 0);
                b / d
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

        let result = run_program(&program, "run_c_divided_by_a", &[]).return_value;
        let expected_qm31 = (c.clone() / a.clone()).unwrap();
        assert_eq!(
            result,
            jit_enum!(0, jit_struct!(Value::from(&expected_qm31)))
        );

        let result = run_program(&program, "run_a_divided_by_b", &[]).return_value;
        let expected_qm31 = (a.clone() / b.clone()).unwrap();
        assert_eq!(
            result,
            jit_enum!(0, jit_struct!(Value::from(&expected_qm31)))
        );

        let result = run_program(&program, "run_b_divided_by_c", &[]).return_value;
        let expected_qm31 = (b / c).unwrap();
        assert_eq!(
            result,
            jit_enum!(0, jit_struct!(Value::from(&expected_qm31)))
        );

        let result = run_program(&program, "run_b_divided_by_d", &[]).return_value;
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

        let m31_range = Range::closed(0, 2147483646);
        let result = run_program(
            &program,
            "run_test_with_0",
            &[Value::BoundedInt {
                value: Felt252::zero(),
                range: m31_range.clone(),
            }],
        )
        .return_value;
        assert_eq!(result, Value::QM31(0, 0, 0, 0));

        let result = run_program(
            &program,
            "run_test_with_1",
            &[Value::BoundedInt {
                value: Felt252::one(),
                range: m31_range.clone(),
            }],
        )
        .return_value;
        assert_eq!(result, Value::QM31(1, 0, 0, 0));

        let result = run_program(
            &program,
            "run_test_with_big_coefficient",
            &[Value::BoundedInt {
                value: Felt252::one(),
                range: m31_range,
            }],
        )
        .return_value;
        assert_eq!(result, Value::QM31(0x60713d44, 0, 0, 0));
    }
}
