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
    dialect::{arith::CmpiPredicate, llvm},
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
/// m31 are always between 0 and 2**31 - 2
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
    let qm31_ty = llvm::r#type::r#struct(context, &[m31_ty, m31_ty, m31_ty, m31_ty], false);

    let m31_0 = entry.const_int_from_type(context, location, info.w0, m31_ty)?;
    let m31_1 = entry.const_int_from_type(context, location, info.w1, m31_ty)?;
    let m31_2 = entry.const_int_from_type(context, location, info.w2, m31_ty)?;
    let m31_3 = entry.const_int_from_type(context, location, info.w3, m31_ty)?;

    let qm31 = entry.append_op_result(llvm::undef(qm31_ty, location))?;
    let qm31 = entry.insert_value(context, location, qm31, m31_0, 0)?;
    let qm31 = entry.insert_value(context, location, qm31, m31_1, 1)?;
    let qm31 = entry.insert_value(context, location, qm31, m31_2, 2)?;
    let qm31 = entry.insert_value(context, location, qm31, m31_3, 3)?;

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
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let qm31 = entry.arg(0)?;
    let m31_ty = IntegerType::new(context, 31).into();
    let qm31_ty = llvm::r#type::r#struct(context, &[m31_ty, m31_ty, m31_ty, m31_ty], false);
    let qm31_ptr = entry.alloca1(context, location, qm31_ty, get_integer_layout(31).align())?;
    let cond_ptr = entry.alloca1(
        context,
        location,
        IntegerType::new(context, 1).into(),
        get_integer_layout(1).align(),
    )?;

    entry.store(context, location, qm31_ptr, qm31)?;

    metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?
        .libfunc_qm31_is_zero(context, helper, entry, qm31_ptr, cond_ptr, location)?;

    let cond = entry.load(
        context,
        location,
        cond_ptr,
        IntegerType::new(context, 1).into(),
    )?;

    let k1 = entry.const_int(context, location, 1, 1)?;
    let cond2 = entry.cmpi(context, CmpiPredicate::Eq, cond, k1, location)?;

    helper.cond_br(
        context,
        entry,
        cond2,
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
    let qm31_ty = llvm::r#type::r#struct(context, &[m31_ty, m31_ty, m31_ty, m31_ty], false);

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
/// m31 are always between 0 and 2**31 - 2
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
    let qm31_ty = llvm::r#type::r#struct(context, &[m31_ty, m31_ty, m31_ty, m31_ty], false);

    let qm31 = entry.append_op_result(llvm::undef(qm31_ty, location))?;
    let qm31 = entry.insert_value(context, location, qm31, m31_0, 0)?;
    let qm31 = entry.insert_value(context, location, qm31, m31_1, 1)?;
    let qm31 = entry.insert_value(context, location, qm31, m31_2, 2)?;
    let qm31 = entry.insert_value(context, location, qm31, m31_3, 3)?;

    helper.br(entry, 0, &[qm31], location)
}

/// Generate MLIR operations for the `qm31_unpack` libfunc.
///
/// Receives a qm31 and unpacks it, returning an array with the
/// four m31.
///
/// # Constraints
///
/// m31 are always between 0 and 2**31 - 2
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
/// m31 are always between 0 and 2**31 - 2
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
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let m31 = entry.arg(0)?;

    let m31_ty = IntegerType::new(context, 31).into();
    let qm31_ty = llvm::r#type::r#struct(context, &[m31_ty, m31_ty, m31_ty, m31_ty], false);

    let m31_ptr = entry.alloca1(context, location, m31_ty, get_integer_layout(31).align())?;
    let qm31_ptr = entry.alloca1(context, location, qm31_ty, get_integer_layout(31).align())?;

    entry.store(context, location, m31_ptr, m31)?;

    metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?
        .libfunc_qm31_from_m31(context, helper, entry, m31_ptr, qm31_ptr, location)?;

    let qm31 = entry.load(context, location, qm31_ptr, qm31_ty)?;

    helper.br(entry, 0, &[qm31], location)
}

#[cfg(test)]
mod test {
    use ark_ff::{One, Zero};
    use cairo_lang_sierra::extensions::utils::Range;
    use cairo_vm::Felt252;
    use num_bigint::BigInt;

    use crate::{jit_enum, jit_struct, load_cairo, utils::testing::run_program, Value};

    impl From<&starknet_types_core::qm31::QM31> for Value {
        fn from(qm31: &starknet_types_core::qm31::QM31) -> Self {
            let coefficients = qm31.to_coefficients();
            Value::QM31(
                coefficients.0,
                coefficients.1,
                coefficients.2,
                coefficients.3,
            )
        }
    }

    #[test]
    fn run_unpack() {
        let program = load_cairo! {
            use core::qm31::{QM31Trait, m31, qm31};

            fn run_test(input: qm31) -> [m31;4] {
                let unpacked_qm31 = input.unpack();

                unpacked_qm31
            }
        };

        let result = run_program(&program, "run_test", &[Value::QM31(1, 2, 3, 4)]).return_value;
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

        let result = run_program(
            &program,
            "run_test",
            &[Value::QM31(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2)],
        )
        .return_value;
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
        };

        let result_with_zero =
            run_program(&program, "run_test", &[Value::QM31(0, 0, 0, 0)]).return_value;
        assert_eq!(result_with_zero, jit_enum!(0, jit_struct!()));

        let result_without_zero =
            run_program(&program, "run_test", &[Value::QM31(0, 0, 1, 0)]).return_value;
        assert_eq!(result_without_zero, jit_enum!(1, Value::QM31(0, 0, 1, 0)));

        let result_without_zero =
            run_program(&program, "run_test", &[Value::QM31(0, 0, 1, 0)]).return_value;
        assert_eq!(result_without_zero, jit_enum!(1, Value::QM31(0, 0, 1, 0)));

        let result_big = run_program(
            &program,
            "run_test",
            &[Value::QM31(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2)],
        )
        .return_value;
        assert_eq!(
            result_big,
            jit_enum!(
                1,
                Value::QM31(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2)
            )
        )
    }

    #[test]
    fn run_add() {
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
    }

    #[test]
    fn run_sub() {
        let program = load_cairo! {
            use core::qm31::{QM31Trait, qm31, m31};

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
    fn run_mul() {
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
    fn run_div() {
        let program = load_cairo! {
            use core::qm31::{QM31Trait, qm31};

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
        let d = starknet_types_core::qm31::QM31::zero();

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
        let expected_qm31 = (b / c).unwrap();
        assert_eq!(
            result,
            jit_enum!(0, jit_struct!(Value::from(&expected_qm31)))
        );

        let result =
            run_program(&program, "run_test", &[Value::from(&a), Value::from(&d)]).return_value;
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
