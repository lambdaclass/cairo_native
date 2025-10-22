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
    let qm31_ty = llvm::r#type::r#struct(
        context,
        &[m31_ty, m31_ty, m31_ty, m31_ty],
        false, // TODO: Confirm this
    );

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
    let qm31_ty = llvm::r#type::r#struct(
        context,
        &[m31_ty, m31_ty, m31_ty, m31_ty],
        false, // TODO: Confirm this
    );
    let qm31_ptr =
        helper
            .init_block
            .alloca1(context, location, qm31_ty, get_integer_layout(31).align())?;
    let cond_ptr = helper.init_block.alloca1(
        // TODO: Check if we need this ptr really
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
    let qm31_ty = llvm::r#type::r#struct(
        context,
        &[m31_ty, m31_ty, m31_ty, m31_ty],
        false, // TODO: Confirm this
    );
    let runtime_bindings_meta = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?;

    let lhs = entry.arg(0)?;
    let rhs = entry.arg(1)?;

    let lhs_ptr =
        helper
            .init_block
            .alloca1(context, location, qm31_ty, get_integer_layout(31).align())?;
    let rhs_ptr =
        helper
            .init_block
            .alloca1(context, location, qm31_ty, get_integer_layout(31).align())?;
    let res_ptr = helper.init_block.alloca1(
        // TODO: This may not be necessary
        context,
        location,
        qm31_ty,
        get_integer_layout(31).align(),
    )?;

    entry.store(context, location, lhs_ptr, lhs)?;
    entry.store(context, location, rhs_ptr, rhs)?;

    match info.operator {
        cairo_lang_sierra::extensions::qm31::QM31BinaryOperator::Add => runtime_bindings_meta
            .libfunc_qm31_add(context, helper, entry, lhs_ptr, rhs_ptr, res_ptr, location)?,
        cairo_lang_sierra::extensions::qm31::QM31BinaryOperator::Sub => runtime_bindings_meta
            .libfunc_qm31_sub(context, helper, entry, lhs_ptr, rhs_ptr, res_ptr, location)?,
        cairo_lang_sierra::extensions::qm31::QM31BinaryOperator::Mul => runtime_bindings_meta
            .libfunc_qm31_mul(context, helper, entry, lhs_ptr, rhs_ptr, res_ptr, location)?,
        cairo_lang_sierra::extensions::qm31::QM31BinaryOperator::Div => runtime_bindings_meta
            .libfunc_qm31_div(context, helper, entry, lhs_ptr, rhs_ptr, res_ptr, location)?,
    };

    let result = entry.load(context, location, res_ptr, qm31_ty)?;

    helper.br(entry, 0, &[result], location)
}

pub fn build_pack<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let m31_ty = IntegerType::new(context, 31).into();
    let qm31_ty = llvm::r#type::r#struct(
        context,
        &[m31_ty, m31_ty, m31_ty, m31_ty],
        false, // TODO: Confirm this
    );

    let m31_0 = entry.arg(0)?;
    let m31_1 = entry.arg(1)?;
    let m31_2 = entry.arg(2)?;
    let m31_3 = entry.arg(3)?;

    let m31_0_ptr =
        helper
            .init_block
            .alloca1(context, location, m31_ty, get_integer_layout(31).align())?;
    let m31_1_ptr =
        helper
            .init_block
            .alloca1(context, location, m31_ty, get_integer_layout(31).align())?;
    let m31_2_ptr =
        helper
            .init_block
            .alloca1(context, location, m31_ty, get_integer_layout(31).align())?;
    let m31_3_ptr =
        helper
            .init_block
            .alloca1(context, location, m31_ty, get_integer_layout(31).align())?;
    let qm31_ptr =
        helper
            .init_block
            .alloca1(context, location, qm31_ty, get_integer_layout(31).align())?;

    entry.store(context, location, m31_0_ptr, m31_0)?;
    entry.store(context, location, m31_1_ptr, m31_1)?;
    entry.store(context, location, m31_2_ptr, m31_2)?;
    entry.store(context, location, m31_3_ptr, m31_3)?;

    metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?
        .libfunc_qm31_pack(
            context, helper, entry, m31_0_ptr, m31_1_ptr, m31_2_ptr, m31_3_ptr, qm31_ptr, location,
        )?;

    let qm31 = entry.load(context, location, qm31_ptr, qm31_ty)?;

    helper.br(entry, 0, &[qm31], location)
}

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
    let qm31_ty = llvm::r#type::r#struct(
        context,
        &[m31_ty, m31_ty, m31_ty, m31_ty],
        false, // TODO: Confirm this
    );

    let m31_ptr =
        helper
            .init_block
            .alloca1(context, location, m31_ty, get_integer_layout(31).align())?;
    let qm31_ptr =
        helper
            .init_block
            .alloca1(context, location, qm31_ty, get_integer_layout(31).align())?;

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
        let Value::Struct {
            // TODO: Find a way to make this asserts nicer
            fields,
            debug_name: _,
        } = result
        else {
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
        let Value::Struct {
            fields,
            debug_name: _,
        } = result
        else {
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
        let program = load_cairo! { // TODO: Check if we can pass the m31 as arguments so we reduce repeated code
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

            fn run_test() -> qm31 {
                let lhs = QM31Trait::new(1,2,3,4);
                let rhs = QM31Trait::new(1,2,3,4);

                lhs + rhs
            }
        };

        let result = run_program(&program, "run_test", &[]).return_value;
        assert_eq!(result, Value::QM31(2, 4, 6, 8));
    }

    #[test]
    fn run_sub() {
        let program = load_cairo! {
            use core::qm31::{QM31Trait, qm31, m31};

            fn run_test_c_minus_a() -> qm31 {
                let a = QM31Trait::new(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2);
                let c = QM31Trait::new(0x1de1328d, 0x3b882f32, 0x47ae3cbc, 0x2a074017);

                c - a
            }

            fn run_test_c_minus_b() -> qm31 {
                let c = QM31Trait::new(0x1de1328d, 0x3b882f32, 0x47ae3cbc, 0x2a074017);
                let b = QM31Trait::new(0x499602d2, 0x544b2fba, 0x673cff77, 0x60713d44);

                c - b
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

        let result_c_minus_a = run_program(&program, "run_test_c_minus_a", &[]).return_value;
        let c_minus_a_coefficients = (c.clone() - a.clone()).to_coefficients();
        assert_eq!(
            result_c_minus_a,
            Value::QM31(
                c_minus_a_coefficients.0,
                c_minus_a_coefficients.1,
                c_minus_a_coefficients.2,
                c_minus_a_coefficients.3
            )
        );

        let result_c_minus_b = run_program(&program, "run_test_c_minus_b", &[]).return_value;
        let c_minus_b_coefficients = (c - b).to_coefficients();
        assert_eq!(
            result_c_minus_b,
            Value::QM31(
                c_minus_b_coefficients.0,
                c_minus_b_coefficients.1,
                c_minus_b_coefficients.2,
                c_minus_b_coefficients.3
            )
        );
    }

    #[test]
    fn run_mul() {
        let program = load_cairo! {
            use core::qm31::{QM31Trait, qm31, m31};

            fn run_test_a_times_b() -> qm31 {
                let a = QM31Trait::new(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2);
                let b = QM31Trait::new(0x499602d2, 0x544b2fba, 0x673cff77, 0x60713d44);
                a * b
            }

            fn run_test_c_times_a() -> qm31 {
                let c = QM31Trait::new(0x1de1328d, 0x3b882f32, 0x47ae3cbc, 0x2a074017);
                let a = QM31Trait::new(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2);

                c * a
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

        let result_a_times_b = run_program(&program, "run_test_a_times_b", &[]).return_value;
        let a_times_b_coefficients = (a.clone() * b).to_coefficients();
        assert_eq!(
            result_a_times_b,
            Value::QM31(
                a_times_b_coefficients.0,
                a_times_b_coefficients.1,
                a_times_b_coefficients.2,
                a_times_b_coefficients.3
            )
        );

        let result_c_times_a = run_program(&program, "run_test_c_times_a", &[]).return_value;
        let c_times_a_coefficients = (c * a).to_coefficients();
        assert_eq!(
            result_c_times_a,
            Value::QM31(
                c_times_a_coefficients.0,
                c_times_a_coefficients.1,
                c_times_a_coefficients.2,
                c_times_a_coefficients.3
            )
        );
    }

    #[test]
    fn run_div() {
        let program = load_cairo! { // TODO: Check if we can pass arguments so we reduce repeated code
            use core::qm31::{QM31Trait, qm31};

            fn run_test_c_divided_by_a() -> qm31 {
                let a = QM31Trait::new(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2);
                let c = QM31Trait::new(0x1de1328d, 0x3b882f32, 0x47ae3cbc, 0x2a074017);
                c / a
            }

            fn run_test_a_divided_by_b() -> qm31 {
                let a = QM31Trait::new(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2);
                let b = QM31Trait::new(0x499602d2, 0x544b2fba, 0x673cff77, 0x60713d44);
                a / b
            }
        };
        // TODO: Check if these can be const so we dont repeat them on each test
        let a = starknet_types_core::qm31::QM31::from_coefficients(
            0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2,
        );
        let b = starknet_types_core::qm31::QM31::from_coefficients(
            0x499602d2, 0x544b2fba, 0x673cff77, 0x60713d44,
        );
        let c = starknet_types_core::qm31::QM31::from_coefficients(
            0x1de1328d, 0x3b882f32, 0x47ae3cbc, 0x2a074017,
        );

        let result_c_div_by_a = run_program(&program, "run_test_c_divided_by_a", &[]).return_value;
        let c_div_by_a_coefficients = (c / a.clone()).unwrap().to_coefficients();
        assert_eq!(
            result_c_div_by_a,
            jit_enum!(
                0,
                jit_struct!(Value::QM31(
                    c_div_by_a_coefficients.0,
                    c_div_by_a_coefficients.1,
                    c_div_by_a_coefficients.2,
                    c_div_by_a_coefficients.3
                ))
            )
        );

        let result_a_div_by_b = run_program(&program, "run_test_a_divided_by_b", &[]).return_value;
        let a_div_by_b_coefficients = (a / b).unwrap().to_coefficients();
        assert_eq!(
            result_a_div_by_b,
            jit_enum!(
                0,
                jit_struct!(Value::QM31(
                    a_div_by_b_coefficients.0,
                    a_div_by_b_coefficients.1,
                    a_div_by_b_coefficients.2,
                    a_div_by_b_coefficients.3
                ))
            )
        );
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

            fn run_test_with_big_number() -> qm31 {
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
            "run_test_with_big_number",
            &[Value::BoundedInt {
                value: Felt252::one(),
                range: m31_range,
            }],
        )
        .return_value;
        assert_eq!(result, Value::QM31(0x60713d44, 0, 0, 0));
    }
}
