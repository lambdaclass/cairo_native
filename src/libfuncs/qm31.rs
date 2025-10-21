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
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let m31_ty = IntegerType::new(context, 31);
    let qm31 = entry.arg(0)?;

    let m31_0 = entry.extract_value(context, location, qm31, m31_ty.into(), 0)?;
    let m31_1 = entry.extract_value(context, location, qm31, m31_ty.into(), 1)?;
    let m31_2 = entry.extract_value(context, location, qm31, m31_ty.into(), 2)?;
    let m31_3 = entry.extract_value(context, location, qm31, m31_ty.into(), 3)?;

    let k0 = entry.const_int(context, location, 0, 31)?;

    let condition_0 = entry.cmpi(context, CmpiPredicate::Eq, m31_0, k0, location)?;
    let condition_1 = entry.cmpi(context, CmpiPredicate::Eq, m31_1, k0, location)?;
    let condition_2 = entry.cmpi(context, CmpiPredicate::Eq, m31_2, k0, location)?;
    let condition_3 = entry.cmpi(context, CmpiPredicate::Eq, m31_3, k0, location)?;

    let condition_01 = entry.cmpi(
        context,
        CmpiPredicate::Eq,
        condition_0,
        condition_1,
        location,
    )?;
    let condition_23 = entry.cmpi(
        context,
        CmpiPredicate::Eq,
        condition_2,
        condition_3,
        location,
    )?;

    let condition_0123 = entry.cmpi(
        context,
        CmpiPredicate::Eq,
        condition_01,
        condition_23,
        location,
    )?;

    helper.cond_br(
        context,
        entry,
        condition_0123,
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
    match info.operator {
        cairo_lang_sierra::extensions::qm31::QM31BinaryOperator::Add => {
            let lhs = entry.arg(0)?;
            let rhs = entry.arg(1)?;

            let lhs_ptr = helper.init_block.alloca1(
                context,
                location,
                qm31_ty,
                get_integer_layout(31).align(),
            )?;
            let rhs_ptr = helper.init_block.alloca1(
                context,
                location,
                qm31_ty,
                get_integer_layout(31).align(),
            )?;
            let res_ptr = helper.init_block.alloca1(
                // TODO: This may not be necessary
                context,
                location,
                qm31_ty,
                get_integer_layout(31).align(),
            )?;

            entry.store(context, location, lhs_ptr, lhs)?;
            entry.store(context, location, rhs_ptr, rhs)?;

            runtime_bindings_meta
                .libfunc_qm31_add(context, helper, entry, lhs_ptr, rhs_ptr, res_ptr, location)?;

            let result = entry.load(context, location, res_ptr, qm31_ty)?;

            helper.br(entry, 0, &[result], location)
        }
        cairo_lang_sierra::extensions::qm31::QM31BinaryOperator::Sub => {
            // TODO: Almost the same implementation as add. See how to unify both
            let lhs = entry.arg(0)?;
            let rhs = entry.arg(1)?;

            let lhs_ptr = helper.init_block.alloca1(
                context,
                location,
                qm31_ty,
                get_integer_layout(31).align(),
            )?;
            let rhs_ptr = helper.init_block.alloca1(
                context,
                location,
                qm31_ty,
                get_integer_layout(31).align(),
            )?;
            let res_ptr = helper.init_block.alloca1(
                // TODO: This may not be necessary
                context,
                location,
                qm31_ty,
                get_integer_layout(31).align(),
            )?;

            entry.store(context, location, lhs_ptr, lhs)?;
            entry.store(context, location, rhs_ptr, rhs)?;

            runtime_bindings_meta
                .libfunc_qm31_sub(context, helper, entry, lhs_ptr, rhs_ptr, res_ptr, location)?;

            let result = entry.load(context, location, res_ptr, qm31_ty)?;

            helper.br(entry, 0, &[result], location)
        }
        cairo_lang_sierra::extensions::qm31::QM31BinaryOperator::Mul => todo!(),
        cairo_lang_sierra::extensions::qm31::QM31BinaryOperator::Div => todo!(),
    }
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

    println!("########################");

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
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _entry: &'this Block<'ctx>,
    _location: Location<'ctx>,
    _helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    todo!()
}

#[cfg(test)]
mod test {
    use cairo_lang_sierra::extensions::utils::Range;
    use cairo_vm::Felt252;
    use num_bigint::BigInt;

    use crate::{jit_enum, jit_struct, load_cairo, utils::testing::run_program, Value};

    #[test]
    fn run_unpack() {
        let program = load_cairo! {
            use core::qm31::{QM31Trait, m31, qm31};

            fn run_test() -> [m31;4] {
                let qm31 = QM31Trait::new(1, 2, 3, 4);

                let unpacked_qm31 = qm31.unpack();

                unpacked_qm31
            }
        };
        let result = run_program(&program, "run_test", &[]).return_value;

        let m31_range = Range::closed(0, BigInt::from(2147483646));
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
                    range: m31_range
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
        };

        let result = run_program(&program, "run_test", &[]).return_value;
        assert_eq!(result, Value::QM31(1, 2, 3, 4));
    }

    #[test]
    fn run_const() {
        let program = load_cairo! {
            use core::qm31::{qm31_const, qm31};

            fn run_test() -> qm31 {
                let qm31 = qm31_const::<1, 2, 3, 4>();

                qm31
            }
        };

        let result = run_program(&program, "run_test", &[]).return_value;
        assert_eq!(result, Value::QM31(1, 2, 3, 4));
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
        assert_eq!(result_without_zero, jit_enum!(1, Value::QM31(0, 0, 1, 0)))
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

        let result_c_minus_a = run_program(&program, "run_test_c_minus_a", &[]).return_value;

        let a = starknet_types_core::qm31::QM31::from_coefficients(
            0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2,
        );
        let b = starknet_types_core::qm31::QM31::from_coefficients(
            0x499602d2, 0x544b2fba, 0x673cff77, 0x60713d44,
        );
        let c = starknet_types_core::qm31::QM31::from_coefficients(
            0x1de1328d, 0x3b882f32, 0x47ae3cbc, 0x2a074017,
        );
        let c_minus_a_coefficients = (c.clone() - a).to_coefficients();

        // println!("{:?}", c - a == b);
        // println!("b: {:?}", b.to_coefficients());
        // println!("result: {:?}", result_c_minus_a);
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
}
