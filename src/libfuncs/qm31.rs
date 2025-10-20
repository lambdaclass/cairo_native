use crate::error::Result;
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
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _entry: &'this Block<'ctx>,
    _location: Location<'ctx>,
    _helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &QM31BinaryOpConcreteLibfunc,
) -> Result<()> {
    todo!()
}

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

    // TODO: Check if there is a nicer way to get the type. I think
    // something can be done with the branch signatures or something like that
    let m31_ty = IntegerType::new(context, 31).into();
    let qm31_ty = llvm::r#type::r#struct(
        context,
        &[m31_ty, m31_ty, m31_ty, m31_ty],
        false, // TODO: Confirm this
    );

    let qm31 = entry.append_op_result(llvm::undef(qm31_ty, location))?;
    let qm31 = entry.insert_value(context, location, qm31, m31_0, 0)?;
    let qm31 = entry.insert_value(context, location, qm31, m31_1, 1)?;
    let qm31 = entry.insert_value(context, location, qm31, m31_2, 2)?;
    let qm31 = entry.insert_value(context, location, qm31, m31_3, 3)?;

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
        super::increment_builtin_counter_by(context, entry, location, entry.arg(0)?.into(), 5)?;

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
        let program = load_cairo! { // TODO: To remove spme code, the entrypoint could receive arguments from outside
            use core::qm31::{QM31Trait, qm31, qm31_is_zero};
            use core::internal::OptionRev;

            fn run_test_with_zero() -> OptionRev<NonZero<qm31>> {
                let qm31 = QM31Trait::new(0, 0, 0, 0);
                qm31_is_zero(qm31)
            }

            fn run_test_without_zero() -> OptionRev<NonZero<qm31>> {
                let qm31 = QM31Trait::new(0, 0, 1, 0);
                qm31_is_zero(qm31)
            }
        };

        let result_with_zero = run_program(&program, "run_test_with_zero", &[]).return_value;
        assert_eq!(result_with_zero, jit_enum!(0, jit_struct!()));

        let result_without_zero = run_program(&program, "run_test_without_zero", &[]).return_value;
        assert_eq!(result_without_zero, jit_enum!(1, Value::QM31(0, 0, 1, 0)))
    }
}
