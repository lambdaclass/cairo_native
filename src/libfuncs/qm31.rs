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
    dialect::llvm,
    helpers::{BuiltinBlockExt, LlvmBlockExt},
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
    _info: &QM31ConstConcreteLibfunc,
) -> Result<()> {
    // TODO: This is the same implementation as pack. Should it be diferent since
    // we have CONST arguments here?
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

pub fn build_is_zero<'ctx, 'this>(
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

    use crate::{
        utils::test::{load_cairo, run_program},
        Value,
    };

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
                    range: m31_range.clone()
                },
            ]
        );
    }

    #[test]
    fn run_pack() {
        let program = load_cairo! {
            use core::qm31::{QM31Trait, qm31};

            fn run_test() -> [m31;4] {
                let qm31 = QM31Trait::new(1, 2, 3, 4);

                qm31
            }
        };

        let _result = run_program(&program, "run_test", &[]).return_value;

        assert!(false);
    }
}
