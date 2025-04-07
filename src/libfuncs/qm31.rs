use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        qm31::{QM31Concrete, QM31ConstConcreteLibfunc},
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::arith::CmpiPredicate,
    ir::{r#type::IntegerType, Block, BlockLike, Location},
    Context,
};
use num_bigint::BigInt;

use crate::{error::Result, metadata::MetadataStorage, utils::BlockExt};

use super::LibfuncHelper;

// An M31 is the quarter part of a QM31, is a 36-bit integer
// whose 5 least significant bits are 0s.
const M31_SIZE: u8 = 36;

/// Select and call the correct libfunc builder function from the selector.
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
        QM31Concrete::Pack(info) => {
            build_qm31_pack(context, registry, entry, location, helper, metadata, info)
        }
        QM31Concrete::Unpack(_) => {
            todo!("impl qm31_unpack");
        }
        QM31Concrete::Const(info) => {
            build_qm31_const(context, registry, entry, location, helper, metadata, info)
        }
        QM31Concrete::FromM31(_) => {
            todo!("impl qm31_from_m31");
        }
        QM31Concrete::IsZero(info) => {
            build_qm31_is_zero(context, registry, entry, location, helper, metadata, info)
        }
        QM31Concrete::BinaryOperation(_) => {
            todo!("impl qm31_bin_op");
        }
    }
}

pub fn build_qm31_pack<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let i144_ty = IntegerType::new(context, 144).into();
    let m31_size = entry.const_int(context, location, M31_SIZE, 144)?;

    let m31_0 = entry.arg(0)?;
    let m31_1 = entry.arg(1)?;
    let m31_2 = entry.arg(2)?;
    let m31_3 = entry.arg(3)?;

    // Extend every limb to 144 bits
    let m31_0 = entry.extui(m31_0, i144_ty, location)?;
    let m31_1 = entry.extui(m31_1, i144_ty, location)?;
    let m31_2 = entry.extui(m31_2, i144_ty, location)?;

    // We first crate the qm31 with its most significant bits
    let qm31 = entry.extui(m31_3, i144_ty, location)?;

    let qm31 = entry.shli(qm31, m31_size, location)?;
    let qm31 = entry.addi(qm31, m31_2, location)?;

    let qm31 = entry.shli(qm31, m31_size, location)?;
    let qm31 = entry.addi(qm31, m31_1, location)?;

    let qm31 = entry.shli(qm31, m31_size, location)?;
    let qm31 = entry.addi(qm31, m31_0, location)?;

    entry.append_operation(helper.br(0, &[qm31], location));

    Ok(())
}

pub fn build_qm31_const<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &QM31ConstConcreteLibfunc,
) -> Result<()> {
    let mut qm31 = BigInt::from(info.w3);
    qm31 <<= M31_SIZE;
    qm31 += BigInt::from(info.w2);
    qm31 <<= M31_SIZE;
    qm31 += BigInt::from(info.w1);
    qm31 <<= M31_SIZE;
    qm31 += BigInt::from(info.w0);

    let qm31 = entry.const_int(context, location, qm31, 144)?;

    entry.append_operation(helper.br(0, &[qm31], location));

    Ok(())
}

pub fn build_qm31_is_zero<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let value = entry.arg(0)?;
    let k0 = entry.const_int(context, location, 0, 144)?;

    let is_zero = entry.cmpi(context, CmpiPredicate::Eq, value, k0, location)?;

    entry.append_operation(helper.cond_br(context, is_zero, [0, 1], [&[], &[value]], location));

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{
        utils::test::{load_cairo, run_program},
        Value,
    };
    use num_bigint::BigUint;
    use num_traits::Num;

    #[test]
    fn test_qm31_pack() {
        let program = load_cairo! {
            use core::qm31::{QM31Trait, qm31};

            fn run_test() -> qm31 {
                QM31Trait::new(1, 2, 3, 4)
            }
        };

        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            Value::QM31(BigUint::from_str_radix("4000000003000000002000000001", 16).unwrap())
        );
    }

    #[test]
    fn test_qm31_const() {
        let program = load_cairo! {
            use core::qm31::{qm31, m31};

            pub extern fn qm31_const<
                const W0: m31, const W1: m31, const W2: m31, const W3: m31,
            >() -> qm31 nopanic;

            fn run_test() -> qm31 {
                qm31_const::<1, 2, 3, 4>()
            }
        };

        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            Value::QM31(BigUint::from_str_radix("4000000003000000002000000001", 16).unwrap())
        );
    }
}
