use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        qm31::{QM31BinaryOpConcreteLibfunc, QM31Concrete, QM31ConstConcreteLibfunc},
    },
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Block, Location},
    Context,
};

use crate::{error::Result, metadata::MetadataStorage};

use super::LibfuncHelper;

const M31_SIZE: u32 = 36;
const M31_MAX: u64 = 1 << M31_SIZE;

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
            todo!("impl qm31_pack");
        }
        QM31Concrete::Unpack(info) => {
            todo!("impl qm31_unpack");

        }
        QM31Concrete::Const(info) => {
            todo!("impl qm31_const");
        }
        QM31Concrete::FromM31(info) => {
            todo!("impl qm31_from_m31");
        }
        QM31Concrete::IsZero(info) => {
            todo!("impl qm31_is_zero");
        }
        QM31Concrete::BinaryOperation(info) => {
            todo!("impl qm31_binary_operation");
        }
    }
}
