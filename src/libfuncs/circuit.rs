//! # Circuit libfuncs

use super::LibfuncHelper;
use crate::{block_ext::BlockExt, error::Result, metadata::MetadataStorage};
use cairo_lang_sierra::{
    extensions::{
        circuit::{CircuitConcreteLibfunc, CircuitTypeConcrete},
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        lib_func::SignatureAndTypeConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use itertools::Itertools;
use melior::{
    dialect::ods::llvm,
    ir::{Block, Location},
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &CircuitConcreteLibfunc,
) -> Result<()> {
    match selector {
        CircuitConcreteLibfunc::AddInput(_) => todo!(),
        CircuitConcreteLibfunc::Eval(_) => todo!(),
        CircuitConcreteLibfunc::GetDescriptor(_) => todo!(),
        CircuitConcreteLibfunc::InitCircuitData(info) => {
            build_init_circuit_data(context, registry, entry, location, helper, metadata, info)
        }
        CircuitConcreteLibfunc::GetOutput(_) => todo!(),
        CircuitConcreteLibfunc::TryIntoCircuitModulus(_) => todo!(),
        CircuitConcreteLibfunc::FailureGuaranteeVerify(_) => todo!(),
        CircuitConcreteLibfunc::IntoU96Guarantee(_) => todo!(),
        CircuitConcreteLibfunc::U96GuaranteeVerify(_) => todo!(),
        CircuitConcreteLibfunc::U96LimbsLessThanGuaranteeVerify(_) => todo!(),
        CircuitConcreteLibfunc::U96SingleLimbLessThanGuaranteeVerify(_) => todo!(),
    }
}

/// Generate MLIR operations for the `bounded_int_add` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_init_circuit_data<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let circuit_type = {
        let circuit_type = registry.get_type(&info.ty)?;

        if let CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(circuit_type)) = circuit_type
        {
            circuit_type
        } else {
            todo!()
        }
    };

    dbg!(&circuit_type.info);
    dbg!(&circuit_type.circuit_info);

    let params = info
        .param_signatures()
        .iter()
        .map(|p| p.ty.debug_name.clone())
        .collect_vec();
    dbg!(params);

    let branches = info.branch_signatures();
    dbg!(branches);

    let dummy = entry.const_int(context, location, 1, 64)?;

    entry.append_operation(helper.br(0, &[entry.argument(0)?.into(), dummy], location));

    Ok(())
}
