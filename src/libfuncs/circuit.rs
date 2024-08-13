//! # Circuit libfuncs

use super::LibfuncHelper;
use crate::{block_ext::BlockExt, error::Result, metadata::MetadataStorage};
use cairo_lang_sierra::{
    extensions::{
        circuit::{
            CircuitConcreteLibfunc, CircuitTypeConcrete, ConcreteGetOutputLibFunc,
            ConcreteU96LimbsLessThanGuaranteeVerifyLibfunc,
        },
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        lib_func::{SignatureAndTypeConcreteLibfunc, SignatureOnlyConcreteLibfunc},
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use itertools::Itertools;
use melior::{
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
        CircuitConcreteLibfunc::AddInput(info) => {
            build_add_input(context, registry, entry, location, helper, metadata, info)
        }
        CircuitConcreteLibfunc::Eval(info) => {
            build_eval(context, registry, entry, location, helper, metadata, info)
        }
        CircuitConcreteLibfunc::GetDescriptor(info) => {
            build_get_descriptor(context, registry, entry, location, helper, metadata, info)
        }
        CircuitConcreteLibfunc::InitCircuitData(info) => {
            build_init_circuit_data(context, registry, entry, location, helper, metadata, info)
        }
        CircuitConcreteLibfunc::GetOutput(info) => {
            build_get_output(context, registry, entry, location, helper, metadata, info)
        }
        CircuitConcreteLibfunc::TryIntoCircuitModulus(info) => build_try_into_circuit_modulus(
            context, registry, entry, location, helper, metadata, info,
        ),
        CircuitConcreteLibfunc::FailureGuaranteeVerify(info) => build_failure_guarantee_verify(
            context, registry, entry, location, helper, metadata, info,
        ),
        CircuitConcreteLibfunc::IntoU96Guarantee(info) => {
            build_into_u96_guarantee(context, registry, entry, location, helper, metadata, info)
        }
        CircuitConcreteLibfunc::U96GuaranteeVerify(info) => {
            build_u96_guarantee_verify(context, registry, entry, location, helper, metadata, info)
        }
        CircuitConcreteLibfunc::U96LimbsLessThanGuaranteeVerify(info) => {
            build_u96_limbs_less_than_guarantee_verify(
                context, registry, entry, location, helper, metadata, info,
            )
        }
        CircuitConcreteLibfunc::U96SingleLimbLessThanGuaranteeVerify(info) => {
            build_u96_single_limb_less_than_guarantee_verify(
                context, registry, entry, location, helper, metadata, info,
            )
        }
    }
}

/// Generate MLIR operations for the `init_circuit_data` libfunc.
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

/// Generate MLIR operations for the `into_u96_guarantee` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_into_u96_guarantee<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let info_type = info.ty.debug_name.clone();
    dbg!(&info_type);

    let params = info
        .param_signatures()
        .iter()
        .map(|p| p.ty.debug_name.clone())
        .collect_vec();
    dbg!(params);

    let branches = info.branch_signatures();
    dbg!(branches);

    let dummy = entry.const_int(context, location, 1, 64)?;
    entry.append_operation(helper.br(0, &[dummy], location));
    Ok(())
}

/// Generate MLIR operations for the `add_circuit_input` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_add_input<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let info_type = info.ty.debug_name.clone();
    dbg!(&info_type);

    let params = info
        .param_signatures()
        .iter()
        .map(|p| p.ty.debug_name.clone())
        .collect_vec();
    dbg!(params);

    let branches = info.branch_signatures();
    dbg!(branches);

    let dummy = entry.const_int(context, location, 1, 64)?;
    entry.append_operation(helper.br(0, &[dummy], location));
    entry.append_operation(helper.br(1, &[dummy], location));
    Ok(())
}

/// Generate MLIR operations for the `try_into_circuit_modulus` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_try_into_circuit_modulus<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let params = info
        .param_signatures()
        .iter()
        .map(|p| p.ty.debug_name.clone())
        .collect_vec();
    dbg!(params);

    let branches = info.branch_signatures();
    dbg!(branches);

    let dummy = entry.const_int(context, location, 1, 64)?;
    entry.append_operation(helper.br(0, &[dummy], location));
    Ok(())
}

/// Generate MLIR operations for the `get_circuit_descriptor` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_get_descriptor<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let info_type = info.ty.debug_name.clone();
    dbg!(&info_type);

    let params = info
        .param_signatures()
        .iter()
        .map(|p| p.ty.debug_name.clone())
        .collect_vec();
    dbg!(params);

    let branches = info.branch_signatures();
    dbg!(branches);

    let dummy = entry.const_int(context, location, 1, 64)?;
    entry.append_operation(helper.br(0, &[dummy], location));
    Ok(())
}

/// Generate MLIR operations for the `eval_circuit` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_eval<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let info_type = info.ty.debug_name.clone();
    dbg!(&info_type);

    let params = info
        .param_signatures()
        .iter()
        .map(|p| p.ty.debug_name.clone())
        .collect_vec();
    dbg!(params);

    let branches = info.branch_signatures();
    dbg!(branches);

    let dummy = entry.const_int(context, location, 1, 64)?;
    entry.append_operation(helper.br(
        0,
        &[entry.argument(0)?.into(), entry.argument(1)?.into(), dummy],
        location,
    ));
    entry.append_operation(helper.br(
        1,
        &[
            entry.argument(0)?.into(),
            entry.argument(1)?.into(),
            dummy,
            dummy,
        ],
        location,
    ));
    Ok(())
}

/// Generate MLIR operations for the `circuit_failure_guarantee_verify` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_failure_guarantee_verify<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let params = info
        .param_signatures()
        .iter()
        .map(|p| p.ty.debug_name.clone())
        .collect_vec();
    dbg!(params);

    let branches = info.branch_signatures();
    dbg!(branches);

    let dummy = entry.const_int(context, location, 1, 64)?;
    entry.append_operation(helper.br(
        0,
        &[entry.argument(0)?.into(), entry.argument(1)?.into(), dummy],
        location,
    ));
    Ok(())
}

/// Generate MLIR operations for the `u96_limbs_less_than_guarantee_verify` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_u96_limbs_less_than_guarantee_verify<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &ConcreteU96LimbsLessThanGuaranteeVerifyLibfunc,
) -> Result<()> {
    dbg!(&info.limb_count);

    let params = info
        .param_signatures()
        .iter()
        .map(|p| p.ty.debug_name.clone())
        .collect_vec();
    dbg!(params);

    let branches = info.branch_signatures();
    dbg!(branches);

    let dummy = entry.const_int(context, location, 1, 64)?;
    entry.append_operation(helper.br(0, &[dummy], location));
    entry.append_operation(helper.br(1, &[dummy], location));
    Ok(())
}

/// Generate MLIR operations for the `u96_guarantee_verify` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_u96_guarantee_verify<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let params = info
        .param_signatures()
        .iter()
        .map(|p| p.ty.debug_name.clone())
        .collect_vec();
    dbg!(params);

    let branches = info.branch_signatures();
    dbg!(branches);

    entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));
    Ok(())
}

/// Generate MLIR operations for the `u96_single_limb_less_than_guarantee_verify` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_u96_single_limb_less_than_guarantee_verify<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let params = info
        .param_signatures()
        .iter()
        .map(|p| p.ty.debug_name.clone())
        .collect_vec();
    dbg!(params);

    let branches = info.branch_signatures();
    dbg!(branches);

    entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));
    Ok(())
}

/// Generate MLIR operations for the `get_circuit_output` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_get_output<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &ConcreteGetOutputLibFunc,
) -> Result<()> {
    dbg!(&info.circuit_ty.debug_name);
    dbg!(&info.output_ty.debug_name);

    let params = info
        .param_signatures()
        .iter()
        .map(|p| p.ty.debug_name.clone())
        .collect_vec();
    dbg!(params);

    let branches = info.branch_signatures();
    dbg!(branches);

    let dummy = entry.const_int(context, location, 1, 64)?;
    entry.append_operation(helper.br(0, &[dummy, dummy], location));
    Ok(())
}
