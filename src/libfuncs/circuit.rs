//! # Circuit libfuncs

use super::LibfuncHelper;
use crate::{
    block_ext::BlockExt,
    error::Result,
    libfuncs::r#struct::build_struct_value,
    metadata::MetadataStorage,
    types::{circuit::CIRCUIT_INPUT_SIZE, TypeBuilder},
    utils::get_integer_layout,
};
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
    dialect::{
        arith, cf,
        llvm::{self},
        ods::llvm::{inttoptr, ptrtoint},
    },
    ir::{
        attribute::DenseI32ArrayAttribute, r#type::IntegerType, Block, Location, Value, ValueLike,
    },
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
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let k0 = entry.const_int(context, location, 0, 64)?;
    let accumulator_ty = &info.branch_signatures()[0].vars[1].ty;
    let accumulator = build_struct_value(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        accumulator_ty,
        &[k0],
    )?;

    entry.append_operation(helper.br(0, &[entry.argument(0)?.into(), accumulator], location));

    Ok(())
}

/// Generate MLIR operations for the `into_u96_guarantee` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_into_u96_guarantee<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    // input is a BoundedInt<0, 79228162514264337593543950335>
    // output is a U96Guarantee
    // they have the same type (i96), so we just return its argument
    // should we debug_assert this?

    entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));

    Ok(())
}

/// Generate MLIR operations for the `add_circuit_input` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_add_input<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let n_inputs = match registry.get_type(&info.ty).unwrap() {
        CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(info)) => info.circuit_info.n_inputs,
        _ => unreachable!(),
    };
    let accumulator_type_id = &info.param_signatures()[0].ty;
    let accumulator_ctype = registry.get_type(accumulator_type_id)?;
    let accumulator_layout = accumulator_ctype.layout(registry)?;

    let accumulator: Value = entry.argument(0)?.into();

    // Get current accumulator length
    let current_length = entry.extract_value(
        context,
        location,
        accumulator,
        IntegerType::new(context, 64).into(),
        0,
    )?;

    // Check if last insert: current_length == number_of_inputs - 1
    let n_inputs_minus_1 = entry.const_int(context, location, n_inputs - 1, 64)?;
    let last_insert = entry.append_op_result(arith::cmpi(
        context,
        arith::CmpiPredicate::Eq,
        current_length,
        n_inputs_minus_1,
        location,
    ))?;

    let middle_insert_block = helper.append_block(Block::new(&[]));
    let last_insert_block = helper.append_block(Block::new(&[]));
    entry.append_operation(cf::cond_br(
        context,
        last_insert,
        last_insert_block,
        middle_insert_block,
        &[],
        &[],
        location,
    ));

    {
        // If not last insert, then:

        // Update length
        let k1 = middle_insert_block.const_int(context, location, 1, 64)?;
        let next_length =
            middle_insert_block.append_op_result(arith::addi(current_length, k1, location))?;
        let accumulator =
            middle_insert_block.insert_value(context, location, accumulator, next_length, 0)?;

        // Get accumulator pointer so that we can do pointer arithmetic
        let accumulator_ptr = middle_insert_block.alloca1(
            context,
            location,
            accumulator.r#type(),
            accumulator_layout.align(),
        )?;
        middle_insert_block.store(context, location, accumulator_ptr, accumulator)?;

        // because GEP instruction only supports constant indexes, me must implement
        // calculate the desired pointer manually:

        let next_input_ptr = {
            // Get first input pointer
            let first_input_ptr = middle_insert_block.append_op_result(llvm::get_element_ptr(
                context,
                accumulator_ptr,
                DenseI32ArrayAttribute::new(context, &[0, 1]),
                accumulator.r#type(),
                llvm::r#type::pointer(context, 0),
                location,
            ))?;
            let first_input_ptr = middle_insert_block.append_op_result(
                ptrtoint(
                    context,
                    IntegerType::new(context, 64).into(),
                    first_input_ptr,
                    location,
                )
                .into(),
            )?;

            // Get offset of next input pointer from first input
            let k_u384_strafe = middle_insert_block.const_int(
                context,
                location,
                get_integer_layout(CIRCUIT_INPUT_SIZE as u32).size(),
                64,
            )?;
            let next_input_offset = middle_insert_block.append_op_result(arith::muli(
                k_u384_strafe,
                current_length,
                location,
            ))?;

            // Calculate next input pointer by adding the offset to the first input pointer
            let next_input_ptr = middle_insert_block.append_op_result(arith::addi(
                first_input_ptr,
                next_input_offset,
                location,
            ))?;

            middle_insert_block.append_op_result(
                inttoptr(
                    context,
                    llvm::r#type::pointer(context, 0),
                    next_input_ptr,
                    location,
                )
                .into(),
            )?
        };

        // Interpret u384 struct as u384 integer
        let u384_struct = entry.argument(1)?.into();
        let new_input =
            u384_struct_to_integer(context, middle_insert_block, u384_struct, location)?;

        // Store input in next input pointer
        middle_insert_block.store(context, location, next_input_ptr, new_input)?;

        // Load accumulator from pointer
        let accumulator =
            middle_insert_block.load(context, location, accumulator_ptr, accumulator.r#type())?;

        middle_insert_block.append_operation(helper.br(1, &[accumulator], location));
    }

    {
        // If is last insert, then:

        let data_type_id = &info.branch_signatures()[0].vars[0].ty;

        // Extract all values from accumulator
        let mut values = vec![];
        for id in 1..n_inputs {
            let value = last_insert_block.extract_value(
                context,
                location,
                accumulator,
                IntegerType::new(context, CIRCUIT_INPUT_SIZE as u32).into(),
                id,
            )?;
            values.push(value)
        }

        // Interpret u384 struct as u384 integer
        let u384_struct = entry.argument(1)?.into();
        let new_input = u384_struct_to_integer(context, last_insert_block, u384_struct, location)?;
        values.push(new_input);

        // Create CircuitData struct
        let data = build_struct_value(
            context,
            registry,
            last_insert_block,
            location,
            helper,
            metadata,
            data_type_id,
            &values,
        )?;

        last_insert_block.append_operation(helper.br(0, &[data], location));
    }

    Ok(())
}

fn u384_struct_to_integer<'a>(
    context: &'a Context,
    block: &'a Block<'a>,
    u384_struct: Value<'a, 'a>,
    location: Location<'a>,
) -> Result<Value<'a, 'a>> {
    // todo! take into account other limbs
    let u384_limb1 = block.append_op_result(arith::extui(
        block.extract_value(
            context,
            location,
            u384_struct,
            IntegerType::new(context, 96).into(),
            0,
        )?,
        IntegerType::new(context, CIRCUIT_INPUT_SIZE as u32).into(),
        location,
    ))?;
    Ok(u384_limb1)
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

    todo!()
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

    todo!()
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

    todo!()
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

    todo!()
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
    todo!()
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

    todo!()
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

    todo!()
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

    todo!()
}
