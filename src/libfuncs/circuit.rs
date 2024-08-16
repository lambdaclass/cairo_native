//! # Circuit libfuncs

use super::LibfuncHelper;
use crate::{
    block_ext::BlockExt,
    error::Result,
    libfuncs::r#struct::build_struct_value,
    metadata::MetadataStorage,
    types::{circuit::CIRCUIT_INPUT_SIZE, TypeBuilder},
    utils::{get_integer_layout, layout_repeat, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        circuit::{
            self, CircuitConcreteLibfunc, CircuitTypeConcrete, ConcreteGetOutputLibFunc,
            ConcreteU96LimbsLessThanGuaranteeVerifyLibfunc,
        },
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        lib_func::{SignatureAndTypeConcreteLibfunc, SignatureOnlyConcreteLibfunc},
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith, cf, llvm},
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

    // Get accumulator current length
    let current_length = entry.extract_value(
        context,
        location,
        accumulator,
        IntegerType::new(context, 64).into(),
        0,
    )?;

    // Check if last_insert: current_length == number_of_inputs - 1
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

    // If not last insert, then:
    {
        // Calculate next length: next_length = current_length + 1
        let k1 = middle_insert_block.const_int(context, location, 1, 64)?;
        let next_length =
            middle_insert_block.append_op_result(arith::addi(current_length, k1, location))?;

        // Insert next_length into accumulator
        let accumulator =
            middle_insert_block.insert_value(context, location, accumulator, next_length, 0)?;

        // Get pointer to accumulator with alloc and store
        let accumulator_ptr = middle_insert_block.alloca1(
            context,
            location,
            accumulator.r#type(),
            accumulator_layout.align(),
        )?;
        middle_insert_block.store(context, location, accumulator_ptr, accumulator)?;

        // Get pointer to next input to insert
        let k0 = middle_insert_block.const_int(context, location, 0, 64)?;
        let next_input_ptr =
            middle_insert_block.append_op_result(llvm::get_element_ptr_dynamic(
                context,
                accumulator_ptr,
                &[k0, k1, current_length],
                accumulator.r#type(),
                llvm::r#type::pointer(context, 0),
                location,
            ))?;

        // Interpret u384 struct (input) as u384 integer
        let u384_struct = entry.argument(1)?.into();
        let new_input =
            u384_struct_to_integer(context, middle_insert_block, location, u384_struct)?;

        // Store the u384 into next input pointer
        middle_insert_block.store(context, location, next_input_ptr, new_input)?;

        // Load accumulator from pointer
        let accumulator =
            middle_insert_block.load(context, location, accumulator_ptr, accumulator.r#type())?;

        middle_insert_block.append_operation(helper.br(1, &[accumulator], location));
    }

    // If is last insert, then:
    {
        let data_type_id = &info.branch_signatures()[0].vars[0].ty;
        let (data_type, data_layout) =
            registry.build_type_with_layout(context, helper, registry, metadata, data_type_id)?;

        // Alloc return data
        let data_ptr =
            last_insert_block.alloca1(context, location, data_type, data_layout.align())?;

        // Get pointer to accumulator with alloc and store
        let accumulator_ptr = last_insert_block.alloca1(
            context,
            location,
            accumulator.r#type(),
            accumulator_layout.align(),
        )?;
        last_insert_block.store(context, location, accumulator_ptr, accumulator)?;

        // Get pointer to accumulator input
        let k0 = last_insert_block.const_int(context, location, 0, 64)?;
        let k1 = last_insert_block.const_int(context, location, 1, 64)?;
        let accumulator_input_ptr =
            last_insert_block.append_op_result(llvm::get_element_ptr_dynamic(
                context,
                accumulator_ptr,
                &[k0, k1],
                accumulator.r#type(),
                llvm::r#type::pointer(context, 0),
                location,
            ))?;

        // Copy accumulator input into return data
        let accumulator_input_length = last_insert_block.const_int(
            context,
            location,
            layout_repeat(&get_integer_layout(CIRCUIT_INPUT_SIZE as u32), n_inputs - 1)?
                .0
                .size(),
            64,
        )?;
        last_insert_block.memcpy(
            context,
            location,
            accumulator_input_ptr,
            data_ptr,
            accumulator_input_length,
        );

        // Interpret u384 struct (input) as u384 integer
        let u384_struct = entry.argument(1)?.into();
        let new_input = u384_struct_to_integer(context, last_insert_block, location, u384_struct)?;

        // Get pointer to data end
        let data_end_ptr = last_insert_block.append_op_result(llvm::get_element_ptr(
            context,
            data_ptr,
            DenseI32ArrayAttribute::new(context, &[0, n_inputs as i32 - 1]),
            data_type,
            llvm::r#type::pointer(context, 0),
            location,
        ))?;

        // Store the u384 into next input pointer
        last_insert_block.store(context, location, data_end_ptr, new_input)?;

        // Load data from pointer
        let data = last_insert_block.load(context, location, data_ptr, data_type)?;

        last_insert_block.append_operation(helper.br(0, &[data], location));
    }

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
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let modulus = u384_struct_to_integer(context, entry, location, entry.argument(0)?.into())?;
    let k1 = entry.const_int(context, location, 1, CIRCUIT_INPUT_SIZE as u32)?;

    let is_valid = entry.append_op_result(arith::cmpi(
        context,
        arith::CmpiPredicate::Ugt,
        modulus,
        k1,
        location,
    ))?;

    entry.append_operation(helper.cond_br(context, is_valid, [0, 1], [&[modulus], &[]], location));

    Ok(())
}

/// Generate MLIR operations for the `get_circuit_descriptor` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_get_descriptor<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let descriptor_type_id = &info.branch_signatures()[0].vars[0].ty;
    let descriptor_type =
        registry.build_type(context, helper, registry, metadata, descriptor_type_id)?;

    let unit = entry.append_op_result(llvm::undef(descriptor_type, location))?;

    entry.append_operation(helper.br(0, &[unit], location));

    Ok(())
}

/// Generate MLIR operations for the `eval_circuit` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_eval<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let circuit_info = match registry.get_type(&info.ty).unwrap() {
        CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(info)) => &info.circuit_info,
        _ => unreachable!(),
    };
    // todo! remove this debug prints
    // dbg!(circuit_info);

    let circuit_data = entry.argument(3)?.into();
    let circuit_modulus = entry.argument(4)?.into();

    // todo! should arguments 5 and 6 be used?
    // let zero = entry.argument(5)?;
    // let one = entry.argument(6)?;

    // Fill input values
    let mut values = vec![None; 1 + circuit_info.n_inputs + circuit_info.values.len()];
    values[0] = Some(entry.const_int(context, location, 1, CIRCUIT_INPUT_SIZE as u32)?);
    for i in 0..circuit_info.n_inputs {
        values[i + 1] = Some(entry.extract_value(
            context,
            location,
            circuit_data,
            IntegerType::new(context, CIRCUIT_INPUT_SIZE as u32).into(),
            i,
        )?);
    }

    // Evaluate circuit
    {
        let mut add_offsets = circuit_info.add_offsets.iter().peekable();
        let mut mul_offsets = circuit_info.mul_offsets.iter();

        loop {
            while let Some(&&circuit::GateOffsets { lhs, rhs, output }) = add_offsets.peek() {
                let lhs_value = values[lhs];
                let rhs_value = values[rhs];
                let output_value = values[output];
                match (lhs_value, rhs_value, output_value) {
                    // ADD: lhs + rhs = out
                    (Some(lhs_value), Some(rhs_value), None) => {
                        let value =
                            entry.append_op_result(arith::addi(lhs_value, rhs_value, location))?;
                        let value = entry.append_op_result(arith::remui(
                            value,
                            circuit_modulus,
                            location,
                        ))?;
                        values[output] = Some(value);
                    }
                    // SUB: lhs = out - rhs
                    (None, Some(rhs_value), Some(output_value)) => {
                        let value = entry.append_op_result(arith::addi(
                            output_value,
                            circuit_modulus,
                            location,
                        ))?;
                        let value =
                            entry.append_op_result(arith::subi(value, rhs_value, location))?;
                        let value = entry.append_op_result(arith::remui(
                            value,
                            circuit_modulus,
                            location,
                        ))?;
                        values[lhs] = Some(value);
                    }
                    _ => break,
                }

                add_offsets.next();
            }

            if let Some(&circuit::GateOffsets { lhs, rhs, output }) = mul_offsets.next() {
                let lhs_value = values[lhs];
                let rhs_value = values[rhs];
                let output_value = values[output];
                match (lhs_value, rhs_value, output_value) {
                    // MUL: lhs * rhs = out
                    (Some(lhs), Some(rhs), None) => {
                        let value = entry.append_op_result(arith::muli(lhs, rhs, location))?;
                        let value = entry.append_op_result(arith::remui(
                            value,
                            circuit_modulus,
                            location,
                        ))?;
                        values[output] = Some(value)
                    }
                    // DIV: lhs = out / rhs
                    (None, Some(_rhs), Some(_output)) => {
                        todo!()
                    }
                    _ => unreachable!(),
                }
            } else {
                break;
            }
        }
    }

    // Validate all values are calculated
    let values = values
        .into_iter()
        .skip(1 + circuit_info.n_inputs)
        .collect::<Option<Vec<Value>>>()
        .expect("circuit should be solvable");

    // Build output struct
    let outputs_type_id = &info.branch_signatures()[0].vars[2].ty;
    let outputs = build_struct_value(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        outputs_type_id,
        &values,
    )?;

    // {
    //     // todo! remove this debug print
    //     let (_outputs_type, outputs_layout) = registry.build_type_with_layout(
    //         context,
    //         helper,
    //         registry,
    //         metadata,
    //         outputs_type_id,
    //     )?;
    //     let outputs_ptr =
    //         entry.alloca1(context, location, outputs.r#type(), outputs_layout.align())?;
    //     entry.store(context, location, outputs_ptr, outputs)?;
    //     metadata.get_mut::<DebugUtils>().unwrap().dump_mem(
    //         context,
    //         helper,
    //         entry,
    //         outputs_ptr,
    //         outputs_layout.size(),
    //         location,
    //     )?;
    // }

    let ktrue = entry.const_int(context, location, 1, 64)?;
    let partial_type_id = &info.branch_signatures()[1].vars[2].ty;
    let partial = entry.append_op_result(llvm::undef(
        registry.build_type(context, helper, registry, metadata, partial_type_id)?,
        location,
    ))?;
    let failure_type_id = &info.branch_signatures()[1].vars[3].ty;
    let failure = entry.append_op_result(llvm::undef(
        registry.build_type(context, helper, registry, metadata, failure_type_id)?,
        location,
    ))?;
    entry.append_operation(helper.cond_br(
        context,
        ktrue,
        [0, 1],
        [
            &[
                entry.argument(0)?.into(),
                entry.argument(1)?.into(),
                outputs,
            ],
            &[
                entry.argument(0)?.into(),
                entry.argument(1)?.into(),
                partial,
                failure,
            ],
        ],
        location,
    ));

    Ok(())
}

/// Generate MLIR operations for the `circuit_failure_guarantee_verify` libfunc.
/// NOOP
#[allow(clippy::too_many_arguments)]
fn build_failure_guarantee_verify<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let guarantee_type_id = &info.branch_signatures()[0].vars[2].ty;
    let guarantee_type =
        registry.build_type(context, helper, registry, metadata, guarantee_type_id)?;

    let guarantee = entry.append_op_result(llvm::undef(guarantee_type, location))?;

    entry.append_operation(helper.br(
        0,
        &[
            entry.argument(0)?.into(),
            entry.argument(1)?.into(),
            guarantee,
        ],
        location,
    ));

    Ok(())
}

/// Generate MLIR operations for the `u96_limbs_less_than_guarantee_verify` libfunc.
/// NOOP
#[allow(clippy::too_many_arguments)]
fn build_u96_limbs_less_than_guarantee_verify<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &ConcreteU96LimbsLessThanGuaranteeVerifyLibfunc,
) -> Result<()> {
    let guarantee_type_id = &info.branch_signatures()[0].vars[0].ty;
    let guarantee_type =
        registry.build_type(context, helper, registry, metadata, guarantee_type_id)?;

    let guarantee = entry.append_op_result(llvm::undef(guarantee_type, location))?;

    let u96_type_id = &info.branch_signatures()[1].vars[0].ty;
    let u96_type = registry.build_type(context, helper, registry, metadata, u96_type_id)?;

    let u96 = entry.append_op_result(llvm::undef(u96_type, location))?;

    let kfalse = entry.const_int(context, location, 0, 64)?;
    entry.append_operation(helper.cond_br(
        context,
        kfalse,
        [0, 1],
        [&[guarantee], &[u96]],
        location,
    ));

    Ok(())
}

/// Generate MLIR operations for the `u96_guarantee_verify` libfunc.
/// NOOP
#[allow(clippy::too_many_arguments)]
fn build_u96_guarantee_verify<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));

    Ok(())
}

/// Generate MLIR operations for the `u96_single_limb_less_than_guarantee_verify` libfunc.
/// NOOP
#[allow(clippy::too_many_arguments)]
fn build_u96_single_limb_less_than_guarantee_verify<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let u96_type_id = &info.branch_signatures()[0].vars[0].ty;
    let u96_type = registry.build_type(context, helper, registry, metadata, u96_type_id)?;
    let u96 = entry.append_op_result(llvm::undef(u96_type, location))?;

    entry.append_operation(helper.br(0, &[u96], location));

    Ok(())
}

/// Generate MLIR operations for the `get_circuit_output` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_get_output<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &ConcreteGetOutputLibFunc,
) -> Result<()> {
    let circuit_info = match registry.get_type(&info.circuit_ty).unwrap() {
        CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(info)) => &info.circuit_info,
        _ => unreachable!(),
    };
    let output_type_id = &info.output_ty;

    let Some(&output_offset_idx) = circuit_info.values.get(output_type_id) else {
        unreachable!()
    };
    let output_idx = output_offset_idx - circuit_info.n_inputs - 1;

    let outputs = entry.argument(0)?.into();
    let output_integer = entry.extract_value(
        context,
        location,
        outputs,
        IntegerType::new(context, CIRCUIT_INPUT_SIZE as u32).into(),
        output_idx,
    )?;
    let output_struct = u384_integer_to_struct(context, entry, location, output_integer)?;

    let guarantee_type_id = &info.branch_signatures()[0].vars[1].ty;
    let guarantee_type =
        registry.build_type(context, helper, registry, metadata, guarantee_type_id)?;
    let guarantee = entry.append_op_result(llvm::undef(guarantee_type, location))?;

    entry.append_operation(helper.br(0, &[output_struct, guarantee], location));

    Ok(())
}

fn u384_struct_to_integer<'a>(
    context: &'a Context,
    block: &'a Block<'a>,
    location: Location<'a>,
    u384_struct: Value<'a, 'a>,
) -> Result<Value<'a, 'a>> {
    let u96_type = IntegerType::new(context, 96).into();

    let limb1 = block.append_op_result(arith::extui(
        block.extract_value(context, location, u384_struct, u96_type, 0)?,
        IntegerType::new(context, CIRCUIT_INPUT_SIZE as u32).into(),
        location,
    ))?;

    let limb2 = {
        let limb = block.append_op_result(arith::extui(
            block.extract_value(context, location, u384_struct, u96_type, 1)?,
            IntegerType::new(context, CIRCUIT_INPUT_SIZE as u32).into(),
            location,
        ))?;
        let k96 = block.const_int(context, location, 96, CIRCUIT_INPUT_SIZE as u32)?;
        block.append_op_result(arith::shli(limb, k96, location))?
    };

    let limb3 = {
        let limb = block.append_op_result(arith::extui(
            block.extract_value(context, location, u384_struct, u96_type, 2)?,
            IntegerType::new(context, CIRCUIT_INPUT_SIZE as u32).into(),
            location,
        ))?;
        let k192 = block.const_int(context, location, 96 * 2, CIRCUIT_INPUT_SIZE as u32)?;
        block.append_op_result(arith::shli(limb, k192, location))?
    };

    let limb4 = {
        let limb = block.append_op_result(arith::extui(
            block.extract_value(context, location, u384_struct, u96_type, 3)?,
            IntegerType::new(context, CIRCUIT_INPUT_SIZE as u32).into(),
            location,
        ))?;
        let k288 = block.const_int(context, location, 96 * 3, CIRCUIT_INPUT_SIZE as u32)?;
        block.append_op_result(arith::shli(limb, k288, location))?
    };

    let value = block.append_op_result(arith::ori(limb1, limb2, location))?;
    let value = block.append_op_result(arith::ori(value, limb3, location))?;
    let value = block.append_op_result(arith::ori(value, limb4, location))?;

    Ok(value)
}

fn u384_integer_to_struct<'a>(
    context: &'a Context,
    block: &'a Block<'a>,
    location: Location<'a>,
    integer: Value<'a, 'a>,
) -> Result<Value<'a, 'a>> {
    let u96_type = IntegerType::new(context, 96).into();

    let limb1 = block.append_op_result(arith::trunci(
        integer,
        IntegerType::new(context, 96).into(),
        location,
    ))?;
    let limb2 = {
        let k96 = block.const_int(context, location, 96, CIRCUIT_INPUT_SIZE as u32)?;
        let limb = block.append_op_result(arith::shrui(integer, k96, location))?;
        block.append_op_result(arith::trunci(limb, u96_type, location))?
    };
    let limb3 = {
        let k192 = block.const_int(context, location, 96 * 2, CIRCUIT_INPUT_SIZE as u32)?;
        let limb = block.append_op_result(arith::shrui(integer, k192, location))?;
        block.append_op_result(arith::trunci(limb, u96_type, location))?
    };
    let limb4 = {
        let k288 = block.const_int(context, location, 96 * 3, CIRCUIT_INPUT_SIZE as u32)?;
        let limb = block.append_op_result(arith::shrui(integer, k288, location))?;
        block.append_op_result(arith::trunci(limb, u96_type, location))?
    };

    let struct_type = llvm::r#type::r#struct(
        context,
        &[
            IntegerType::new(context, 96).into(),
            IntegerType::new(context, 96).into(),
            IntegerType::new(context, 96).into(),
            IntegerType::new(context, 96).into(),
        ],
        false,
    );
    let struct_value = block.append_op_result(llvm::undef(struct_type, location))?;

    block.insert_values(
        context,
        location,
        struct_value,
        &[limb1, limb2, limb3, limb4],
    )
}
