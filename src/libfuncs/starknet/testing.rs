use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        starknet::testing::CheatcodeConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use cairo_lang_utils::bigint::BigIntAsHex;
use melior::{
    dialect::{
        arith,
        llvm::{self, LoadStoreOptions},
    },
    ir::{
        attribute::{
            DenseI32ArrayAttribute, DenseI64ArrayAttribute, IntegerAttribute, TypeAttribute,
        },
        operation::OperationBuilder,
        r#type::IntegerType,
        Block, Identifier, Location, Value,
    },
    Context,
};

use crate::{
    error::Result, libfuncs::LibfuncHelper, metadata::MetadataStorage,
    starknet::handler::StarknetSyscallHandlerCallbacks, utils::get_integer_layout,
};

pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &CheatcodeConcreteLibfunc,
) -> Result<()> {
    let selector_as_hex = BigIntAsHex {
        value: selector.selector.clone(),
    };
    let selector_bytes = &selector_as_hex.value.to_bytes_be().1;
    let selector_str = std::str::from_utf8(&selector_bytes).unwrap();

    // Extract self pointer.
    let ptr: Value = entry
        .append_operation(llvm::load(
            context,
            entry.argument(1)?.into(),
            llvm::r#type::opaque_pointer(context),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                // The branch is deliberately duplicated because:
                //   - There is no `[0].vars[2]` (it returns `()`).
                //   - We need a variant to make the length be 2.
                //   - It requires a `ConcreteTypeId`, we can't pass an MLIR type.
                selector.branch_signatures()[1].vars[2].ty.clone(),
                selector.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let constant_one = helper
        .init_block()
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
            location,
        ))
        .result(0)?
        .into();

    let result_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[
                    (
                        Identifier::new(context, "alignment"),
                        IntegerAttribute::new(
                            IntegerType::new(context, 64).into(),
                            result_layout.align().try_into()?,
                        )
                        .into(),
                    ),
                    (
                        Identifier::new(context, "elem_type"),
                        TypeAttribute::new(llvm::r#type::r#struct(
                            context,
                            &[
                                result_tag_ty,
                                llvm::r#type::array(
                                    IntegerType::new(context, 8).into(),
                                    (result_layout.size() - 1).try_into()?,
                                ),
                            ],
                            false,
                        ))
                        .into(),
                    ),
                ])
                .add_operands(&[constant_one])
                .add_results(&[llvm::r#type::opaque_pointer(context)])
                .build()?,
        )
        .result(0)?
        .into();

    // Allocate space and write the current gas.
    let gas_builtin_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        IntegerType::new(context, 64).into(),
                        result_layout.align().try_into()?,
                    )
                    .into(),
                )])
                .add_operands(&[constant_one])
                .add_results(&[llvm::r#type::pointer(
                    IntegerType::new(context, 128).into(),
                    0,
                )])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(0)?.into(),
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    match selector_str {
        "set_sequencer_address" => {
            // Allocate `address` argument and write the value.
            let address_arg_ptr_ty =
                llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0);
            let address_arg_ptr = helper
                .init_block()
                .append_operation(
                    OperationBuilder::new("llvm.alloca", location)
                        .add_attributes(&[(
                            Identifier::new(context, "alignment"),
                            IntegerAttribute::new(
                                IntegerType::new(context, 64).into(),
                                get_integer_layout(252).align().try_into().unwrap(),
                            )
                            .into(),
                        )])
                        .add_operands(&[constant_one])
                        .add_results(&[address_arg_ptr_ty])
                        .build()?,
                )
                .result(0)?
                .into();
            entry.append_operation(llvm::store(
                context,
                entry.argument(3)?.into(),
                address_arg_ptr,
                location,
                LoadStoreOptions::default(),
            ));

            // Extract function pointer.
            let fn_ptr_ty = llvm::r#type::function(
                llvm::r#type::void(context),
                &[
                    llvm::r#type::opaque_pointer(context),
                    llvm::r#type::opaque_pointer(context),
                    llvm::r#type::pointer(IntegerType::new(context, 128).into(), 0),
                    address_arg_ptr_ty,
                ],
                false,
            );

            let fn_ptr = entry
                .append_operation(llvm::get_element_ptr(
                    context,
                    entry.argument(1)?.into(),
                    DenseI32ArrayAttribute::new(
                        context,
                        &[StarknetSyscallHandlerCallbacks::<()>::SET_SEQUENCER_ADDRESS
                            .try_into()?],
                    ),
                    llvm::r#type::opaque_pointer(context),
                    llvm::r#type::opaque_pointer(context),
                    location,
                ))
                .result(0)?
                .into();
            let fn_ptr = entry
                .append_operation(llvm::load(
                    context,
                    fn_ptr,
                    llvm::r#type::pointer(fn_ptr_ty, 0),
                    location,
                    LoadStoreOptions::default(),
                ))
                .result(0)?
                .into();

            entry.append_operation(
                OperationBuilder::new("llvm.call", location)
                    .add_operands(&[
                        fn_ptr,
                        result_ptr,
                        ptr,
                        gas_builtin_ptr,
                        entry.argument(2)?.into(),
                        address_arg_ptr,
                    ])
                    .build()?,
            );

            let result = entry
                .append_operation(llvm::load(
                    context,
                    result_ptr,
                    llvm::r#type::r#struct(
                        context,
                        &[
                            result_tag_ty,
                            llvm::r#type::array(
                                IntegerType::new(context, 8).into(),
                                (result_layout.size() - 1).try_into()?,
                            ),
                        ],
                        false,
                    ),
                    location,
                    LoadStoreOptions::default(),
                ))
                .result(0)?
                .into();
            let result_tag = entry
                .append_operation(llvm::extract_value(
                    context,
                    result,
                    DenseI64ArrayAttribute::new(context, &[0]),
                    IntegerType::new(context, 1).into(),
                    location,
                ))
                .result(0)?
                .into();

            let payload_ok = {
                let ptr = entry
                    .append_operation(
                        OperationBuilder::new("llvm.getelementptr", location)
                            .add_attributes(&[
                                (
                                    Identifier::new(context, "rawConstantIndices"),
                                    DenseI32ArrayAttribute::new(
                                        context,
                                        &[result_tag_layout
                                            .extend(variant_tys[0].1)?
                                            .1
                                            .try_into()?],
                                    )
                                    .into(),
                                ),
                                (
                                    Identifier::new(context, "elem_type"),
                                    TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                                ),
                            ])
                            .add_operands(&[result_ptr])
                            .add_results(&[llvm::r#type::opaque_pointer(context)])
                            .build()?,
                    )
                    .result(0)?
                    .into();
                entry
                    .append_operation(llvm::load(
                        context,
                        ptr,
                        variant_tys[0].0,
                        location,
                        LoadStoreOptions::default(),
                    ))
                    .result(0)?
                    .into()
            };
            let payload_err = {
                let ptr = entry
                    .append_operation(
                        OperationBuilder::new("llvm.getelementptr", location)
                            .add_attributes(&[
                                (
                                    Identifier::new(context, "rawConstantIndices"),
                                    DenseI32ArrayAttribute::new(
                                        context,
                                        &[result_tag_layout
                                            .extend(variant_tys[1].1)?
                                            .1
                                            .try_into()?],
                                    )
                                    .into(),
                                ),
                                (
                                    Identifier::new(context, "elem_type"),
                                    TypeAttribute::new(IntegerType::new(context, 8).into()).into(),
                                ),
                            ])
                            .add_operands(&[result_ptr])
                            .add_results(&[llvm::r#type::opaque_pointer(context)])
                            .build()?,
                    )
                    .result(0)?
                    .into();
                entry
                    .append_operation(llvm::load(
                        context,
                        ptr,
                        variant_tys[1].0,
                        location,
                        LoadStoreOptions::default(),
                    ))
                    .result(0)?
                    .into()
            };

            let remaining_gas = entry
                .append_operation(llvm::load(
                    context,
                    gas_builtin_ptr,
                    IntegerType::new(context, 128).into(),
                    location,
                    LoadStoreOptions::default(),
                ))
                .result(0)?
                .into();

            entry.append_operation(helper.cond_br(
                context,
                result_tag,
                [1, 0],
                [
                    &[remaining_gas, entry.argument(1)?.into(), payload_err],
                    &[remaining_gas, entry.argument(1)?.into(), payload_ok],
                ],
                location,
            ));
        }
        "set_block_number" => {
            // self.starknet_state.exec_info.block_info.block_number = as_single_input(inputs)?;
        }
        "set_block_timestamp" => {
            // self.starknet_state.exec_info.block_info.block_timestamp = as_single_input(inputs)?;
        }
        "set_caller_address" => {
            // self.starknet_state.exec_info.caller_address = as_single_input(inputs)?;
        }
        "set_contract_address" => {
            // self.starknet_state.exec_info.contract_address = as_single_input(inputs)?;
        }
        "set_version" => {
            // self.starknet_state.exec_info.tx_info.version = as_single_input(inputs)?;
        }
        "set_account_contract_address" => {
            // self.starknet_state.exec_info.tx_info.account_contract_address =
            // as_single_input(inputs)?;
        }
        "set_max_fee" => {
            // self.starknet_state.exec_info.tx_info.max_fee = as_single_input(inputs)?;
        }
        "set_transaction_hash" => {
            // self.starknet_state.exec_info.tx_info.transaction_hash = as_single_input(inputs)?;
        }
        "set_chain_id" => {
            // self.starknet_state.exec_info.tx_info.chain_id = as_single_input(inputs)?;
        }
        "set_nonce" => {
            // self.starknet_state.exec_info.tx_info.nonce = as_single_input(inputs)?;
        }
        "set_signature" => {
            // self.starknet_state.exec_info.tx_info.signature = inputs;
        }
        "pop_log" => {
            // let contract_logs = self.starknet_state.logs.get_mut(&as_single_input(inputs)?);
            // if let Some((keys, data)) =
            //     contract_logs.and_then(|contract_logs| contract_logs.events.pop_front())
            // {
            //     res_segment.write(keys.len())?;
            //     res_segment.write_data(keys.iter())?;
            //     res_segment.write(data.len())?;
            //     res_segment.write_data(data.iter())?;
            // }
        }
        "pop_l2_to_l1_message" => {
            // let contract_logs = self.starknet_state.logs.get_mut(&as_single_input(inputs)?);
            // if let Some((to_address, payload)) = contract_logs
            //     .and_then(|contract_logs| contract_logs.l2_to_l1_messages.pop_front())
            // {
            //     res_segment.write(to_address)?;
            //     res_segment.write(payload.len())?;
            //     res_segment.write_data(payload.iter())?;
            // }
        }
        _ => unimplemented!(), // Err(HintError::CustomHint(Box::from(format!("Unknown cheatcode selector: {selector}"))))?,
    }

    Ok(())
}
