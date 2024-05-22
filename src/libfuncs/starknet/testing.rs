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
    dialect::llvm::{self, LoadStoreOptions},
    ir::{
        attribute::{
            DenseI32ArrayAttribute, DenseI64ArrayAttribute, IntegerAttribute, TypeAttribute,
        },
        operation::OperationBuilder,
        r#type::IntegerType,
        Block, Identifier, Location,
    },
    Context,
};

use crate::{
    block_ext::BlockExt, error::Result, libfuncs::LibfuncHelper, metadata::MetadataStorage,
    starknet::handler::StarknetSyscallHandlerCallbacks,
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
    dbg!("executing build for cheatcode");
    let selector_as_hex = BigIntAsHex {
        value: selector.selector.clone(),
    };
    let selector_bytes = &selector_as_hex.value.to_bytes_be().1;
    let selector_str = std::str::from_utf8(selector_bytes).unwrap();

    dbg!(selector_str);

    // TODO(juanbono): save the selector and the span to memory (check keccak syscall)

    // Extract self pointer.
    // let ptr = entry
    //     .append_operation(llvm::load(
    //         context,
    //         entry.argument(0)?.into(),
    //         llvm::r#type::opaque_pointer(context),
    //         location,
    //         LoadStoreOptions::default(),
    //     ))
    //     .result(0)?
    //     .into();

    // this may be not needed since the self pointer is not passed as arg.

    dbg!("allocating space for return value");
    // TODO(juanbono): Check this
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
                selector.branch_signatures()[0].vars[0].ty.clone(),
                selector.branch_signatures()[0].vars[0].ty.clone(),
            ],
        )?;

    dbg!(result_layout);
    dbg!(result_tag_ty);
    dbg!(result_tag_layout);
    dbg!(&variant_tys);

    let u64_type = IntegerType::new(context, 64);
    let u8_type = IntegerType::new(context, 8);
    let u252_type = IntegerType::new(context, 252);
    let u32_type = IntegerType::new(context, 32);
    let const_1 = entry.const_int_from_type(context, location, 1, u64_type.into())?;

    dbg!("creating result pointer");

    // create the result pointer for the array of felts returned

    // looks like this is wrong
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
                .add_operands(&[const_1])
                .add_results(&[llvm::r#type::opaque_pointer(context)])
                .build()?,
        )
        .result(0)?
        .into();

    // Allocate space and write the current gas.

    dbg!("matching selector");
    match selector_str {
        "set_sequencer_address" => {
            dbg!("generating code for set_sequencer_address");
            // Allocate for the array argument and write the value.
            let array_arg_ptr_ty = llvm::r#type::pointer(
                llvm::r#type::r#struct(
                    context,
                    &[llvm::r#type::r#struct(
                        context,
                        &[
                            llvm::r#type::pointer(u252_type.into(), 0),
                            u32_type.into(),
                            u32_type.into(),
                            u32_type.into(),
                        ],
                        false,
                    )],
                    false,
                ),
                0,
            );
            eprintln!("######## {array_arg_ptr_ty}");
            dbg!("array_arg_ptr");
            let array_arg_ptr = helper
                .init_block()
                .append_operation(
                    OperationBuilder::new("llvm.alloca", location)
                        .add_attributes(&[(
                            Identifier::new(context, "alignment"),
                            IntegerAttribute::new(u64_type.into(), 8).into(),
                        )])
                        .add_operands(&[const_1])
                        .add_results(&[array_arg_ptr_ty])
                        .build()?,
                )
                .result(0)?
                .into();
            dbg!("storing array_arg_ptr");
            entry.append_operation(llvm::store(
                context,
                entry.argument(0)?.into(),
                array_arg_ptr,
                location,
                LoadStoreOptions::default(),
            ));

            dbg!("extracting function pointer");
            // Extract function pointer.
            let fn_ptr_ty = llvm::r#type::function(
                llvm::r#type::void(context),
                &[
                    // TODO(juanbono): check if this args are needed
                    llvm::r#type::opaque_pointer(context), // return_ptr
                    llvm::r#type::opaque_pointer(context), // self
                    // llvm::r#type::pointer(IntegerType::new(context, 128).into(), 0),
                    array_arg_ptr_ty,
                ],
                false,
            );

            dbg!("fn_ptr");
            let fn_ptr = entry
                .append_operation(llvm::get_element_ptr(
                    context,
                    entry.argument(0)?.into(),
                    DenseI32ArrayAttribute::new(
                        context,
                        &[StarknetSyscallHandlerCallbacks::<()>::CHEATCODE.try_into()?],
                    ),
                    llvm::r#type::opaque_pointer(context), // type elem 
                    llvm::r#type::opaque_pointer(context), // type result
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

            dbg!("llvm.call execution");
            entry.append_operation(
                OperationBuilder::new("llvm.call", location)
                    .add_operands(&[
                        fn_ptr,
                        result_ptr,
                        // ptr,
                        // gas_builtin_ptr,
                        // entry.argument(0)?.into(),
                        array_arg_ptr,
                    ])
                    .build()?,
            );

            dbg!("writing result");

            let result = entry
                .append_operation(llvm::load(
                    context,
                    result_ptr,
                    llvm::r#type::r#struct(
                        context,
                        &[
                            result_tag_ty,
                            llvm::r#type::array(
                                u8_type.into(),
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
            dbg!("extracting result_tag");
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

            dbg!("payload_ok");
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
                                    TypeAttribute::new(u8_type.into()).into(),
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
            dbg!("payload_err");
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
                                    TypeAttribute::new(u8_type.into()).into(),
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

            // let remaining_gas = entry
            //     .append_operation(llvm::load(
            //         context,
            //         gas_builtin_ptr,
            //         IntegerType::new(context, 128).into(),
            //         location,
            //         LoadStoreOptions::default(),
            //     ))
            //     .result(0)?
            //     .into();

            dbg!("cond_br");
            // TODO(juanbono) replace with helper.br
            entry.append_operation(helper.cond_br(
                context,
                result_tag,
                [0, 0],
                [
                    // &[remaining_gas, entry.argument(1)?.into(), payload_err],
                    // &[remaining_gas, entry.argument(1)?.into(), payload_ok],
                    &[entry.argument(0)?.into(), payload_err],
                    &[entry.argument(0)?.into(), payload_ok],
                ],
                location,
            ));

            dbg!("compiled sucessfully");
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
