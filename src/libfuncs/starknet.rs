//! # Starknet libfuncs

use super::LibfuncHelper;
use crate::{
    error::Result,
    ffi::get_struct_field_type_at,
    metadata::MetadataStorage,
    starknet::handler::StarknetSyscallHandlerCallbacks,
    types::felt252::PRIME,
    utils::{get_integer_layout, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        consts::SignatureAndConstConcreteLibfunc,
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        starknet::StarkNetConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        llvm::{self, LoadStoreOptions},
    },
    ir::{
        attribute::{
            DenseI32ArrayAttribute, DenseI64ArrayAttribute, IntegerAttribute, TypeAttribute,
        },
        operation::OperationBuilder,
        r#type::IntegerType,
        Attribute, Block, Identifier, Location, Type, ValueLike,
    },
    Context,
};
use num_bigint::{Sign, ToBigUint};
use std::alloc::Layout;

mod secp256;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &StarkNetConcreteLibfunc,
) -> Result<()> {
    match selector {
        StarkNetConcreteLibfunc::CallContract(info) => {
            build_call_contract(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::ClassHashConst(info) => {
            build_class_hash_const(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::ClassHashTryFromFelt252(info) => {
            build_class_hash_try_from_felt252(
                context, registry, entry, location, helper, metadata, info,
            )
        }
        StarkNetConcreteLibfunc::ClassHashToFelt252(info) => {
            build_class_hash_to_felt252(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::ContractAddressConst(info) => {
            build_contract_address_const(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::ContractAddressTryFromFelt252(info) => {
            build_contract_address_try_from_felt252(
                context, registry, entry, location, helper, metadata, info,
            )
        }
        StarkNetConcreteLibfunc::ContractAddressToFelt252(info) => {
            build_contract_address_to_felt252(
                context, registry, entry, location, helper, metadata, info,
            )
        }
        StarkNetConcreteLibfunc::StorageRead(info) => {
            build_storage_read(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::StorageWrite(info) => {
            build_storage_write(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::StorageBaseAddressConst(info) => build_storage_base_address_const(
            context, registry, entry, location, helper, metadata, info,
        ),
        StarkNetConcreteLibfunc::StorageBaseAddressFromFelt252(info) => {
            build_storage_base_address_from_felt252(
                context, registry, entry, location, helper, metadata, info,
            )
        }
        StarkNetConcreteLibfunc::StorageAddressFromBase(info) => build_storage_address_from_base(
            context, registry, entry, location, helper, metadata, info,
        ),
        StarkNetConcreteLibfunc::StorageAddressFromBaseAndOffset(info) => {
            build_storage_address_from_base_and_offset(
                context, registry, entry, location, helper, metadata, info,
            )
        }
        StarkNetConcreteLibfunc::StorageAddressToFelt252(info) => build_storage_address_to_felt252(
            context, registry, entry, location, helper, metadata, info,
        ),
        StarkNetConcreteLibfunc::StorageAddressTryFromFelt252(info) => {
            build_storage_address_try_from_felt252(
                context, registry, entry, location, helper, metadata, info,
            )
        }
        StarkNetConcreteLibfunc::EmitEvent(info) => {
            build_emit_event(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::GetBlockHash(info) => {
            build_get_block_hash(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::GetExecutionInfo(info) => {
            build_get_execution_info(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::GetExecutionInfoV2(info) => {
            build_get_execution_info_v2(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::Deploy(info) => {
            build_deploy(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::Keccak(info) => {
            build_keccak(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::LibraryCall(info) => {
            build_library_call(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::ReplaceClass(info) => {
            build_replace_class(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::SendMessageToL1(info) => {
            build_send_message_to_l1(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::Secp256(selector) => self::secp256::build(
            context, registry, entry, location, helper, metadata, selector,
        ),
        StarkNetConcreteLibfunc::Testing(_) => todo!("implement starknet testing libfunc"),
    }
}

pub fn build_call_contract<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry
        .append_operation(llvm::load(
            context,
            entry.argument(1)?.into(),
            llvm::r#type::opaque_pointer(context),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let k1 = helper
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
                .add_operands(&[k1])
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
                .add_operands(&[k1])
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

    // Allocate `address` argument and write the value.
    let address_arg_ptr_ty = llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0);
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
                .add_operands(&[k1])
                .add_results(&[address_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        address_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `entry_point_selector` argument and write the value.
    let entry_point_selector_arg_ptr_ty =
        llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0);
    let entry_point_selector_arg_ptr = helper
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
                .add_operands(&[k1])
                .add_results(&[entry_point_selector_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        entry_point_selector_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `calldata` argument and write the value.
    let calldata_arg_ptr_ty = llvm::r#type::pointer(
        llvm::r#type::r#struct(
            context,
            &[llvm::r#type::r#struct(
                context,
                &[
                    llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                ],
                false,
            )],
            false,
        ),
        0,
    );
    let calldata_arg_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(IntegerType::new(context, 64).into(), 8).into(),
                )])
                .add_operands(&[k1])
                .add_results(&[calldata_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(4)?.into(),
        calldata_arg_ptr,
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
            entry_point_selector_arg_ptr_ty,
            calldata_arg_ptr_ty,
        ],
        false,
    );
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::CALL_CONTRACT.try_into()?],
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
                address_arg_ptr,
                entry_point_selector_arg_ptr,
                calldata_arg_ptr,
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
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
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
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
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
    Ok(())
}

pub fn build_class_hash_const<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureAndConstConcreteLibfunc,
) -> Result<()> {
    let value = match info.c.sign() {
        Sign::Minus => PRIME.to_biguint().unwrap() - info.c.to_biguint().unwrap(),
        _ => info.c.to_biguint().unwrap(),
    };

    let value = entry
        .append_operation(arith::constant(
            context,
            Attribute::parse(context, &format!("{value} : i252")).unwrap(),
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

pub fn build_class_hash_to_felt252<'ctx, 'this>(
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

pub fn build_class_hash_try_from_felt252<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let value = entry.argument(1)?.into();

    let limit = entry
        .append_operation(arith::constant(
            context,
            Attribute::parse(
                context,
                "3618502788666131106986593281521497120414687020801267626233049500247285301248 : i252",
            )
            .unwrap(),
            location,
        ))
        .result(0)?
        .into();
    let is_in_range = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Ult,
            value,
            limit,
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(
        context,
        is_in_range,
        [0, 1],
        [&[range_check, value], &[range_check]],
        location,
    ));
    Ok(())
}

pub fn build_contract_address_const<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureAndConstConcreteLibfunc,
) -> Result<()> {
    let value = match info.c.sign() {
        Sign::Minus => PRIME.to_biguint().unwrap() - info.c.to_biguint().unwrap(),
        _ => info.c.to_biguint().unwrap(),
    };

    let value = entry
        .append_operation(arith::constant(
            context,
            Attribute::parse(context, &format!("{value} : i252")).unwrap(),
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

pub fn build_contract_address_try_from_felt252<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let value = entry.argument(1)?.into();

    let limit = entry
        .append_operation(arith::constant(
            context,
            Attribute::parse(
                context,
                "3618502788666131106986593281521497120414687020801267626233049500247285301248 : i252",
            )
            .unwrap(),
            location,
        ))
        .result(0)?
        .into();
    let is_in_range = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Ult,
            value,
            limit,
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(
        context,
        is_in_range,
        [0, 1],
        [&[range_check, value], &[range_check]],
        location,
    ));
    Ok(())
}

pub fn build_contract_address_to_felt252<'ctx, 'this>(
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

pub fn build_storage_read<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry
        .append_operation(llvm::load(
            context,
            entry.argument(1)?.into(),
            llvm::r#type::opaque_pointer(context),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let k1 = helper
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
                .add_operands(&[k1])
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
                .add_operands(&[k1])
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

    // Allocate `address` argument and write the value.
    let address_arg_ptr_ty = llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0);
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
                .add_operands(&[k1])
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
            IntegerType::new(context, 32).into(),
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
                &[StarknetSyscallHandlerCallbacks::<()>::STORAGE_READ.try_into()?],
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
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
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
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
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
    Ok(())
}

pub fn build_storage_write<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry
        .append_operation(llvm::load(
            context,
            entry.argument(1)?.into(),
            llvm::r#type::opaque_pointer(context),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
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
                info.branch_signatures()[1].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let k1 = helper
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
                .add_operands(&[k1])
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
                .add_operands(&[k1])
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

    // Allocate `address` argument and write the value.
    let address_arg_ptr_ty = llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0);
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
                .add_operands(&[k1])
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

    // Allocate `value` argument and write the value.
    let value_arg_ptr_ty = llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0);
    let value_arg_ptr = helper
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
                .add_operands(&[k1])
                .add_results(&[value_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(4)?.into(),
        value_arg_ptr,
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
            IntegerType::new(context, 32).into(),
            address_arg_ptr_ty,
            value_arg_ptr_ty,
        ],
        false,
    );
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::STORAGE_WRITE.try_into()?],
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
                value_arg_ptr,
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
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
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
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
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
    Ok(())
}

pub fn build_storage_base_address_const<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureAndConstConcreteLibfunc,
) -> Result<()> {
    let value = match info.c.sign() {
        Sign::Minus => PRIME.to_biguint().unwrap() - info.c.to_biguint().unwrap(),
        _ => info.c.to_biguint().unwrap(),
    };

    let value = entry
        .append_operation(arith::constant(
            context,
            Attribute::parse(context, &format!("{value} : i252")).unwrap(),
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

pub fn build_storage_base_address_from_felt252<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let k_limit = entry
        .append_operation(arith::constant(
            context,
            Attribute::parse(
                context,
                "3618502788666131106986593281521497120414687020801267626233049500247285300992 : i252",
            )
            .unwrap(),
            location,
        ))
        .result(0)?
        .into();

    let limited_value = entry
        .append_operation(arith::subi(entry.argument(1)?.into(), k_limit, location))
        .result(0)?
        .into();

    let is_within_limit = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Ult,
            entry.argument(1)?.into(),
            k_limit,
            location,
        ))
        .result(0)?
        .into();
    let value = entry
        .append_operation(
            OperationBuilder::new("arith.select", location)
                .add_operands(&[is_within_limit, entry.argument(1)?.into(), limited_value])
                .add_results(&[IntegerType::new(context, 252).into()])
                .build()?,
        )
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[range_check, value], location));
    Ok(())
}

pub fn build_storage_address_from_base<'ctx, 'this>(
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

pub fn build_storage_address_from_base_and_offset<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let offset = entry
        .append_operation(arith::extui(
            entry.argument(1)?.into(),
            entry.argument(0)?.r#type(),
            location,
        ))
        .result(0)?
        .into();
    let addr = entry
        .append_operation(arith::addi(entry.argument(0)?.into(), offset, location))
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[addr], location));
    Ok(())
}

pub fn build_storage_address_to_felt252<'ctx, 'this>(
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

pub fn build_storage_address_try_from_felt252<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let value = entry.argument(1)?.into();

    let limit = entry
        .append_operation(arith::constant(
            context,
            Attribute::parse(
                context,
                "3618502788666131106986593281521497120414687020801267626233049500247285301248 : i252",
            )
            .unwrap(),
            location,
        ))
        .result(0)?
        .into();
    let is_in_range = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Ult,
            value,
            limit,
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(
        context,
        is_in_range,
        [0, 1],
        [&[range_check, value], &[range_check]],
        location,
    ));
    Ok(())
}

pub fn build_emit_event<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry
        .append_operation(llvm::load(
            context,
            entry.argument(1)?.into(),
            llvm::r#type::opaque_pointer(context),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
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
                info.branch_signatures()[1].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let k1 = helper
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
                .add_operands(&[k1])
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
                .add_operands(&[k1])
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

    // Allocate `keys` argument and write the value.
    let keys_arg_ptr_ty = llvm::r#type::pointer(
        llvm::r#type::r#struct(
            context,
            &[llvm::r#type::r#struct(
                context,
                &[
                    llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                ],
                false,
            )],
            false,
        ),
        0,
    );
    let keys_arg_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(IntegerType::new(context, 64).into(), 8).into(),
                )])
                .add_operands(&[k1])
                .add_results(&[keys_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        keys_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `data` argument and write the value.
    let data_arg_ptr_ty = llvm::r#type::pointer(
        llvm::r#type::r#struct(
            context,
            &[llvm::r#type::r#struct(
                context,
                &[
                    llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                ],
                false,
            )],
            false,
        ),
        0,
    );
    let data_arg_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(IntegerType::new(context, 64).into(), 8).into(),
                )])
                .add_operands(&[k1])
                .add_results(&[data_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        data_arg_ptr,
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
            keys_arg_ptr_ty,
            data_arg_ptr_ty,
        ],
        false,
    );
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::EMIT_EVENT.try_into()?],
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
                keys_arg_ptr,
                data_arg_ptr,
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
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
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
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
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
    Ok(())
}

pub fn build_get_block_hash<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry
        .append_operation(llvm::load(
            context,
            entry.argument(1)?.into(),
            llvm::r#type::opaque_pointer(context),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let k1 = helper
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
                .add_operands(&[k1])
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
                .add_operands(&[k1])
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

    // Extract function pointer.
    let fn_ptr_ty = llvm::r#type::function(
        llvm::r#type::void(context),
        &[
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::pointer(IntegerType::new(context, 128).into(), 0),
            IntegerType::new(context, 64).into(),
        ],
        false,
    );
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::GET_BLOCK_HASH.try_into()?],
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
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
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
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
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
    Ok(())
}

pub fn build_get_execution_info<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry
        .append_operation(llvm::load(
            context,
            entry.argument(1)?.into(),
            llvm::r#type::opaque_pointer(context),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let k1 = helper
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
                .add_operands(&[k1])
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
                .add_operands(&[k1])
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

    // Extract function pointer.
    let fn_ptr_ty = llvm::r#type::function(
        llvm::r#type::void(context),
        &[
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::pointer(IntegerType::new(context, 128).into(), 0),
        ],
        false,
    );
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::GET_EXECUTION_INFO.try_into()?],
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
            .add_operands(&[fn_ptr, result_ptr, ptr, gas_builtin_ptr])
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
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
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
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
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
    Ok(())
}

pub fn build_get_execution_info_v2<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry
        .append_operation(llvm::load(
            context,
            entry.argument(1)?.into(),
            llvm::r#type::opaque_pointer(context),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let k1 = helper
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
                .add_operands(&[k1])
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
                .add_operands(&[k1])
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

    // Extract function pointer.
    let fn_ptr_ty = llvm::r#type::function(
        llvm::r#type::void(context),
        &[
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::opaque_pointer(context),
            llvm::r#type::pointer(IntegerType::new(context, 128).into(), 0),
        ],
        false,
    );
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::GET_EXECUTION_INFOV2.try_into()?],
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
            .add_operands(&[fn_ptr, result_ptr, ptr, gas_builtin_ptr])
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
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
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
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
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
    Ok(())
}

pub fn build_deploy<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry
        .append_operation(llvm::load(
            context,
            entry.argument(1)?.into(),
            llvm::r#type::opaque_pointer(context),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) = {
        let tag_layout = get_integer_layout(1);
        let tag_ty: Type = IntegerType::new(context, 1).into();

        let mut layout = tag_layout;
        let output = [
            {
                let (p0_ty, p0_layout) = registry.build_type_with_layout(
                    context,
                    helper,
                    registry,
                    metadata,
                    &info.branch_signatures()[0].vars[2].ty,
                )?;
                let (p1_ty, p1_layout) = registry.build_type_with_layout(
                    context,
                    helper,
                    registry,
                    metadata,
                    &info.branch_signatures()[0].vars[3].ty,
                )?;

                let payload_ty = llvm::r#type::r#struct(context, &[p0_ty, p1_ty], false);
                let payload_layout = p0_layout.extend(p1_layout)?.0;

                let full_layout = tag_layout.extend(payload_layout)?.0;
                layout = Layout::from_size_align(
                    layout.size().max(full_layout.size()),
                    layout.align().max(full_layout.align()),
                )?;

                (payload_ty, payload_layout)
            },
            {
                let (payload_ty, payload_layout) = registry.build_type_with_layout(
                    context,
                    helper,
                    registry,
                    metadata,
                    &info.branch_signatures()[1].vars[2].ty,
                )?;

                let full_layout = tag_layout.extend(payload_layout)?.0;
                layout = Layout::from_size_align(
                    layout.size().max(full_layout.size()),
                    layout.align().max(full_layout.align()),
                )?;

                (payload_ty, payload_layout)
            },
        ];

        (layout, (tag_ty, tag_layout), output)
    };

    let k1 = helper
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
                .add_operands(&[k1])
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
                        get_integer_layout(64).align().try_into()?,
                    )
                    .into(),
                )])
                .add_operands(&[k1])
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

    // Allocate `class_hash` argument and write the value.
    let class_hash_arg_ptr_ty = llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0);
    let class_hash_arg_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        IntegerType::new(context, 64).into(),
                        get_integer_layout(252).align().try_into()?,
                    )
                    .into(),
                )])
                .add_operands(&[k1])
                .add_results(&[class_hash_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        class_hash_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `entry_point_selector` argument and write the value.
    let contract_address_salt_arg_ptr_ty =
        llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0);
    let contract_address_salt_arg_ptr = helper
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
                .add_operands(&[k1])
                .add_results(&[contract_address_salt_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        contract_address_salt_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `calldata` argument and write the value.
    let calldata_arg_ptr_ty = llvm::r#type::pointer(
        llvm::r#type::r#struct(
            context,
            &[llvm::r#type::r#struct(
                context,
                &[
                    llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                ],
                false,
            )],
            false,
        ),
        0,
    );
    let calldata_arg_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(IntegerType::new(context, 64).into(), 8).into(),
                )])
                .add_operands(&[k1])
                .add_results(&[calldata_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(4)?.into(),
        calldata_arg_ptr,
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
            class_hash_arg_ptr_ty,
            contract_address_salt_arg_ptr_ty,
            calldata_arg_ptr_ty,
            IntegerType::new(context, 1).into(),
        ],
        false,
    );
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::DEPLOY.try_into()?],
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
                class_hash_arg_ptr,
                contract_address_salt_arg_ptr,
                calldata_arg_ptr,
                entry
                    .append_operation(llvm::extract_value(
                        context,
                        entry.argument(5)?.into(),
                        DenseI64ArrayAttribute::new(context, &[0]),
                        IntegerType::new(context, 1).into(),
                        location,
                    ))
                    .result(0)?
                    .into(),
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
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
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
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
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

    entry.append_operation(
        helper.cond_br(
            context,
            result_tag,
            [1, 0],
            [
                &[remaining_gas, entry.argument(1)?.into(), payload_err],
                &[
                    remaining_gas,
                    entry.argument(1)?.into(),
                    entry
                        .append_operation(llvm::extract_value(
                            context,
                            payload_ok,
                            DenseI64ArrayAttribute::new(context, &[0]),
                            get_struct_field_type_at(&variant_tys[0].0, 0),
                            location,
                        ))
                        .result(0)?
                        .into(),
                    entry
                        .append_operation(llvm::extract_value(
                            context,
                            payload_ok,
                            DenseI64ArrayAttribute::new(context, &[1]),
                            get_struct_field_type_at(&variant_tys[0].0, 1),
                            location,
                        ))
                        .result(0)?
                        .into(),
                ],
            ],
            location,
        ),
    );
    Ok(())
}

pub fn build_keccak<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry
        .append_operation(llvm::load(
            context,
            entry.argument(1)?.into(),
            llvm::r#type::opaque_pointer(context),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let k1 = helper
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
                .add_operands(&[k1])
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
                .add_operands(&[k1])
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

    // Allocate `input` argument and write the value.
    let input_arg_ptr_ty = llvm::r#type::pointer(
        llvm::r#type::r#struct(
            context,
            &[
                llvm::r#type::pointer(IntegerType::new(context, 64).into(), 0),
                IntegerType::new(context, 32).into(),
                IntegerType::new(context, 32).into(),
                IntegerType::new(context, 32).into(),
            ],
            false,
        ),
        0,
    );
    let input_arg_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(IntegerType::new(context, 64).into(), 8).into(),
                )])
                .add_operands(&[k1])
                .add_results(&[input_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        input_arg_ptr,
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
            input_arg_ptr_ty,
        ],
        false,
    );
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::KECCAK.try_into()?],
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
            .add_operands(&[fn_ptr, result_ptr, ptr, gas_builtin_ptr, input_arg_ptr])
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
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
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
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
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
    Ok(())
}

pub fn build_library_call<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry
        .append_operation(llvm::load(
            context,
            entry.argument(1)?.into(),
            llvm::r#type::opaque_pointer(context),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let k1 = helper
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
                .add_operands(&[k1])
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
                .add_operands(&[k1])
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

    // Allocate `class_hash` argument and write the value.
    let class_hash_arg_ptr_ty = llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0);
    let class_hash_arg_ptr = helper
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
                .add_operands(&[k1])
                .add_results(&[class_hash_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        class_hash_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `entry_point_selector` argument and write the value.
    let function_selector_arg_ptr_ty =
        llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0);
    let function_selector_arg_ptr = helper
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
                .add_operands(&[k1])
                .add_results(&[function_selector_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        function_selector_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `calldata` argument and write the value.
    let calldata_arg_ptr_ty = llvm::r#type::pointer(
        llvm::r#type::r#struct(
            context,
            &[llvm::r#type::r#struct(
                context,
                &[
                    llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                ],
                false,
            )],
            false,
        ),
        0,
    );
    let calldata_arg_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(IntegerType::new(context, 64).into(), 8).into(),
                )])
                .add_operands(&[k1])
                .add_results(&[calldata_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(4)?.into(),
        calldata_arg_ptr,
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
            class_hash_arg_ptr_ty,
            function_selector_arg_ptr_ty,
            calldata_arg_ptr_ty,
        ],
        false,
    );
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::LIBRARY_CALL.try_into()?],
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
                class_hash_arg_ptr,
                function_selector_arg_ptr,
                calldata_arg_ptr,
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
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
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
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
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
    Ok(())
}

pub fn build_replace_class<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry
        .append_operation(llvm::load(
            context,
            entry.argument(1)?.into(),
            llvm::r#type::opaque_pointer(context),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
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
                info.branch_signatures()[1].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let k1 = helper
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
                .add_operands(&[k1])
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
                .add_operands(&[k1])
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

    // Allocate `class_hash` argument and write the value.
    let class_hash_arg_ptr_ty = llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0);
    let class_hash_arg_ptr = helper
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
                .add_operands(&[k1])
                .add_results(&[class_hash_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        class_hash_arg_ptr,
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
            class_hash_arg_ptr_ty,
        ],
        false,
    );
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::REPLACE_CLASS.try_into()?],
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
            .add_operands(&[fn_ptr, result_ptr, ptr, gas_builtin_ptr, class_hash_arg_ptr])
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
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
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
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
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
    Ok(())
}

pub fn build_send_message_to_l1<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry
        .append_operation(llvm::load(
            context,
            entry.argument(1)?.into(),
            llvm::r#type::opaque_pointer(context),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
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
                info.branch_signatures()[1].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let k1 = helper
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
                .add_operands(&[k1])
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
                .add_operands(&[k1])
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

    // Allocate `to_address` argument and write the value.
    let to_address_arg_ptr_ty = llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0);
    let to_address_arg_ptr = helper
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
                .add_operands(&[k1])
                .add_results(&[to_address_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        to_address_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `payload` argument and write the value.
    let payload_arg_ptr_ty = llvm::r#type::pointer(
        llvm::r#type::r#struct(
            context,
            &[llvm::r#type::r#struct(
                context,
                &[
                    llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                ],
                false,
            )],
            false,
        ),
        0,
    );
    let payload_arg_ptr = helper
        .init_block()
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(IntegerType::new(context, 64).into(), 8).into(),
                )])
                .add_operands(&[k1])
                .add_results(&[payload_arg_ptr_ty])
                .build()?,
        )
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        payload_arg_ptr,
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
            to_address_arg_ptr_ty,
            payload_arg_ptr_ty,
        ],
        false,
    );
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::SEND_MESSAGE_TO_L1.try_into()?],
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
                to_address_arg_ptr,
                payload_arg_ptr,
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
                                &[result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?],
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
                                &[result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?],
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
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::{jit_enum, jit_struct, load_cairo, run_program_assert_output};
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;
    use starknet_types_core::felt::Felt;

    lazy_static! {
        static ref STORAGE_BASE_ADDRESS_FROM_FELT252: (String, Program) = load_cairo! {
            use starknet::storage_access::{StorageBaseAddress, storage_base_address_from_felt252};

            fn run_program(value: felt252) -> StorageBaseAddress {
                storage_base_address_from_felt252(value)
            }
        };
        static ref STORAGE_ADDRESS_FROM_BASE: (String, Program) = load_cairo! {
            use starknet::storage_access::{StorageAddress, StorageBaseAddress, storage_address_from_base};

            fn run_program(value: StorageBaseAddress) -> StorageAddress {
                storage_address_from_base(value)
            }
        };
        static ref STORAGE_ADDRESS_FROM_BASE_AND_OFFSET: (String, Program) = load_cairo! {
            use starknet::storage_access::{StorageAddress, StorageBaseAddress, storage_address_from_base_and_offset};

            fn run_program(addr: StorageBaseAddress, offset: u8) -> StorageAddress {
                storage_address_from_base_and_offset(addr, offset)
            }
        };
        static ref STORAGE_ADDRESS_TO_FELT252: (String, Program) = load_cairo! {
            use starknet::storage_access::{StorageAddress, storage_address_to_felt252};

            fn run_program(value: StorageAddress) -> felt252 {
                storage_address_to_felt252(value)
            }
        };
        static ref STORAGE_ADDRESS_TRY_FROM_FELT252: (String, Program) = load_cairo! {
            use starknet::storage_access::{StorageAddress, storage_address_try_from_felt252};

            fn run_program(value: felt252) -> Option<StorageAddress> {
                storage_address_try_from_felt252(value)
            }
        };
        static ref CLASS_HASH_CONST: (String, Program) = load_cairo! {
            use starknet::class_hash::{class_hash_const, ClassHash};

            fn run_program() -> ClassHash {
                class_hash_const::<0>()
            }
        };
    }

    #[test]
    fn class_hash_const() {
        run_program_assert_output(&CLASS_HASH_CONST, "run_program", &[], Felt::ZERO.into())
    }

    #[test]
    #[cfg_attr(target_arch = "aarch64", ignore = "LLVM code generation bug")]
    fn storage_base_address_from_felt252() {
        run_program_assert_output(
            &STORAGE_BASE_ADDRESS_FROM_FELT252,
            "run_program",
            &[Felt::ZERO.into()],
            Felt::ZERO.into(),
        );
        run_program_assert_output(
            &STORAGE_BASE_ADDRESS_FROM_FELT252,
            "run_program",
            &[Felt::ONE.into()],
            Felt::ONE.into(),
        );
        run_program_assert_output(
            &STORAGE_BASE_ADDRESS_FROM_FELT252,
            "run_program",
            &[Felt::from(-1).into()],
            Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                .unwrap()
                .into(),
        );
        run_program_assert_output(
            &STORAGE_BASE_ADDRESS_FROM_FELT252,
            "run_program",
            &[Felt::from_dec_str(
                "3618502788666131106986593281521497120414687020801267626233049500247285300992",
            )
            .unwrap()
            .into()],
            Felt::ZERO.into(),
        );
    }

    #[test]
    fn storage_address_from_base() {
        run_program_assert_output(
            &STORAGE_ADDRESS_FROM_BASE,
            "run_program",
            &[Felt::ZERO.into()],
            Felt::ZERO.into(),
        );
        run_program_assert_output(
            &STORAGE_ADDRESS_FROM_BASE,
            "run_program",
            &[Felt::ONE.into()],
            Felt::ONE.into(),
        );
        run_program_assert_output(
            &STORAGE_ADDRESS_FROM_BASE,
            "run_program",
            &[
                Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                    .unwrap()
                    .into(),
            ],
            Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                .unwrap()
                .into(),
        );
    }

    #[test]
    fn storage_address_from_base_and_offset() {
        run_program_assert_output(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            "run_program",
            &[Felt::ZERO.into(), 0u8.into()],
            Felt::ZERO.into(),
        );
        run_program_assert_output(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            "run_program",
            &[Felt::ONE.into(), 0u8.into()],
            Felt::ONE.into(),
        );
        run_program_assert_output(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            "run_program",
            &[
                Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                    .unwrap()
                    .into(),
                0u8.into(),
            ],
            Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                .unwrap()
                .into(),
        );

        run_program_assert_output(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            "run_program",
            &[Felt::ZERO.into(), 1u8.into()],
            Felt::ONE.into(),
        );
        run_program_assert_output(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            "run_program",
            &[Felt::ONE.into(), 1u8.into()],
            Felt::from(2).into(),
        );
        run_program_assert_output(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            "run_program",
            &[
                Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                    .unwrap()
                    .into(),
                1u8.into(),
            ],
            Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719489")
                .unwrap()
                .into(),
        );

        run_program_assert_output(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            "run_program",
            &[Felt::ZERO.into(), 255u8.into()],
            Felt::from(255).into(),
        );
        run_program_assert_output(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            "run_program",
            &[Felt::ONE.into(), 255u8.into()],
            Felt::from(256).into(),
        );

        run_program_assert_output(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            "run_program",
            &[
                Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                    .unwrap()
                    .into(),
                255u8.into(),
            ],
            Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719743")
                .unwrap()
                .into(),
        );
    }

    #[test]
    fn storage_address_to_felt252() {
        run_program_assert_output(
            &STORAGE_ADDRESS_TO_FELT252,
            "run_program",
            &[Felt::ZERO.into()],
            Felt::ZERO.into(),
        );
        run_program_assert_output(
            &STORAGE_ADDRESS_TO_FELT252,
            "run_program",
            &[Felt::ONE.into()],
            Felt::ONE.into(),
        );
        run_program_assert_output(
            &STORAGE_ADDRESS_TO_FELT252,
            "run_program",
            &[
                Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                    .unwrap()
                    .into(),
            ],
            Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                .unwrap()
                .into(),
        );
    }

    #[test]
    fn storage_address_try_from_felt252() {
        run_program_assert_output(
            &STORAGE_ADDRESS_TRY_FROM_FELT252,
            "run_program",
            &[Felt::ZERO.into()],
            jit_enum!(0, Felt::ZERO.into()),
        );
        run_program_assert_output(
            &STORAGE_ADDRESS_TRY_FROM_FELT252,
            "run_program",
            &[Felt::ONE.into()],
            jit_enum!(0, Felt::ONE.into()),
        );
        run_program_assert_output(
            &STORAGE_ADDRESS_TRY_FROM_FELT252,
            "run_program",
            &[
                Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                    .unwrap()
                    .into(),
            ],
            jit_enum!(
                0,
                Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                    .unwrap()
                    .into()
            ),
        );

        run_program_assert_output(
            &STORAGE_ADDRESS_TRY_FROM_FELT252,
            "run_program",
            &[Felt::from(-1).into()],
            jit_enum!(1, jit_struct!()),
        );
    }
}
