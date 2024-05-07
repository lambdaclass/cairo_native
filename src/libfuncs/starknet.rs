//! # Starknet libfuncs

use super::LibfuncHelper;
use crate::{
    block_ext::BlockExt, error::Result, metadata::MetadataStorage,
    starknet::handler::StarknetSyscallHandlerCallbacks, types::felt252::PRIME,
    utils::get_integer_layout,
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
        ods,
    },
    ir::{
        attribute::{DenseI32ArrayAttribute, TypeAttribute},
        operation::OperationBuilder,
        r#type::IntegerType,
        Attribute, Block, Identifier, Location, Value, ValueLike,
    },
    Context,
};
use num_bigint::{Sign, ToBigUint};

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

#[allow(clippy::too_many_arguments)]
pub fn build_generic_syscall<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
    function_offset: usize,
    arguments: &[Value],
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry
        .append_operation(llvm::load(
            context,
            entry.argument(1)?.into(),
            llvm::r#type::pointer(context, 0),
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

    let result_ptr = helper.init_block().alloca1(
        context,
        location,
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
        Some(result_layout.align()),
    )?;

    // Allocate space and write the current gas.
    let gas_builtin_ptr = helper.init_block().alloca1(
        context,
        location,
        IntegerType::new(context, 128).into(),
        Some(get_integer_layout(128).align()),
    )?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(0)?.into(),
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Extract function pointer.
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(context, &[function_offset.try_into()?]),
            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
            location,
        ))
        .result(0)?
        .into();
    let fn_ptr = entry
        .append_operation(llvm::load(
            context,
            fn_ptr,
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    let mut args = vec![fn_ptr, result_ptr, ptr, gas_builtin_ptr];
    args.extend(arguments);

    entry.append_operation(ods::llvm::call(context, &args, location).into());

    let result = entry.load(
        context,
        location,
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
        None,
    )?;

    let result_tag = entry.extract_value(
        context,
        location,
        result,
        IntegerType::new(context, 1).into(),
        0,
    )?;

    // Load the two variants of the result returned by the syscall handler.
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
                    .add_results(&[llvm::r#type::pointer(context, 0)])
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
                    .add_results(&[llvm::r#type::pointer(context, 0)])
                    .build()?,
            )
            .result(0)?
            .into();
        entry.load(context, location, ptr, variant_tys[1].0, None)?
    };

    let remaining_gas = entry.load(
        context,
        location,
        gas_builtin_ptr,
        IntegerType::new(context, 128).into(),
        None,
    )?;

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

pub fn build_call_contract<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Allocate `address` argument and write the value.
    let address_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        address_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `entry_point_selector` argument and write the value.
    let entry_point_selector_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        entry_point_selector_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `calldata` argument and write the value.
    let calldata_arg_ty = llvm::r#type::r#struct(
        context,
        &[llvm::r#type::r#struct(
            context,
            &[
                llvm::r#type::pointer(context, 0), // ptr to felt
                IntegerType::new(context, 32).into(),
                IntegerType::new(context, 32).into(),
                IntegerType::new(context, 32).into(),
            ],
            false,
        )],
        false,
    );
    let calldata_arg_ptr = helper
        .init_block()
        .alloca1(context, location, calldata_arg_ty, None)?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(4)?.into(),
        calldata_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    build_generic_syscall(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        info,
        StarknetSyscallHandlerCallbacks::<()>::CALL_CONTRACT,
        &[
            address_arg_ptr,
            entry_point_selector_arg_ptr,
            calldata_arg_ptr,
        ],
    )
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
    // Allocate `address` argument and write the value.
    let address_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.store(
        context,
        location,
        address_arg_ptr,
        entry.argument(3)?.into(),
        None,
    );

    build_generic_syscall(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        info,
        StarknetSyscallHandlerCallbacks::<()>::STORAGE_READ,
        &[address_arg_ptr],
    )
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
    // Allocate `address` argument and write the value.
    let address_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.store(
        context,
        location,
        address_arg_ptr,
        entry.argument(3)?.into(),
        None,
    );

    // Allocate `value` argument and write the value.
    let value_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.store(
        context,
        location,
        value_arg_ptr,
        entry.argument(4)?.into(),
        None,
    );

    build_generic_syscall(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        info,
        StarknetSyscallHandlerCallbacks::<()>::STORAGE_WRITE,
        &[address_arg_ptr, value_arg_ptr],
    )
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

    let value = entry.const_int(context, location, value, 252)?;

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
    // Allocate `keys` argument and write the value.
    let keys_arg_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[llvm::r#type::r#struct(
                context,
                &[
                    llvm::r#type::pointer(context, 0), // ptr to felt
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                ],
                false,
            )],
            false,
        ),
        None,
    )?;
    entry.store(
        context,
        location,
        keys_arg_ptr,
        entry.argument(2)?.into(),
        None,
    );

    // Allocate `data` argument and write the value.
    let data_arg_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[llvm::r#type::r#struct(
                context,
                &[
                    llvm::r#type::pointer(context, 0), // ptr to felt
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                ],
                false,
            )],
            false,
        ),
        None,
    )?;
    entry.store(
        context,
        location,
        data_arg_ptr,
        entry.argument(3)?.into(),
        None,
    );

    build_generic_syscall(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        info,
        StarknetSyscallHandlerCallbacks::<()>::EMIT_EVENT,
        &[keys_arg_ptr, data_arg_ptr],
    )
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
    build_generic_syscall(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        info,
        StarknetSyscallHandlerCallbacks::<()>::GET_BLOCK_HASH,
        &[entry.argument(2)?.into()],
    )
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
    build_generic_syscall(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        info,
        StarknetSyscallHandlerCallbacks::<()>::GET_EXECUTION_INFO,
        &[],
    )
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
    build_generic_syscall(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        info,
        StarknetSyscallHandlerCallbacks::<()>::GET_EXECUTION_INFOV2,
        &[],
    )
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
    // Allocate `class_hash` argument and write the value.
    let class_hash_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.store(
        context,
        location,
        class_hash_arg_ptr,
        entry.argument(2)?.into(),
        None,
    );

    // Allocate `entry_point_selector` argument and write the value.
    let contract_address_salt_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.store(
        context,
        location,
        contract_address_salt_arg_ptr,
        entry.argument(3)?.into(),
        None,
    );

    // Allocate `calldata` argument and write the value.
    let calldata_arg_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[llvm::r#type::r#struct(
                context,
                &[
                    llvm::r#type::pointer(context, 0), // ptr to felt
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                ],
                false,
            )],
            false,
        ),
        None,
    )?;
    entry.store(
        context,
        location,
        calldata_arg_ptr,
        entry.argument(4)?.into(),
        None,
    );

    build_generic_syscall(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        info,
        StarknetSyscallHandlerCallbacks::<()>::DEPLOY,
        &[
            class_hash_arg_ptr,
            contract_address_salt_arg_ptr,
            calldata_arg_ptr,
        ],
    )
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
    // Allocate `input` argument and write the value.
    let input_arg_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[
                llvm::r#type::pointer(context, 0), // ptr to u64
                IntegerType::new(context, 32).into(),
                IntegerType::new(context, 32).into(),
                IntegerType::new(context, 32).into(),
            ],
            false,
        ),
        None,
    )?;
    entry.store(
        context,
        location,
        input_arg_ptr,
        entry.argument(2)?.into(),
        None,
    );

    build_generic_syscall(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        info,
        StarknetSyscallHandlerCallbacks::<()>::KECCAK,
        &[input_arg_ptr],
    )
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
    // Allocate `class_hash` argument and write the value.
    let class_hash_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        class_hash_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `entry_point_selector` argument and write the value.
    let function_selector_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        function_selector_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `calldata` argument and write the value.
    let calldata_arg_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[llvm::r#type::r#struct(
                context,
                &[
                    llvm::r#type::pointer(context, 0), // ptr to felt
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                ],
                false,
            )],
            false,
        ),
        None,
    )?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(4)?.into(),
        calldata_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    build_generic_syscall(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        info,
        StarknetSyscallHandlerCallbacks::<()>::LIBRARY_CALL,
        &[
            class_hash_arg_ptr,
            function_selector_arg_ptr,
            calldata_arg_ptr,
        ],
    )
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
    // Allocate `class_hash` argument and write the value.
    let class_hash_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        class_hash_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    build_generic_syscall(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        info,
        StarknetSyscallHandlerCallbacks::<()>::REPLACE_CLASS,
        &[class_hash_arg_ptr],
    )
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
    // Allocate `to_address` argument and write the value.
    let to_address_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        to_address_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `payload` argument and write the value.
    let payload_arg_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[llvm::r#type::r#struct(
                context,
                &[
                    llvm::r#type::pointer(context, 0), // ptr to felt
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                ],
                false,
            )],
            false,
        ),
        None,
    )?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        payload_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    build_generic_syscall(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        info,
        StarknetSyscallHandlerCallbacks::<()>::SEND_MESSAGE_TO_L1,
        &[to_address_arg_ptr, payload_arg_ptr],
    )
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
        run_program_assert_output(&CLASS_HASH_CONST, "run_program", &[], Felt::from(0).into())
    }

    #[test]
    #[cfg_attr(target_arch = "aarch64", ignore = "LLVM code generation bug")]
    fn storage_base_address_from_felt252() {
        run_program_assert_output(
            &STORAGE_BASE_ADDRESS_FROM_FELT252,
            "run_program",
            &[Felt::from(0).into()],
            Felt::from(0).into(),
        );
        run_program_assert_output(
            &STORAGE_BASE_ADDRESS_FROM_FELT252,
            "run_program",
            &[Felt::from(1).into()],
            Felt::from(1).into(),
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
            Felt::from(0).into(),
        );
    }

    #[test]
    fn storage_address_from_base() {
        run_program_assert_output(
            &STORAGE_ADDRESS_FROM_BASE,
            "run_program",
            &[Felt::from(0).into()],
            Felt::from(0).into(),
        );
        run_program_assert_output(
            &STORAGE_ADDRESS_FROM_BASE,
            "run_program",
            &[Felt::from(1).into()],
            Felt::from(1).into(),
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
            &[Felt::from(0).into(), 0u8.into()],
            Felt::from(0).into(),
        );
        run_program_assert_output(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            "run_program",
            &[Felt::from(1).into(), 0u8.into()],
            Felt::from(1).into(),
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
            &[Felt::from(0).into(), 1u8.into()],
            Felt::from(1).into(),
        );
        run_program_assert_output(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            "run_program",
            &[Felt::from(1).into(), 1u8.into()],
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
            &[Felt::from(0).into(), 255u8.into()],
            Felt::from(255).into(),
        );
        run_program_assert_output(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            "run_program",
            &[Felt::from(1).into(), 255u8.into()],
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
            &[Felt::from(0).into()],
            Felt::from(0).into(),
        );
        run_program_assert_output(
            &STORAGE_ADDRESS_TO_FELT252,
            "run_program",
            &[Felt::from(1).into()],
            Felt::from(1).into(),
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
            &[Felt::from(0).into()],
            jit_enum!(0, Felt::from(0).into()),
        );
        run_program_assert_output(
            &STORAGE_ADDRESS_TRY_FROM_FELT252,
            "run_program",
            &[Felt::from(1).into()],
            jit_enum!(0, Felt::from(1).into()),
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
