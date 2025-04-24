use super::EvalAction;
use crate::{
    starknet::{Secp256r1Point, StarknetSyscallHandler, U256},
    Value,
};
use cairo_lang_sierra::{
    extensions::{
        consts::SignatureAndConstConcreteLibfunc,
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        lib_func::SignatureOnlyConcreteLibfunc,
        starknet::StarknetConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use num_traits::One;
use smallvec::smallvec;
use starknet_types_core::felt::Felt;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &StarknetConcreteLibfunc,
    args: Vec<Value>,
    syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    match selector {
        StarknetConcreteLibfunc::CallContract(info) => {
            self::eval_call_contract(registry, info, args, syscall_handler)
        }
        StarknetConcreteLibfunc::ClassHashConst(info) => {
            eval_class_hash_const(registry, info, args)
        }
        StarknetConcreteLibfunc::ClassHashTryFromFelt252(info) => {
            eval_class_hash_try_from_felt(registry, info, args)
        }
        StarknetConcreteLibfunc::ClassHashToFelt252(info) => {
            eval_class_hash_to_felt(registry, info, args)
        }
        StarknetConcreteLibfunc::ContractAddressConst(info) => {
            eval_contract_address_const(registry, info, args)
        }
        StarknetConcreteLibfunc::ContractAddressTryFromFelt252(info) => {
            eval_contract_address_try_from_felt(registry, info, args)
        }
        StarknetConcreteLibfunc::ContractAddressToFelt252(info) => {
            eval_contract_address_to_felt(registry, info, args)
        }
        StarknetConcreteLibfunc::StorageRead(info) => {
            eval_storage_read(registry, info, args, syscall_handler)
        }
        StarknetConcreteLibfunc::StorageWrite(info) => {
            eval_storage_write(registry, info, args, syscall_handler)
        }
        StarknetConcreteLibfunc::StorageBaseAddressConst(info) => {
            eval_storage_base_address_const(registry, info, args)
        }
        StarknetConcreteLibfunc::StorageBaseAddressFromFelt252(info) => {
            eval_storage_base_address_from_felt(registry, info, args)
        }
        StarknetConcreteLibfunc::StorageAddressFromBase(info) => {
            eval_storage_address_from_base(registry, info, args)
        }
        StarknetConcreteLibfunc::StorageAddressFromBaseAndOffset(info) => {
            eval_storage_address_from_base_and_offset(registry, info, args)
        }
        StarknetConcreteLibfunc::StorageAddressToFelt252(info) => {
            eval_storage_address_to_felt(registry, info, args)
        }
        StarknetConcreteLibfunc::StorageAddressTryFromFelt252(info) => {
            eval_storage_address_try_from_felt(registry, info, args)
        }
        StarknetConcreteLibfunc::EmitEvent(info) => {
            eval_emit_event(registry, info, args, syscall_handler)
        }
        StarknetConcreteLibfunc::GetBlockHash(info) => {
            eval_get_block_hash(registry, info, args, syscall_handler)
        }
        StarknetConcreteLibfunc::GetExecutionInfo(info) => {
            eval_get_execution_info(registry, info, args, syscall_handler)
        }
        StarknetConcreteLibfunc::GetExecutionInfoV2(info) => {
            eval_get_execution_info_v2(registry, info, args, syscall_handler)
        }
        StarknetConcreteLibfunc::Deploy(info) => eval_deploy(registry, info, args, syscall_handler),
        StarknetConcreteLibfunc::Keccak(info) => eval_keccak(registry, info, args, syscall_handler),
        StarknetConcreteLibfunc::Sha256ProcessBlock(info) => {
            eval_sha256_process_block(registry, info, args, syscall_handler)
        }
        StarknetConcreteLibfunc::Sha256StateHandleInit(info) => {
            eval_sha256_state_handle_init(registry, info, args, syscall_handler)
        }
        StarknetConcreteLibfunc::Sha256StateHandleDigest(info) => {
            eval_sha256_state_handle_digest(registry, info, args, syscall_handler)
        }
        StarknetConcreteLibfunc::LibraryCall(info) => {
            eval_library_call(registry, info, args, syscall_handler)
        }
        StarknetConcreteLibfunc::ReplaceClass(info) => {
            eval_replace_class(registry, info, args, syscall_handler)
        }
        StarknetConcreteLibfunc::SendMessageToL1(info) => {
            eval_send_message_to_l1(registry, info, args, syscall_handler)
        }
        StarknetConcreteLibfunc::Testing(_info) => todo!(),
        StarknetConcreteLibfunc::Secp256(info) => match info {
            cairo_lang_sierra::extensions::starknet::secp256::Secp256ConcreteLibfunc::K1(info) => {
                match info {
                    cairo_lang_sierra::extensions::starknet::secp256::Secp256OpConcreteLibfunc::New(_) => todo!(),
                    cairo_lang_sierra::extensions::starknet::secp256::Secp256OpConcreteLibfunc::Add(_) => todo!(),
                    cairo_lang_sierra::extensions::starknet::secp256::Secp256OpConcreteLibfunc::Mul(_) => todo!(),
                    cairo_lang_sierra::extensions::starknet::secp256::Secp256OpConcreteLibfunc::GetPointFromX(_) => todo!(),
                    cairo_lang_sierra::extensions::starknet::secp256::Secp256OpConcreteLibfunc::GetXy(_) => todo!(),
                }
            }
            cairo_lang_sierra::extensions::starknet::secp256::Secp256ConcreteLibfunc::R1(info) => {
                match info {
                    cairo_lang_sierra::extensions::starknet::secp256::Secp256OpConcreteLibfunc::New(info) => eval_secp_r_new(registry, info, args, syscall_handler),
                    cairo_lang_sierra::extensions::starknet::secp256::Secp256OpConcreteLibfunc::Add(info) => eval_secp_r_add(registry, info, args, syscall_handler),
                    cairo_lang_sierra::extensions::starknet::secp256::Secp256OpConcreteLibfunc::Mul(info) => eval_secp_r_mul(registry, info, args, syscall_handler),
                    cairo_lang_sierra::extensions::starknet::secp256::Secp256OpConcreteLibfunc::GetPointFromX(info) => eval_secp_r_get_point_from_x(registry, info, args, syscall_handler),
                    cairo_lang_sierra::extensions::starknet::secp256::Secp256OpConcreteLibfunc::GetXy(info) => secp_r_get_xy(registry, info, args, syscall_handler),
                }
            }
        },
        StarknetConcreteLibfunc::GetClassHashAt(_) => todo!(),
        StarknetConcreteLibfunc::MetaTxV0(_) => todo!(),
    }
}

fn eval_secp_r_add(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    let [Value::U64(mut gas), system @ Value::Unit, x, y]: [Value; 4] = args.try_into().unwrap()
    else {
        panic!()
    };

    let x = Secp256r1Point::from_value(x);
    let y = Secp256r1Point::from_value(y);

    match syscall_handler.secp256r1_add(x, y, &mut gas) {
        Ok(x) => EvalAction::NormalBranch(0, smallvec![Value::U64(gas), system, x.into_value()]),
        Err(r) => {
            let r = Value::Struct(r.into_iter().map(Value::Felt).collect::<Vec<_>>());
            EvalAction::NormalBranch(1, smallvec![Value::U64(gas), system, r])
        }
    }
}

fn eval_secp_r_mul(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    let [Value::U64(mut gas), system @ Value::Unit, x, n]: [Value; 4] = args.try_into().unwrap()
    else {
        panic!()
    };

    let x = Secp256r1Point::from_value(x);
    let n = U256::from_value(n);

    match syscall_handler.secp256r1_mul(x, n, &mut gas) {
        Ok(x) => EvalAction::NormalBranch(0, smallvec![Value::U64(gas), system, x.into_value()]),
        Err(r) => {
            let r = Value::Struct(r.into_iter().map(Value::Felt).collect::<Vec<_>>());
            EvalAction::NormalBranch(1, smallvec![Value::U64(gas), system, r])
        }
    }
}

fn eval_secp_r_new(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    let [Value::U64(mut gas), system @ Value::Unit, Value::Struct(x), Value::Struct(y)]: [Value;
        4] = args.try_into().unwrap()
    else {
        panic!()
    };

    let Value::U128(x_lo) = x[0] else { panic!() };
    let Value::U128(x_hi) = x[1] else { panic!() };
    let x = U256 { lo: x_lo, hi: x_hi };
    let Value::U128(y_lo) = y[0] else { panic!() };
    let Value::U128(y_hi) = y[1] else { panic!() };
    let y = U256 { lo: y_lo, hi: y_hi };

    match syscall_handler.secp256r1_new(x, y, &mut gas) {
        Ok(p) => {
            let enum_ty = &info.branch_signatures()[0].vars[2].ty;
            let value = match p {
                Some(p) => Value::Enum {
                    self_ty: enum_ty.clone(),
                    index: 0,
                    payload: Box::new(p.into_value()),
                },
                None => Value::Enum {
                    self_ty: enum_ty.clone(),
                    index: 1,
                    payload: Box::new(Value::Unit),
                },
            };

            EvalAction::NormalBranch(0, smallvec![Value::U64(gas), system, value])
        }
        Err(payload) => {
            // get felt type from the error branch array
            let felt_ty = {
                match registry
                    .get_type(&info.branch_signatures()[1].vars[2].ty)
                    .unwrap()
                {
                    CoreTypeConcrete::Array(info) => info.ty.clone(),
                    _ => unreachable!(),
                }
            };

            let value = payload.into_iter().map(Value::Felt).collect::<Vec<_>>();
            EvalAction::NormalBranch(
                1,
                smallvec![
                    Value::U64(gas),
                    system,
                    Value::Array {
                        ty: felt_ty,
                        data: value
                    }
                ],
            )
        }
    }
}

fn eval_secp_r_get_point_from_x(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    let [Value::U64(mut gas), system @ Value::Unit, Value::Struct(x), Value::Enum {
        index: y_parity, ..
    }]: [Value; 4] = args.try_into().unwrap()
    else {
        panic!()
    };

    let Value::U128(x_lo) = x[0] else { panic!() };
    let Value::U128(x_hi) = x[1] else { panic!() };
    let x = U256 { lo: x_lo, hi: x_hi };
    let y_parity = y_parity.is_one();

    match syscall_handler.secp256r1_get_point_from_x(x, y_parity, &mut gas) {
        Ok(p) => {
            let enum_ty = &info.branch_signatures()[0].vars[2].ty;
            let value = match p {
                Some(p) => Value::Enum {
                    self_ty: enum_ty.clone(),
                    index: 0,
                    payload: Box::new(p.into_value()),
                },
                None => Value::Enum {
                    self_ty: enum_ty.clone(),
                    index: 1,
                    payload: Box::new(Value::Unit),
                },
            };

            EvalAction::NormalBranch(0, smallvec![Value::U64(gas), system, value])
        }
        Err(payload) => {
            // get felt type from the error branch array
            let felt_ty = {
                match registry
                    .get_type(&info.branch_signatures()[1].vars[2].ty)
                    .unwrap()
                {
                    CoreTypeConcrete::Array(info) => info.ty.clone(),
                    _ => unreachable!(),
                }
            };

            let value = payload.into_iter().map(Value::Felt).collect::<Vec<_>>();
            EvalAction::NormalBranch(
                1,
                smallvec![
                    Value::U64(gas),
                    system,
                    Value::Array {
                        ty: felt_ty,
                        data: value
                    }
                ],
            )
        }
    }
}

fn secp_r_get_xy(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    let [Value::U64(mut gas), system @ Value::Unit, secp_value]: [Value; 3] =
        args.try_into().unwrap()
    else {
        panic!()
    };

    let secp_value = Secp256r1Point::from_value(secp_value);

    match syscall_handler.secp256r1_get_xy(secp_value, &mut gas) {
        Ok(payload) => {
            let (x, y) = (payload.0.into_value(), payload.1.into_value());
            EvalAction::NormalBranch(0, smallvec![Value::U64(gas), system, x, y])
        }
        Err(payload) => {
            let felt_ty = {
                match registry
                    .get_type(&info.branch_signatures()[1].vars[2].ty)
                    .unwrap()
                {
                    CoreTypeConcrete::Array(info) => info.ty.clone(),
                    _ => unreachable!(),
                }
            };

            let payload = payload.into_iter().map(Value::Felt).collect::<Vec<_>>();
            EvalAction::NormalBranch(
                0,
                smallvec![
                    Value::U64(gas),
                    system,
                    Value::Array {
                        ty: felt_ty,
                        data: payload
                    }
                ],
            )
        }
    }
}

fn eval_class_hash_const(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureAndConstConcreteLibfunc,
    _args: Vec<Value>,
) -> EvalAction {
    EvalAction::NormalBranch(0, smallvec![Value::Felt(info.c.clone().into())])
}

fn eval_storage_base_address_const(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureAndConstConcreteLibfunc,
    _args: Vec<Value>,
) -> EvalAction {
    EvalAction::NormalBranch(0, smallvec![Value::Felt(info.c.clone().into())])
}

fn eval_contract_address_const(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureAndConstConcreteLibfunc,
    _args: Vec<Value>,
) -> EvalAction {
    EvalAction::NormalBranch(0, smallvec![Value::Felt(info.c.clone().into())])
}

fn eval_class_hash_try_from_felt(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    // 2 ** 251 = 3618502788666131106986593281521497120414687020801267626233049500247285301248

    let [range_check @ Value::Unit, Value::Felt(value)]: [Value; 2] = args.try_into().unwrap()
    else {
        panic!()
    };

    if value
        < Felt::from_dec_str(
            "3618502788666131106986593281521497120414687020801267626233049500247285301248",
        )
        .unwrap()
    {
        EvalAction::NormalBranch(0, smallvec![range_check, Value::Felt(value)])
    } else {
        EvalAction::NormalBranch(1, smallvec![range_check])
    }
}

fn eval_contract_address_try_from_felt(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    // 2 ** 251 = 3618502788666131106986593281521497120414687020801267626233049500247285301248

    let [range_check @ Value::Unit, Value::Felt(value)]: [Value; 2] = args.try_into().unwrap()
    else {
        panic!()
    };

    if value
        < Felt::from_dec_str(
            "3618502788666131106986593281521497120414687020801267626233049500247285301248",
        )
        .unwrap()
    {
        EvalAction::NormalBranch(0, smallvec![range_check, Value::Felt(value)])
    } else {
        EvalAction::NormalBranch(1, smallvec![range_check])
    }
}

fn eval_storage_address_try_from_felt(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    // 2 ** 251 = 3618502788666131106986593281521497120414687020801267626233049500247285301248

    let [range_check @ Value::Unit, Value::Felt(value)]: [Value; 2] = args.try_into().unwrap()
    else {
        panic!()
    };

    if value
        < Felt::from_dec_str(
            "3618502788666131106986593281521497120414687020801267626233049500247285301248",
        )
        .unwrap()
    {
        EvalAction::NormalBranch(0, smallvec![range_check, Value::Felt(value)])
    } else {
        EvalAction::NormalBranch(1, smallvec![range_check])
    }
}

fn eval_storage_base_address_from_felt(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [range_check, value] = args.try_into().unwrap();
    EvalAction::NormalBranch(0, smallvec![range_check, value])
}

fn eval_storage_address_to_felt(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value] = args.try_into().unwrap();
    EvalAction::NormalBranch(0, smallvec![value])
}

fn eval_contract_address_to_felt(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value] = args.try_into().unwrap();
    EvalAction::NormalBranch(0, smallvec![value])
}

fn eval_class_hash_to_felt(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value] = args.try_into().unwrap();
    EvalAction::NormalBranch(0, smallvec![value])
}

fn eval_storage_address_from_base(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value] = args.try_into().unwrap();
    EvalAction::NormalBranch(0, smallvec![value])
}

fn eval_storage_address_from_base_and_offset(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Felt(value), Value::U8(offset)]: [Value; 2] = args.try_into().unwrap() else {
        panic!()
    };

    EvalAction::NormalBranch(0, smallvec![Value::Felt(value + Felt::from(offset))])
}

fn eval_call_contract(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    let [Value::U64(mut gas), system, Value::Felt(address), Value::Felt(entry_point_selector), Value::Struct(calldata)]: [Value; 5] =
        args.try_into().unwrap()
    else {
        panic!()
    };

    let [Value::Array {
        ty: _,
        data: calldata,
    }]: [Value; 1] = calldata.try_into().unwrap()
    else {
        panic!()
    };

    let calldata = calldata
        .into_iter()
        .map(|x| match x {
            Value::Felt(x) => x,
            _ => unreachable!(),
        })
        .collect();

    // get felt type from the error branch array
    let felt_ty = {
        match registry
            .get_type(&info.branch_signatures()[1].vars[2].ty)
            .unwrap()
        {
            CoreTypeConcrete::Array(info) => info.ty.clone(),
            _ => unreachable!(),
        }
    };

    let result = syscall_handler.call_contract(address, entry_point_selector, calldata, &mut gas);

    match result {
        Ok(return_values) => EvalAction::NormalBranch(
            0,
            smallvec![
                Value::U64(gas),
                system,
                Value::Struct(vec![Value::Array {
                    ty: felt_ty,
                    data: return_values
                        .into_iter()
                        .map(Value::Felt)
                        .collect::<Vec<_>>(),
                }])
            ],
        ),
        Err(e) => EvalAction::NormalBranch(
            1,
            smallvec![
                Value::U64(gas),
                system,
                Value::Array {
                    ty: felt_ty,
                    data: e.into_iter().map(Value::Felt).collect::<Vec<_>>(),
                }
            ],
        ),
    }
}

fn eval_storage_read(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    let [Value::U64(mut gas), system, Value::U32(address_domain), Value::Felt(storage_key)]: [Value;
        4] = args.try_into().unwrap()
    else {
        panic!()
    };
    let error_felt_ty = {
        match registry
            .get_type(&info.branch_signatures()[1].vars[2].ty)
            .unwrap()
        {
            CoreTypeConcrete::Array(info) => info.ty.clone(),
            _ => unreachable!(),
        }
    };

    let result = syscall_handler.storage_read(address_domain, storage_key, &mut gas);

    match result {
        Ok(value) => {
            EvalAction::NormalBranch(0, smallvec![Value::U64(gas), system, Value::Felt(value)])
        }
        Err(e) => EvalAction::NormalBranch(
            1,
            smallvec![
                Value::U64(gas),
                system,
                Value::Array {
                    ty: error_felt_ty,
                    data: e.into_iter().map(Value::Felt).collect::<Vec<_>>(),
                }
            ],
        ),
    }
}

fn eval_storage_write(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    let [Value::U64(mut gas), system, Value::U32(address_domain), Value::Felt(storage_key), Value::Felt(value)]: [Value; 5] = args.try_into().unwrap() else {
        panic!()
    };

    let error_felt_ty = {
        match registry
            .get_type(&info.branch_signatures()[1].vars[2].ty)
            .unwrap()
        {
            CoreTypeConcrete::Array(info) => info.ty.clone(),
            _ => unreachable!(),
        }
    };

    let result = syscall_handler.storage_write(address_domain, storage_key, value, &mut gas);

    match result {
        Ok(_) => EvalAction::NormalBranch(0, smallvec![Value::U64(gas), system]),
        Err(e) => EvalAction::NormalBranch(
            1,
            smallvec![
                Value::U64(gas),
                system,
                Value::Array {
                    ty: error_felt_ty,
                    data: e.into_iter().map(Value::Felt).collect::<Vec<_>>(),
                }
            ],
        ),
    }
}

fn eval_emit_event(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    let [Value::U64(mut gas), system, Value::Struct(key_arr), Value::Struct(data_arr)]: [Value; 4] =
        args.try_into().unwrap()
    else {
        panic!()
    };

    let [Value::Array { ty: _, data: keys }]: [Value; 1] = key_arr.try_into().unwrap() else {
        panic!()
    };

    let [Value::Array { ty: _, data }]: [Value; 1] = data_arr.try_into().unwrap() else {
        panic!()
    };

    let error_felt_ty = {
        match registry
            .get_type(&info.branch_signatures()[1].vars[2].ty)
            .unwrap()
        {
            CoreTypeConcrete::Array(info) => info.ty.clone(),
            _ => unreachable!(),
        }
    };

    let keys = keys
        .into_iter()
        .map(|x| match x {
            Value::Felt(x) => x,
            _ => unreachable!(),
        })
        .collect();
    let data = data
        .into_iter()
        .map(|x| match x {
            Value::Felt(x) => x,
            _ => unreachable!(),
        })
        .collect();

    let result = syscall_handler.emit_event(keys, data, &mut gas);

    match result {
        Ok(_) => EvalAction::NormalBranch(0, smallvec![Value::U64(gas), system]),
        Err(e) => EvalAction::NormalBranch(
            1,
            smallvec![
                Value::U64(gas),
                system,
                Value::Array {
                    ty: error_felt_ty,
                    data: e.into_iter().map(Value::Felt).collect::<Vec<_>>(),
                }
            ],
        ),
    }
}

fn eval_get_block_hash(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    let [Value::U64(mut gas), system, Value::U64(block_number)]: [Value; 3] =
        args.try_into().unwrap()
    else {
        panic!()
    };
    let error_felt_ty = {
        match registry
            .get_type(&info.branch_signatures()[1].vars[2].ty)
            .unwrap()
        {
            CoreTypeConcrete::Array(info) => info.ty.clone(),
            _ => unreachable!(),
        }
    };

    let result = syscall_handler.get_block_hash(block_number, &mut gas);

    match result {
        Ok(res) => {
            EvalAction::NormalBranch(0, smallvec![Value::U64(gas), system, Value::Felt(res)])
        }
        Err(e) => EvalAction::NormalBranch(
            1,
            smallvec![
                Value::U64(gas),
                system,
                Value::Array {
                    ty: error_felt_ty,
                    data: e.into_iter().map(Value::Felt).collect::<Vec<_>>(),
                }
            ],
        ),
    }
}

fn eval_get_execution_info(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    let [Value::U64(mut gas), system]: [Value; 2] = args.try_into().unwrap() else {
        panic!()
    };
    // get felt type from the error branch array
    let felt_ty = {
        match registry
            .get_type(&info.branch_signatures()[1].vars[2].ty)
            .unwrap()
        {
            CoreTypeConcrete::Array(info) => info.ty.clone(),
            _ => unreachable!(),
        }
    };

    let result = syscall_handler.get_execution_info(&mut gas);

    match result {
        Ok(res) => EvalAction::NormalBranch(
            0,
            smallvec![Value::U64(gas), system, res.into_value(felt_ty)],
        ),
        Err(e) => EvalAction::NormalBranch(
            1,
            smallvec![
                Value::U64(gas),
                system,
                Value::Array {
                    ty: felt_ty,
                    data: e.into_iter().map(Value::Felt).collect::<Vec<_>>(),
                }
            ],
        ),
    }
}

fn eval_get_execution_info_v2(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    let [Value::U64(mut gas), system]: [Value; 2] = args.try_into().unwrap() else {
        panic!()
    };
    // get felt type from the error branch array
    let felt_ty = {
        match registry
            .get_type(&info.branch_signatures()[1].vars[2].ty)
            .unwrap()
        {
            CoreTypeConcrete::Array(info) => info.ty.clone(),
            _ => unreachable!(),
        }
    };

    let result = syscall_handler.get_execution_info_v2(&mut gas);

    let mut out_ty = registry
        .get_type(&info.branch_signatures()[0].vars[2].ty)
        .unwrap();
    let mut out_ty_id = &info.branch_signatures()[0].vars[2].ty;

    if let CoreTypeConcrete::Box(inner) = out_ty {
        out_ty_id = &inner.ty;
        out_ty = registry.get_type(&inner.ty).unwrap();
    };

    if let CoreTypeConcrete::Struct(inner) = out_ty {
        out_ty_id = &inner.members[1];
        out_ty = registry.get_type(&inner.members[1]).unwrap();
    };

    if let CoreTypeConcrete::Box(inner) = out_ty {
        out_ty_id = &inner.ty;
        out_ty = registry.get_type(&inner.ty).unwrap();
    };

    if let CoreTypeConcrete::Struct(inner) = out_ty {
        out_ty_id = &inner.members[7];
        out_ty = registry.get_type(&inner.members[7]).unwrap();
    };

    if let CoreTypeConcrete::Struct(inner) = out_ty {
        out_ty_id = &inner.members[0];
        out_ty = registry.get_type(&inner.members[0]).unwrap();
    };
    if let CoreTypeConcrete::Snapshot(inner) = out_ty {
        out_ty_id = &inner.ty;
        out_ty = registry.get_type(&inner.ty).unwrap();
    };
    if let CoreTypeConcrete::Array(inner) = out_ty {
        out_ty_id = &inner.ty;
    };

    match result {
        Ok(res) => EvalAction::NormalBranch(
            0,
            smallvec![
                Value::U64(gas),
                system,
                res.into_value(felt_ty, out_ty_id.clone())
            ],
        ),
        Err(e) => EvalAction::NormalBranch(
            1,
            smallvec![
                Value::U64(gas),
                system,
                Value::Array {
                    ty: felt_ty,
                    data: e.into_iter().map(Value::Felt).collect::<Vec<_>>(),
                }
            ],
        ),
    }
}

fn eval_deploy(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    let [Value::U64(mut gas), system, Value::Felt(class_hash), Value::Felt(contract_address_salt), Value::Struct(calldata), Value::Enum {
        self_ty: _,
        index: deploy_from_zero,
        payload: _,
        ..
    }]: [Value; 6] = args.try_into().unwrap()
    else {
        panic!()
    };

    let deploy_from_zero = deploy_from_zero != 0;

    let [Value::Array {
        ty: _,
        data: calldata,
    }]: [Value; 1] = calldata.try_into().unwrap()
    else {
        panic!()
    };

    let calldata = calldata
        .into_iter()
        .map(|x| match x {
            Value::Felt(x) => x,
            _ => unreachable!(),
        })
        .collect();

    // get felt type from the error branch array
    let felt_ty = {
        match registry
            .get_type(&info.branch_signatures()[1].vars[2].ty)
            .unwrap()
        {
            CoreTypeConcrete::Array(info) => info.ty.clone(),
            _ => unreachable!(),
        }
    };

    let result = syscall_handler.deploy(
        class_hash,
        contract_address_salt,
        calldata,
        deploy_from_zero,
        &mut gas,
    );

    match result {
        Ok((contract_address, return_values)) => EvalAction::NormalBranch(
            0,
            smallvec![
                Value::U64(gas),
                system,
                Value::Felt(contract_address),
                Value::Struct(vec![Value::Array {
                    ty: felt_ty,
                    data: return_values
                        .into_iter()
                        .map(Value::Felt)
                        .collect::<Vec<_>>(),
                }])
            ],
        ),
        Err(e) => EvalAction::NormalBranch(
            1,
            smallvec![
                Value::U64(gas),
                system,
                Value::Array {
                    ty: felt_ty,
                    data: e.into_iter().map(Value::Felt).collect::<Vec<_>>(),
                }
            ],
        ),
    }
}

fn eval_keccak(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    let [Value::U64(mut gas), system, Value::Struct(input)]: [Value; 3] = args.try_into().unwrap()
    else {
        panic!()
    };

    let [Value::Array { ty: _, data: input }]: [Value; 1] = input.try_into().unwrap() else {
        panic!()
    };

    let input = input
        .into_iter()
        .map(|x| match x {
            Value::U64(x) => x,
            _ => unreachable!(),
        })
        .collect();

    let result = syscall_handler.keccak(input, &mut gas);

    // get felt type from the error branch array
    let felt_ty = {
        match registry
            .get_type(&info.branch_signatures()[1].vars[2].ty)
            .unwrap()
        {
            CoreTypeConcrete::Array(info) => info.ty.clone(),
            _ => unreachable!(),
        }
    };

    match result {
        Ok(res) => {
            EvalAction::NormalBranch(0, smallvec![Value::U64(gas), system, res.into_value()])
        }
        Err(e) => EvalAction::NormalBranch(
            1,
            smallvec![
                Value::U64(gas),
                system,
                Value::Array {
                    ty: felt_ty,
                    data: e.into_iter().map(Value::Felt).collect::<Vec<_>>(),
                }
            ],
        ),
    }
}

fn eval_library_call(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    let [Value::U64(mut gas), system, Value::Felt(class_hash), Value::Felt(function_selector), Value::Struct(calldata)]: [Value; 5] =
        args.try_into().unwrap()
    else {
        panic!()
    };

    let [Value::Array {
        ty: _,
        data: calldata,
    }]: [Value; 1] = calldata.try_into().unwrap()
    else {
        panic!()
    };

    let calldata = calldata
        .into_iter()
        .map(|x| match x {
            Value::Felt(x) => x,
            _ => unreachable!(),
        })
        .collect();

    // get felt type from the error branch array
    let felt_ty = {
        match registry
            .get_type(&info.branch_signatures()[1].vars[2].ty)
            .unwrap()
        {
            CoreTypeConcrete::Array(info) => info.ty.clone(),
            _ => unreachable!(),
        }
    };

    let result = syscall_handler.library_call(class_hash, function_selector, calldata, &mut gas);

    match result {
        Ok(return_values) => EvalAction::NormalBranch(
            0,
            smallvec![
                Value::U64(gas),
                system,
                Value::Struct(vec![Value::Array {
                    ty: felt_ty,
                    data: return_values
                        .into_iter()
                        .map(Value::Felt)
                        .collect::<Vec<_>>(),
                }])
            ],
        ),
        Err(e) => EvalAction::NormalBranch(
            1,
            smallvec![
                Value::U64(gas),
                system,
                Value::Array {
                    ty: felt_ty,
                    data: e.into_iter().map(Value::Felt).collect::<Vec<_>>(),
                }
            ],
        ),
    }
}

fn eval_replace_class(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    let [Value::U64(mut gas), system, Value::Felt(class_hash)]: [Value; 3] =
        args.try_into().unwrap()
    else {
        panic!()
    };
    // get felt type from the error branch array
    let felt_ty = {
        match registry
            .get_type(&info.branch_signatures()[1].vars[2].ty)
            .unwrap()
        {
            CoreTypeConcrete::Array(info) => info.ty.clone(),
            _ => unreachable!(),
        }
    };

    let result = syscall_handler.replace_class(class_hash, &mut gas);

    match result {
        Ok(()) => EvalAction::NormalBranch(0, smallvec![Value::U64(gas), system]),
        Err(e) => EvalAction::NormalBranch(
            1,
            smallvec![
                Value::U64(gas),
                system,
                Value::Array {
                    ty: felt_ty,
                    data: e.into_iter().map(Value::Felt).collect::<Vec<_>>(),
                }
            ],
        ),
    }
}

fn eval_send_message_to_l1(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    let [Value::U64(mut gas), system, Value::Felt(address), Value::Struct(payload)]: [Value; 4] =
        args.try_into().unwrap()
    else {
        panic!()
    };

    let [Value::Array {
        ty: _,
        data: payload,
    }]: [Value; 1] = payload.try_into().unwrap()
    else {
        panic!()
    };

    let payload = payload
        .into_iter()
        .map(|x| match x {
            Value::Felt(x) => x,
            _ => unreachable!(),
        })
        .collect();

    // get felt type from the error branch array
    let felt_ty = {
        match registry
            .get_type(&info.branch_signatures()[1].vars[2].ty)
            .unwrap()
        {
            CoreTypeConcrete::Array(info) => info.ty.clone(),
            _ => unreachable!(),
        }
    };

    let result = syscall_handler.send_message_to_l1(address, payload, &mut gas);

    match result {
        Ok(()) => EvalAction::NormalBranch(0, smallvec![Value::U64(gas), system]),
        Err(e) => EvalAction::NormalBranch(
            1,
            smallvec![
                Value::U64(gas),
                system,
                Value::Array {
                    ty: felt_ty,
                    data: e.into_iter().map(Value::Felt).collect::<Vec<_>>(),
                }
            ],
        ),
    }
}

fn eval_sha256_state_handle_init(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    _syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    let [value]: [Value; 1] = args.try_into().unwrap();

    EvalAction::NormalBranch(0, smallvec![value])
}

fn eval_sha256_state_handle_digest(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    _syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    let [value]: [Value; 1] = args.try_into().unwrap();

    EvalAction::NormalBranch(0, smallvec![value])
}

fn eval_sha256_process_block(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    syscall_handler: &mut impl StarknetSyscallHandler,
) -> EvalAction {
    let [Value::U64(mut gas), system, Value::Struct(prev_state), Value::Struct(current_block)]: [Value; 4] = args.try_into().unwrap() else {
        panic!()
    };

    let prev_state: [u32; 8] = prev_state
        .into_iter()
        .map(|v| {
            let Value::U32(v) = v else { panic!() };
            v
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let current_block: [u32; 16] = current_block
        .into_iter()
        .map(|v| {
            let Value::U32(v) = v else { panic!() };
            v
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    // get felt type from the error branch array
    let felt_ty = {
        match registry
            .get_type(&info.branch_signatures()[1].vars[2].ty)
            .unwrap()
        {
            CoreTypeConcrete::Array(info) => info.ty.clone(),
            _ => unreachable!(),
        }
    };

    match syscall_handler.sha256_process_block(prev_state, current_block, &mut gas) {
        Ok(payload) => {
            let payload = payload.into_iter().map(Value::U32).collect::<Vec<_>>();
            EvalAction::NormalBranch(
                0,
                smallvec![Value::U64(gas), system, Value::Struct(payload)],
            )
        }
        Err(payload) => EvalAction::NormalBranch(
            1,
            smallvec![
                Value::U64(gas),
                system,
                Value::Array {
                    ty: felt_ty,
                    data: payload.into_iter().map(Value::Felt).collect::<Vec<_>>(),
                }
            ],
        ),
    }
}
