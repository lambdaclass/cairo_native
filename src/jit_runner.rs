//! # JIT runner

use crate::{
    error::{
        jit_engine::{
            make_insufficient_gas_error, make_missing_parameter, make_type_builder_error,
            make_unexpected_value_error,
        },
        JitRunnerError,
    },
    execution_result::ContractExecutionResult,
    starknet::{
        handler::{
            CoroutineState, StarkNetSyscallHandlerCallbacks, StarknetRequest, StarknetResponse,
        },
        StarkNetSyscallHandler,
    },
    types::TypeBuilder,
    utils::generate_function_name,
    values::{JitValue, ValueBuilder},
};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        starknet::StarkNetTypeConcrete,
    },
    ids::FunctionId,
    program_registry::ProgramRegistry,
};
use generator::Gn;
use melior::ExecutionEngine;
use std::{
    alloc::Layout,
    iter::once,
    ptr::{addr_of_mut, NonNull},
};
use tracing::debug;

const STACK_SIZE: usize = 65536;

/// The result of the JIT execution.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub remaining_gas: Option<u128>,
    pub return_values: Vec<JitValue>,
}

pub(crate) fn execute_inner(
    engine: &ExecutionEngine,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    function_id: &FunctionId,
    params: &[JitValue],
    required_initial_gas: Option<u128>,
    gas: Option<u128>,
    syscall_handler: Option<&mut dyn StarkNetSyscallHandler>,
) -> Result<ExecutionResult, JitRunnerError> {
    let arena = Bump::new();

    let entry_point = registry.get_function(function_id)?;
    debug!(
        "executing entry_point with the following required parameters: {:?}",
        entry_point.signature.param_types
    );

    let mut params_ptrs: Vec<NonNull<()>> = Vec::new();

    let mut params_it = params.iter();

    let mut syscall_handler_wrapper = None;
    for param_type_id in &entry_point.signature.param_types {
        let ty = registry.get_type(param_type_id)?;

        match ty {
            CoreTypeConcrete::Array(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JitValue::Array(_)) {
                    Err(make_unexpected_value_error("JITValue::Array".to_string()))?;
                }

                params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
            }
            CoreTypeConcrete::Bitwise(_)
            | CoreTypeConcrete::RangeCheck(_)
            | CoreTypeConcrete::SegmentArena(_)
            | CoreTypeConcrete::Pedersen(_)
            | CoreTypeConcrete::Poseidon(_)
            | CoreTypeConcrete::Uint128MulGuarantee(_)
            | CoreTypeConcrete::EcOp(_)
            | CoreTypeConcrete::BuiltinCosts(_) => {
                params_ptrs.push(arena.alloc_layout(Layout::new::<()>()).cast())
            }
            CoreTypeConcrete::Box(_) => {
                todo!()
            }
            CoreTypeConcrete::EcPoint(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JitValue::EcPoint(..)) {
                    Err(make_unexpected_value_error("JITValue::EcPoint".to_string()))?;
                }

                params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
            }
            CoreTypeConcrete::EcState(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JitValue::EcState(..)) {
                    Err(make_unexpected_value_error("JITValue::EcState".to_string()))?;
                }

                params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
            }
            CoreTypeConcrete::Felt252(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JitValue::Felt252(_)) {
                    Err(make_unexpected_value_error("JITValue::Felt252".to_string()))?;
                }

                params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
            }
            CoreTypeConcrete::GasBuiltin(_) => {
                let ptr = arena.alloc_layout(Layout::new::<u128>()).cast();
                let gas_builtin = ptr.cast::<u128>().as_ptr();
                let gas = gas.unwrap_or(0);

                // If program has a required initial gas, check if a gas builtin exists and check if the passed
                // gas was enough, if so, deduct the required gas before execution.
                if let Some(required_initial_gas) = required_initial_gas {
                    if gas < required_initial_gas {
                        return Err(make_insufficient_gas_error(required_initial_gas, gas));
                    }

                    let starting_gas = gas - required_initial_gas;

                    unsafe { gas_builtin.write(starting_gas) };
                }

                params_ptrs.push(ptr);
            }
            CoreTypeConcrete::Uint8(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JitValue::Uint8(_)) {
                    Err(make_unexpected_value_error("JITValue::Uint8".to_string()))?;
                }

                params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
            }
            CoreTypeConcrete::Uint16(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JitValue::Uint16(_)) {
                    Err(make_unexpected_value_error("JITValue::Uint16".to_string()))?;
                }

                params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
            }
            CoreTypeConcrete::Uint32(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JitValue::Uint32(_)) {
                    Err(make_unexpected_value_error("JITValue::Uint32".to_string()))?;
                }

                params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
            }
            CoreTypeConcrete::Uint64(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JitValue::Uint64(_)) {
                    Err(make_unexpected_value_error("JITValue::Uint64".to_string()))?;
                }

                params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
            }
            CoreTypeConcrete::Uint128(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JitValue::Uint128(_)) {
                    Err(make_unexpected_value_error("JITValue::Uint128".to_string()))?;
                }

                params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
            }
            CoreTypeConcrete::Sint8(_) => todo!(),
            CoreTypeConcrete::Sint16(_) => todo!(),
            CoreTypeConcrete::Sint32(_) => todo!(),
            CoreTypeConcrete::Sint64(_) => todo!(),
            CoreTypeConcrete::Sint128(_) => todo!(),
            CoreTypeConcrete::NonZero(info) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                params_ptrs.push(next.to_jit(&arena, registry, &info.ty)?);
            }
            CoreTypeConcrete::Nullable(_) => todo!(),
            CoreTypeConcrete::Uninitialized(_) => todo!(),
            CoreTypeConcrete::Enum(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JitValue::Enum { .. }) {
                    Err(make_unexpected_value_error("JITValue::Enum".to_string()))?;
                }

                params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
            }
            CoreTypeConcrete::Struct(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JitValue::Struct { .. }) {
                    Err(make_unexpected_value_error("JITValue::Struct".to_string()))?;
                }

                params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
            }
            CoreTypeConcrete::Felt252Dict(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JitValue::Felt252Dict { .. }) {
                    Err(make_unexpected_value_error(
                        "JITValue::Felt252Dict".to_string(),
                    ))?;
                }

                params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
            }
            CoreTypeConcrete::Felt252DictEntry(_) => todo!(),
            CoreTypeConcrete::SquashedFelt252Dict(_) => todo!(),
            CoreTypeConcrete::Span(_) => {
                todo!()
            }
            CoreTypeConcrete::StarkNet(selector) => {
                match selector {
                    StarkNetTypeConcrete::StorageAddress(_)
                    | StarkNetTypeConcrete::StorageBaseAddress(_)
                    | StarkNetTypeConcrete::ContractAddress(_)
                    | StarkNetTypeConcrete::ClassHash(_) => {
                        let next = params_it
                            .next()
                            .ok_or_else(|| make_missing_parameter(param_type_id))?;

                        if !matches!(next, JitValue::Felt252(_)) {
                            Err(make_unexpected_value_error("JITValue::Felt252".to_string()))?;
                        }

                        params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
                    }
                    StarkNetTypeConcrete::Secp256Point(_) => todo!(),
                    StarkNetTypeConcrete::System(_) => {
                        let mut syscall_handler = syscall_handler_wrapper
                            .get_or_insert_with(StarkNetSyscallHandlerCallbacks::new);

                        params_ptrs
                            .push(*arena.alloc(
                                NonNull::new(addr_of_mut!(syscall_handler)).unwrap().cast(),
                            ));
                    }
                };
            }
            CoreTypeConcrete::Snapshot(_) => todo!(),
            CoreTypeConcrete::Bytes31(_) => todo!(),
        }
    }

    let mut complex_results = entry_point.signature.ret_types.len() > 1;
    let (layout, offsets) = entry_point.signature.ret_types.iter().try_fold(
        (Option::<Layout>::None, Vec::new()),
        |(acc, mut offsets), id| {
            let ty = registry.get_type(id)?;
            let ty_layout = ty.layout(registry).map_err(make_type_builder_error(id))?;

            let (layout, offset) = match acc {
                Some(layout) => layout.extend(ty_layout)?,
                None => (ty_layout, 0),
            };

            offsets.push(offset);
            complex_results |= ty.is_complex();

            Result::<_, JitRunnerError>::Ok((Some(layout), offsets))
        },
    )?;

    let layout = layout.unwrap_or(Layout::new::<()>());
    let ret_ptr = arena.alloc_layout(layout).cast::<()>();

    let function_name = generate_function_name(function_id);
    let mut io_pointers = if complex_results {
        let ret_ptr_ptr = arena.alloc(ret_ptr) as *mut NonNull<()>;
        once(ret_ptr_ptr as *mut ())
            .chain(params_ptrs.into_iter().map(NonNull::as_ptr))
            .collect::<Vec<_>>()
    } else {
        params_ptrs
            .into_iter()
            .map(NonNull::as_ptr)
            .chain(once(ret_ptr.as_ptr()))
            .collect::<Vec<_>>()
    };

    match syscall_handler_wrapper {
        Some(mut syscall_handler_wrapper) => {
            let mut coroutine =
                Gn::<StarknetResponse>::new_scoped_opt_local(STACK_SIZE, |mut scope| unsafe {
                    syscall_handler_wrapper.scope = addr_of_mut!(scope) as usize as *mut _;
                    CoroutineState::Finished(engine.invoke_packed(&function_name, &mut io_pointers))
                });

            let syscall_handler = syscall_handler.unwrap();
            loop {
                match coroutine.resume().unwrap() {
                    CoroutineState::Request(request) => match request {
                        StarknetRequest::GetBlockHash {
                            mut gas,
                            block_number,
                        } => {
                            let result = syscall_handler.get_block_hash(block_number, &mut gas);
                            coroutine.set_para(StarknetResponse::GetBlockHash { gas, result });
                        }
                        StarknetRequest::GetExecutionInfo { mut gas } => {
                            let result = syscall_handler.get_execution_info(&mut gas);
                            coroutine.set_para(StarknetResponse::GetExecutionInfo { gas, result });
                        }
                        StarknetRequest::Deploy {
                            mut gas,
                            class_hash,
                            contract_address_salt,
                            calldata,
                            deploy_from_zero,
                        } => {
                            let result = syscall_handler.deploy(
                                class_hash,
                                contract_address_salt,
                                &calldata,
                                deploy_from_zero,
                                &mut gas,
                            );
                            coroutine.set_para(StarknetResponse::Deploy { gas, result });
                        }
                        StarknetRequest::ReplaceClass {
                            mut gas,
                            class_hash,
                        } => {
                            let result = syscall_handler.replace_class(class_hash, &mut gas);
                            coroutine.set_para(StarknetResponse::ReplaceClass { gas, result });
                        }
                        StarknetRequest::LibraryCall {
                            mut gas,
                            class_hash,
                            function_selector,
                            calldata,
                        } => {
                            let result = syscall_handler.library_call(
                                class_hash,
                                function_selector,
                                &calldata,
                                &mut gas,
                            );
                            coroutine.set_para(StarknetResponse::LibraryCall { gas, result });
                        }
                        StarknetRequest::CallContract {
                            mut gas,
                            address,
                            entry_point_selector,
                            calldata,
                        } => {
                            let result = syscall_handler.call_contract(
                                address,
                                entry_point_selector,
                                &calldata,
                                &mut gas,
                            );
                            coroutine.set_para(StarknetResponse::CallContract { gas, result });
                        }
                        StarknetRequest::StorageRead {
                            mut gas,
                            address_domain,
                            address,
                        } => {
                            let result =
                                syscall_handler.storage_read(address_domain, address, &mut gas);
                            coroutine.set_para(StarknetResponse::StorageRead { gas, result });
                        }
                        StarknetRequest::StorageWrite {
                            mut gas,
                            address_domain,
                            address,
                            value,
                        } => {
                            let result = syscall_handler.storage_write(
                                address_domain,
                                address,
                                value,
                                &mut gas,
                            );
                            coroutine.set_para(StarknetResponse::StorageWrite { gas, result });
                        }
                        StarknetRequest::EmitEvent {
                            mut gas,
                            keys,
                            data,
                        } => {
                            let result = syscall_handler.emit_event(&keys, &data, &mut gas);
                            coroutine.set_para(StarknetResponse::EmitEvent { gas, result });
                        }
                        StarknetRequest::SendMessageToL1 {
                            mut gas,
                            to_address,
                            payload,
                        } => {
                            let result =
                                syscall_handler.send_message_to_l1(to_address, &payload, &mut gas);
                            coroutine.set_para(StarknetResponse::SendMessageToL1 { gas, result });
                        }
                        StarknetRequest::Keccak { mut gas, input } => {
                            let result = syscall_handler.keccak(&input, &mut gas);
                            coroutine.set_para(StarknetResponse::Keccak { gas, result });
                        }
                    },
                    CoroutineState::Finished(result) => {
                        assert!(coroutine.is_done());
                        break result;
                    }
                }
            }?
        }
        None => unsafe { engine.invoke_packed(&function_name, &mut io_pointers)? },
    }

    let mut returns = Vec::new();

    let mut remaining_gas = None;

    for (type_id, offset) in entry_point.signature.ret_types.iter().zip(offsets) {
        let ty = registry.get_type(type_id)?;

        let ptr = NonNull::new(((ret_ptr.as_ptr() as usize) + offset) as *mut _).unwrap();

        match ty {
            CoreTypeConcrete::GasBuiltin(_) => {
                remaining_gas = Some(*unsafe { ptr.cast::<u128>().as_ref() });
            }
            CoreTypeConcrete::Pedersen(_)
            | CoreTypeConcrete::Poseidon(_)
            | CoreTypeConcrete::Bitwise(_)
            | CoreTypeConcrete::BuiltinCosts(_)
            | CoreTypeConcrete::RangeCheck(_)
            | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::System(_))
            | CoreTypeConcrete::EcOp(_)
            | CoreTypeConcrete::SegmentArena(_) => {
                // ignore returned builtins
            }
            _ => {
                let value = JitValue::from_jit(ptr, type_id, registry);
                returns.push(value);
            }
        };
    }

    Ok(ExecutionResult {
        remaining_gas,
        return_values: returns,
    })
}

/// Execute a function on an engine loaded with a Sierra program.
///
/// The JIT execution of a Sierra program requires an [`ExecutionEngine`] already configured with
/// the compiled module. This has been designed this way because it allows reusing the engine, as
/// opposed to building a different engine every time a function is called and therefore losing all
/// potential optimizations that are already present.
///
/// The registry is needed to convert the params and return values into and from the JIT ABI. Check
/// out [the values module](crate::values) for more information about the de/serialization process.
///
/// The function's arguments and return values are passed around using [`JITValue`].
pub fn execute(
    engine: &ExecutionEngine,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    function_id: &FunctionId,
    params: &[JitValue],
    required_initial_gas: Option<u128>,
    gas: Option<u128>,
) -> Result<ExecutionResult, JitRunnerError> {
    execute_inner(
        engine,
        registry,
        function_id,
        params,
        required_initial_gas,
        gas,
        None,
    )
}

/// Utility function to make it easier to execute contracts. Please see the [`execute`] for more information about program execution.
///
/// To call a contract, the calldata needs to be inside a struct inside a array, this helper function does that for you.
///
/// The calldata passed should all be felt252 values.
pub fn execute_contract<T>(
    engine: &ExecutionEngine,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    function_id: &FunctionId,
    calldata: &[JitValue],
    required_initial_gas: Option<u128>,
    gas: Option<u128>,
    syscall_handler: &mut T,
) -> Result<ContractExecutionResult, JitRunnerError>
where
    T: StarkNetSyscallHandler,
{
    let params = vec![JitValue::Struct {
        fields: vec![JitValue::Array(calldata.to_vec())],
        debug_name: None,
    }];

    ContractExecutionResult::from_execution_result(execute_inner(
        engine,
        registry,
        function_id,
        &params,
        required_initial_gas,
        gas,
        Some(syscall_handler),
    )?)
}
