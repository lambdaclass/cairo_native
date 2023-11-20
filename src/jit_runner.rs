//! # JIT runner

use crate::{
    error::{
        jit_engine::{
            make_insufficient_gas_error, make_missing_parameter, make_type_builder_error,
            make_unexpected_value_error, ErrorImpl,
        },
        JitRunnerError,
    },
    execution_result::ContractExecutionResult,
    metadata::syscall_handler::SyscallHandlerMeta,
    types::TypeBuilder,
    utils::generate_function_name,
    values::{JITValue, ValueBuilder},
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
use melior::ExecutionEngine;
use std::{alloc::Layout, iter::once, ptr::NonNull};
use tracing::debug;

/// The result of the JIT execution.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub remaining_gas: Option<u128>,
    pub return_values: Vec<JITValue>,
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
    params: &[JITValue],
    required_initial_gas: Option<u128>,
    gas: Option<u128>,
    syscall_handler: Option<&SyscallHandlerMeta>,
) -> Result<ExecutionResult, JitRunnerError> {
    let arena = Bump::new();

    let entry_point = registry.get_function(function_id)?;
    debug!(
        "executing entry_point with the following required parameters: {:?}",
        entry_point.signature.param_types
    );

    let mut params_ptrs: Vec<NonNull<()>> = Vec::new();

    let mut params_it = params.iter();

    for param_type_id in &entry_point.signature.param_types {
        let ty = registry.get_type(param_type_id)?;

        match ty {
            CoreTypeConcrete::Array(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JITValue::Array(_)) {
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

                if !matches!(next, JITValue::EcPoint(..)) {
                    Err(make_unexpected_value_error("JITValue::EcPoint".to_string()))?;
                }

                params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
            }
            CoreTypeConcrete::EcState(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JITValue::EcState(..)) {
                    Err(make_unexpected_value_error("JITValue::EcState".to_string()))?;
                }

                params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
            }
            CoreTypeConcrete::Felt252(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JITValue::Felt252(_)) {
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

                if !matches!(next, JITValue::Uint8(_)) {
                    Err(make_unexpected_value_error("JITValue::Uint8".to_string()))?;
                }

                params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
            }
            CoreTypeConcrete::Uint16(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JITValue::Uint16(_)) {
                    Err(make_unexpected_value_error("JITValue::Uint16".to_string()))?;
                }

                params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
            }
            CoreTypeConcrete::Uint32(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JITValue::Uint32(_)) {
                    Err(make_unexpected_value_error("JITValue::Uint32".to_string()))?;
                }

                params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
            }
            CoreTypeConcrete::Uint64(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JITValue::Uint64(_)) {
                    Err(make_unexpected_value_error("JITValue::Uint64".to_string()))?;
                }

                params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
            }
            CoreTypeConcrete::Uint128(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JITValue::Uint128(_)) {
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

                if !matches!(next, JITValue::Enum { .. }) {
                    Err(make_unexpected_value_error("JITValue::Enum".to_string()))?;
                }

                params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
            }
            CoreTypeConcrete::Struct(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JITValue::Struct { .. }) {
                    Err(make_unexpected_value_error("JITValue::Struct".to_string()))?;
                }

                params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
            }
            CoreTypeConcrete::Felt252Dict(_) => {
                let next = params_it
                    .next()
                    .ok_or_else(|| make_missing_parameter(param_type_id))?;

                if !matches!(next, JITValue::Felt252Dict { .. }) {
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

                        if !matches!(next, JITValue::Felt252(_)) {
                            Err(make_unexpected_value_error("JITValue::Felt252".to_string()))?;
                        }

                        params_ptrs.push(next.to_jit(&arena, registry, param_type_id)?);
                    }
                    StarkNetTypeConcrete::Secp256Point(_) => todo!(),
                    StarkNetTypeConcrete::System(_) => {
                        let syscall_addr = syscall_handler
                            .ok_or(JitRunnerError::from(ErrorImpl::MissingSyscallHandler))?
                            .as_ptr()
                            .as_ptr() as *const ()
                            as usize;

                        params_ptrs.push(
                            arena
                                .alloc(NonNull::new(syscall_addr as *mut ()).unwrap())
                                .cast(),
                        );
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
                Some(layout) => layout.extend(ty_layout).unwrap(),
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

    unsafe {
        engine
            .invoke_packed(&function_name, &mut io_pointers)
            .unwrap();
    }

    let mut returns = Vec::new();

    let mut remaining_gas = None;

    for (type_id, offset) in entry_point.signature.ret_types.iter().zip(offsets) {
        let ty = registry.get_type(type_id).unwrap();

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
                let value = JITValue::from_jit(ptr, type_id, registry);
                returns.push(value);
            }
        };
    }

    Ok(ExecutionResult {
        remaining_gas,
        return_values: returns,
    })
}

/// Utility function to make it easier to execute contracts. Please see the [`execute`] for more information about program execution.
///
/// To call a contract, the calldata needs to be inside a struct inside a array, this helper function does that for you.
///
/// The calldata passed should all be felt252 values.
pub fn execute_contract(
    engine: &ExecutionEngine,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    function_id: &FunctionId,
    calldata: &[JITValue],
    required_initial_gas: Option<u128>,
    gas: u128,
    syscall_handler: &SyscallHandlerMeta,
) -> Result<ContractExecutionResult, JitRunnerError> {
    let params = vec![JITValue::Struct {
        fields: vec![JITValue::Array(calldata.to_vec())],
        debug_name: None,
    }];

    ContractExecutionResult::from_execution_result(execute(
        engine,
        registry,
        function_id,
        &params,
        required_initial_gas,
        Some(gas),
        Some(syscall_handler),
    )?)
}
