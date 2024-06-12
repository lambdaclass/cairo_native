////! # Executors
//! # Executors
////!
//!
////! This module provides methods to execute the programs, either via JIT or compiled ahead
//! This module provides methods to execute the programs, either via JIT or compiled ahead
////! of time. It also provides a cache to avoid recompiling previously compiled programs.
//! of time. It also provides a cache to avoid recompiling previously compiled programs.
//

//pub use self::{aot::AotNativeExecutor, jit::JitNativeExecutor};
pub use self::{aot::AotNativeExecutor, jit::JitNativeExecutor};
//use crate::{
use crate::{
//    error::Error,
    error::Error,
//    execution_result::{BuiltinStats, ContractExecutionResult, ExecutionResult},
    execution_result::{BuiltinStats, ContractExecutionResult, ExecutionResult},
//    starknet::{handler::StarknetSyscallHandlerCallbacks, StarknetSyscallHandler},
    starknet::{handler::StarknetSyscallHandlerCallbacks, StarknetSyscallHandler},
//    types::TypeBuilder,
    types::TypeBuilder,
//    utils::get_integer_layout,
    utils::get_integer_layout,
//    values::JitValue,
    values::JitValue,
//};
};
//use bumpalo::Bump;
use bumpalo::Bump;
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
//        starknet::StarkNetTypeConcrete,
        starknet::StarkNetTypeConcrete,
//    },
    },
//    ids::{ConcreteTypeId, FunctionId},
    ids::{ConcreteTypeId, FunctionId},
//    program::FunctionSignature,
    program::FunctionSignature,
//    program_registry::{ProgramRegistry, ProgramRegistryError},
    program_registry::{ProgramRegistry, ProgramRegistryError},
//};
};
//use libc::c_void;
use libc::c_void;
//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
//use std::{
use std::{
//    alloc::Layout,
    alloc::Layout,
//    arch::global_asm,
    arch::global_asm,
//    ptr::{null_mut, NonNull},
    ptr::{null_mut, NonNull},
//    rc::Rc,
    rc::Rc,
//};
};
//

//mod aot;
mod aot;
//mod jit;
mod jit;
//

//#[cfg(target_arch = "aarch64")]
#[cfg(target_arch = "aarch64")]
//global_asm!(include_str!("arch/aarch64.s"));
global_asm!(include_str!("arch/aarch64.s"));
//#[cfg(target_arch = "x86_64")]
#[cfg(target_arch = "x86_64")]
//global_asm!(include_str!("arch/x86_64.s"));
global_asm!(include_str!("arch/x86_64.s"));
//

//extern "C" {
extern "C" {
//    /// Invoke an AOT or JIT-compiled function.
    /// Invoke an AOT or JIT-compiled function.
//    ///
    ///
//    /// The `ret_ptr` argument is only used when the first argument (the actual return pointer) is
    /// The `ret_ptr` argument is only used when the first argument (the actual return pointer) is
//    /// unused. Used for u8, u16, u32, u64, u128 and felt252, but not for arrays, enums or structs.
    /// unused. Used for u8, u16, u32, u64, u128 and felt252, but not for arrays, enums or structs.
//    #[cfg_attr(not(target_os = "macos"), link_name = "_invoke_trampoline")]
    #[cfg_attr(not(target_os = "macos"), link_name = "_invoke_trampoline")]
//    fn invoke_trampoline(
    fn invoke_trampoline(
//        fn_ptr: *const c_void,
        fn_ptr: *const c_void,
//        args_ptr: *const u64,
        args_ptr: *const u64,
//        args_len: usize,
        args_len: usize,
//        ret_ptr: *mut u64,
        ret_ptr: *mut u64,
//    );
    );
//}
}
//

///// The cairo native executor, either AOT or JIT based.
/// The cairo native executor, either AOT or JIT based.
//#[derive(Debug, Clone)]
#[derive(Debug, Clone)]
//pub enum NativeExecutor<'m> {
pub enum NativeExecutor<'m> {
//    Aot(Rc<AotNativeExecutor>),
    Aot(Rc<AotNativeExecutor>),
//    Jit(Rc<JitNativeExecutor<'m>>),
    Jit(Rc<JitNativeExecutor<'m>>),
//}
}
//

//impl<'a> NativeExecutor<'a> {
impl<'a> NativeExecutor<'a> {
//    /// Invoke the given function by its function id, with the given arguments and gas.
    /// Invoke the given function by its function id, with the given arguments and gas.
//    pub fn invoke_dynamic(
    pub fn invoke_dynamic(
//        &self,
        &self,
//        function_id: &FunctionId,
        function_id: &FunctionId,
//        args: &[JitValue],
        args: &[JitValue],
//        gas: Option<u128>,
        gas: Option<u128>,
//    ) -> Result<ExecutionResult, Error> {
    ) -> Result<ExecutionResult, Error> {
//        match self {
        match self {
//            NativeExecutor::Aot(executor) => executor.invoke_dynamic(function_id, args, gas),
            NativeExecutor::Aot(executor) => executor.invoke_dynamic(function_id, args, gas),
//            NativeExecutor::Jit(executor) => executor.invoke_dynamic(function_id, args, gas),
            NativeExecutor::Jit(executor) => executor.invoke_dynamic(function_id, args, gas),
//        }
        }
//    }
    }
//

//    /// Invoke the given function by its function id, with the given arguments and gas.
    /// Invoke the given function by its function id, with the given arguments and gas.
//    /// This should be used for programs which require a syscall handler, whose
    /// This should be used for programs which require a syscall handler, whose
//    /// implementation should be passed on.
    /// implementation should be passed on.
//    pub fn invoke_dynamic_with_syscall_handler(
    pub fn invoke_dynamic_with_syscall_handler(
//        &self,
        &self,
//        function_id: &FunctionId,
        function_id: &FunctionId,
//        args: &[JitValue],
        args: &[JitValue],
//        gas: Option<u128>,
        gas: Option<u128>,
//        syscall_handler: impl StarknetSyscallHandler,
        syscall_handler: impl StarknetSyscallHandler,
//    ) -> Result<ExecutionResult, Error> {
    ) -> Result<ExecutionResult, Error> {
//        match self {
        match self {
//            NativeExecutor::Aot(executor) => executor.invoke_dynamic_with_syscall_handler(
            NativeExecutor::Aot(executor) => executor.invoke_dynamic_with_syscall_handler(
//                function_id,
                function_id,
//                args,
                args,
//                gas,
                gas,
//                syscall_handler,
                syscall_handler,
//            ),
            ),
//            NativeExecutor::Jit(executor) => executor.invoke_dynamic_with_syscall_handler(
            NativeExecutor::Jit(executor) => executor.invoke_dynamic_with_syscall_handler(
//                function_id,
                function_id,
//                args,
                args,
//                gas,
                gas,
//                syscall_handler,
                syscall_handler,
//            ),
            ),
//        }
        }
//    }
    }
//

//    /// Invoke the given function by its function id, with the given arguments and gas.
    /// Invoke the given function by its function id, with the given arguments and gas.
//    /// This should be used for starknet contracts which require a syscall handler, whose
    /// This should be used for starknet contracts which require a syscall handler, whose
//    /// implementation should be passed on.
    /// implementation should be passed on.
//    pub fn invoke_contract_dynamic(
    pub fn invoke_contract_dynamic(
//        &self,
        &self,
//        function_id: &FunctionId,
        function_id: &FunctionId,
//        args: &[Felt],
        args: &[Felt],
//        gas: Option<u128>,
        gas: Option<u128>,
//        syscall_handler: impl StarknetSyscallHandler,
        syscall_handler: impl StarknetSyscallHandler,
//    ) -> Result<ContractExecutionResult, Error> {
    ) -> Result<ContractExecutionResult, Error> {
//        match self {
        match self {
//            NativeExecutor::Aot(executor) => {
            NativeExecutor::Aot(executor) => {
//                executor.invoke_contract_dynamic(function_id, args, gas, syscall_handler)
                executor.invoke_contract_dynamic(function_id, args, gas, syscall_handler)
//            }
            }
//            NativeExecutor::Jit(executor) => {
            NativeExecutor::Jit(executor) => {
//                executor.invoke_contract_dynamic(function_id, args, gas, syscall_handler)
                executor.invoke_contract_dynamic(function_id, args, gas, syscall_handler)
//            }
            }
//        }
        }
//    }
    }
//}
}
//

//impl<'m> From<AotNativeExecutor> for NativeExecutor<'m> {
impl<'m> From<AotNativeExecutor> for NativeExecutor<'m> {
//    fn from(value: AotNativeExecutor) -> Self {
    fn from(value: AotNativeExecutor) -> Self {
//        Self::Aot(Rc::new(value))
        Self::Aot(Rc::new(value))
//    }
    }
//}
}
//

//impl<'m> From<JitNativeExecutor<'m>> for NativeExecutor<'m> {
impl<'m> From<JitNativeExecutor<'m>> for NativeExecutor<'m> {
//    fn from(value: JitNativeExecutor<'m>) -> Self {
    fn from(value: JitNativeExecutor<'m>) -> Self {
//        Self::Jit(Rc::new(value))
        Self::Jit(Rc::new(value))
//    }
    }
//}
}
//

///// Internal method.
/// Internal method.
/////
///
///// Invokes the given function by constructing the function call depending on the arguments given.
/// Invokes the given function by constructing the function call depending on the arguments given.
///// Usually calling a function requires knowing it's signature at compile time, but we need to be
/// Usually calling a function requires knowing it's signature at compile time, but we need to be
///// able to call any given function provided it's signatue (arguments and return type) at runtime,
/// able to call any given function provided it's signatue (arguments and return type) at runtime,
///// to do so we have a "trampoline" in the given platform assembly (x86_64, aarch64) which
/// to do so we have a "trampoline" in the given platform assembly (x86_64, aarch64) which
///// constructs the function call in place.
/// constructs the function call in place.
/////
///
///// To pass the arguments, they are stored in a arena.
/// To pass the arguments, they are stored in a arena.
//fn invoke_dynamic(
fn invoke_dynamic(
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    function_ptr: *const c_void,
    function_ptr: *const c_void,
//    function_signature: &FunctionSignature,
    function_signature: &FunctionSignature,
//    args: &[JitValue],
    args: &[JitValue],
//    gas: u128,
    gas: u128,
//    mut syscall_handler: Option<impl StarknetSyscallHandler>,
    mut syscall_handler: Option<impl StarknetSyscallHandler>,
//) -> Result<ExecutionResult, Error> {
) -> Result<ExecutionResult, Error> {
//    tracing::info!("Invoking function with signature: {function_signature:?}.");
    tracing::info!("Invoking function with signature: {function_signature:?}.");
//

//    let arena = Bump::new();
    let arena = Bump::new();
//    let mut invoke_data = ArgumentMapper::new(&arena, registry);
    let mut invoke_data = ArgumentMapper::new(&arena, registry);
//

//    // Generate return pointer (if necessary).
    // Generate return pointer (if necessary).
//    //
    //
//    // Generated when either:
    // Generated when either:
//    //   - There are more than one non-zst return values.
    //   - There are more than one non-zst return values.
//    //     - All builtins except GasBuiltin and Starknet are ZST.
    //     - All builtins except GasBuiltin and Starknet are ZST.
//    //     - The unit struct is a ZST.
    //     - The unit struct is a ZST.
//    //   - The return argument is complex.
    //   - The return argument is complex.
//    let mut ret_types_iter = function_signature
    let mut ret_types_iter = function_signature
//        .ret_types
        .ret_types
//        .iter()
        .iter()
//        .filter(|id| {
        .filter(|id| {
//            let info = registry.get_type(id).unwrap();
            let info = registry.get_type(id).unwrap();
//            !(info.is_builtin() && info.is_zst(registry))
            !(info.is_builtin() && info.is_zst(registry))
//        })
        })
//        .peekable();
        .peekable();
//

//    let num_return_args = ret_types_iter.clone().count();
    let num_return_args = ret_types_iter.clone().count();
//    let mut return_ptr = if num_return_args > 1
    let mut return_ptr = if num_return_args > 1
//        || ret_types_iter
        || ret_types_iter
//            .peek()
            .peek()
//            .is_some_and(|id| registry.get_type(id).unwrap().is_complex(registry))
            .is_some_and(|id| registry.get_type(id).unwrap().is_complex(registry))
//    {
    {
//        let layout = ret_types_iter.fold(Layout::new::<()>(), |layout, id| {
        let layout = ret_types_iter.fold(Layout::new::<()>(), |layout, id| {
//            let type_info = registry.get_type(id).unwrap();
            let type_info = registry.get_type(id).unwrap();
//            layout
            layout
//                .extend(type_info.layout(registry).unwrap())
                .extend(type_info.layout(registry).unwrap())
//                .unwrap()
                .unwrap()
//                .0
                .0
//        });
        });
//

//        let return_ptr = arena.alloc_layout(layout).cast::<()>();
        let return_ptr = arena.alloc_layout(layout).cast::<()>();
//        invoke_data.push_aligned(
        invoke_data.push_aligned(
//            get_integer_layout(64).align(),
            get_integer_layout(64).align(),
//            &[return_ptr.as_ptr() as u64],
            &[return_ptr.as_ptr() as u64],
//        );
        );
//

//        Some(return_ptr)
        Some(return_ptr)
//    } else {
    } else {
//        None
        None
//    };
    };
//

//    // Generate argument list.
    // Generate argument list.
//    let mut iter = args.iter();
    let mut iter = args.iter();
//    for type_id in function_signature.param_types.iter().filter(|id| {
    for type_id in function_signature.param_types.iter().filter(|id| {
//        let info = registry.get_type(id).unwrap();
        let info = registry.get_type(id).unwrap();
//        !info.is_zst(registry)
        !info.is_zst(registry)
//    }) {
    }) {
//        // Process gas requirements and syscall handler.
        // Process gas requirements and syscall handler.
//        match registry.get_type(type_id).unwrap() {
        match registry.get_type(type_id).unwrap() {
//            CoreTypeConcrete::GasBuiltin(_) => invoke_data.push_aligned(
            CoreTypeConcrete::GasBuiltin(_) => invoke_data.push_aligned(
//                get_integer_layout(128).align(),
                get_integer_layout(128).align(),
//                &[gas as u64, (gas >> 64) as u64],
                &[gas as u64, (gas >> 64) as u64],
//            ),
            ),
//            CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::System(_)) => {
            CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::System(_)) => {
//                match syscall_handler.as_mut() {
                match syscall_handler.as_mut() {
//                    Some(syscall_handler) => {
                    Some(syscall_handler) => {
//                        let syscall_handler =
                        let syscall_handler =
//                            arena.alloc(StarknetSyscallHandlerCallbacks::new(syscall_handler));
                            arena.alloc(StarknetSyscallHandlerCallbacks::new(syscall_handler));
//                        invoke_data.push_aligned(
                        invoke_data.push_aligned(
//                            get_integer_layout(64).align(),
                            get_integer_layout(64).align(),
//                            &[syscall_handler as *mut _ as u64],
                            &[syscall_handler as *mut _ as u64],
//                        )
                        )
//                    }
                    }
//                    None => {
                    None => {
//                        panic!("Syscall handler is required");
                        panic!("Syscall handler is required");
//                    }
                    }
//                }
                }
//            }
            }
//            type_info => invoke_data
            type_info => invoke_data
//                .push(
                .push(
//                    type_id,
                    type_id,
//                    type_info,
                    type_info,
//                    if type_info.is_builtin() {
                    if type_info.is_builtin() {
//                        &JitValue::Uint64(0)
                        &JitValue::Uint64(0)
//                    } else {
                    } else {
//                        iter.next().unwrap()
                        iter.next().unwrap()
//                    },
                    },
//                )
                )
//                .unwrap(),
                .unwrap(),
//        }
        }
//    }
    }
//

//    // Invoke the trampoline.
    // Invoke the trampoline.
//    #[cfg(target_arch = "x86_64")]
    #[cfg(target_arch = "x86_64")]
//    let mut ret_registers = [0; 2];
    let mut ret_registers = [0; 2];
//    #[cfg(target_arch = "aarch64")]
    #[cfg(target_arch = "aarch64")]
//    let mut ret_registers = [0; 4];
    let mut ret_registers = [0; 4];
//

//    unsafe {
    unsafe {
//        invoke_trampoline(
        invoke_trampoline(
//            function_ptr,
            function_ptr,
//            invoke_data.invoke_data().as_ptr(),
            invoke_data.invoke_data().as_ptr(),
//            invoke_data.invoke_data().len(),
            invoke_data.invoke_data().len(),
//            ret_registers.as_mut_ptr(),
            ret_registers.as_mut_ptr(),
//        );
        );
//    }
    }
//

//    // Parse final gas.
    // Parse final gas.
//    unsafe fn read_value<T>(ptr: &mut NonNull<()>) -> &T {
    unsafe fn read_value<T>(ptr: &mut NonNull<()>) -> &T {
//        let align_offset = ptr
        let align_offset = ptr
//            .cast::<u8>()
            .cast::<u8>()
//            .as_ptr()
            .as_ptr()
//            .align_offset(std::mem::align_of::<T>());
            .align_offset(std::mem::align_of::<T>());
//        let value_ptr = ptr.cast::<u8>().as_ptr().add(align_offset).cast::<T>();
        let value_ptr = ptr.cast::<u8>().as_ptr().add(align_offset).cast::<T>();
//

//        *ptr = NonNull::new_unchecked(value_ptr.add(1)).cast();
        *ptr = NonNull::new_unchecked(value_ptr.add(1)).cast();
//        &*value_ptr
        &*value_ptr
//    }
    }
//

//    let mut remaining_gas = None;
    let mut remaining_gas = None;
//    let mut builtin_stats = BuiltinStats::default();
    let mut builtin_stats = BuiltinStats::default();
//    for type_id in &function_signature.ret_types {
    for type_id in &function_signature.ret_types {
//        let type_info = registry.get_type(type_id).unwrap();
        let type_info = registry.get_type(type_id).unwrap();
//        match type_info {
        match type_info {
//            CoreTypeConcrete::GasBuiltin(_) => {
            CoreTypeConcrete::GasBuiltin(_) => {
//                remaining_gas = Some(match &mut return_ptr {
                remaining_gas = Some(match &mut return_ptr {
//                    Some(return_ptr) => unsafe { *read_value::<u128>(return_ptr) },
                    Some(return_ptr) => unsafe { *read_value::<u128>(return_ptr) },
//                    None => {
                    None => {
//                        // If there's no return ptr then the function only returned the gas. We don't
                        // If there's no return ptr then the function only returned the gas. We don't
//                        // need to bother with the syscall handler builtin.
                        // need to bother with the syscall handler builtin.
//                        ((ret_registers[1] as u128) << 64) | ret_registers[0] as u128
                        ((ret_registers[1] as u128) << 64) | ret_registers[0] as u128
//                    }
                    }
//                });
                });
//            }
            }
//            CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::System(_)) => match &mut return_ptr {
            CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::System(_)) => match &mut return_ptr {
//                Some(return_ptr) => unsafe {
                Some(return_ptr) => unsafe {
//                    let ptr = return_ptr.cast::<*mut ()>();
                    let ptr = return_ptr.cast::<*mut ()>();
//                    *return_ptr = NonNull::new_unchecked(ptr.as_ptr().add(1)).cast();
                    *return_ptr = NonNull::new_unchecked(ptr.as_ptr().add(1)).cast();
//                },
                },
//                None => {}
                None => {}
//            },
            },
//            _ if type_info.is_builtin() => {
            _ if type_info.is_builtin() => {
//                if !type_info.is_zst(registry) {
                if !type_info.is_zst(registry) {
//                    let value = match &mut return_ptr {
                    let value = match &mut return_ptr {
//                        Some(return_ptr) => unsafe { *read_value::<u64>(return_ptr) },
                        Some(return_ptr) => unsafe { *read_value::<u64>(return_ptr) },
//                        None => ret_registers[0],
                        None => ret_registers[0],
//                    } as usize;
                    } as usize;
//

//                    match type_info {
                    match type_info {
//                        CoreTypeConcrete::Bitwise(_) => builtin_stats.bitwise = value,
                        CoreTypeConcrete::Bitwise(_) => builtin_stats.bitwise = value,
//                        CoreTypeConcrete::EcOp(_) => builtin_stats.ec_op = value,
                        CoreTypeConcrete::EcOp(_) => builtin_stats.ec_op = value,
//                        CoreTypeConcrete::RangeCheck(_) => builtin_stats.range_check = value,
                        CoreTypeConcrete::RangeCheck(_) => builtin_stats.range_check = value,
//                        CoreTypeConcrete::Pedersen(_) => builtin_stats.pedersen = value,
                        CoreTypeConcrete::Pedersen(_) => builtin_stats.pedersen = value,
//                        CoreTypeConcrete::Poseidon(_) => builtin_stats.poseidon = value,
                        CoreTypeConcrete::Poseidon(_) => builtin_stats.poseidon = value,
//                        CoreTypeConcrete::SegmentArena(_) => builtin_stats.segment_arena = value,
                        CoreTypeConcrete::SegmentArena(_) => builtin_stats.segment_arena = value,
//                        _ => unreachable!("{type_id:?}"),
                        _ => unreachable!("{type_id:?}"),
//                    }
                    }
//                }
                }
//            }
            }
//            _ => break,
            _ => break,
//        }
        }
//    }
    }
//

//    // Parse return values.
    // Parse return values.
//    let return_value = function_signature
    let return_value = function_signature
//        .ret_types
        .ret_types
//        .last()
        .last()
//        .map(|ret_type| {
        .map(|ret_type| {
//            parse_result(
            parse_result(
//                ret_type,
                ret_type,
//                registry,
                registry,
//                return_ptr,
                return_ptr,
//                ret_registers,
                ret_registers,
//                // TODO: Consider returning an Option<JitValue> as return_value instead
                // TODO: Consider returning an Option<JitValue> as return_value instead
//                // As cairo functions can not have a return value
                // As cairo functions can not have a return value
//            )
            )
//        })
        })
//        .unwrap_or_else(|| {
        .unwrap_or_else(|| {
//            Ok(JitValue::Struct {
            Ok(JitValue::Struct {
//                fields: vec![],
                fields: vec![],
//                debug_name: None,
                debug_name: None,
//            })
            })
//        })?;
        })?;
//

//    // FIXME: Arena deallocation.
    // FIXME: Arena deallocation.
//    std::mem::forget(arena);
    std::mem::forget(arena);
//

//    Ok(ExecutionResult {
    Ok(ExecutionResult {
//        remaining_gas,
        remaining_gas,
//        return_value,
        return_value,
//        builtin_stats,
        builtin_stats,
//    })
    })
//}
}
//

//pub struct ArgumentMapper<'a> {
pub struct ArgumentMapper<'a> {
//    arena: &'a Bump,
    arena: &'a Bump,
//    registry: &'a ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &'a ProgramRegistry<CoreType, CoreLibfunc>,
//

//    invoke_data: Vec<u64>,
    invoke_data: Vec<u64>,
//}
}
//

//impl<'a> ArgumentMapper<'a> {
impl<'a> ArgumentMapper<'a> {
//    pub fn new(arena: &'a Bump, registry: &'a ProgramRegistry<CoreType, CoreLibfunc>) -> Self {
    pub fn new(arena: &'a Bump, registry: &'a ProgramRegistry<CoreType, CoreLibfunc>) -> Self {
//        Self {
        Self {
//            arena,
            arena,
//            registry,
            registry,
//            invoke_data: Vec::new(),
            invoke_data: Vec::new(),
//        }
        }
//    }
    }
//

//    pub fn invoke_data(&self) -> &[u64] {
    pub fn invoke_data(&self) -> &[u64] {
//        &self.invoke_data
        &self.invoke_data
//    }
    }
//

//    pub fn push_aligned(&mut self, align: usize, mut values: &[u64]) {
    pub fn push_aligned(&mut self, align: usize, mut values: &[u64]) {
//        assert!(align.is_power_of_two());
        assert!(align.is_power_of_two());
//        assert!(align <= 16);
        assert!(align <= 16);
//

//        #[cfg(target_arch = "x86_64")]
        #[cfg(target_arch = "x86_64")]
//        const NUM_REGISTER_ARGS: usize = 6;
        const NUM_REGISTER_ARGS: usize = 6;
//        #[cfg(target_arch = "aarch64")]
        #[cfg(target_arch = "aarch64")]
//        const NUM_REGISTER_ARGS: usize = 8;
        const NUM_REGISTER_ARGS: usize = 8;
//

//        if align == 16 {
        if align == 16 {
//            // This works because on both aarch64 and x86_64 the stack is already aligned to
            // This works because on both aarch64 and x86_64 the stack is already aligned to
//            // 16 bytes when the trampoline starts pushing values.
            // 16 bytes when the trampoline starts pushing values.
//

//            // Whenever a value spans across multiple registers, if it's in a position where it would be split between
            // Whenever a value spans across multiple registers, if it's in a position where it would be split between
//            // registers and the stack it must be padded so that the entire value is stored within the stack.
            // registers and the stack it must be padded so that the entire value is stored within the stack.
//            if self.invoke_data.len() >= NUM_REGISTER_ARGS {
            if self.invoke_data.len() >= NUM_REGISTER_ARGS {
//                if self.invoke_data.len() & 1 != 0 {
                if self.invoke_data.len() & 1 != 0 {
//                    self.invoke_data.push(0);
                    self.invoke_data.push(0);
//                }
                }
//            } else if self.invoke_data.len() + 1 >= NUM_REGISTER_ARGS {
            } else if self.invoke_data.len() + 1 >= NUM_REGISTER_ARGS {
//                self.invoke_data.push(0);
                self.invoke_data.push(0);
//            } else {
            } else {
//                let new_len = self.invoke_data.len() + values.len();
                let new_len = self.invoke_data.len() + values.len();
//                if new_len >= NUM_REGISTER_ARGS && new_len % 2 != 0 {
                if new_len >= NUM_REGISTER_ARGS && new_len % 2 != 0 {
//                    let chunk;
                    let chunk;
//                    (chunk, values) = if values.len() >= 4 {
                    (chunk, values) = if values.len() >= 4 {
//                        values.split_at(4)
                        values.split_at(4)
//                    } else {
                    } else {
//                        (values, [].as_slice())
                        (values, [].as_slice())
//                    };
                    };
//                    self.invoke_data.extend(chunk);
                    self.invoke_data.extend(chunk);
//                    self.invoke_data.push(0);
                    self.invoke_data.push(0);
//                }
                }
//            }
            }
//        }
        }
//

//        self.invoke_data.extend(values);
        self.invoke_data.extend(values);
//    }
    }
//

//    pub fn push(
    pub fn push(
//        &mut self,
        &mut self,
//        type_id: &ConcreteTypeId,
        type_id: &ConcreteTypeId,
//        type_info: &CoreTypeConcrete,
        type_info: &CoreTypeConcrete,
//        value: &JitValue,
        value: &JitValue,
//    ) -> Result<(), Box<ProgramRegistryError>> {
    ) -> Result<(), Box<ProgramRegistryError>> {
//        match (type_info, value) {
        match (type_info, value) {
//            (CoreTypeConcrete::Array(info), JitValue::Array(values)) => {
            (CoreTypeConcrete::Array(info), JitValue::Array(values)) => {
//                // TODO: Assert that `info.ty` matches all the values' types.
                // TODO: Assert that `info.ty` matches all the values' types.
//

//                let type_info = self.registry.get_type(&info.ty)?;
                let type_info = self.registry.get_type(&info.ty)?;
//                let type_layout = type_info.layout(self.registry).unwrap().pad_to_align();
                let type_layout = type_info.layout(self.registry).unwrap().pad_to_align();
//

//                // This needs to be a heap-allocated pointer because it's the actual array data.
                // This needs to be a heap-allocated pointer because it's the actual array data.
//                let ptr = if values.is_empty() {
                let ptr = if values.is_empty() {
//                    null_mut()
                    null_mut()
//                } else {
                } else {
//                    unsafe { libc::realloc(null_mut(), type_layout.size() * values.len()) }
                    unsafe { libc::realloc(null_mut(), type_layout.size() * values.len()) }
//                };
                };
//

//                for (idx, value) in values.iter().enumerate() {
                for (idx, value) in values.iter().enumerate() {
//                    unsafe {
                    unsafe {
//                        std::ptr::copy_nonoverlapping(
                        std::ptr::copy_nonoverlapping(
//                            value
                            value
//                                .to_jit(self.arena, self.registry, &info.ty)
                                .to_jit(self.arena, self.registry, &info.ty)
//                                .unwrap()
                                .unwrap()
//                                .cast()
                                .cast()
//                                .as_ptr(),
                                .as_ptr(),
//                            (ptr as usize + type_layout.size() * idx) as *mut u8,
                            (ptr as usize + type_layout.size() * idx) as *mut u8,
//                            type_layout.size(),
                            type_layout.size(),
//                        );
                        );
//                    }
                    }
//                }
                }
//

//                self.push_aligned(
                self.push_aligned(
//                    get_integer_layout(64).align(),
                    get_integer_layout(64).align(),
//                    &[ptr as u64, 0, values.len() as u64, values.len() as u64],
                    &[ptr as u64, 0, values.len() as u64, values.len() as u64],
//                );
                );
//            }
            }
//            (CoreTypeConcrete::EcPoint(_), JitValue::EcPoint(a, b)) => {
            (CoreTypeConcrete::EcPoint(_), JitValue::EcPoint(a, b)) => {
//                let align = get_integer_layout(252).align();
                let align = get_integer_layout(252).align();
//                self.push_aligned(align, &a.to_le_digits());
                self.push_aligned(align, &a.to_le_digits());
//                self.push_aligned(align, &b.to_le_digits());
                self.push_aligned(align, &b.to_le_digits());
//            }
            }
//            (CoreTypeConcrete::EcState(_), JitValue::EcState(a, b, c, d)) => {
            (CoreTypeConcrete::EcState(_), JitValue::EcState(a, b, c, d)) => {
//                let align = get_integer_layout(252).align();
                let align = get_integer_layout(252).align();
//                self.push_aligned(align, &a.to_le_digits());
                self.push_aligned(align, &a.to_le_digits());
//                self.push_aligned(align, &b.to_le_digits());
                self.push_aligned(align, &b.to_le_digits());
//                self.push_aligned(align, &c.to_le_digits());
                self.push_aligned(align, &c.to_le_digits());
//                self.push_aligned(align, &d.to_le_digits());
                self.push_aligned(align, &d.to_le_digits());
//            }
            }
//            (CoreTypeConcrete::Enum(info), JitValue::Enum { tag, value, .. }) => {
            (CoreTypeConcrete::Enum(info), JitValue::Enum { tag, value, .. }) => {
//                if type_info.is_memory_allocated(self.registry) {
                if type_info.is_memory_allocated(self.registry) {
//                    let (layout, tag_layout, variant_layouts) =
                    let (layout, tag_layout, variant_layouts) =
//                        crate::types::r#enum::get_layout_for_variants(
                        crate::types::r#enum::get_layout_for_variants(
//                            self.registry,
                            self.registry,
//                            &info.variants,
                            &info.variants,
//                        )
                        )
//                        .unwrap();
                        .unwrap();
//

//                    let ptr = self.arena.alloc_layout(layout);
                    let ptr = self.arena.alloc_layout(layout);
//                    unsafe {
                    unsafe {
//                        match tag_layout.size() {
                        match tag_layout.size() {
//                            0 => {}
                            0 => {}
//                            1 => *ptr.cast::<u8>().as_mut() = *tag as u8,
                            1 => *ptr.cast::<u8>().as_mut() = *tag as u8,
//                            2 => *ptr.cast::<u16>().as_mut() = *tag as u16,
                            2 => *ptr.cast::<u16>().as_mut() = *tag as u16,
//                            4 => *ptr.cast::<u32>().as_mut() = *tag as u32,
                            4 => *ptr.cast::<u32>().as_mut() = *tag as u32,
//                            8 => *ptr.cast::<u64>().as_mut() = *tag as u64,
                            8 => *ptr.cast::<u64>().as_mut() = *tag as u64,
//                            _ => unreachable!(),
                            _ => unreachable!(),
//                        }
                        }
//                    }
                    }
//

//                    let offset = tag_layout.extend(variant_layouts[*tag]).unwrap().1;
                    let offset = tag_layout.extend(variant_layouts[*tag]).unwrap().1;
//                    let payload_ptr = value
                    let payload_ptr = value
//                        .to_jit(self.arena, self.registry, &info.variants[*tag])
                        .to_jit(self.arena, self.registry, &info.variants[*tag])
//                        .unwrap();
                        .unwrap();
//                    unsafe {
                    unsafe {
//                        std::ptr::copy_nonoverlapping(
                        std::ptr::copy_nonoverlapping(
//                            payload_ptr.cast::<u8>().as_ptr(),
                            payload_ptr.cast::<u8>().as_ptr(),
//                            ptr.cast::<u8>().as_ptr().add(offset),
                            ptr.cast::<u8>().as_ptr().add(offset),
//                            variant_layouts[*tag].size(),
                            variant_layouts[*tag].size(),
//                        );
                        );
//                    }
                    }
//

//                    self.invoke_data.push(ptr.as_ptr() as u64);
                    self.invoke_data.push(ptr.as_ptr() as u64);
//                } else {
                } else {
//                    // Write the tag.
                    // Write the tag.
//                    match (info.variants.len().next_power_of_two().trailing_zeros() + 7) / 8 {
                    match (info.variants.len().next_power_of_two().trailing_zeros() + 7) / 8 {
//                        0 => {}
                        0 => {}
//                        _ => self.invoke_data.push(*tag as u64),
                        _ => self.invoke_data.push(*tag as u64),
//                    }
                    }
//

//                    // Write the payload.
                    // Write the payload.
//                    let type_info = self.registry.get_type(&info.variants[*tag]).unwrap();
                    let type_info = self.registry.get_type(&info.variants[*tag]).unwrap();
//                    self.push(&info.variants[*tag], type_info, value)?;
                    self.push(&info.variants[*tag], type_info, value)?;
//                }
                }
//            }
            }
//            (
            (
//                CoreTypeConcrete::Felt252(_)
                CoreTypeConcrete::Felt252(_)
//                | CoreTypeConcrete::StarkNet(
                | CoreTypeConcrete::StarkNet(
//                    StarkNetTypeConcrete::ClassHash(_)
                    StarkNetTypeConcrete::ClassHash(_)
//                    | StarkNetTypeConcrete::ContractAddress(_)
                    | StarkNetTypeConcrete::ContractAddress(_)
//                    | StarkNetTypeConcrete::StorageAddress(_)
                    | StarkNetTypeConcrete::StorageAddress(_)
//                    | StarkNetTypeConcrete::StorageBaseAddress(_),
                    | StarkNetTypeConcrete::StorageBaseAddress(_),
//                ),
                ),
//                JitValue::Felt252(value),
                JitValue::Felt252(value),
//            ) => {
            ) => {
//                self.push_aligned(get_integer_layout(252).align(), &value.to_le_digits());
                self.push_aligned(get_integer_layout(252).align(), &value.to_le_digits());
//            }
            }
//            (CoreTypeConcrete::Bytes31(_), JitValue::Bytes31(value)) => {
            (CoreTypeConcrete::Bytes31(_), JitValue::Bytes31(value)) => {
//                self.push_aligned(
                self.push_aligned(
//                    get_integer_layout(248).align(),
                    get_integer_layout(248).align(),
//                    &Felt::from_bytes_be_slice(value).to_le_digits(),
                    &Felt::from_bytes_be_slice(value).to_le_digits(),
//                );
                );
//            }
            }
//            (CoreTypeConcrete::Felt252Dict(_), JitValue::Felt252Dict { .. }) => {
            (CoreTypeConcrete::Felt252Dict(_), JitValue::Felt252Dict { .. }) => {
//                #[cfg(not(feature = "with-runtime"))]
                #[cfg(not(feature = "with-runtime"))]
//                unimplemented!("enable the `with-runtime` feature to use felt252 dicts");
                unimplemented!("enable the `with-runtime` feature to use felt252 dicts");
//

//                // TODO: Assert that `info.ty` matches all the values' types.
                // TODO: Assert that `info.ty` matches all the values' types.
//

//                self.invoke_data.push(
                self.invoke_data.push(
//                    value
                    value
//                        .to_jit(self.arena, self.registry, type_id)
                        .to_jit(self.arena, self.registry, type_id)
//                        .unwrap()
                        .unwrap()
//                        .as_ptr() as u64,
                        .as_ptr() as u64,
//                );
                );
//            }
            }
//            (CoreTypeConcrete::Struct(info), JitValue::Struct { fields, .. }) => {
            (CoreTypeConcrete::Struct(info), JitValue::Struct { fields, .. }) => {
//                for (field_type_id, field_value) in info.members.iter().zip(fields) {
                for (field_type_id, field_value) in info.members.iter().zip(fields) {
//                    self.push(
                    self.push(
//                        field_type_id,
                        field_type_id,
//                        self.registry.get_type(field_type_id)?,
                        self.registry.get_type(field_type_id)?,
//                        field_value,
                        field_value,
//                    )?;
                    )?;
//                }
                }
//            }
            }
//            (CoreTypeConcrete::Uint128(_), JitValue::Uint128(value)) => self.push_aligned(
            (CoreTypeConcrete::Uint128(_), JitValue::Uint128(value)) => self.push_aligned(
//                get_integer_layout(128).align(),
                get_integer_layout(128).align(),
//                &[*value as u64, (value >> 64) as u64],
                &[*value as u64, (value >> 64) as u64],
//            ),
            ),
//            (CoreTypeConcrete::Uint64(_), JitValue::Uint64(value)) => {
            (CoreTypeConcrete::Uint64(_), JitValue::Uint64(value)) => {
//                self.push_aligned(get_integer_layout(64).align(), &[*value]);
                self.push_aligned(get_integer_layout(64).align(), &[*value]);
//            }
            }
//            (CoreTypeConcrete::Uint32(_), JitValue::Uint32(value)) => {
            (CoreTypeConcrete::Uint32(_), JitValue::Uint32(value)) => {
//                self.push_aligned(get_integer_layout(32).align(), &[*value as u64]);
                self.push_aligned(get_integer_layout(32).align(), &[*value as u64]);
//            }
            }
//            (CoreTypeConcrete::Uint16(_), JitValue::Uint16(value)) => {
            (CoreTypeConcrete::Uint16(_), JitValue::Uint16(value)) => {
//                self.push_aligned(get_integer_layout(16).align(), &[*value as u64]);
                self.push_aligned(get_integer_layout(16).align(), &[*value as u64]);
//            }
            }
//            (CoreTypeConcrete::Uint8(_), JitValue::Uint8(value)) => {
            (CoreTypeConcrete::Uint8(_), JitValue::Uint8(value)) => {
//                self.push_aligned(get_integer_layout(8).align(), &[*value as u64]);
                self.push_aligned(get_integer_layout(8).align(), &[*value as u64]);
//            }
            }
//            (CoreTypeConcrete::Sint128(_), JitValue::Sint128(value)) => {
            (CoreTypeConcrete::Sint128(_), JitValue::Sint128(value)) => {
//                self.push_aligned(
                self.push_aligned(
//                    get_integer_layout(128).align(),
                    get_integer_layout(128).align(),
//                    &[*value as u64, (value >> 64) as u64],
                    &[*value as u64, (value >> 64) as u64],
//                );
                );
//            }
            }
//            (CoreTypeConcrete::Sint64(_), JitValue::Sint64(value)) => {
            (CoreTypeConcrete::Sint64(_), JitValue::Sint64(value)) => {
//                self.push_aligned(get_integer_layout(64).align(), &[*value as u64]);
                self.push_aligned(get_integer_layout(64).align(), &[*value as u64]);
//            }
            }
//            (CoreTypeConcrete::Sint32(_), JitValue::Sint32(value)) => {
            (CoreTypeConcrete::Sint32(_), JitValue::Sint32(value)) => {
//                self.push_aligned(get_integer_layout(32).align(), &[*value as u64]);
                self.push_aligned(get_integer_layout(32).align(), &[*value as u64]);
//            }
            }
//            (CoreTypeConcrete::Sint16(_), JitValue::Sint16(value)) => {
            (CoreTypeConcrete::Sint16(_), JitValue::Sint16(value)) => {
//                self.push_aligned(get_integer_layout(16).align(), &[*value as u64]);
                self.push_aligned(get_integer_layout(16).align(), &[*value as u64]);
//            }
            }
//            (CoreTypeConcrete::Sint8(_), JitValue::Sint8(value)) => {
            (CoreTypeConcrete::Sint8(_), JitValue::Sint8(value)) => {
//                self.push_aligned(get_integer_layout(8).align(), &[*value as u64]);
                self.push_aligned(get_integer_layout(8).align(), &[*value as u64]);
//            }
            }
//            (CoreTypeConcrete::NonZero(info), _) => {
            (CoreTypeConcrete::NonZero(info), _) => {
//                // TODO: Check that the value is indeed non-zero.
                // TODO: Check that the value is indeed non-zero.
//                let type_info = self.registry.get_type(&info.ty)?;
                let type_info = self.registry.get_type(&info.ty)?;
//                self.push(&info.ty, type_info, value)?;
                self.push(&info.ty, type_info, value)?;
//            }
            }
//            (CoreTypeConcrete::Snapshot(info), _) => {
            (CoreTypeConcrete::Snapshot(info), _) => {
//                let type_info = self.registry.get_type(&info.ty)?;
                let type_info = self.registry.get_type(&info.ty)?;
//                self.push(&info.ty, type_info, value)?;
                self.push(&info.ty, type_info, value)?;
//            }
            }
//            (
            (
//                CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::Secp256Point(_)),
                CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::Secp256Point(_)),
//                JitValue::Secp256K1Point { x, y } | JitValue::Secp256R1Point { x, y },
                JitValue::Secp256K1Point { x, y } | JitValue::Secp256R1Point { x, y },
//            ) => {
            ) => {
//                let x_data = unsafe { std::mem::transmute::<[u128; 2], [u64; 4]>([x.0, x.1]) };
                let x_data = unsafe { std::mem::transmute::<[u128; 2], [u64; 4]>([x.0, x.1]) };
//                let y_data = unsafe { std::mem::transmute::<[u128; 2], [u64; 4]>([y.0, y.1]) };
                let y_data = unsafe { std::mem::transmute::<[u128; 2], [u64; 4]>([y.0, y.1]) };
//

//                self.push_aligned(get_integer_layout(252).align(), &x_data);
                self.push_aligned(get_integer_layout(252).align(), &x_data);
//                self.push_aligned(get_integer_layout(252).align(), &y_data);
                self.push_aligned(get_integer_layout(252).align(), &y_data);
//            }
            }
//            (CoreTypeConcrete::Bitwise(_), JitValue::Uint64(value))
            (CoreTypeConcrete::Bitwise(_), JitValue::Uint64(value))
//            | (CoreTypeConcrete::BuiltinCosts(_), JitValue::Uint64(value))
            | (CoreTypeConcrete::BuiltinCosts(_), JitValue::Uint64(value))
//            | (CoreTypeConcrete::EcOp(_), JitValue::Uint64(value))
            | (CoreTypeConcrete::EcOp(_), JitValue::Uint64(value))
//            | (CoreTypeConcrete::Pedersen(_), JitValue::Uint64(value))
            | (CoreTypeConcrete::Pedersen(_), JitValue::Uint64(value))
//            | (CoreTypeConcrete::Poseidon(_), JitValue::Uint64(value))
            | (CoreTypeConcrete::Poseidon(_), JitValue::Uint64(value))
//            | (CoreTypeConcrete::RangeCheck(_), JitValue::Uint64(value))
            | (CoreTypeConcrete::RangeCheck(_), JitValue::Uint64(value))
//            | (CoreTypeConcrete::SegmentArena(_), JitValue::Uint64(value)) => {
            | (CoreTypeConcrete::SegmentArena(_), JitValue::Uint64(value)) => {
//                self.push_aligned(get_integer_layout(64).align(), &[*value])
                self.push_aligned(get_integer_layout(64).align(), &[*value])
//            }
            }
//            (_, _) => todo!(),
            (_, _) => todo!(),
//        }
        }
//

//        Ok(())
        Ok(())
//    }
    }
//}
}
//

///// Parses the result by reading from the return ptr the given type.
/// Parses the result by reading from the return ptr the given type.
//fn parse_result(
fn parse_result(
//    type_id: &ConcreteTypeId,
    type_id: &ConcreteTypeId,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    mut return_ptr: Option<NonNull<()>>,
    mut return_ptr: Option<NonNull<()>>,
//    #[cfg(target_arch = "x86_64")] mut ret_registers: [u64; 2],
    #[cfg(target_arch = "x86_64")] mut ret_registers: [u64; 2],
//    #[cfg(target_arch = "aarch64")] mut ret_registers: [u64; 4],
    #[cfg(target_arch = "aarch64")] mut ret_registers: [u64; 4],
//) -> Result<JitValue, Error> {
) -> Result<JitValue, Error> {
//    let type_info = registry.get_type(type_id).unwrap();
    let type_info = registry.get_type(type_id).unwrap();
//

//    // Align the pointer to the actual return value.
    // Align the pointer to the actual return value.
//    if let Some(return_ptr) = &mut return_ptr {
    if let Some(return_ptr) = &mut return_ptr {
//        let layout = type_info.layout(registry).unwrap();
        let layout = type_info.layout(registry).unwrap();
//        let align_offset = return_ptr
        let align_offset = return_ptr
//            .cast::<u8>()
            .cast::<u8>()
//            .as_ptr()
            .as_ptr()
//            .align_offset(layout.align());
            .align_offset(layout.align());
//

//        *return_ptr = unsafe {
        *return_ptr = unsafe {
//            NonNull::new(return_ptr.cast::<u8>().as_ptr().add(align_offset))
            NonNull::new(return_ptr.cast::<u8>().as_ptr().add(align_offset))
//                .expect("nonnull is null")
                .expect("nonnull is null")
//                .cast()
                .cast()
//        };
        };
//    }
    }
//

//    match type_info {
    match type_info {
//        CoreTypeConcrete::Array(_) => {
        CoreTypeConcrete::Array(_) => {
//            Ok(JitValue::from_jit(return_ptr.unwrap(), type_id, registry))
            Ok(JitValue::from_jit(return_ptr.unwrap(), type_id, registry))
//        }
        }
//        CoreTypeConcrete::Box(info) => unsafe {
        CoreTypeConcrete::Box(info) => unsafe {
//            let ptr = return_ptr.unwrap_or(NonNull::new_unchecked(ret_registers[0] as *mut ()));
            let ptr = return_ptr.unwrap_or(NonNull::new_unchecked(ret_registers[0] as *mut ()));
//            let value = JitValue::from_jit(ptr, &info.ty, registry);
            let value = JitValue::from_jit(ptr, &info.ty, registry);
//            libc::free(ptr.cast().as_ptr());
            libc::free(ptr.cast().as_ptr());
//            Ok(value)
            Ok(value)
//        },
        },
//        CoreTypeConcrete::EcPoint(_) => {
        CoreTypeConcrete::EcPoint(_) => {
//            Ok(JitValue::from_jit(return_ptr.unwrap(), type_id, registry))
            Ok(JitValue::from_jit(return_ptr.unwrap(), type_id, registry))
//        }
        }
//        CoreTypeConcrete::EcState(_) => {
        CoreTypeConcrete::EcState(_) => {
//            Ok(JitValue::from_jit(return_ptr.unwrap(), type_id, registry))
            Ok(JitValue::from_jit(return_ptr.unwrap(), type_id, registry))
//        }
        }
//        CoreTypeConcrete::Felt252(_)
        CoreTypeConcrete::Felt252(_)
//        | CoreTypeConcrete::StarkNet(
        | CoreTypeConcrete::StarkNet(
//            StarkNetTypeConcrete::ClassHash(_)
            StarkNetTypeConcrete::ClassHash(_)
//            | StarkNetTypeConcrete::ContractAddress(_)
            | StarkNetTypeConcrete::ContractAddress(_)
//            | StarkNetTypeConcrete::StorageAddress(_)
            | StarkNetTypeConcrete::StorageAddress(_)
//            | StarkNetTypeConcrete::StorageBaseAddress(_),
            | StarkNetTypeConcrete::StorageBaseAddress(_),
//        ) => match return_ptr {
        ) => match return_ptr {
//            Some(return_ptr) => Ok(JitValue::from_jit(return_ptr, type_id, registry)),
            Some(return_ptr) => Ok(JitValue::from_jit(return_ptr, type_id, registry)),
//            None => {
            None => {
//                #[cfg(target_arch = "x86_64")]
                #[cfg(target_arch = "x86_64")]
//                // Since x86_64's return values hold at most two different 64bit registers,
                // Since x86_64's return values hold at most two different 64bit registers,
//                // everything bigger than u128 will be returned by memory, therefore making
                // everything bigger than u128 will be returned by memory, therefore making
//                // this branch is unreachable on that architecture.
                // this branch is unreachable on that architecture.
//                return Err(Error::ParseAttributeError);
                return Err(Error::ParseAttributeError);
//

//                #[cfg(target_arch = "aarch64")]
                #[cfg(target_arch = "aarch64")]
//                Ok(JitValue::Felt252(
                Ok(JitValue::Felt252(
//                    starknet_types_core::felt::Felt::from_bytes_le(unsafe {
                    starknet_types_core::felt::Felt::from_bytes_le(unsafe {
//                        std::mem::transmute::<&[u64; 4], &[u8; 32]>(&ret_registers)
                        std::mem::transmute::<&[u64; 4], &[u8; 32]>(&ret_registers)
//                    }),
                    }),
//                ))
                ))
//            }
            }
//        },
        },
//        CoreTypeConcrete::Bytes31(_) => match return_ptr {
        CoreTypeConcrete::Bytes31(_) => match return_ptr {
//            Some(return_ptr) => Ok(JitValue::from_jit(return_ptr, type_id, registry)),
            Some(return_ptr) => Ok(JitValue::from_jit(return_ptr, type_id, registry)),
//            None => {
            None => {
//                #[cfg(target_arch = "x86_64")]
                #[cfg(target_arch = "x86_64")]
//                // Since x86_64's return values hold at most two different 64bit registers,
                // Since x86_64's return values hold at most two different 64bit registers,
//                // everything bigger than u128 will be returned by memory, therefore making
                // everything bigger than u128 will be returned by memory, therefore making
//                // this branch is unreachable on that architecture.
                // this branch is unreachable on that architecture.
//                return Err(Error::ParseAttributeError);
                return Err(Error::ParseAttributeError);
//

//                #[cfg(target_arch = "aarch64")]
                #[cfg(target_arch = "aarch64")]
//                Ok(JitValue::Bytes31(unsafe {
                Ok(JitValue::Bytes31(unsafe {
//                    *std::mem::transmute::<&[u64; 4], &[u8; 31]>(&ret_registers)
                    *std::mem::transmute::<&[u64; 4], &[u8; 31]>(&ret_registers)
//                }))
                }))
//            }
            }
//        },
        },
//        CoreTypeConcrete::Uint8(_) => match return_ptr {
        CoreTypeConcrete::Uint8(_) => match return_ptr {
//            Some(return_ptr) => Ok(JitValue::Uint8(unsafe { *return_ptr.cast().as_ref() })),
            Some(return_ptr) => Ok(JitValue::Uint8(unsafe { *return_ptr.cast().as_ref() })),
//            None => Ok(JitValue::Uint8(ret_registers[0] as u8)),
            None => Ok(JitValue::Uint8(ret_registers[0] as u8)),
//        },
        },
//        CoreTypeConcrete::Uint16(_) => match return_ptr {
        CoreTypeConcrete::Uint16(_) => match return_ptr {
//            Some(return_ptr) => Ok(JitValue::Uint16(unsafe { *return_ptr.cast().as_ref() })),
            Some(return_ptr) => Ok(JitValue::Uint16(unsafe { *return_ptr.cast().as_ref() })),
//            None => Ok(JitValue::Uint16(ret_registers[0] as u16)),
            None => Ok(JitValue::Uint16(ret_registers[0] as u16)),
//        },
        },
//        CoreTypeConcrete::Uint32(_) => match return_ptr {
        CoreTypeConcrete::Uint32(_) => match return_ptr {
//            Some(return_ptr) => Ok(JitValue::Uint32(unsafe { *return_ptr.cast().as_ref() })),
            Some(return_ptr) => Ok(JitValue::Uint32(unsafe { *return_ptr.cast().as_ref() })),
//            None => Ok(JitValue::Uint32(ret_registers[0] as u32)),
            None => Ok(JitValue::Uint32(ret_registers[0] as u32)),
//        },
        },
//        CoreTypeConcrete::Uint64(_) => match return_ptr {
        CoreTypeConcrete::Uint64(_) => match return_ptr {
//            Some(return_ptr) => Ok(JitValue::Uint64(unsafe { *return_ptr.cast().as_ref() })),
            Some(return_ptr) => Ok(JitValue::Uint64(unsafe { *return_ptr.cast().as_ref() })),
//            None => Ok(JitValue::Uint64(ret_registers[0])),
            None => Ok(JitValue::Uint64(ret_registers[0])),
//        },
        },
//        CoreTypeConcrete::Uint128(_) => match return_ptr {
        CoreTypeConcrete::Uint128(_) => match return_ptr {
//            Some(return_ptr) => Ok(JitValue::Uint128(unsafe { *return_ptr.cast().as_ref() })),
            Some(return_ptr) => Ok(JitValue::Uint128(unsafe { *return_ptr.cast().as_ref() })),
//            None => Ok(JitValue::Uint128(
            None => Ok(JitValue::Uint128(
//                ((ret_registers[1] as u128) << 64) | ret_registers[0] as u128,
                ((ret_registers[1] as u128) << 64) | ret_registers[0] as u128,
//            )),
            )),
//        },
        },
//        CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
        CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
//        CoreTypeConcrete::Sint8(_) => match return_ptr {
        CoreTypeConcrete::Sint8(_) => match return_ptr {
//            Some(return_ptr) => Ok(JitValue::Sint8(unsafe { *return_ptr.cast().as_ref() })),
            Some(return_ptr) => Ok(JitValue::Sint8(unsafe { *return_ptr.cast().as_ref() })),
//            None => Ok(JitValue::Sint8(ret_registers[0] as i8)),
            None => Ok(JitValue::Sint8(ret_registers[0] as i8)),
//        },
        },
//        CoreTypeConcrete::Sint16(_) => match return_ptr {
        CoreTypeConcrete::Sint16(_) => match return_ptr {
//            Some(return_ptr) => Ok(JitValue::Sint16(unsafe { *return_ptr.cast().as_ref() })),
            Some(return_ptr) => Ok(JitValue::Sint16(unsafe { *return_ptr.cast().as_ref() })),
//            None => Ok(JitValue::Sint16(ret_registers[0] as i16)),
            None => Ok(JitValue::Sint16(ret_registers[0] as i16)),
//        },
        },
//        CoreTypeConcrete::Sint32(_) => match return_ptr {
        CoreTypeConcrete::Sint32(_) => match return_ptr {
//            Some(return_ptr) => Ok(JitValue::Sint32(unsafe { *return_ptr.cast().as_ref() })),
            Some(return_ptr) => Ok(JitValue::Sint32(unsafe { *return_ptr.cast().as_ref() })),
//            None => Ok(JitValue::Sint32(ret_registers[0] as i32)),
            None => Ok(JitValue::Sint32(ret_registers[0] as i32)),
//        },
        },
//        CoreTypeConcrete::Sint64(_) => match return_ptr {
        CoreTypeConcrete::Sint64(_) => match return_ptr {
//            Some(return_ptr) => Ok(JitValue::Uint64(unsafe { *return_ptr.cast().as_ref() })),
            Some(return_ptr) => Ok(JitValue::Uint64(unsafe { *return_ptr.cast().as_ref() })),
//            None => Ok(JitValue::Sint64(ret_registers[0] as i64)),
            None => Ok(JitValue::Sint64(ret_registers[0] as i64)),
//        },
        },
//        CoreTypeConcrete::Sint128(_) => match return_ptr {
        CoreTypeConcrete::Sint128(_) => match return_ptr {
//            Some(return_ptr) => Ok(JitValue::Uint128(unsafe { *return_ptr.cast().as_ref() })),
            Some(return_ptr) => Ok(JitValue::Uint128(unsafe { *return_ptr.cast().as_ref() })),
//            None => Ok(JitValue::Sint128(
            None => Ok(JitValue::Sint128(
//                ((ret_registers[1] as i128) << 64) | ret_registers[0] as i128,
                ((ret_registers[1] as i128) << 64) | ret_registers[0] as i128,
//            )),
            )),
//        },
        },
//        CoreTypeConcrete::NonZero(info) => {
        CoreTypeConcrete::NonZero(info) => {
//            parse_result(&info.ty, registry, return_ptr, ret_registers)
            parse_result(&info.ty, registry, return_ptr, ret_registers)
//        }
        }
//        CoreTypeConcrete::Nullable(info) => unsafe {
        CoreTypeConcrete::Nullable(info) => unsafe {
//            let ptr = return_ptr.map_or(ret_registers[0] as *mut (), |x| {
            let ptr = return_ptr.map_or(ret_registers[0] as *mut (), |x| {
//                *x.cast::<*mut ()>().as_ref()
                *x.cast::<*mut ()>().as_ref()
//            });
            });
//            if ptr.is_null() {
            if ptr.is_null() {
//                Ok(JitValue::Null)
                Ok(JitValue::Null)
//            } else {
            } else {
//                let ptr = NonNull::new_unchecked(ptr);
                let ptr = NonNull::new_unchecked(ptr);
//                let value = JitValue::from_jit(ptr, &info.ty, registry);
                let value = JitValue::from_jit(ptr, &info.ty, registry);
//                libc::free(ptr.as_ptr().cast());
                libc::free(ptr.as_ptr().cast());
//                Ok(value)
                Ok(value)
//            }
            }
//        },
        },
//        CoreTypeConcrete::Uninitialized(_) => todo!(),
        CoreTypeConcrete::Uninitialized(_) => todo!(),
//        CoreTypeConcrete::Enum(info) => {
        CoreTypeConcrete::Enum(info) => {
//            let (_, tag_layout, variant_layouts) =
            let (_, tag_layout, variant_layouts) =
//                crate::types::r#enum::get_layout_for_variants(registry, &info.variants).unwrap();
                crate::types::r#enum::get_layout_for_variants(registry, &info.variants).unwrap();
//

//            let (tag, ptr) = if type_info.is_memory_allocated(registry) || return_ptr.is_some() {
            let (tag, ptr) = if type_info.is_memory_allocated(registry) || return_ptr.is_some() {
//                let ptr = return_ptr.unwrap();
                let ptr = return_ptr.unwrap();
//

//                let tag = unsafe {
                let tag = unsafe {
//                    match tag_layout.size() {
                    match tag_layout.size() {
//                        0 => 0,
                        0 => 0,
//                        1 => *ptr.cast::<u8>().as_ref() as usize,
                        1 => *ptr.cast::<u8>().as_ref() as usize,
//                        2 => *ptr.cast::<u16>().as_ref() as usize,
                        2 => *ptr.cast::<u16>().as_ref() as usize,
//                        4 => *ptr.cast::<u32>().as_ref() as usize,
                        4 => *ptr.cast::<u32>().as_ref() as usize,
//                        8 => *ptr.cast::<u64>().as_ref() as usize,
                        8 => *ptr.cast::<u64>().as_ref() as usize,
//                        _ => return Err(Error::ParseAttributeError),
                        _ => return Err(Error::ParseAttributeError),
//                    }
                    }
//                };
                };
//

//                (
                (
//                    tag,
                    tag,
//                    Ok(unsafe {
                    Ok(unsafe {
//                        NonNull::new_unchecked(
                        NonNull::new_unchecked(
//                            ptr.cast::<u8>()
                            ptr.cast::<u8>()
//                                .as_ptr()
                                .as_ptr()
//                                .add(tag_layout.extend(variant_layouts[tag]).unwrap().1),
                                .add(tag_layout.extend(variant_layouts[tag]).unwrap().1),
//                        )
                        )
//                        .cast()
                        .cast()
//                    }),
                    }),
//                )
                )
//            } else {
            } else {
//                match info.variants.len() {
                match info.variants.len() {
//                    0 | 1 => (0, Err(0)),
                    0 | 1 => (0, Err(0)),
//                    _ => (
                    _ => (
//                        match tag_layout.size() {
                        match tag_layout.size() {
//                            1 => ret_registers[0] as u8 as usize,
                            1 => ret_registers[0] as u8 as usize,
//                            2 => ret_registers[0] as u16 as usize,
                            2 => ret_registers[0] as u16 as usize,
//                            4 => ret_registers[0] as u32 as usize,
                            4 => ret_registers[0] as u32 as usize,
//                            8 => ret_registers[0] as usize,
                            8 => ret_registers[0] as usize,
//                            _ => return Err(Error::ParseAttributeError),
                            _ => return Err(Error::ParseAttributeError),
//                        },
                        },
//                        Err(1),
                        Err(1),
//                    ),
                    ),
//                }
                }
//            };
            };
//

//            let value = match ptr {
            let value = match ptr {
//                Ok(ptr) => Box::new(JitValue::from_jit(ptr, &info.variants[tag], registry)),
                Ok(ptr) => Box::new(JitValue::from_jit(ptr, &info.variants[tag], registry)),
//                Err(offset) => {
                Err(offset) => {
//                    ret_registers.copy_within(offset.., 0);
                    ret_registers.copy_within(offset.., 0);
//                    Box::new(parse_result(
                    Box::new(parse_result(
//                        &info.variants[tag],
                        &info.variants[tag],
//                        registry,
                        registry,
//                        None,
                        None,
//                        ret_registers,
                        ret_registers,
//                    )?)
                    )?)
//                }
                }
//            };
            };
//

//            Ok(JitValue::Enum {
            Ok(JitValue::Enum {
//                tag,
                tag,
//                value,
                value,
//                debug_name: type_id.debug_name.as_deref().map(ToString::to_string),
                debug_name: type_id.debug_name.as_deref().map(ToString::to_string),
//            })
            })
//        }
        }
//        CoreTypeConcrete::Struct(info) => {
        CoreTypeConcrete::Struct(info) => {
//            if info.members.is_empty() {
            if info.members.is_empty() {
//                Ok(JitValue::Struct {
                Ok(JitValue::Struct {
//                    fields: Vec::new(),
                    fields: Vec::new(),
//                    debug_name: type_id.debug_name.as_deref().map(ToString::to_string),
                    debug_name: type_id.debug_name.as_deref().map(ToString::to_string),
//                })
                })
//            } else {
            } else {
//                Ok(JitValue::from_jit(return_ptr.unwrap(), type_id, registry))
                Ok(JitValue::from_jit(return_ptr.unwrap(), type_id, registry))
//            }
            }
//        }
        }
//        CoreTypeConcrete::Felt252Dict(_) => match return_ptr {
        CoreTypeConcrete::Felt252Dict(_) => match return_ptr {
//            Some(return_ptr) => Ok(JitValue::from_jit(
            Some(return_ptr) => Ok(JitValue::from_jit(
//                unsafe { *return_ptr.cast::<NonNull<()>>().as_ref() },
                unsafe { *return_ptr.cast::<NonNull<()>>().as_ref() },
//                type_id,
                type_id,
//                registry,
                registry,
//            )),
            )),
//            None => Ok(JitValue::from_jit(
            None => Ok(JitValue::from_jit(
//                NonNull::new(ret_registers[0] as *mut ()).unwrap(),
                NonNull::new(ret_registers[0] as *mut ()).unwrap(),
//                type_id,
                type_id,
//                registry,
                registry,
//            )),
            )),
//        },
        },
//        CoreTypeConcrete::Felt252DictEntry(_) => todo!(),
        CoreTypeConcrete::Felt252DictEntry(_) => todo!(),
//        CoreTypeConcrete::SquashedFelt252Dict(_) => match return_ptr {
        CoreTypeConcrete::SquashedFelt252Dict(_) => match return_ptr {
//            Some(return_ptr) => Ok(JitValue::from_jit(
            Some(return_ptr) => Ok(JitValue::from_jit(
//                unsafe { *return_ptr.cast::<NonNull<()>>().as_ref() },
                unsafe { *return_ptr.cast::<NonNull<()>>().as_ref() },
//                type_id,
                type_id,
//                registry,
                registry,
//            )),
            )),
//            None => Ok(JitValue::from_jit(
            None => Ok(JitValue::from_jit(
//                NonNull::new(ret_registers[0] as *mut ()).unwrap(),
                NonNull::new(ret_registers[0] as *mut ()).unwrap(),
//                type_id,
                type_id,
//                registry,
                registry,
//            )),
            )),
//        },
        },
//        CoreTypeConcrete::Span(_) => todo!(),
        CoreTypeConcrete::Span(_) => todo!(),
//        CoreTypeConcrete::Snapshot(_) => todo!(),
        CoreTypeConcrete::Snapshot(_) => todo!(),
//        CoreTypeConcrete::Bitwise(_) => todo!(),
        CoreTypeConcrete::Bitwise(_) => todo!(),
//        CoreTypeConcrete::Const(_) => todo!(),
        CoreTypeConcrete::Const(_) => todo!(),
//        CoreTypeConcrete::EcOp(_) => todo!(),
        CoreTypeConcrete::EcOp(_) => todo!(),
//        CoreTypeConcrete::GasBuiltin(_) => Ok(JitValue::Struct {
        CoreTypeConcrete::GasBuiltin(_) => Ok(JitValue::Struct {
//            fields: Vec::new(),
            fields: Vec::new(),
//            debug_name: type_id.debug_name.as_deref().map(ToString::to_string),
            debug_name: type_id.debug_name.as_deref().map(ToString::to_string),
//        }),
        }),
//        CoreTypeConcrete::BuiltinCosts(_) => todo!(),
        CoreTypeConcrete::BuiltinCosts(_) => todo!(),
//        CoreTypeConcrete::RangeCheck(_) => todo!(),
        CoreTypeConcrete::RangeCheck(_) => todo!(),
//        CoreTypeConcrete::Pedersen(_) => todo!(),
        CoreTypeConcrete::Pedersen(_) => todo!(),
//        CoreTypeConcrete::Poseidon(_) => todo!(),
        CoreTypeConcrete::Poseidon(_) => todo!(),
//        CoreTypeConcrete::SegmentArena(_) => todo!(),
        CoreTypeConcrete::SegmentArena(_) => todo!(),
//        CoreTypeConcrete::BoundedInt(_) => todo!(),
        CoreTypeConcrete::BoundedInt(_) => todo!(),
//        _ => todo!(),
        _ => todo!(),
//    }
    }
//}
}
