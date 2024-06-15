//! # Executors
//!
//! This module provides methods to execute the programs, either via JIT or compiled ahead
//! of time. It also provides a cache to avoid recompiling previously compiled programs.
//! PLT: maybe extract the cache to a separate module to avoid mixing mechanism with policy.

pub use self::{aot::AotNativeExecutor, jit::JitNativeExecutor};
use crate::{
    error::Error,
    execution_result::{BuiltinStats, ContractExecutionResult, ExecutionResult},
    starknet::{handler::StarknetSyscallHandlerCallbacks, StarknetSyscallHandler},
    types::TypeBuilder,
    utils::get_integer_layout,
    values::JitValue,
};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        starknet::StarkNetTypeConcrete,
    },
    ids::{ConcreteTypeId, FunctionId},
    program::FunctionSignature,
    program_registry::{ProgramRegistry, ProgramRegistryError},
};
use libc::c_void;
use starknet_types_core::felt::Felt;
use std::{
    alloc::Layout,
    arch::global_asm,
    ptr::{addr_of_mut, null_mut, NonNull},
    rc::Rc,
};

mod aot;
mod jit;

#[cfg(target_arch = "aarch64")]
global_asm!(include_str!("arch/aarch64.s"));
#[cfg(target_arch = "x86_64")]
global_asm!(include_str!("arch/x86_64.s"));
// PLT: do we need a third `cfg` to fail compilation when trying
// to build for an unsupported arch?

extern "C" {
    /// Invoke an AOT or JIT-compiled function.
    ///
    /// The `ret_ptr` argument is only used when the first argument (the actual return pointer) is
    /// unused. Used for u8, u16, u32, u64, u128 and felt252, but not for arrays, enums or structs.
    /// PLT: what happens on MacOS? What happens on *BSD? I think this is implicitly for Linux
    /// only. Unsupported targets should have an explicit build time failure with a reasonable
    /// message.
    #[cfg_attr(not(target_os = "macos"), link_name = "_invoke_trampoline")]
    fn invoke_trampoline(
        fn_ptr: *const c_void,
        args_ptr: *const u64,
        args_len: usize,
        ret_ptr: *mut u64,
    );
}

/// The cairo native executor, either AOT or JIT based.
#[derive(Debug, Clone)]
pub enum NativeExecutor<'m> {
    Aot(Rc<AotNativeExecutor>),
    Jit(Rc<JitNativeExecutor<'m>>),
}

impl<'a> NativeExecutor<'a> {
    /// Invoke the given function by its function id, with the given arguments and gas.
    pub fn invoke_dynamic(
        &self,
        function_id: &FunctionId,
        args: &[JitValue],
        gas: Option<u128>,
    ) -> Result<ExecutionResult, Error> {
        match self {
            NativeExecutor::Aot(executor) => executor.invoke_dynamic(function_id, args, gas),
            NativeExecutor::Jit(executor) => executor.invoke_dynamic(function_id, args, gas),
        }
    }

    /// Invoke the given function by its function id, with the given arguments and gas.
    /// This should be used for programs which require a syscall handler, whose
    /// implementation should be passed on.
    pub fn invoke_dynamic_with_syscall_handler(
        &self,
        function_id: &FunctionId,
        args: &[JitValue],
        gas: Option<u128>,
        syscall_handler: impl StarknetSyscallHandler,
    ) -> Result<ExecutionResult, Error> {
        match self {
            NativeExecutor::Aot(executor) => executor.invoke_dynamic_with_syscall_handler(
                function_id,
                args,
                gas,
                syscall_handler,
            ),
            NativeExecutor::Jit(executor) => executor.invoke_dynamic_with_syscall_handler(
                function_id,
                args,
                gas,
                syscall_handler,
            ),
        }
    }

    /// Invoke the given function by its function id, with the given arguments and gas.
    /// This should be used for starknet contracts which require a syscall handler, whose
    /// implementation should be passed on.
    pub fn invoke_contract_dynamic(
        &self,
        function_id: &FunctionId,
        args: &[Felt],
        gas: Option<u128>,
        syscall_handler: impl StarknetSyscallHandler,
    ) -> Result<ContractExecutionResult, Error> {
        match self {
            NativeExecutor::Aot(executor) => {
                executor.invoke_contract_dynamic(function_id, args, gas, syscall_handler)
            }
            NativeExecutor::Jit(executor) => {
                executor.invoke_contract_dynamic(function_id, args, gas, syscall_handler)
            }
        }
    }
}

impl<'m> From<AotNativeExecutor> for NativeExecutor<'m> {
    fn from(value: AotNativeExecutor) -> Self {
        Self::Aot(Rc::new(value))
    }
}

impl<'m> From<JitNativeExecutor<'m>> for NativeExecutor<'m> {
    fn from(value: JitNativeExecutor<'m>) -> Self {
        Self::Jit(Rc::new(value))
    }
}

/// Internal method.
///
/// Invokes the given function by constructing the function call depending on the arguments given.
/// Usually calling a function requires knowing it's signature at compile time, but we need to be
/// PLT: s/signatue/signature/
/// able to call any given function provided it's signatue (arguments and return type) at runtime,
/// to do so we have a "trampoline" in the given platform assembly (x86_64, aarch64) which
/// constructs the function call in place.
///
/// To pass the arguments, they are stored in a arena.
fn invoke_dynamic(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    function_ptr: *const c_void,
    function_signature: &FunctionSignature,
    args: &[JitValue],
    gas: u128,
    mut syscall_handler: Option<impl StarknetSyscallHandler>,
) -> Result<ExecutionResult, Error> {
    tracing::info!("Invoking function with signature: {function_signature:?}.");
    let arena = Bump::new();
    // PLT: check what `ArgumentMapper` does with `arena`.
    // PLT: check if we may need to preallocate here. Or if there is any real advantage from using
    // bumpalo here.
    let mut invoke_data = ArgumentMapper::new(&arena, registry);

    // Generate return pointer (if necessary).
    //
    // Generated when either:
    //   - There are more than one non-zst return values.
    //     - All builtins except GasBuiltin and Starknet are ZST.
    //     - The unit struct is a ZST.
    //     PLT: empty enums and structs are ZSTs. May be obvious, but since we list unit struct I
    //     guess we're trying to be exhaustive here.
    //     PLT: it may be sensible to talk about "compound" rather than "complex" types.
    //   - The return argument is complex.
    let mut ret_types_iter = function_signature
        .ret_types
        .iter()
        .filter(|id| {
            let info = registry.get_type(id).unwrap();
            !(info.is_builtin() && info.is_zst(registry))
        })
        .peekable();

    let num_return_args = ret_types_iter.clone().count();
    // PLT: I think we could get away with implementing these type-bound methods for
    // `Option<T: Type>` with `Some(v)` forwarding and `None` returning values that make sense
    // (say, `None` may behave like the unit struct).
    // That would cleanly remove all the `unwrap`s here.
    // Alternatively, at least add a `debug_assert!` (or even `assert!` if we feel confident) that
    // checks that all types in the function signature have a matching entry in the registry.
    let mut return_ptr = if num_return_args > 1
        // PLT: this is rather hard to grok here, maybe extract to a variable.
        || ret_types_iter
            .peek()
            .is_some_and(|id| registry.get_type(id).unwrap().is_complex(registry))
    {
        let layout = ret_types_iter.fold(Layout::new::<()>(), |layout, id| {
            let type_info = registry.get_type(id).unwrap();
            layout
                .extend(type_info.layout(registry).unwrap())
                .unwrap()
                .0
        });

        let return_ptr = arena.alloc_layout(layout).cast::<()>();
        invoke_data.push_aligned(
            get_integer_layout(64).align(),
            &[return_ptr.as_ptr() as u64],
        );

        Some(return_ptr)
    } else {
        None
    };

    // The Cairo compiler doesn't specify that the cheatcode syscall needs the syscall handler,
    // so we must always allocate it in case it needs it, regardless of whether it's passed
    // as an argument to the entry point or not.
    let mut syscall_handler = syscall_handler
        .as_mut()
        .map(|syscall_handler| StarknetSyscallHandlerCallbacks::new(syscall_handler));
    // We only care for the previous syscall handler if we actually modify it
    #[cfg(feature = "with-cheatcode")]
    let previous_syscall_handler = syscall_handler.as_mut().map(|syscall_handler| {
        let previous_syscall_handler = crate::starknet::SYSCALL_HANDLER_VTABLE.get();
        let syscall_handler_ptr = std::ptr::addr_of!(*syscall_handler) as *mut ();
        crate::starknet::SYSCALL_HANDLER_VTABLE.set(syscall_handler_ptr);

        previous_syscall_handler
    });

    // Generate argument list.
    let mut iter = args.iter();
    for type_id in function_signature.param_types.iter().filter(|id| {
        let info = registry.get_type(id).unwrap();
        !info.is_zst(registry)
    }) {
        // Process gas requirements and syscall handler.
        match registry.get_type(type_id).unwrap() {
            CoreTypeConcrete::GasBuiltin(_) => invoke_data.push_aligned(
                get_integer_layout(128).align(),
                &[gas as u64, (gas >> 64) as u64],
            ),
            CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::System(_)) => {
                let syscall_handler = syscall_handler
                    .as_mut()
                    .expect("syscall handler is required");

                invoke_data.push_aligned(
                    get_integer_layout(64).align(),
                    &[syscall_handler as *mut _ as u64],
                );
            }
            type_info => invoke_data
                .push(
                    type_id,
                    type_info,
                    if type_info.is_builtin() {
                        &JitValue::Uint64(0)
                    } else {
                        // PLT: how do we ensure this is safe?
                        iter.next().unwrap()
                    },
                )
                .unwrap(),
        }
    }

    // Invoke the trampoline.
    #[cfg(target_arch = "x86_64")]
    let mut ret_registers = [0; 2];
    #[cfg(target_arch = "aarch64")]
    let mut ret_registers = [0; 4];

    unsafe {
        invoke_trampoline(
            function_ptr,
            invoke_data.invoke_data().as_ptr(),
            invoke_data.invoke_data().len(),
            ret_registers.as_mut_ptr(),
        );
    }

    // If the syscall handler was changed, then reset the previous one.
    // It's only necessary to restore the pointer if it's been modified i.e. if previous_syscall_handler is Some(...)
    #[cfg(feature = "with-cheatcode")]
    if let Some(previous_syscall_handler) = previous_syscall_handler {
        // PLT: will this work properly in concurrent settings?
        // Maybe we should pass an explicit vtable instead via context so they always belong only
        // to the executing program.
        crate::starknet::SYSCALL_HANDLER_VTABLE.set(previous_syscall_handler);
    }

    // Parse final gas.
    // PLT: this seems to implement an iterator-like logic for typed pointers. I would suggest
    // renaming to something like `next_value` or `consume_value`.
    unsafe fn read_value<T>(ptr: &mut NonNull<()>) -> &T {
        let align_offset = ptr
            .cast::<u8>()
            .as_ptr()
            .align_offset(std::mem::align_of::<T>());
        let value_ptr = ptr.cast::<u8>().as_ptr().add(align_offset).cast::<T>();

        *ptr = NonNull::new_unchecked(value_ptr.add(1)).cast();
        &*value_ptr
    }

    let mut remaining_gas = None;
    let mut builtin_stats = BuiltinStats::default();
    for type_id in &function_signature.ret_types {
        let type_info = registry.get_type(type_id).unwrap();
        match type_info {
            CoreTypeConcrete::GasBuiltin(_) => {
                remaining_gas = Some(match &mut return_ptr {
                    Some(return_ptr) => unsafe { *read_value::<u128>(return_ptr) },
                    None => {
                        // If there's no return ptr then the function only returned the gas. We don't
                        // need to bother with the syscall handler builtin.
                        ((ret_registers[1] as u128) << 64) | ret_registers[0] as u128
                    }
                });
            }
            CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::System(_)) => match &mut return_ptr {
                Some(return_ptr) => unsafe {
                    let ptr = return_ptr.cast::<*mut ()>();
                    *return_ptr = NonNull::new_unchecked(ptr.as_ptr().add(1)).cast();
                },
                None => {}
            },
            _ if type_info.is_builtin() => {
                if !type_info.is_zst(registry) {
                    let value = match &mut return_ptr {
                        Some(return_ptr) => unsafe { *read_value::<u64>(return_ptr) },
                        None => ret_registers[0],
                    } as usize;

                    match type_info {
                        CoreTypeConcrete::Bitwise(_) => builtin_stats.bitwise = value,
                        CoreTypeConcrete::EcOp(_) => builtin_stats.ec_op = value,
                        CoreTypeConcrete::RangeCheck(_) => builtin_stats.range_check = value,
                        CoreTypeConcrete::Pedersen(_) => builtin_stats.pedersen = value,
                        CoreTypeConcrete::Poseidon(_) => builtin_stats.poseidon = value,
                        CoreTypeConcrete::SegmentArena(_) => builtin_stats.segment_arena = value,
                        _ => unreachable!("{type_id:?}"),
                    }
                }
            }
            // PLT: is the order guaranteed? Otherwise, a non-builtin return coming before the
            // others would end the loop early and leave that data unhandled.
            _ => break,
        }
    }

    // Parse return values.
    let return_value = function_signature
        .ret_types
        .last()
        .and_then(|ret_type| {
            let type_info = registry.get_type(ret_type).unwrap();
            if type_info.is_builtin() {
                None
            } else {
                Some(parse_result(
                    ret_type,
                    registry,
                    return_ptr,
                    ret_registers,
                    // PLT: "can not have" means it is impossible for them to have one or that it
                    // is optional for them? I'd rephrase to "may not" in the latter case.
                    // TODO: Consider returning an Option<JitValue> as return_value instead
                    // As cairo functions can not have a return value
                ))
            }
        })
        .unwrap_or_else(|| {
            Ok(JitValue::Struct {
                fields: vec![],
                debug_name: None,
            })
        })?;

    // PLT: when will we fix this? Can the arena be privately part of `ExecutionResult` so we can
    // drop it when we're done with it?
    // FIXME: Arena deallocation.
    std::mem::forget(arena);

    Ok(ExecutionResult {
        remaining_gas,
        return_value,
        builtin_stats,
    })
}

pub struct ArgumentMapper<'a> {
    arena: &'a Bump,
    registry: &'a ProgramRegistry<CoreType, CoreLibfunc>,

    invoke_data: Vec<u64>,
}

impl<'a> ArgumentMapper<'a> {
    pub fn new(arena: &'a Bump, registry: &'a ProgramRegistry<CoreType, CoreLibfunc>) -> Self {
        Self {
            arena,
            registry,
            invoke_data: Vec::new(),
        }
    }

    pub fn invoke_data(&self) -> &[u64] {
        &self.invoke_data
    }

    pub fn push_aligned(&mut self, align: usize, mut values: &[u64]) {
        // PLT: I'd advise against panic in a public function for a library. Consider returning a
        // `Result` for invalid arguments.
        assert!(align.is_power_of_two());
        // PLT: is a >16 bytes alignment really forbidden? I've seen it used for structures meant
        // for SIMD operation.
        assert!(align <= 16);

        #[cfg(target_arch = "x86_64")]
        const NUM_REGISTER_ARGS: usize = 6;
        #[cfg(target_arch = "aarch64")]
        const NUM_REGISTER_ARGS: usize = 8;

        if align == 16 {
            // This works because on both aarch64 and x86_64 the stack is already aligned to
            // 16 bytes when the trampoline starts pushing values.

            // Whenever a value spans across multiple registers, if it's in a position where it would be split between
            // registers and the stack it must be padded so that the entire value is stored within the stack.
            // PLT: this needs better documentation.
            if self.invoke_data.len() >= NUM_REGISTER_ARGS {
                if self.invoke_data.len() & 1 != 0 {
                    self.invoke_data.push(0);
                }
            } else if self.invoke_data.len() + 1 >= NUM_REGISTER_ARGS {
                self.invoke_data.push(0);
            } else {
                let new_len = self.invoke_data.len() + values.len();
                if new_len >= NUM_REGISTER_ARGS && new_len % 2 != 0 {
                    let chunk;
                    // PLT: maybe use:
                    // ```
                    // (chunk, values) = values.split_at_checked(4)
                    //     .unwrap_or_else(|| (values, [].as_slice())
                    // ```
                    (chunk, values) = if values.len() >= 4 {
                        values.split_at(4)
                    } else {
                        (values, [].as_slice())
                    };
                    self.invoke_data.extend(chunk);
                    self.invoke_data.push(0);
                }
            }
        }

        self.invoke_data.extend(values);
    }

    pub fn push(
        &mut self,
        type_id: &ConcreteTypeId,
        type_info: &CoreTypeConcrete,
        value: &JitValue,
    ) -> Result<(), Box<ProgramRegistryError>> {
        match (type_info, value) {
            (CoreTypeConcrete::Array(info), JitValue::Array(values)) => {
                // PLT: yes, check and return an error.
                // TODO: Assert that `info.ty` matches all the values' types.

                let type_info = self.registry.get_type(&info.ty)?;
                // PLT: is success guaranteed?
                let type_layout = type_info.layout(self.registry).unwrap().pad_to_align();

                // This needs to be a heap-allocated pointer because it's the actual array data.
                // PLT: the if is not needed. Asking `malloc/realloc` for zero-sized allocations
                // is well defined. It will return either `NULL` or a different, unique pointer
                // that can be passed to `free`. For access we need to check bounds either way.
                let ptr = if values.is_empty() {
                    null_mut()
                } else {
                    // PLT: why not malloc?
                    // PLT: are ZST working correctly here? It needs to check on access to avoid
                    // accessing a zero-sized allocation later on.
                    unsafe { libc::realloc(null_mut(), type_layout.size() * values.len()) }
                };

                for (idx, value) in values.iter().enumerate() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            value
                                .to_jit(self.arena, self.registry, &info.ty)
                                .unwrap()
                                .cast()
                                .as_ptr(),
                            // PLT: this doesn't satisfy alignment AFAICT.
                            (ptr as usize + type_layout.size() * idx) as *mut u8,
                            type_layout.size(),
                        );
                    }
                }

                self.push_aligned(
                    get_integer_layout(64).align(),
                    &[ptr as u64, 0, values.len() as u64, values.len() as u64],
                );
            }
            (CoreTypeConcrete::EcPoint(_), JitValue::EcPoint(a, b)) => {
                let align = get_integer_layout(252).align();
                self.push_aligned(align, &a.to_le_digits());
                self.push_aligned(align, &b.to_le_digits());
            }
            (CoreTypeConcrete::EcState(_), JitValue::EcState(a, b, c, d)) => {
                let align = get_integer_layout(252).align();
                self.push_aligned(align, &a.to_le_digits());
                self.push_aligned(align, &b.to_le_digits());
                self.push_aligned(align, &c.to_le_digits());
                self.push_aligned(align, &d.to_le_digits());
            }
            (CoreTypeConcrete::Enum(info), JitValue::Enum { tag, value, .. }) => {
                if type_info.is_memory_allocated(self.registry) {
                    let (layout, tag_layout, variant_layouts) =
                        crate::types::r#enum::get_layout_for_variants(
                            self.registry,
                            &info.variants,
                        )
                        .unwrap();

                    let ptr = self.arena.alloc_layout(layout);
                    unsafe {
                        match tag_layout.size() {
                            0 => {}
                            1 => *ptr.cast::<u8>().as_mut() = *tag as u8,
                            2 => *ptr.cast::<u16>().as_mut() = *tag as u16,
                            4 => *ptr.cast::<u32>().as_mut() = *tag as u32,
                            8 => *ptr.cast::<u64>().as_mut() = *tag as u64,
                            // PLT: is repr(u128) forbidden? Signed reprs?
                            _ => unreachable!(),
                        }
                    }

                    let offset = tag_layout.extend(variant_layouts[*tag]).unwrap().1;
                    let payload_ptr = value
                        .to_jit(self.arena, self.registry, &info.variants[*tag])
                        .unwrap();
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            payload_ptr.cast::<u8>().as_ptr(),
                            ptr.cast::<u8>().as_ptr().add(offset),
                            variant_layouts[*tag].size(),
                        );
                    }

                    self.invoke_data.push(ptr.as_ptr() as u64);
                } else {
                    // Write the tag.
                    match (info.variants.len().next_power_of_two().trailing_zeros() + 7) / 8 {
                        0 => {}
                        _ => self.invoke_data.push(*tag as u64),
                    }

                    // Write the payload.
                    let type_info = self.registry.get_type(&info.variants[*tag]).unwrap();
                    self.push(&info.variants[*tag], type_info, value)?;
                }
            }
            (
                CoreTypeConcrete::Felt252(_)
                | CoreTypeConcrete::StarkNet(
                    StarkNetTypeConcrete::ClassHash(_)
                    | StarkNetTypeConcrete::ContractAddress(_)
                    | StarkNetTypeConcrete::StorageAddress(_)
                    | StarkNetTypeConcrete::StorageBaseAddress(_),
                ),
                JitValue::Felt252(value),
            ) => {
                self.push_aligned(get_integer_layout(252).align(), &value.to_le_digits());
            }
            (CoreTypeConcrete::Bytes31(_), JitValue::Bytes31(value)) => {
                self.push_aligned(
                    get_integer_layout(248).align(),
                    &Felt::from_bytes_be_slice(value).to_le_digits(),
                );
            }
            (CoreTypeConcrete::Felt252Dict(_), JitValue::Felt252Dict { .. }) => {
                // PLT: are we planning on implementing it?
                #[cfg(not(feature = "with-runtime"))]
                unimplemented!("enable the `with-runtime` feature to use felt252 dicts");

                // PLT: yes, do it. But return an error, don't panic.
                // TODO: Assert that `info.ty` matches all the values' types.

                self.invoke_data.push(
                    value
                        .to_jit(self.arena, self.registry, type_id)
                        // PLT: is success guaranteed?
                        .unwrap()
                        .as_ptr() as u64,
                );
            }
            (CoreTypeConcrete::Struct(info), JitValue::Struct { fields, .. }) => {
                for (field_type_id, field_value) in info.members.iter().zip(fields) {
                    self.push(
                        field_type_id,
                        self.registry.get_type(field_type_id)?,
                        field_value,
                    )?;
                }
            }
            // PLT: signedness doesn't matter here, so we can pair the arms.
            // E.g.: (CoreTypeConcrete::Uint64(_), JitValue::Uint64(value))|(CoreTypeConcrete::Sint64(_), JitValue::Sint64(value))
            (CoreTypeConcrete::Uint128(_), JitValue::Uint128(value)) => self.push_aligned(
                get_integer_layout(128).align(),
                &[*value as u64, (value >> 64) as u64],
            ),
            (CoreTypeConcrete::Uint64(_), JitValue::Uint64(value)) => {
                self.push_aligned(get_integer_layout(64).align(), &[*value]);
            }
            (CoreTypeConcrete::Uint32(_), JitValue::Uint32(value)) => {
                self.push_aligned(get_integer_layout(32).align(), &[*value as u64]);
            }
            (CoreTypeConcrete::Uint16(_), JitValue::Uint16(value)) => {
                self.push_aligned(get_integer_layout(16).align(), &[*value as u64]);
            }
            (CoreTypeConcrete::Uint8(_), JitValue::Uint8(value)) => {
                self.push_aligned(get_integer_layout(8).align(), &[*value as u64]);
            }
            (CoreTypeConcrete::Sint128(_), JitValue::Sint128(value)) => {
                self.push_aligned(
                    get_integer_layout(128).align(),
                    &[*value as u64, (value >> 64) as u64],
                );
            }
            (CoreTypeConcrete::Sint64(_), JitValue::Sint64(value)) => {
                self.push_aligned(get_integer_layout(64).align(), &[*value as u64]);
            }
            (CoreTypeConcrete::Sint32(_), JitValue::Sint32(value)) => {
                self.push_aligned(get_integer_layout(32).align(), &[*value as u64]);
            }
            (CoreTypeConcrete::Sint16(_), JitValue::Sint16(value)) => {
                self.push_aligned(get_integer_layout(16).align(), &[*value as u64]);
            }
            (CoreTypeConcrete::Sint8(_), JitValue::Sint8(value)) => {
                self.push_aligned(get_integer_layout(8).align(), &[*value as u64]);
            }
            (CoreTypeConcrete::NonZero(info), _) => {
                // PLT: yes, check it.
                // TODO: Check that the value is indeed non-zero.
                let type_info = self.registry.get_type(&info.ty)?;
                self.push(&info.ty, type_info, value)?;
            }
            (CoreTypeConcrete::Snapshot(info), _) => {
                let type_info = self.registry.get_type(&info.ty)?;
                self.push(&info.ty, type_info, value)?;
            }
            (
                CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::Secp256Point(_)),
                JitValue::Secp256K1Point { x, y } | JitValue::Secp256R1Point { x, y },
            ) => {
                let x_data = unsafe { std::mem::transmute::<[u128; 2], [u64; 4]>([x.0, x.1]) };
                let y_data = unsafe { std::mem::transmute::<[u128; 2], [u64; 4]>([y.0, y.1]) };

                self.push_aligned(get_integer_layout(252).align(), &x_data);
                self.push_aligned(get_integer_layout(252).align(), &y_data);
            }
            (CoreTypeConcrete::Bitwise(_), JitValue::Uint64(value))
            | (CoreTypeConcrete::BuiltinCosts(_), JitValue::Uint64(value))
            | (CoreTypeConcrete::EcOp(_), JitValue::Uint64(value))
            | (CoreTypeConcrete::Pedersen(_), JitValue::Uint64(value))
            | (CoreTypeConcrete::Poseidon(_), JitValue::Uint64(value))
            | (CoreTypeConcrete::RangeCheck(_), JitValue::Uint64(value))
            | (CoreTypeConcrete::SegmentArena(_), JitValue::Uint64(value)) => {
                self.push_aligned(get_integer_layout(64).align(), &[*value])
            }
            (_, _) => todo!(),
        }

        Ok(())
    }
}

/// Parses the result by reading from the return ptr the given type.
fn parse_result(
    type_id: &ConcreteTypeId,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    mut return_ptr: Option<NonNull<()>>,
    // PLT: can we use the global const instead of cfg tags?
    #[cfg(target_arch = "x86_64")] mut ret_registers: [u64; 2],
    #[cfg(target_arch = "aarch64")] mut ret_registers: [u64; 4],
) -> Result<JitValue, Error> {
    let type_info = registry.get_type(type_id).unwrap();

    // Align the pointer to the actual return value.
    if let Some(return_ptr) = &mut return_ptr {
        let layout = type_info.layout(registry).unwrap();
        let align_offset = return_ptr
            .cast::<u8>()
            .as_ptr()
            .align_offset(layout.align());

        *return_ptr = unsafe {
            NonNull::new(return_ptr.cast::<u8>().as_ptr().add(align_offset))
                .expect("nonnull is null")
                .cast()
        };
    }

    match type_info {
        CoreTypeConcrete::Array(_) => {
            Ok(JitValue::from_jit(return_ptr.unwrap(), type_id, registry))
        }
        CoreTypeConcrete::Box(info) => unsafe {
            let ptr = return_ptr.unwrap_or(NonNull::new_unchecked(ret_registers[0] as *mut ()));
            let value = JitValue::from_jit(ptr, &info.ty, registry);
            libc::free(ptr.cast().as_ptr());
            Ok(value)
        },
        CoreTypeConcrete::EcPoint(_) | CoreTypeConcrete::EcState(_) => {
            Ok(JitValue::from_jit(return_ptr.unwrap(), type_id, registry))
        }
        CoreTypeConcrete::Felt252(_)
        | CoreTypeConcrete::StarkNet(
            StarkNetTypeConcrete::ClassHash(_)
            | StarkNetTypeConcrete::ContractAddress(_)
            | StarkNetTypeConcrete::StorageAddress(_)
            | StarkNetTypeConcrete::StorageBaseAddress(_),
        ) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::from_jit(return_ptr, type_id, registry)),
            None => {
                #[cfg(target_arch = "x86_64")]
                // Since x86_64's return values hold at most two different 64bit registers,
                // everything bigger than u128 will be returned by memory, therefore making
                // this branch is unreachable on that architecture.
                // PLT: consistency: either remove the other `unwrap`s in cases where data is
                // guaranteed to spill to RAM, or make this one an `unreachable!` instead of
                // an error.
                return Err(Error::ParseAttributeError);

                #[cfg(target_arch = "aarch64")]
                Ok(JitValue::Felt252(
                    starknet_types_core::felt::Felt::from_bytes_le(unsafe {
                        std::mem::transmute::<&[u64; 4], &[u8; 32]>(&ret_registers)
                    }),
                ))
            }
        },
        CoreTypeConcrete::Bytes31(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::from_jit(return_ptr, type_id, registry)),
            None => {
                #[cfg(target_arch = "x86_64")]
                // Since x86_64's return values hold at most two different 64bit registers,
                // everything bigger than u128 will be returned by memory, therefore making
                // this branch is unreachable on that architecture.
                // PLT: consistency: either remove the other `unwrap`s in cases where data is
                // guaranteed to spill to RAM, or make this one an `unreachable!` instead of
                // an error.
                return Err(Error::ParseAttributeError);

                #[cfg(target_arch = "aarch64")]
                Ok(JitValue::Bytes31(unsafe {
                    *std::mem::transmute::<&[u64; 4], &[u8; 31]>(&ret_registers)
                }))
            }
        },
        CoreTypeConcrete::Uint8(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::Uint8(unsafe { *return_ptr.cast().as_ref() })),
            None => Ok(JitValue::Uint8(ret_registers[0] as u8)),
        },
        CoreTypeConcrete::Uint16(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::Uint16(unsafe { *return_ptr.cast().as_ref() })),
            None => Ok(JitValue::Uint16(ret_registers[0] as u16)),
        },
        CoreTypeConcrete::Uint32(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::Uint32(unsafe { *return_ptr.cast().as_ref() })),
            None => Ok(JitValue::Uint32(ret_registers[0] as u32)),
        },
        CoreTypeConcrete::Uint64(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::Uint64(unsafe { *return_ptr.cast().as_ref() })),
            None => Ok(JitValue::Uint64(ret_registers[0])),
        },
        CoreTypeConcrete::Uint128(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::Uint128(unsafe { *return_ptr.cast().as_ref() })),
            None => Ok(JitValue::Uint128(
                ((ret_registers[1] as u128) << 64) | ret_registers[0] as u128,
            )),
        },
        CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
        CoreTypeConcrete::Sint8(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::Sint8(unsafe { *return_ptr.cast().as_ref() })),
            None => Ok(JitValue::Sint8(ret_registers[0] as i8)),
        },
        CoreTypeConcrete::Sint16(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::Sint16(unsafe { *return_ptr.cast().as_ref() })),
            None => Ok(JitValue::Sint16(ret_registers[0] as i16)),
        },
        CoreTypeConcrete::Sint32(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::Sint32(unsafe { *return_ptr.cast().as_ref() })),
            None => Ok(JitValue::Sint32(ret_registers[0] as i32)),
        },
        CoreTypeConcrete::Sint64(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::Uint64(unsafe { *return_ptr.cast().as_ref() })),
            None => Ok(JitValue::Sint64(ret_registers[0] as i64)),
        },
        CoreTypeConcrete::Sint128(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::Uint128(unsafe { *return_ptr.cast().as_ref() })),
            // PLT: check signed shift semantics for Rust.
            None => Ok(JitValue::Sint128(
                ((ret_registers[1] as i128) << 64) | ret_registers[0] as i128,
            )),
        },
        CoreTypeConcrete::NonZero(info) => {
            parse_result(&info.ty, registry, return_ptr, ret_registers)
        }
        CoreTypeConcrete::Nullable(info) => unsafe {
            let ptr = return_ptr.map_or(ret_registers[0] as *mut (), |x| {
                *x.cast::<*mut ()>().as_ref()
            });
            if ptr.is_null() {
                Ok(JitValue::Null)
            } else {
                let ptr = NonNull::new_unchecked(ptr);
                // PLT: TODO: check that `JitValue` doesn't keep a copy of the pointer.
                let value = JitValue::from_jit(ptr, &info.ty, registry);
                libc::free(ptr.as_ptr().cast());
                Ok(value)
            }
        },
        CoreTypeConcrete::Enum(info) => {
            let (_, tag_layout, variant_layouts) =
                crate::types::r#enum::get_layout_for_variants(registry, &info.variants).unwrap();

            let (tag, ptr) = if type_info.is_memory_allocated(registry) || return_ptr.is_some() {
                // PLT: this logic seems repeated for most of the enum handling, maybe it could be
                // abstracted into a function.
                let ptr = return_ptr.unwrap();

                let tag = unsafe {
                    match tag_layout.size() {
                        0 => 0,
                        1 => *ptr.cast::<u8>().as_ref() as usize,
                        2 => *ptr.cast::<u16>().as_ref() as usize,
                        4 => *ptr.cast::<u32>().as_ref() as usize,
                        8 => *ptr.cast::<u64>().as_ref() as usize,
                        // PLT: is repr(u64) the max?
                        _ => return Err(Error::ParseAttributeError),
                    }
                };

                (
                    tag,
                    Ok(unsafe {
                        NonNull::new_unchecked(
                            ptr.cast::<u8>()
                                .as_ptr()
                                .add(tag_layout.extend(variant_layouts[tag]).unwrap().1),
                        )
                        .cast()
                    }),
                )
            } else {
                match info.variants.len() {
                    0 | 1 => (0, Err(0)),
                    _ => (
                        match tag_layout.size() {
                            1 => ret_registers[0] as u8 as usize,
                            2 => ret_registers[0] as u16 as usize,
                            4 => ret_registers[0] as u32 as usize,
                            8 => ret_registers[0] as usize,
                            _ => return Err(Error::ParseAttributeError),
                        },
                        Err(1),
                    ),
                }
            };

            let value = match ptr {
                Ok(ptr) => Box::new(JitValue::from_jit(ptr, &info.variants[tag], registry)),
                Err(offset) => {
                    // PLT: this feels a bit magical and I can't really be sure about its
                    // correctness. The fact that we encode the offset to copy from as an error
                    // looks really unintuitive.
                    ret_registers.copy_within(offset.., 0);
                    Box::new(parse_result(
                        &info.variants[tag],
                        registry,
                        None,
                        ret_registers,
                    )?)
                }
            };

            Ok(JitValue::Enum {
                tag,
                value,
                debug_name: type_id.debug_name.as_deref().map(ToString::to_string),
            })
        }
        CoreTypeConcrete::Struct(info) => {
            if info.members.is_empty() {
                Ok(JitValue::Struct {
                    fields: Vec::new(),
                    debug_name: type_id.debug_name.as_deref().map(ToString::to_string),
                })
            } else {
                Ok(JitValue::from_jit(return_ptr.unwrap(), type_id, registry))
            }
        }
        CoreTypeConcrete::Felt252Dict(_) | CoreTypeConcrete::SquashedFelt252Dict(_) => unsafe {
            let ptr = return_ptr.unwrap_or(NonNull::new_unchecked(
                addr_of_mut!(ret_registers[0]) as *mut ()
            ));
            let value = JitValue::from_jit(ptr, type_id, registry);
            Ok(value)
        },

        // Builtins are handled before the call to parse_result
        // and should not be reached here.
        CoreTypeConcrete::Bitwise(_)
        | CoreTypeConcrete::Const(_)
        | CoreTypeConcrete::EcOp(_)
        | CoreTypeConcrete::GasBuiltin(_)
        | CoreTypeConcrete::BuiltinCosts(_)
        | CoreTypeConcrete::RangeCheck(_)
        | CoreTypeConcrete::Pedersen(_)
        | CoreTypeConcrete::Poseidon(_)
        | CoreTypeConcrete::SegmentArena(_) => unreachable!(),
        // PLT: what is missing for these ones?
        CoreTypeConcrete::Felt252DictEntry(_)
        | CoreTypeConcrete::Span(_)
        | CoreTypeConcrete::Snapshot(_)
        | CoreTypeConcrete::BoundedInt(_)
        | CoreTypeConcrete::Uninitialized(_)
        | CoreTypeConcrete::Coupon(_)
        | CoreTypeConcrete::StarkNet(_) => todo!(),
    }
}

// PLT: exactly 0 tests. Are we verifying somewhere else?
// PLT: potential optimization: intern `debug_name`s or store a function pointer that returns it,
// rather than forcing a string allocation for each value created. It's typically used in an error
// path only. Or maybe even split the debug data to somewhere else or use a function to create the
// strings.
// PLT: ACK
