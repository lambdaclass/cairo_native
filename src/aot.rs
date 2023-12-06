use crate::{
    types::TypeBuilder,
    utils::generate_function_name,
    values::{JITValue, ValueBuilder},
};
use cairo_lang_sierra::{
    extensions::{GenericLibfunc, GenericType},
    ids::FunctionId,
    program_registry::ProgramRegistry,
};
use libc::c_void;
use libloading::Library;
use std::{alloc::Layout, arch::global_asm};

#[cfg(target_arch = "aarch64")]
global_asm!(include_str!("arch/aarch64.s"));
#[cfg(target_arch = "x86_64")]
global_asm!(include_str!("arch/x86_64.s"));

extern "C" {
    fn aot_trampoline(fn_ptr: *mut c_void, args_ptr: *const u64, args_len: usize);
}

pub struct AotNativeExecutor<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
{
    library: Library,
    registry: ProgramRegistry<TType, TLibfunc>,
}

impl<TType, TLibfunc> AotNativeExecutor<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder,
{
    pub fn new(library: Library, registry: ProgramRegistry<TType, TLibfunc>) -> Self {
        Self { library, registry }
    }

    pub fn invoke_dynamic(&self, function_id: &FunctionId, args: &[JITValue]) -> Vec<JITValue> {
        let function_name = generate_function_name(function_id);
        let function_name = format!("_mlir_ciface_{function_name}");

        // Arguments and return values are hardcoded since they'll be handled by the trampoline.
        let function_ptr = unsafe {
            self.library
                .get::<extern "C" fn()>(function_name.as_bytes())
                .unwrap()
        };

        let function_signature = &self.registry.get_function(function_id).unwrap().signature;
        let mut invoke_data = Vec::new();

        // Generate return pointer (if necessary).
        let return_ptr = if function_signature.ret_types.len() > 1
            || function_signature
                .ret_types
                .first()
                .is_some_and(|id| self.registry.get_type(id).unwrap().is_complex())
        {
            let layout =
                function_signature
                    .ret_types
                    .iter()
                    .fold(Layout::new::<()>(), |layout, id| {
                        let type_info = self.registry.get_type(id).unwrap();
                        layout
                            .extend(type_info.layout(&self.registry).unwrap())
                            .unwrap()
                            .0
                    });

            let return_ptr = unsafe { std::alloc::alloc(layout) };
            invoke_data.push(return_ptr as u64);

            Some((layout, return_ptr))
        } else {
            invoke_data.push(0);
            None
        };

        for arg in args {
            match arg {
                JITValue::Felt252(value) => invoke_data.extend(value.to_le_digits()),
                JITValue::Array(_) => todo!(),
                JITValue::Struct { .. } => todo!(),
                JITValue::Enum { .. } => todo!(),
                JITValue::Felt252Dict { .. } => todo!(),
                JITValue::Uint8(value) => invoke_data.push(*value as u64),
                JITValue::Uint16(value) => invoke_data.push(*value as u64),
                JITValue::Uint32(value) => invoke_data.push(*value as u64),
                JITValue::Uint64(value) => invoke_data.push(*value),
                JITValue::Uint128(value) => {
                    invoke_data.push(*value as u64);
                    invoke_data.push((value >> 64) as u64);
                }
                JITValue::EcPoint(_, _) => todo!(),
                JITValue::EcState(_, _, _, _) => todo!(),
            }
        }

        // TODO: Invoke the trampoline.
        unsafe {
            aot_trampoline(
                function_ptr.into_raw().into_raw(),
                invoke_data.as_ptr(),
                invoke_data.len(),
            );
        }

        if let Some((layout, return_ptr)) = return_ptr {
            unsafe {
                std::alloc::dealloc(return_ptr, layout);
            }
        }

        vec![]
    }
}
