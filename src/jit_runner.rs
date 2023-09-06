//! # JIT runner

use crate::{
    error::{
        jit_engine::{
            make_deserializer_error, make_insuficient_gas_error, make_serializer_error,
            make_type_builder_error,
        },
        JitRunnerError,
    },
    libfuncs::LibfuncBuilder,
    types::TypeBuilder,
    utils::generate_function_name,
    values::{ValueBuilder, ValueDeserializer, ValueSerializer},
};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{GenericLibfunc, GenericType},
    ids::{ConcreteTypeId, FunctionId},
    program_registry::ProgramRegistry,
};
use melior::ExecutionEngine;
use serde::{de::Visitor, ser::SerializeSeq, Deserializer, Serializer};
use std::{alloc::Layout, fmt, iter::once, ptr::NonNull};
use tracing::debug;

/// Execute a function on an engine loaded with a Sierra program.
///
/// The JIT execution of a Sierra program requires an [`ExecutionEngine`] already configured with
/// the compiled module. This has been designed this way because it allows engine reusal, as opposed
/// to building a different engine every time a function is called and therefore losing all
/// potential optimizations that are already present.
///
/// The registry is needed to convert the params and return values into and from the JIT ABI. Check
/// out [the values module](crate::values) for more information about the de/serialization process.
///
/// The function's arguments and return values are passed using a [`Deserializer`] and a
/// [`Serializer`] respectively. This method provides an easy way to process the values while also
/// not requiring recompilation every time the function's signature changes.
pub fn execute<'de, TType, TLibfunc, D, S>(
    engine: &ExecutionEngine,
    registry: &ProgramRegistry<TType, TLibfunc>,
    function_id: &FunctionId,
    params: D,
    returns: S,
    required_initial_gas: Option<u64>,
) -> Result<S::Ok, JitRunnerError<'de, TType, TLibfunc, D, S>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
    S: Serializer,
{
    let arena = Bump::new();

    let entry_point = registry.get_function(function_id)?;
    debug!(
        "executing entry_point with the following required parameters: {:?}",
        entry_point.signature.param_types
    );

    let params = params
        .deserialize_seq(ArgsVisitor {
            arena: &arena,
            registry,
            params: &entry_point.signature.param_types,
        })
        .map_err(make_deserializer_error)?;

    // If program has a required initial gas, check if a gas builting exists
    // and check if the passed gas was enough, if so, deduct the required gas before execution.
    if let Some(required_initial_gas) = required_initial_gas {
        for (id, param) in entry_point.signature.param_types.iter().zip(params.iter()) {
            if id.debug_name.as_deref() == Some("GasBuiltin") {
                let gas_builtin = unsafe { *param.cast::<u64>().as_ptr() };

                if gas_builtin < required_initial_gas {
                    return Err(make_insuficient_gas_error(
                        required_initial_gas,
                        gas_builtin,
                    ));
                }

                let starting_gas = gas_builtin - required_initial_gas;

                unsafe {
                    // update gas with the starting gas
                    param.cast::<u64>().as_ptr().write(starting_gas);
                }
            }
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

            Result::<_, JitRunnerError<'de, TType, TLibfunc, D, S>>::Ok((Some(layout), offsets))
        },
    )?;

    let layout = layout.unwrap_or(Layout::new::<()>());
    let ret_ptr = arena.alloc_layout(layout).cast::<()>();

    let function_name = generate_function_name(function_id);
    let mut io_pointers = if complex_results {
        let ret_ptr_ptr = arena.alloc(ret_ptr) as *mut NonNull<()>;
        once(ret_ptr_ptr as *mut ())
            .chain(params.into_iter().map(NonNull::as_ptr))
            .collect::<Vec<_>>()
    } else {
        params
            .into_iter()
            .map(NonNull::as_ptr)
            .chain(once(ret_ptr.as_ptr()))
            .collect::<Vec<_>>()
    };

    unsafe {
        engine.invoke_packed(&function_name, &mut io_pointers)?;
    }

    let mut return_seq = returns
        .serialize_seq(Some(entry_point.signature.ret_types.len()))
        .map_err(make_serializer_error)?;

    for (id, offset) in entry_point.signature.ret_types.iter().zip(offsets) {
        let ty = registry.get_type(id)?;
        type ParamSerializer<'a, TType, TLibfunc> =
            <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Serializer<'a>;

        return_seq
            .serialize_element(&<ParamSerializer<TType, TLibfunc> as ValueSerializer<
                TType,
                TLibfunc,
            >>::new(
                // ret_ptr.map_addr(|addr| unsafe { addr.unchecked_add(offset) }),
                NonNull::new(((ret_ptr.as_ptr() as usize) + offset) as *mut _).unwrap(),
                registry,
                ty,
            ))
            .map_err(make_serializer_error)?;

        // TODO: Drop if necessary (ex. arrays).
    }

    return_seq.end().map_err(make_serializer_error)
}

struct ArgsVisitor<'a, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
{
    arena: &'a Bump,
    registry: &'a ProgramRegistry<TType, TLibfunc>,
    params: &'a [ConcreteTypeId],
}

impl<'a, 'de, TType, TLibfunc> Visitor<'de> for ArgsVisitor<'a, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
{
    type Value = Vec<NonNull<()>>;

    fn expecting(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut param_ptr_list = Vec::with_capacity(self.params.len());
        for (idx, param_type_id) in self.params.iter().enumerate() {
            let param_ty = self.registry.get_type(param_type_id).unwrap();

            type ParamDeserializer<'a, TType, TLibfunc> =
                <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Deserializer<
                    'a,
                >;
            let deserializer =
                ParamDeserializer::<TType, TLibfunc>::new(self.arena, self.registry, param_ty);

            let ptr = seq
                .next_element_seed::<ParamDeserializer<TType, TLibfunc>>(deserializer)?
                .unwrap_or_else(|| {
                    let param_list: Vec<_> = self
                        .params
                        .iter()
                        .map(|x| x.debug_name.as_ref().unwrap().as_str()).collect();

                    panic!(
                        "Missing input parameter of type '{}' (param index {idx}), required parameters: {:?}",
                        param_type_id.debug_name.as_ref().unwrap().as_str(),
                        param_list
                    );
                });

            param_ptr_list.push(ptr);
        }

        Ok(param_ptr_list)
    }
}
