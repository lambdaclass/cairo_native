use crate::{
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

pub fn execute<'de, TType, TLibfunc, D, S>(
    engine: &ExecutionEngine,
    registry: &ProgramRegistry<TType, TLibfunc>,
    function_id: &FunctionId,
    params: D,
    returns: S,
) -> Result<S::Ok, Box<dyn std::error::Error>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
    S: Serializer,
{
    let arena = Bump::new();

    let entry_point = registry.get_function(function_id).unwrap();
    let params = params
        .deserialize_seq(ArgsVisitor {
            arena: &arena,
            registry,
            params: &entry_point.signature.param_types,
        })
        .unwrap();

    let mut complex_results = entry_point.signature.ret_types.len() > 1;
    let (layout, offsets) = entry_point.signature.ret_types.iter().fold(
        (Option::<Layout>::None, Vec::new()),
        |(acc, mut offsets), id| {
            let ty = registry.get_type(id).unwrap();
            let ty_layout = ty.layout(registry);

            let (layout, offset) = match acc {
                Some(layout) => layout.extend(ty_layout).unwrap(),
                None => (ty_layout, 0),
            };

            offsets.push(offset);
            complex_results |= ty.is_complex();

            (Some(layout), offsets)
        },
    );

    let layout = layout.unwrap_or(Layout::from_size_align(0, 1).unwrap());
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
        engine
            .invoke_packed(&function_name, &mut io_pointers)
            .unwrap();
    }

    let mut return_seq = returns
        .serialize_seq(Some(entry_point.signature.ret_types.len()))
        .unwrap();

    for (id, offset) in entry_point.signature.ret_types.iter().zip(offsets) {
        let ty = registry.get_type(id).unwrap();
        type ParamSerializer<'a, TType, TLibfunc> =
            <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Serializer<'a>;

        return_seq
            .serialize_element(&<ParamSerializer<TType, TLibfunc> as ValueSerializer<
                TType,
                TLibfunc,
            >>::new(
                ret_ptr.map_addr(|addr| unsafe { addr.unchecked_add(offset) }),
                registry,
                ty,
            ))
            .unwrap();

        // TODO: Drop if necessary (ex. arrays).
    }

    Ok(return_seq.end().unwrap())
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
        for param_type_id in self.params {
            let param_ty = self.registry.get_type(param_type_id).unwrap();

            type ParamDeserializer<'a, TType, TLibfunc> =
                <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Deserializer<
                    'a,
                >;
            let deserializer =
                ParamDeserializer::<TType, TLibfunc>::new(self.arena, self.registry, param_ty);

            let ptr = seq
                .next_element_seed::<ParamDeserializer<TType, TLibfunc>>(deserializer)?
                .unwrap();

            param_ptr_list.push(ptr);
        }

        Ok(param_ptr_list)
    }
}
