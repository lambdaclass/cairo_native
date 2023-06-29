use super::{ValueBuilder, ValueSerializer};
use crate::{types::TypeBuilder, utils::debug_with};
use cairo_lang_sierra::{
    extensions::{structure::StructConcreteType, GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use serde::{ser::SerializeTuple, Deserializer, Serializer};
use std::{alloc::Layout, fmt, ptr::NonNull};
use bumpalo::Bump;

pub unsafe fn deserialize<'de, TType, TLibfunc, D>(
    _deserializer: D,
    _arena: &Bump,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _info: &StructConcreteType,
) -> Result<NonNull<()>, D::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
{
    todo!()
}

pub unsafe fn serialize<TType, TLibfunc, S>(
    serializer: S,
    registry: &ProgramRegistry<TType, TLibfunc>,
    ptr: NonNull<()>,
    info: &StructConcreteType,
) -> Result<S::Ok, S::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
    S: Serializer,
{
    let mut ser = serializer.serialize_tuple(info.members.len())?;

    let mut layout: Option<Layout> = None;
    for member_ty in &info.members {
        let member = registry.get_type(member_ty).unwrap();
        let member_layout = member.layout(registry);

        let (new_layout, offset) = match layout {
            Some(layout) => layout.extend(member_layout).unwrap(),
            None => (member_layout, 0),
        };
        layout = Some(new_layout);

        type ParamSerializer<'a, TType, TLibfunc> =
            <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Serializer<'a>;
        ser.serialize_element(&ParamSerializer::<TType, TLibfunc>::new(
            ptr.map_addr(|addr| addr.unchecked_add(offset)),
            registry,
            member,
        ))?;
    }

    ser.end()
}

pub unsafe fn debug_fmt<TType, TLibfunc>(
    f: &mut fmt::Formatter,
    id: &ConcreteTypeId,
    registry: &ProgramRegistry<TType, TLibfunc>,
    ptr: NonNull<()>,
    info: &StructConcreteType,
) -> fmt::Result
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
{
    let mut fmt = f.debug_tuple(id.debug_name.as_deref().unwrap_or(""));

    let mut layout: Option<Layout> = None;
    for member_ty in &info.members {
        let member = registry.get_type(member_ty).unwrap();
        let member_layout = member.layout(registry);

        let (new_layout, offset) = match layout {
            Some(layout) => layout.extend(member_layout).unwrap(),
            None => (member_layout, 0),
        };
        layout = Some(new_layout);

        fmt.field(&debug_with(|f| {
            member.debug_fmt(
                f,
                member_ty,
                registry,
                ptr.map_addr(|addr| addr.unchecked_add(offset)),
            )
        }));
    }

    fmt.finish()
}
