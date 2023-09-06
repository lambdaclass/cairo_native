use super::{ValueBuilder, ValueDeserializer, ValueSerializer};
use crate::{types::TypeBuilder, utils::debug_with};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{structure::StructConcreteType, GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use serde::{de, ser::SerializeTuple, Deserializer, Serializer};
use std::{alloc::Layout, fmt, ptr::NonNull};

pub unsafe fn deserialize<'de, TType, TLibfunc, D>(
    deserializer: D,
    arena: &Bump,
    registry: &ProgramRegistry<TType, TLibfunc>,
    info: &StructConcreteType,
) -> Result<NonNull<()>, D::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
{
    deserializer.deserialize_tuple(info.members.len(), Visitor::new(arena, registry, info))
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
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
    S: Serializer,
{
    let mut ser = serializer.serialize_tuple(info.members.len())?;

    let mut layout: Option<Layout> = None;
    for member_ty in &info.members {
        let member = registry.get_type(member_ty).unwrap();
        let member_layout = member.layout(registry).unwrap();

        let (new_layout, offset) = match layout {
            Some(layout) => layout.extend(member_layout).unwrap(),
            None => (member_layout, 0),
        };
        layout = Some(new_layout);

        type ParamSerializer<'a, TType, TLibfunc> =
            <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Serializer<'a>;
        ser.serialize_element(&ParamSerializer::<TType, TLibfunc>::new(
            // ptr.map_addr(|addr| addr.unchecked_add(offset)),
            NonNull::new(((ptr.as_ptr() as usize) + offset) as *mut ()).unwrap(),
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
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
{
    let mut fmt = f.debug_tuple(id.debug_name.as_deref().unwrap_or(""));

    let mut layout: Option<Layout> = None;
    for member_ty in &info.members {
        let member = registry.get_type(member_ty).unwrap();
        let member_layout = member.layout(registry).unwrap();

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
                NonNull::new(((ptr.as_ptr() as usize) + offset) as *mut ()).unwrap(),
                // ptr.map_addr(|addr| addr.unchecked_add(offset)),
            )
        }));
    }

    fmt.finish()
}

struct Visitor<'a, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
{
    arena: &'a Bump,
    registry: &'a ProgramRegistry<TType, TLibfunc>,
    info: &'a StructConcreteType,
}

impl<'a, TType, TLibfunc> Visitor<'a, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
{
    fn new(
        arena: &'a Bump,
        registry: &'a ProgramRegistry<TType, TLibfunc>,
        info: &'a StructConcreteType,
    ) -> Self {
        Self {
            arena,
            registry,
            info,
        }
    }
}

impl<'a, 'de, TType, TLibfunc> de::Visitor<'de> for Visitor<'a, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
{
    type Value = NonNull<()>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "A tuple containing the struct's fields")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: de::SeqAccess<'de>,
    {
        let mut layout: Option<Layout> = None;
        let mut data = Vec::with_capacity(self.info.members.len());

        type ParamDeserializer<'a, TType, TLibfunc> =
            <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Deserializer<'a>;
        for member in &self.info.members {
            let member_ty = self.registry.get_type(member).unwrap();
            let member_layout = member_ty.layout(self.registry).unwrap();

            let (new_layout, offset) = match layout {
                Some(layout) => layout.extend(member_layout).unwrap(),
                None => (member_layout, 0),
            };
            layout = Some(new_layout);

            data.push((
                member_layout,
                offset,
                seq.next_element_seed(ParamDeserializer::<TType, TLibfunc>::new(
                    self.arena,
                    self.registry,
                    member_ty,
                ))?
                .unwrap(),
            ));
        }

        let ptr = self
            .arena
            .alloc_layout(layout.unwrap_or(Layout::new::<()>()))
            .cast();

        for (layout, offset, member_ptr) in data {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    member_ptr.cast::<u8>().as_ptr(),
                    NonNull::new(((ptr.as_ptr() as usize) + offset) as *mut u8)
                        .unwrap()
                        .cast()
                        .as_ptr(),
                    // ptr.map_addr(|addr| addr.unchecked_add(offset))
                    //     .cast()
                    // .as_ptr(),
                    layout.size(),
                );
            }
        }

        Ok(ptr)
    }
}
