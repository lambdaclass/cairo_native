use super::{ValueBuilder, ValueDeserializer, ValueSerializer};
use crate::{error::CoreTypeBuilderError, types::TypeBuilder, utils::next_multiple_of_usize};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{enm::EnumConcreteType, GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use serde::{de, ser::SerializeSeq, Deserializer, Serializer};
use std::{fmt, ptr::NonNull};

pub unsafe fn deserialize<'de, TType, TLibfunc, D>(
    deserializer: D,
    arena: &Bump,
    registry: &ProgramRegistry<TType, TLibfunc>,
    info: &EnumConcreteType,
) -> Result<NonNull<()>, D::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete:
        TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError> + ValueBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
{
    deserializer.deserialize_seq(Visitor::new(arena, registry, info))
}

pub unsafe fn serialize<TType, TLibfunc, S>(
    serializer: S,
    registry: &ProgramRegistry<TType, TLibfunc>,
    ptr: NonNull<()>,
    info: &EnumConcreteType,
) -> Result<S::Ok, S::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete:
        TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError> + ValueBuilder<TType, TLibfunc>,
    S: Serializer,
{
    let tag_layout = crate::utils::get_integer_layout(match info.variants.len() {
        0 | 1 => 0,
        num_variants => (next_multiple_of_usize(num_variants.next_power_of_two(), 8) >> 3)
            .try_into()
            .unwrap(),
    });
    let tag_value = match info.variants.len() {
        0 => {
            // An enum without variants is basically the `!` (never) type in Rust.
            panic!("An enum without variants is not a valid type.")
        }
        1 => 0,
        _ => match tag_layout.size() {
            1 => *ptr.cast::<u8>().as_ref() as usize,
            2 => *ptr.cast::<u16>().as_ref() as usize,
            4 => *ptr.cast::<u32>().as_ref() as usize,
            8 => *ptr.cast::<u64>().as_ref() as usize,
            _ => unreachable!(),
        },
    };

    let payload_ty = registry.get_type(&info.variants[tag_value]).unwrap();
    let payload_layout = payload_ty.layout(registry).unwrap();

    type ParamSerializer<'a, TType, TLibfunc> =
        <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Serializer<'a>;

    let mut ser = serializer.serialize_seq(Some(2))?;
    ser.serialize_element(&tag_value)?;
    ser.serialize_element(&ParamSerializer::<TType, TLibfunc>::new(
        // ptr.map_addr(|addr| addr.unchecked_add(tag_layout.extend(payload_layout).unwrap().1)),
        NonNull::new(
            ((ptr.as_ptr() as usize) + tag_layout.extend(payload_layout).unwrap().1) as *mut _,
        )
        .unwrap(),
        registry,
        payload_ty,
    ))?;
    ser.end()
}

pub unsafe fn debug_fmt<TType, TLibfunc>(
    _f: &mut fmt::Formatter,
    _id: &ConcreteTypeId,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _ptr: NonNull<()>,
    _info: &EnumConcreteType,
) -> fmt::Result
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete:
        TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError> + ValueBuilder<TType, TLibfunc>,
{
    todo!()
}

struct Visitor<'a, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
{
    arena: &'a Bump,
    registry: &'a ProgramRegistry<TType, TLibfunc>,
    info: &'a EnumConcreteType,
}

impl<'a, TType, TLibfunc> Visitor<'a, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete:
        TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError> + ValueBuilder<TType, TLibfunc>,
{
    fn new(
        arena: &'a Bump,
        registry: &'a ProgramRegistry<TType, TLibfunc>,
        info: &'a EnumConcreteType,
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
    <TType as GenericType>::Concrete:
        TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError> + ValueBuilder<TType, TLibfunc>,
{
    type Value = NonNull<()>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "A sequence of the discriminant followed by the payload")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: de::SeqAccess<'de>,
    {
        let tag_value = seq.next_element::<usize>()?.unwrap();
        assert!(tag_value <= self.info.variants.len());

        type ParamDeserializer<'a, TType, TLibfunc> =
            <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Deserializer<'a>;

        let payload_ty = self
            .registry
            .get_type(&self.info.variants[tag_value])
            .unwrap();
        let payload = seq
            .next_element_seed(ParamDeserializer::<TType, TLibfunc>::new(
                self.arena,
                self.registry,
                payload_ty,
            ))?
            .unwrap();

        let (layout, tag_layout, variant_layouts) =
            crate::types::r#enum::get_layout_for_variants(self.registry, &self.info.variants)
                .unwrap();
        let ptr = self.arena.alloc_layout(layout).cast();

        match tag_layout.size() {
            1 => *unsafe { ptr.cast::<u8>().as_mut() } = tag_value as u8,
            2 => *unsafe { ptr.cast::<u16>().as_mut() } = tag_value as u16,
            4 => *unsafe { ptr.cast::<u32>().as_mut() } = tag_value as u32,
            8 => *unsafe { ptr.cast::<u64>().as_mut() } = tag_value as u64,
            _ => unreachable!(),
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                payload.cast::<u8>().as_ptr(),
                NonNull::new(
                    ((ptr.as_ptr() as usize)
                        + tag_layout.extend(variant_layouts[tag_value]).unwrap().1)
                        as *mut u8,
                )
                .unwrap()
                // ptr.map_addr(|addr| {
                //     addr.unchecked_add(tag_layout.extend(variant_layouts[tag_value]).unwrap().1)
                // })
                .cast()
                .as_ptr(),
                variant_layouts[tag_value].size(),
            );
        };

        Ok(ptr)
    }
}
