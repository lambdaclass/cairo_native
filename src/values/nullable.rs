use super::ValueBuilder;
use crate::types::TypeBuilder;
use crate::values::ValueDeserializer;
use crate::values::ValueSerializer;
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{types::InfoAndTypeConcreteType, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry,
};
use serde::{
    de::{self, DeserializeSeed},
    Deserializer, Serializer,
};
use std::{
    alloc::Layout,
    fmt,
    ptr::{null_mut, NonNull},
};

pub unsafe fn deserialize<'de, TType, TLibfunc, D>(
    deserializer: D,
    arena: &Bump,
    registry: &ProgramRegistry<TType, TLibfunc>,
    info: &InfoAndTypeConcreteType,
) -> Result<NonNull<()>, D::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
{
    deserializer.deserialize_option(Visitor::new(arena, registry, info))
}

pub unsafe fn serialize<TType, TLibfunc, S>(
    serializer: S,
    registry: &ProgramRegistry<TType, TLibfunc>,
    ptr: NonNull<()>,
    info: &InfoAndTypeConcreteType,
) -> Result<S::Ok, S::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
    S: Serializer,
{
    let inner_ty = registry.get_type(&info.ty).unwrap();

    type ParamSerializer<'a, TType, TLibfunc> =
        <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Serializer<'a>;

    let ptr: NonNull<*mut u8> = ptr.cast();
    let data_ptr = *ptr.as_ptr();

    if data_ptr.is_null() {
        serializer.serialize_none()
    } else {
        serializer.serialize_some(&ParamSerializer::<TType, TLibfunc>::new(
            NonNull::new(data_ptr).unwrap().cast(),
            registry,
            inner_ty,
        ))
    }
}

struct Visitor<'a, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
{
    arena: &'a Bump,
    registry: &'a ProgramRegistry<TType, TLibfunc>,
    info: &'a InfoAndTypeConcreteType,
}

impl<'a, TType, TLibfunc> Visitor<'a, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
{
    fn new(
        arena: &'a Bump,
        registry: &'a ProgramRegistry<TType, TLibfunc>,
        info: &'a InfoAndTypeConcreteType,
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
        write!(f, "An array of values with the same type.")
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let target_ptr: NonNull<*mut u8> = self.arena.alloc_layout(Layout::new::<*mut u8>()).cast();

        unsafe {
            std::ptr::write(target_ptr.as_ptr(), null_mut());
        }

        Ok(target_ptr.cast())
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        let inner_ty = self.registry.get_type(&self.info.ty).unwrap();
        let inner_layout = self
            .registry
            .get_type(&self.info.ty)
            .unwrap()
            .layout(self.registry)
            .unwrap()
            .pad_to_align();

        type ParamDeserializer<'a, TType, TLibfunc> =
            <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Deserializer<'a>;

        let param_de =
            ParamDeserializer::<TType, TLibfunc>::new(self.arena, self.registry, inner_ty);

        let value = param_de.deserialize(deserializer)?;

        let inner_ptr: NonNull<()> = self.arena.alloc_layout(inner_layout).cast();

        unsafe {
            // copy to allocated ptr
            std::ptr::copy_nonoverlapping(
                value.as_ptr().cast::<u8>(),
                inner_ptr.as_ptr().cast(),
                inner_layout.size(),
            );
        }

        let target_ptr: NonNull<NonNull<*mut u8>> = self
            .arena
            .alloc_layout(Layout::new::<NonNull<*mut u8>>())
            .cast();

        unsafe {
            std::ptr::write(target_ptr.as_ptr().cast(), inner_ptr);
        }

        Ok(target_ptr.cast())
    }
}
