use super::ValueBuilder;
use crate::types::TypeBuilder;
use crate::values::ValueDeserializer;
use crate::values::ValueSerializer;
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{
        types::{InfoAndTypeConcreteType, InfoOnlyConcreteType},
        GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use serde::{
    de::{self, DeserializeSeed},
    ser::SerializeSeq,
    Deserialize, Deserializer, Serialize, Serializer,
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
    deserializer.deserialize_seq(Visitor::new(arena, registry, info))

    /*
    let inner_ty = registry.get_type(&info.ty).unwrap();
    let inner_layout = registry
        .get_type(&info.ty)
        .unwrap()
        .layout(registry)
        .unwrap();
    let ptr: NonNull<*mut ()> = arena.alloc_layout(Layout::new::<*mut ()>()).cast();

    type ParamDeserializer<'a, TType, TLibfunc> =
        <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Deserializer<'a>;

    let inner_de = ParamDeserializer::<TType, TLibfunc>::new(arena, registry, inner_ty);

    let seq = deserializer.deserialize_tuple(2, visitor)

    let tag = u8::deserialize(deserializer)?;

    match tag {
        0 => {
            // null
            std::ptr::write(ptr.as_ptr(), null_mut());
        }
        1 => {
            let value = inner_de.deserialize(deserializer)?;

            let inner_ptr: *mut () = unsafe { libc::malloc(inner_layout.size()).cast() };

            std::ptr::write(inner_ptr.cast(), *value.as_ptr());

            std::ptr::copy_nonoverlapping(
                value.cast::<u8>().as_ptr(),
                inner_ptr.cast(),
                inner_layout.size(),
            );
        }
        _ => panic!("invalid tag"),
    };

    Ok(ptr.cast())
     */
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
    let inner_layout = registry
        .get_type(&info.ty)
        .unwrap()
        .layout(registry)
        .unwrap();

    type ParamSerializer<'a, TType, TLibfunc> =
        <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Serializer<'a>;

    let ptr: NonNull<*mut ()> = ptr.cast();
    dbg!("reached");
    dbg!(ptr);
    let data_ptr = *ptr.as_ptr();
    dbg!("reached 2");
    dbg!(data_ptr);
    let seq_len: usize = (!data_ptr.is_null()) as usize + 1usize;

    let mut ser = serializer.serialize_seq(Some(seq_len))?;

    ser.serialize_element(&(!data_ptr.is_null() as u8))?;

    if !data_ptr.is_null() {
        ser.serialize_element(&ParamSerializer::<TType, TLibfunc>::new(
            NonNull::new(data_ptr).unwrap().cast(),
            registry,
            inner_ty,
        ))?;
    }

    ser.end()
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

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: de::SeqAccess<'de>,
    {
        let inner_ty = self.registry.get_type(&self.info.ty).unwrap();
        let inner_layout = self
            .registry
            .get_type(&self.info.ty)
            .unwrap()
            .layout(self.registry)
            .unwrap();
        let mut ptr: *mut () = null_mut();

        type ParamDeserializer<'a, TType, TLibfunc> =
            <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Deserializer<'a>;

        let tag: u8 = seq.next_element().unwrap().unwrap();

        match dbg!(tag) {
            0 => {
                // null
                // unsafe { std::ptr::write(ptr.as_ptr(), null_mut()) };
            }
            1 => {
                let value = seq
                    .next_element_seed(ParamDeserializer::<TType, TLibfunc>::new(
                        self.arena,
                        self.registry,
                        inner_ty,
                    ))?
                    .unwrap();

                ptr = unsafe { libc::malloc(inner_layout.size()).cast() };

                unsafe {
                    // copy to allocated ptr
                    std::ptr::copy_nonoverlapping(
                        value.as_ptr().cast::<u8>(),
                        ptr.cast(),
                        inner_layout.size(),
                    );
                }
            }
            _ => panic!("invalid tag"),
        };

        let target_ptr: NonNull<*mut ()> = self.arena.alloc_layout(Layout::new::<*mut ()>()).cast();
        dbg!(target_ptr);

        unsafe {
            std::ptr::write::<*mut ()>(target_ptr.as_ptr().cast(), ptr);
        }
        dbg!(target_ptr);
        dbg!("end visitor");

        Ok(target_ptr.cast())
    }
}
