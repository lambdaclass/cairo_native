use super::{ValueBuilder, ValueSerializer};
use crate::types::felt252::PRIME;
use crate::values::ValueDeserializer;
use crate::{types::TypeBuilder, utils::debug_with};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{types::InfoAndTypeConcreteType, GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use num_bigint::BigInt;
use num_bigint::BigUint;
use num_bigint::Sign;
use serde::{de, ser::SerializeMap, Deserializer, Serializer};
use std::alloc::Layout;
use std::ops::Neg;
use std::{collections::HashMap, fmt, ptr::NonNull};

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
    deserializer.deserialize_map(Visitor::new(arena, registry, info))
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
    assert!(!ptr.as_ptr().is_null());
    let ptr = ptr.cast::<NonNull<HashMap<[u8; 32], NonNull<std::ffi::c_void>>>>();
    let ptr = *ptr.as_ptr();
    let map = Box::from_raw(ptr.as_ptr());
    let target_type = registry.get_type(&info.ty).unwrap();

    let mut ser = serializer.serialize_map(Some(map.len()))?;

    for (key, val_ptr) in map.iter() {
        assert!(!val_ptr.as_ptr().is_null());

        type ParamSerializer<'a, TType, TLibfunc> =
            <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Serializer<'a>;

        ser.serialize_entry(
            &BigUint::from_bytes_le(key).to_str_radix(10),
            &ParamSerializer::<TType, TLibfunc>::new(val_ptr.cast(), registry, target_type),
        )?;
    }

    Box::leak(map); // we must leak to avoid a double free

    ser.end()
}

pub unsafe fn debug_fmt<TType, TLibfunc>(
    f: &mut fmt::Formatter,
    _id: &ConcreteTypeId,
    registry: &ProgramRegistry<TType, TLibfunc>,
    ptr: NonNull<()>,
    info: &InfoAndTypeConcreteType,
) -> fmt::Result
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
{
    let mut fmt = f.debug_map();
    let ptr = ptr.cast::<NonNull<HashMap<[u8; 32], NonNull<std::ffi::c_void>>>>();
    let ptr = *ptr.as_ptr();
    let map = Box::from_raw(ptr.as_ptr());
    let target_type = registry.get_type(&info.ty).unwrap();

    for (key, val_ptr) in map.iter() {
        assert!(!val_ptr.as_ptr().is_null());

        fmt.entry(
            &BigUint::from_bytes_le(key).to_str_radix(10),
            &debug_with(|f| target_type.debug_fmt(f, &info.ty, registry, val_ptr.cast())),
        );
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
    info: &'a InfoAndTypeConcreteType,
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
        write!(f, "A map containing a sequence of key values.")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: de::MapAccess<'de>,
    {
        type ParamDeserializer<'a, TType, TLibfunc> =
            <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Deserializer<'a>;

        let elem_ty = self.registry.get_type(&self.info.ty).unwrap();
        let elem_layout = elem_ty.layout(self.registry).unwrap().pad_to_align();

        let mut value_map = HashMap::<[u8; 32], NonNull<std::ffi::c_void>>::new();

        // next key must be called before next_value

        while let Some(key) = map.next_key()? {
            let key: String = key;
            let key = key.parse::<BigInt>().unwrap();
            let key = match key.sign() {
                Sign::Minus => &*PRIME - key.neg().to_biguint().unwrap(),
                _ => key.to_biguint().unwrap(),
            };
            let mut key = key.to_bytes_le();
            key.resize(32, 0);
            let key: [u8; 32] = key.try_into().unwrap();

            let value = map.next_value_seed(ParamDeserializer::<TType, TLibfunc>::new(
                self.arena,
                self.registry,
                elem_ty,
            ))?;

            let value_malloc_ptr =
                NonNull::new(unsafe { libc::malloc(dbg!(elem_layout.size())) }).unwrap();

            unsafe {
                std::ptr::copy_nonoverlapping(
                    value.cast::<u8>().as_ptr(),
                    value_malloc_ptr.as_ptr().cast(),
                    elem_layout.size(),
                );
            }

            value_map.insert(key, value_malloc_ptr);
        }

        let target: NonNull<NonNull<HashMap<[u8; 32], NonNull<std::ffi::c_void>>>> = self
            .arena
            .alloc_layout(Layout::new::<
                NonNull<HashMap<[u8; 32], NonNull<std::ffi::c_void>>>,
            >())
            .cast();

        let map_ptr: NonNull<HashMap<[u8; 32], NonNull<std::ffi::c_void>>> = self
            .arena
            .alloc_layout(
                Layout::new::<HashMap<[u8; 32], NonNull<std::ffi::c_void>>>().pad_to_align(),
            )
            .cast();

        unsafe {
            std::ptr::write(map_ptr.as_ptr(), value_map);
            std::ptr::write(target.as_ptr(), map_ptr);
        }

        Ok(target.cast())
    }
}
