use super::ValueBuilder;
use crate::{
    types::TypeBuilder,
    values::{ValueDeserializer, ValueSerializer},
};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{types::InfoAndTypeConcreteType, GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use serde::{de, ser::SerializeSeq, Deserializer, Serializer};
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
    let elem_ty = registry.get_type(&info.ty).unwrap();

    let elem_layout = elem_ty.layout(registry).unwrap();
    let elem_stride = elem_layout.pad_to_align().size();

    type ParamSerializer<'a, TType, TLibfunc> =
        <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Serializer<'a>;

    let ptr_layout = Layout::new::<*mut ()>();
    let len_layout = crate::utils::get_integer_layout(32);

    let len_value = *ptr
        .map_addr(|addr| addr.unchecked_add(ptr_layout.extend(len_layout).unwrap().1))
        .cast::<u32>()
        .as_ref();

    let data_ptr = *ptr.cast::<NonNull<()>>().as_ref();

    let mut ser = serializer.serialize_seq(Some(len_value.try_into().unwrap()))?;
    for i in 0..(len_value as usize) {
        let cur_elem_ptr = data_ptr.map_addr(|addr| addr.unchecked_add(elem_stride * i));

        ser.serialize_element(&ParamSerializer::<TType, TLibfunc>::new(
            cur_elem_ptr.cast(),
            registry,
            elem_ty,
        ))?;
    }
    ser.end()
}

pub unsafe fn debug_fmt<TType, TLibfunc>(
    _f: &mut fmt::Formatter,
    _id: &ConcreteTypeId,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _ptr: NonNull<()>,
    _info: &InfoAndTypeConcreteType,
) -> fmt::Result
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
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
        let elem_ty = self.registry.get_type(&self.info.ty).unwrap();
        let elem_layout = elem_ty.layout(self.registry).unwrap().pad_to_align();

        let mut ptr: *mut NonNull<()> = null_mut();
        let mut len: u32 = 0;
        let mut cap: u32 = 0;

        if let Some(len) = seq.size_hint() {
            ptr = unsafe { libc::realloc(ptr.cast(), elem_layout.size() * len).cast() };
            cap = len.try_into().unwrap();
        }

        type ParamDeserializer<'a, TType, TLibfunc> =
            <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Deserializer<'a>;
        while let Some(elem) = seq.next_element_seed(ParamDeserializer::<TType, TLibfunc>::new(
            self.arena,
            self.registry,
            elem_ty,
        ))? {
            if len == cap {
                let new_cap = (cap + 1).next_power_of_two();

                ptr = unsafe {
                    libc::realloc(ptr.cast(), elem_layout.size() * new_cap as usize).cast()
                };
                cap = new_cap;
            }

            unsafe {
                std::ptr::copy_nonoverlapping(
                    elem.cast::<u8>().as_ptr(),
                    NonNull::new_unchecked(ptr)
                        .map_addr(|addr| addr.unchecked_add(len as usize * elem_layout.size()))
                        .cast()
                        .as_ptr(),
                    elem_layout.size(),
                );
            }

            len += 1;
        }

        unsafe {
            let target = self.arena.alloc_layout(
                Layout::new::<*mut NonNull<()>>()
                    .extend(Layout::new::<u32>())
                    .unwrap()
                    .0
                    .extend(Layout::new::<u32>())
                    .unwrap()
                    .0,
            );

            *target.cast::<*mut NonNull<()>>().as_mut() = ptr;

            let (layout, offset) = Layout::new::<*mut NonNull<()>>()
                .extend(Layout::new::<u32>())
                .unwrap();
            *target
                .map_addr(|addr| addr.unchecked_add(offset))
                .cast()
                .as_mut() = len;

            let (_, offset) = layout.extend(Layout::new::<u32>()).unwrap();
            *target
                .map_addr(|addr| addr.unchecked_add(offset))
                .cast()
                .as_mut() = cap;
        }

        todo!()
    }
}
