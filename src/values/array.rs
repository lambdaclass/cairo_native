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

    // nightly feature - strict_provenance:
    // let len_value = *ptr.map_addr(|addr| addr.unchecked_add(ptr_layout.extend(len_layout).unwrap().1))
    let len_value = *NonNull::new(
        ((ptr.as_ptr() as usize) + ptr_layout.extend(len_layout).unwrap().1) as *mut (),
    )
    .unwrap()
    .cast::<u32>()
    .as_ref();

    let data_ptr = *ptr.cast::<NonNull<()>>().as_ref();

    let mut ser = serializer.serialize_seq(Some(len_value.try_into().unwrap()))?;
    for i in 0..(len_value as usize) {
        // nightly feature - strict_provenance, alloc_layout_extra:
        // let cur_elem_ptr = data_ptr.map_addr(|addr| addr.unchecked_add(elem_stride * i));
        let cur_elem_ptr =
            NonNull::new(((data_ptr.as_ptr() as usize) + elem_stride * i) as *mut ()).unwrap();

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

            // nightly feature - strict_provenance, alloc_layout_extra:
            // unsafe {
            //     let a = NonNull::new_unchecked(ptr)
            //         .map_addr(|addr| addr.unchecked_add(len as usize * elem_layout.size()))
            //         .cast()
            //         .as_ptr();
            // }

            // let a = elem.cast::<u8>().as_ptr();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    elem.cast::<u8>().as_ptr(),
                    // NonNull::new_unchecked(ptr)
                    NonNull::new(
                        ((NonNull::new_unchecked(ptr).as_ptr() as usize)
                            + len as usize * elem_layout.size()) as *mut u8,
                    )
                    .unwrap()
                    // nightly feature - alloc_layout_extra:
                    // .map_addr(|addr| addr.unchecked_add(len as usize * elem_layout.size()))
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
            // nightly feature - alloc_layout_extra:
            // *target
            //     .map_addr(|addr| addr.unchecked_add(offset))
            //     .cast()
            //     .as_mut() = len;
            *NonNull::new(((target.as_ptr() as usize) + offset) as *mut u32)
                .unwrap()
                .cast()
                .as_mut() = len;

            let (_, offset) = layout.extend(Layout::new::<u32>()).unwrap();
            // nightly feature - alloc_layout_extra:
            // *target
            //     .map_addr(|addr| addr.unchecked_add(offset))
            *NonNull::new(((target.as_ptr() as usize) + offset) as *mut u32)
                .unwrap()
                .cast()
                .as_mut() = cap;
            Ok(target.cast())
        }
    }
}
