use crate::types::TypeBuilder;

use super::ValueBuilder;
use crate::values::ValueSerializer;
use cairo_lang_sierra::{
    extensions::{types::InfoAndTypeConcreteType, GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use serde::{ser::SerializeSeq, Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Number;
use std::{alloc::Layout, fmt, ptr::NonNull, str::FromStr};

pub unsafe fn deserialize<'de, TType, TLibfunc, D>(
    deserializer: D,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    ptr: NonNull<()>,
    info: &InfoAndTypeConcreteType,
) -> Result<(), D::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
{
    dbg!(&info.info);
    todo!();
    /*
    let payload_ty = registry.get_type(&info.info.).unwrap();
    let payload_layout = payload_ty.layout(registry);

    let value = <Number as Deserialize>::deserialize(deserializer)?;
    let value: u32 = value.to_string().parse().unwrap();
    std::ptr::write(ptr.cast::<u32>().as_mut(), value);
    */
    Ok(())
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
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
    S: Serializer,
{
    let elem_ty = registry.get_type(&info.ty).unwrap();

    let elem_layout = elem_ty.layout(registry);
    let elem_stride = elem_layout.pad_to_align().size();

    type ParamSerializer<'a, TType, TLibfunc> =
        <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Serializer<'a>;

    let ptr_layout = Layout::new::<*mut ()>();
    let len_layout = crate::utils::get_integer_layout(32);

    let array_data_ptr = ptr.cast::<*mut ()>();
    let len_value = *ptr
        .map_addr(|addr| addr.unchecked_add(ptr_layout.extend(len_layout).unwrap().1))
        .cast::<u32>()
        .as_ref();

    let cap_value = *ptr
        .map_addr(|addr| {
            addr.unchecked_add(
                ptr_layout
                    .extend(len_layout)
                    .unwrap()
                    .0
                    .extend(len_layout)
                    .unwrap()
                    .1,
            )
        })
        .cast::<u32>()
        .as_ref();

    let mut ser = serializer.serialize_seq(Some(len_value.try_into().unwrap()))?;
    let mut cur_elem_ptr = array_data_ptr;

    for i in 0..(len_value as usize) {
        cur_elem_ptr = cur_elem_ptr.map_addr(|addr| addr.unchecked_add(elem_stride * i));

        ser.serialize_element(&ParamSerializer::<TType, TLibfunc>::new(
            cur_elem_ptr.cast(),
            registry,
            elem_ty,
        ))?;
    }
    ser.end()
}

pub unsafe fn debug_fmt<TType, TLibfunc>(
    f: &mut fmt::Formatter,
    _id: &ConcreteTypeId,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    ptr: NonNull<()>,
    _info: &InfoAndTypeConcreteType,
) -> fmt::Result
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
{
    todo!()
}
