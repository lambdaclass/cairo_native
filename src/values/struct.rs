use super::ValueBuilder;
use crate::{types::TypeBuilder, utils::debug_with};
use cairo_lang_sierra::{
    extensions::{structure::StructConcreteType, GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use std::{alloc::Layout, fmt, ptr::NonNull};

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
    <TType as GenericType>::Concrete: ValueBuilder,
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
