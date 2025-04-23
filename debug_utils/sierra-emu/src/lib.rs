use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::ConcreteTypeId,
    program::{GenFunction, Program, StatementIdx},
    program_registry::ProgramRegistry,
};

pub use self::{dump::*, gas::BuiltinCosts, value::*, vm::VirtualMachine};

mod debug;
mod dump;
mod gas;
pub mod starknet;
mod test_utils;
mod value;
mod vm;

pub fn find_entry_point_by_idx(
    program: &Program,
    entry_point_idx: usize,
) -> Option<&GenFunction<StatementIdx>> {
    program
        .funcs
        .iter()
        .find(|x| x.id.id == entry_point_idx as u64)
}

pub fn find_entry_point_by_name<'a>(
    program: &'a Program,
    name: &str,
) -> Option<&'a GenFunction<StatementIdx>> {
    program
        .funcs
        .iter()
        .find(|x| x.id.debug_name.as_ref().map(|x| x.as_str()) == Some(name))
}

// If type is invisible to sierra (i.e. a single element container),
// finds it's actual concrete type recursively.
// If not, returns the current type
pub fn find_real_type(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    ty: &ConcreteTypeId,
) -> ConcreteTypeId {
    match registry.get_type(ty).unwrap() {
        cairo_lang_sierra::extensions::core::CoreTypeConcrete::Box(info) => {
            find_real_type(registry, &info.ty)
        }
        cairo_lang_sierra::extensions::core::CoreTypeConcrete::Uninitialized(info) => {
            find_real_type(registry, &info.ty)
        }
        cairo_lang_sierra::extensions::core::CoreTypeConcrete::Span(info) => {
            find_real_type(registry, &info.ty)
        }
        cairo_lang_sierra::extensions::core::CoreTypeConcrete::Snapshot(info) => {
            find_real_type(registry, &info.ty)
        }
        _ => ty.clone(),
    }
}
