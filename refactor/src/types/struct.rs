use crate::compiler::{TypeFactory, TypeLayout};
use cairo_lang_sierra::program::TypeDeclaration;

/// Builtin type `Struct<ut@T, ...>`.
#[must_use]
pub fn r#struct(_builder: &TypeFactory, type_declaration: &TypeDeclaration) -> TypeLayout {
    assert_eq!(type_declaration.long_id.generic_id.0, "felt252");
    assert!(!type_declaration.long_id.generic_args.is_empty());

    todo!()
}
