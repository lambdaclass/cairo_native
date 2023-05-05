use crate::compiler::{TypeFactory, TypeLayout};
use cairo_lang_sierra::program::TypeDeclaration;

/// Builtin type `Array<T>`.
#[must_use]
pub fn array(_builder: &TypeFactory, type_declaration: &TypeDeclaration) -> TypeLayout {
    assert_eq!(type_declaration.long_id.generic_id.0, "Array");
    assert_eq!(type_declaration.long_id.generic_args.len(), 1);

    todo!()
}
