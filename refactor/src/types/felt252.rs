use crate::compiler::{TypeFactory, TypeLayout};
use cairo_lang_sierra::program::TypeDeclaration;

/// Builtin type `felt252`.
#[must_use]
pub fn felt252(builder: &TypeFactory, type_declaration: &TypeDeclaration) -> TypeLayout {
    assert_eq!(type_declaration.long_id.generic_id.0, "felt252");
    assert!(type_declaration.long_id.generic_args.is_empty());

    builder.integer_type(252)
}
