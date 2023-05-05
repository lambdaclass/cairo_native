use crate::compiler::{LibfuncImpl, TypeFactory};
use cairo_lang_sierra::program::{GenericArg, LibfuncDeclaration};

/// Builtin libfunc `felt252_const<V>`.
#[must_use]
pub fn felt252_const(
    type_factory: &TypeFactory,
    libfunc_declaration: &LibfuncDeclaration,
) -> LibfuncImpl {
    assert_eq!(libfunc_declaration.long_id.generic_id.0, "felt252_const");
    assert_eq!(libfunc_declaration.long_id.generic_args.len(), 1);

    let felt252_type = type_factory.integer_type(252);

    let value = match &libfunc_declaration.long_id.generic_args[0] {
        GenericArg::Value(x) => x.to_biguint().unwrap(),
        _ => todo!(),
    };

    LibfuncImpl::new(move |factory, _args, successors| {
        let op0 = factory
            .builder("arith.constant")
            .add_string_attribute("value", &value.to_string())
            .add_return_value(&felt252_type)
            .build();

        factory
            .builder("cf.br")
            .add_successor(&successors[0])
            .build();

        vec![vec![op0[0].clone()]]
    })
}
