use crate::compiler::{LibfuncImpl, TypeFactory};
use cairo_lang_sierra::program::{GenericArg, LibfuncDeclaration};

/// Builtin libfunc `store_temp<T>`.
#[must_use]
pub fn store_temp(_builder: &TypeFactory, libfunc_declaration: &LibfuncDeclaration) -> LibfuncImpl {
    assert_eq!(libfunc_declaration.long_id.generic_id.0, "store_temp");
    assert_eq!(libfunc_declaration.long_id.generic_args.len(), 1);

    match &libfunc_declaration.long_id.generic_args[0] {
        GenericArg::Type(_) => {}
        _ => todo!(),
    }

    LibfuncImpl::new(move |factory, args, successors| {
        factory
            .builder("cf.br")
            .add_successor(&successors[0])
            .build();

        vec![vec![args[0].clone()]]
    })
}
