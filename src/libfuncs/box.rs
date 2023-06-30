//! # Box libfuncs

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::MetadataStorage,
    types::TypeBuilder,
};
use cairo_lang_sierra::{
    extensions::{
        boxing::BoxConcreteLibfunc, lib_func::SignatureAndTypeConcreteLibfunc, GenericLibfunc,
        GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Block, Location},
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &BoxConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        BoxConcreteLibfunc::Into(info) => {
            build_into_box(context, registry, entry, location, helper, metadata, info)
        }
        BoxConcreteLibfunc::Unbox(info) => {
            build_unbox(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `into_box` libfunc.
pub fn build_into_box<'ctx, 'this, TType, TLibfunc>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));
    Ok(())
}

/// Generate MLIR operations for the `unbox` libfunc.
pub fn build_unbox<'ctx, 'this, TType, TLibfunc>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::run_cairo;
    use serde_json::json;

    #[test]
    fn run_box_unbox() {
        let result = run_cairo! { run_test() in mod {
            use box::BoxTrait;
            use box::BoxImpl;

            fn run_test() -> u32 {
                let x: u32 = 2_u32;
                let box_x: Box<u32> = BoxTrait::new(x);
                box_x.unbox()
            }
        }};

        assert_eq!(result, json!([2]));
    }
}
