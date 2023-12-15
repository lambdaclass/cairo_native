//! # Box libfuncs

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::{realloc_bindings::ReallocBindingsMeta, MetadataStorage},
    types::TypeBuilder,
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        boxing::BoxConcreteLibfunc, lib_func::SignatureAndTypeConcreteLibfunc, GenericLibfunc,
        GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith,
        llvm::{self, r#type::opaque_pointer, LoadStoreOptions},
    },
    ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, Location},
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
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }

    let inner_type = registry.get_type(&info.ty)?;
    let inner_layout = inner_type.layout(registry)?;

    let op = entry.append_operation(llvm::nullptr(opaque_pointer(context), location));
    let nullptr = op.result(0)?.into();

    let op = entry.append_operation(arith::constant(
        context,
        IntegerAttribute::new(
            inner_layout.pad_to_align().size().try_into()?,
            IntegerType::new(context, 64).into(),
        )
        .into(),
        location,
    ));
    let value_len = op.result(0)?.into();

    let op = entry.append_operation(ReallocBindingsMeta::realloc(
        context, nullptr, value_len, location,
    ));

    let ptr = op.result(0)?.into();

    entry.append_operation(llvm::store(
        context,
        entry.argument(0)?.into(),
        ptr,
        location,
        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
            inner_layout.align() as i64,
            IntegerType::new(context, 64).into(),
        ))),
    ));

    entry.append_operation(helper.br(0, &[ptr], location));
    Ok(())
}

/// Generate MLIR operations for the `unbox` libfunc.
pub fn build_unbox<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let (inner_ty, inner_layout) =
        registry.build_type_with_layout(context, helper, registry, metadata, &info.ty)?;

    let op = entry.append_operation(llvm::load(
        context,
        entry.argument(0)?.into(),
        inner_ty,
        location,
        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
            inner_layout.align() as i64,
            IntegerType::new(context, 64).into(),
        ))),
    ));
    entry.append_operation(helper.br(0, &[op.result(0)?.into()], location));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{load_cairo, run_program_assert_output},
        values::JitValue,
    };

    #[test]
    fn run_box_unbox() {
        let program = load_cairo!(
            use box::BoxTrait;
            use box::BoxImpl;

            fn run_test() -> u32 {
                let x: u32 = 2_u32;
                let box_x: Box<u32> = BoxTrait::new(x);
                box_x.unbox()
            }
        );

        run_program_assert_output(&program, "run_test", &[], JitValue::Uint32(2));
    }
}
