#![allow(dead_code)]

//! # Debug libfuncs
//!
//! TODO

// Printable: 9-13, 27, 32, 33-126
//     is_ascii_graphic() -> 33-126
//     is_ascii_whitespace():
//         U+0009 HORIZONTAL TAB
//         U+000A LINE FEED
//         U+000C FORM FEED
//         U+000D CARRIAGE RETURN.
//         U+0020 SPACE

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    types::TypeBuilder,
};
use cairo_lang_sierra::{
    extensions::{
        debug::DebugConcreteLibfunc, lib_func::SignatureOnlyConcreteLibfunc, GenericLibfunc,
        GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith, cf, llvm},
    ir::{
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        r#type::IntegerType,
        Block, Location,
    },
    Context,
};

pub fn build<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, '_>,
    metadata: &mut MetadataStorage,
    selector: &DebugConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        DebugConcreteLibfunc::Print(info) => {
            build_print(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_print<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, '_>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    let stdout_fd = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(1, IntegerType::new(context, 32).into()).into(),
            location,
        ))
        .result(0)?
        .into();

    let values_ptr = entry
        .append_operation(llvm::extract_value(
            context,
            entry.argument(0)?.into(),
            DenseI64ArrayAttribute::new(context, &[0]),
            llvm::r#type::pointer(IntegerType::new(context, 252).into(), 0),
            location,
        ))
        .result(0)?
        .into();
    let values_len = entry
        .append_operation(llvm::extract_value(
            context,
            entry.argument(0)?.into(),
            DenseI64ArrayAttribute::new(context, &[1]),
            IntegerType::new(context, 32).into(),
            location,
        ))
        .result(0)?
        .into();

    let return_code = runtime_bindings.libfunc_debug_print(
        context, helper, entry, stdout_fd, values_ptr, values_len, location,
    )?;

    let k0 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(0, IntegerType::new(context, 32).into()).into(),
            location,
        ))
        .result(0)?
        .into();
    let return_code_is_ok = entry
        .append_operation(arith::cmpi(
            context,
            arith::CmpiPredicate::Eq,
            return_code,
            k0,
            location,
        ))
        .result(0)?
        .into();
    cf::assert(
        context,
        return_code_is_ok,
        "Print libfunc invocation failed.",
        location,
    );

    entry.append_operation(helper.br(0, &[], location));

    Ok(())
}
