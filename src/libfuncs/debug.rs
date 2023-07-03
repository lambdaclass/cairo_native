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
    metadata::MetadataStorage,
    types::TypeBuilder,
};
use cairo_lang_sierra::{
    extensions::{lib_func::SignatureOnlyConcreteLibfunc, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Block, Location},
    Context,
};

fn build<'ctx, TType, TLibfunc>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _entry: &Block<'ctx>,
    _location: Location<'ctx>,
    _helper: &LibfuncHelper<'ctx, '_>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    todo!()
}
