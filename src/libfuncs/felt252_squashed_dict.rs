use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        squashed_felt252_dict::SquashedFelt252DictConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Block, Location},
    Context,
};

use crate::{error::Result, metadata::MetadataStorage};

use super::{build_noop, LibfuncHelper};

pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &SquashedFelt252DictConcreteLibfunc,
) -> Result<()> {
    match selector {
        // This libfunc is an identity operation in cairo-lang, which means
        // it returns the original value. So in cairo-naitve it's a noop.
        SquashedFelt252DictConcreteLibfunc::IntoEntries(info) => build_noop::<1, false>(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            info.param_signatures(),
        ),
    }
}
