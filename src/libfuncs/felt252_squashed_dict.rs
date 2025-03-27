use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        squashed_felt252_dict::SquashedFelt252DictConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    helpers::BuiltinBlockExt,
    ir::{Block, BlockLike, Location},
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
        SquashedFelt252DictConcreteLibfunc::IntoEntries(info) => build_noop::<0, false>(
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
