use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        starknet::testing::CheatcodeConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Block, Location},
    Context,
};

use crate::{error::Result, libfuncs::LibfuncHelper, metadata::MetadataStorage};

pub fn build<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _entry: &'this Block<'ctx>,
    _location: Location<'ctx>,
    _helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &CheatcodeConcreteLibfunc,
) -> Result<()> {
    dbg!("building cheatcode", &info.selector);

    let selector_bytes = info.selector.to_bytes_be().1;
    let selector = std::str::from_utf8(&selector_bytes).unwrap();

    dbg!("selector to string", selector);

    Ok(())
}
