#![cfg(feature = "with-trace-dump")]

use std::collections::HashMap;

use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::{ConcreteTypeId, VarId},
    program::StatementIdx,
    program_registry::ProgramRegistry,
};
use cairo_lang_utils::ordered_hash_map::OrderedHashMap;
use melior::{
    ir::{BlockRef, Location, Module, Value},
    Context,
};

use crate::metadata::MetadataStorage;

#[allow(clippy::too_many_arguments)]
pub fn build_state_snapshot(
    _context: &Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _module: &Module,
    _block: &BlockRef,
    _location: Location,
    _metadata: &mut MetadataStorage,
    statement_idx: StatementIdx,
    state: &OrderedHashMap<VarId, Value>,
    var_types: &HashMap<VarId, ConcreteTypeId>,
) {
    println!("Statement: {}", statement_idx);

    for (var_id, value) in state.iter() {
        let value_type_id = var_types.get(var_id).unwrap();

        println!("- {}: {} = {}", var_id, value_type_id, value);
    }
}
