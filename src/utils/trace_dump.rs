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
    ir::{BlockRef, Location, Module, Value, ValueLike},
    Context,
};

use crate::{
    metadata::{trace_dump::TraceDumpMeta, MetadataStorage},
    types::TypeBuilder,
};

use super::BlockExt;

#[allow(clippy::too_many_arguments)]
pub fn build_state_snapshot(
    context: &Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    module: &Module,
    block: &BlockRef,
    location: Location,
    metadata: &mut MetadataStorage,
    statement_idx: StatementIdx,
    state: &OrderedHashMap<VarId, Value>,
    var_types: &HashMap<VarId, ConcreteTypeId>,
) {
    let trace_dump = metadata.get_or_insert_with(TraceDumpMeta::default);

    for (var_id, value) in state.iter() {
        let value_type_id = var_types.get(var_id).unwrap();
        let value_type = registry.get_type(value_type_id).unwrap();

        let layout = value_type.layout(registry).unwrap();

        let ptr_value = block
            .alloca1(context, location, value.r#type(), layout.align())
            .unwrap();
        block.store(context, location, ptr_value, *value).unwrap();

        trace_dump
            .build_state(
                context,
                module,
                block,
                var_id,
                value_type_id,
                ptr_value,
                location,
            )
            .unwrap();
    }

    trace_dump
        .build_push(context, module, block, statement_idx, location)
        .unwrap();
}
