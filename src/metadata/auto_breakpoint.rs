#![cfg(feature = "with-debug-utils")]

use super::{debug_utils::DebugUtils, MetadataStorage};
use cairo_lang_sierra::ids::ConcreteTypeId;
use melior::ir::{Block, Location};
use std::collections::HashSet;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum BreakpointEvent {
    EnumInit {
        type_id: ConcreteTypeId,
        variant_idx: usize,
    },
}

#[derive(Clone, Debug, Default)]
pub struct AutoBreakpoint {
    events: HashSet<BreakpointEvent>,
}

impl AutoBreakpoint {
    pub fn add_event(&mut self, event: BreakpointEvent) {
        self.events.insert(event);
    }

    pub fn has_event(&self, event: &BreakpointEvent) -> bool {
        self.events.contains(event)
    }

    pub fn maybe_breakpoint(
        &self,
        block: &Block,
        location: Location,
        metadata: &mut MetadataStorage,
        event: &BreakpointEvent,
    ) {
        if self.has_event(event) {
            metadata
                .get_mut::<DebugUtils>()
                .unwrap()
                .debug_breakpoint_trap(block, location)
                .unwrap();
        }
    }
}
