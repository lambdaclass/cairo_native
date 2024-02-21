//! # Enum variants
//!
//! There are some places (ex. the `snapshot_take` libfunc) that require knowing the complete type
//! for some operations (ex. cloning). Sierra provides us with that information for most types, but
//! for enum snapshots we don't have the variants' types.
//!
//! This metadata provides us with a way to retrieve an enum's variants given the snapshot's
//! concrete type id.

use cairo_lang_sierra::ids::ConcreteTypeId;
use std::collections::HashMap;

/// Enum snapshot variants metadata.
#[derive(Default)]
pub struct EnumSnapshotVariantsMeta {
    map: HashMap<ConcreteTypeId, Vec<ConcreteTypeId>>,
}

impl EnumSnapshotVariantsMeta {
    /// Set a mapping from a snapshot enum's concrete type id to its variants.
    pub fn set_mapping(
        &mut self,
        snapshot_id: &ConcreteTypeId,
        enum_variants: Option<&[ConcreteTypeId]>,
    ) {
        if let Some(variants) = enum_variants {
            self.map.insert(snapshot_id.clone(), variants.to_vec());
        }
    }

    /// Retrieve the variants given a snapshot enum's concrete type id.
    pub fn get_variants(&self, snapshot_id: &ConcreteTypeId) -> Option<&Vec<ConcreteTypeId>> {
        self.map.get(snapshot_id)
    }
}
