//use cairo_lang_sierra::ids::ConcreteTypeId;
use cairo_lang_sierra::ids::ConcreteTypeId;
//use std::collections::HashMap;
use std::collections::HashMap;
//

//// Maps a Snapshot<Enum> type to its enum variant types
// Maps a Snapshot<Enum> type to its enum variant types
//

//#[derive(Default)]
#[derive(Default)]
//pub struct EnumSnapshotVariantsMeta {
pub struct EnumSnapshotVariantsMeta {
//    map: HashMap<ConcreteTypeId, Vec<ConcreteTypeId>>,
    map: HashMap<ConcreteTypeId, Vec<ConcreteTypeId>>,
//}
}
//

//impl EnumSnapshotVariantsMeta {
impl EnumSnapshotVariantsMeta {
//    pub fn set_mapping(
    pub fn set_mapping(
//        &mut self,
        &mut self,
//        snapshot_id: &ConcreteTypeId,
        snapshot_id: &ConcreteTypeId,
//        enum_variants: Option<&[ConcreteTypeId]>,
        enum_variants: Option<&[ConcreteTypeId]>,
//    ) {
    ) {
//        if let Some(variants) = enum_variants {
        if let Some(variants) = enum_variants {
//            self.map.insert(snapshot_id.clone(), variants.to_vec());
            self.map.insert(snapshot_id.clone(), variants.to_vec());
//        }
        }
//    }
    }
//

//    pub fn get_variants(&self, snapshot_id: &ConcreteTypeId) -> Option<&Vec<ConcreteTypeId>> {
    pub fn get_variants(&self, snapshot_id: &ConcreteTypeId) -> Option<&Vec<ConcreteTypeId>> {
//        self.map.get(snapshot_id)
        self.map.get(snapshot_id)
//    }
    }
//}
}
