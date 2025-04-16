use crate::Value;
use serde::Serialize;
use starknet_types_core::felt::Felt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct BlockInfo {
    pub block_number: u64,
    pub block_timestamp: u64,
    pub sequencer_address: Felt,
}

impl BlockInfo {
    pub(crate) fn into_value(self) -> Value {
        Value::Struct(vec![
            Value::U64(self.block_number),
            Value::U64(self.block_timestamp),
            Value::Felt(self.sequencer_address),
        ])
    }
}
