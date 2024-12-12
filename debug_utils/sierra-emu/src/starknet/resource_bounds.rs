use crate::Value;
use serde::Serialize;
use starknet_types_core::felt::Felt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct ResourceBounds {
    pub resource: Felt,
    pub max_amount: u64,
    pub max_price_per_unit: u128,
}

impl ResourceBounds {
    pub(crate) fn into_value(self) -> Value {
        Value::Struct(vec![
            Value::Felt(self.resource),
            Value::U64(self.max_amount),
            Value::U128(self.max_price_per_unit),
        ])
    }
}
