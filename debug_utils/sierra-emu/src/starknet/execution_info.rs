use super::{BlockInfo, TxInfo};
use crate::Value;
use cairo_lang_sierra::ids::ConcreteTypeId;
use starknet_types_core::felt::Felt;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExecutionInfo {
    pub block_info: BlockInfo,
    pub tx_info: TxInfo,
    pub caller_address: Felt,
    pub contract_address: Felt,
    pub entry_point_selector: Felt,
}

impl ExecutionInfo {
    pub(crate) fn into_value(self, felt252_ty: ConcreteTypeId) -> Value {
        Value::Struct(vec![
            self.block_info.into_value(),
            self.tx_info.into_value(felt252_ty),
            Value::Felt(self.caller_address),
            Value::Felt(self.contract_address),
            Value::Felt(self.entry_point_selector),
        ])
    }
}
