use crate::Value;
use cairo_lang_sierra::ids::ConcreteTypeId;
use starknet_types_core::felt::Felt;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TxInfo {
    pub version: Felt,
    pub account_contract_address: Felt,
    pub max_fee: u128,
    pub signature: Vec<Felt>,
    pub transaction_hash: Felt,
    pub chain_id: Felt,
    pub nonce: Felt,
}

impl TxInfo {
    pub(crate) fn into_value(self, felt252_ty: ConcreteTypeId) -> Value {
        Value::Struct(vec![
            Value::Felt(self.version),
            Value::Felt(self.account_contract_address),
            Value::U128(self.max_fee),
            Value::Struct(vec![Value::Array {
                ty: felt252_ty,
                data: self.signature.into_iter().map(Value::Felt).collect(),
            }]),
            Value::Felt(self.transaction_hash),
            Value::Felt(self.chain_id),
            Value::Felt(self.nonce),
        ])
    }
}
