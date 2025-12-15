use cairo_lang_sierra::ids::ConcreteTypeId;
use starknet_crypto::Felt;

use crate::{starknet::ResourceBounds, Value};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct TxV3Info {
    pub version: Felt,
    pub account_contract_address: Felt,
    pub max_fee: u128,
    pub signature: Vec<Felt>,
    pub transaction_hash: Felt,
    pub chain_id: Felt,
    pub nonce: Felt,
    pub resource_bounds: Vec<ResourceBounds>,
    pub tip: u128,
    pub paymaster_data: Vec<Felt>,
    pub nonce_data_availability_mode: u32,
    pub fee_data_availability_mode: u32,
    pub account_deployment_data: Vec<Felt>,
    pub proof_facts: Vec<Felt>,
}

impl TxV3Info {
    pub(crate) fn into_value(
        self,
        felt252_ty: ConcreteTypeId,
        resource_bounds_ty: ConcreteTypeId,
    ) -> Value {
        Value::Struct(vec![
            Value::Felt(self.version),
            Value::Felt(self.account_contract_address),
            Value::U128(self.max_fee),
            Value::Struct(vec![Value::Array {
                ty: felt252_ty.clone(),
                data: self.signature.into_iter().map(Value::Felt).collect(),
            }]),
            Value::Felt(self.transaction_hash),
            Value::Felt(self.chain_id),
            Value::Felt(self.nonce),
            Value::Struct(vec![Value::Array {
                ty: resource_bounds_ty,
                data: self
                    .resource_bounds
                    .into_iter()
                    .map(ResourceBounds::into_value)
                    .collect(),
            }]),
            Value::U128(self.tip),
            Value::Struct(vec![Value::Array {
                ty: felt252_ty.clone(),
                data: self.paymaster_data.into_iter().map(Value::Felt).collect(),
            }]),
            Value::U32(self.nonce_data_availability_mode),
            Value::U32(self.fee_data_availability_mode),
            Value::Struct(vec![Value::Array {
                ty: felt252_ty.clone(),
                data: self
                    .account_deployment_data
                    .into_iter()
                    .map(Value::Felt)
                    .collect(),
            }]),
            Value::Array {
                ty: felt252_ty,
                data: self.proof_facts.into_iter().map(Value::Felt).collect(),
            },
        ])
    }
}
