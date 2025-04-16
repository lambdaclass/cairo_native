use std::{fmt::Debug, ops::Deref};

use cairo_lang_runner::token_gas_cost;
use cairo_lang_sierra::{
    extensions::gas::CostTokenType,
    ids::FunctionId,
    program::{Program, StatementIdx},
};
use cairo_lang_sierra_ap_change::{ap_change_info::ApChangeInfo, ApChangeError};
use cairo_lang_sierra_gas::{gas_info::GasInfo, CostError};
use cairo_lang_sierra_to_casm::metadata::{
    calc_metadata, calc_metadata_ap_change_only, Metadata as CairoGasMetadata,
    MetadataComputationConfig, MetadataError as CairoGasMetadataError,
};
use cairo_lang_utils::casts::IntoOrPanic;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct BuiltinCosts {
    pub r#const: u64,
    pub pedersen: u64,
    pub bitwise: u64,
    pub ecop: u64,
    pub poseidon: u64,
    pub add_mod: u64,
    pub mul_mod: u64,
}

impl Default for BuiltinCosts {
    fn default() -> Self {
        Self {
            r#const: token_gas_cost(CostTokenType::Const) as u64,
            pedersen: token_gas_cost(CostTokenType::Pedersen) as u64,
            bitwise: token_gas_cost(CostTokenType::Bitwise) as u64,
            ecop: token_gas_cost(CostTokenType::EcOp) as u64,
            poseidon: token_gas_cost(CostTokenType::Poseidon) as u64,
            add_mod: token_gas_cost(CostTokenType::AddMod) as u64,
            mul_mod: token_gas_cost(CostTokenType::MulMod) as u64,
        }
    }
}

impl From<BuiltinCosts> for [u64; 7] {
    // Order matters, for the libfunc impl
    // https://github.com/starkware-libs/sequencer/blob/1b7252f8a30244d39614d7666aa113b81291808e/crates/blockifier/src/execution/entry_point_execution.rs#L208
    fn from(value: BuiltinCosts) -> Self {
        [
            value.r#const,
            value.pedersen,
            value.bitwise,
            value.ecop,
            value.poseidon,
            value.add_mod,
            value.mul_mod,
        ]
    }
}

/// Holds global gas info.
#[derive(Default)]
pub struct GasMetadata(pub CairoGasMetadata);

/// Error for metadata calculations.
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum GasMetadataError {
    #[error(transparent)]
    ApChangeError(#[from] ApChangeError),
    #[error(transparent)]
    CostError(#[from] CostError),
    #[error("Not enough gas to run the operation. Required: {:?}, Available: {:?}.", gas.0, gas.1)]
    NotEnoughGas { gas: Box<(u64, u64)> },
}

impl From<CairoGasMetadataError> for GasMetadataError {
    fn from(value: CairoGasMetadataError) -> Self {
        match value {
            CairoGasMetadataError::ApChangeError(e) => Self::ApChangeError(e),
            CairoGasMetadataError::CostError(e) => Self::CostError(e),
        }
    }
}

impl Deref for GasMetadata {
    type Target = CairoGasMetadata;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Debug for GasMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GasMetadata")
            .field("ap_change_info", &self.ap_change_info)
            .field("gas_info", &self.gas_info)
            .finish()
    }
}

impl GasMetadata {
    pub fn new(
        sierra_program: &Program,
        config: Option<MetadataComputationConfig>,
    ) -> Result<GasMetadata, GasMetadataError> {
        let metadata = if let Some(metadata_config) = config {
            calc_metadata(sierra_program, metadata_config).map_err(GasMetadataError::from)?
        } else {
            calc_metadata_ap_change_only(sierra_program).map_err(GasMetadataError::from)?
        };

        Ok(Self(metadata))
    }

    /// Returns the initial value for the gas counter.
    /// If `available_gas` is None returns 0.
    pub fn get_initial_available_gas(
        &self,
        func: &FunctionId,
        available_gas: Option<u64>,
    ) -> Result<u64, GasMetadataError> {
        let Some(available_gas) = available_gas else {
            return Ok(0);
        };

        // In case we don't have any costs - it means no gas equations were solved (and we are in
        // the case of no gas checking enabled) - so the gas builtin is irrelevant, and we
        // can return any value.
        let Some(required_gas) = self.initial_required_gas(func) else {
            return Ok(0);
        };

        available_gas
            .checked_sub(required_gas)
            .ok_or(GasMetadataError::NotEnoughGas {
                gas: Box::new((required_gas, available_gas)),
            })
    }

    pub fn initial_required_gas(&self, func: &FunctionId) -> Option<u64> {
        if self.gas_info.function_costs.is_empty() {
            return None;
        }
        Some(
            self.gas_info.function_costs[func]
                .iter()
                .map(|(token_type, val)| val.into_or_panic::<usize>() * token_gas_cost(*token_type))
                .sum::<usize>() as u64,
        )
    }

    pub fn get_gas_costs_for_statement(&self, idx: StatementIdx) -> Vec<(u64, CostTokenType)> {
        let mut costs = Vec::new();
        for cost_type in CostTokenType::iter_casm_tokens() {
            if let Some(cost_count) =
                self.get_gas_cost_for_statement_and_cost_token_type(idx, *cost_type)
            {
                if cost_count > 0 {
                    costs.push((cost_count, *cost_type));
                }
            }
        }
        costs
    }

    pub fn get_gas_cost_for_statement_and_cost_token_type(
        &self,
        idx: StatementIdx,
        cost_type: CostTokenType,
    ) -> Option<u64> {
        self
            .gas_info
            .variable_values
            .get(&(idx, cost_type))
            .copied()
            .map(|x| {
                x.try_into()
                    .expect("gas cost couldn't be converted to u128, should never happen")
            })
    }
}

impl Clone for GasMetadata {
    fn clone(&self) -> Self {
        Self(CairoGasMetadata {
            ap_change_info: ApChangeInfo {
                variable_values: self.ap_change_info.variable_values.clone(),
                function_ap_change: self.ap_change_info.function_ap_change.clone(),
            },
            gas_info: GasInfo {
                variable_values: self.gas_info.variable_values.clone(),
                function_costs: self.gas_info.function_costs.clone(),
            },
        })
    }
}
