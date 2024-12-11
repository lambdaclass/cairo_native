//! This file contains the gas calculation metadata.
//!
//! Each statement has an associated `GasCost`, which represents the cost of
//! executing that statement, in terms of tokens.
//!
//! To calculate the actual cost, the amount of tokens is multiplied by the cost
//! of the given token type. The cost of each token type is specified on runtime,
//! with the `BuiltinCosts` structure.
//!
//! When implementing libfuncs, the `GasCost` metadata entry already contains
//! the `GasCost` for the current sierra statement

use cairo_lang_runner::token_gas_cost;
use cairo_lang_sierra::{
    extensions::gas::CostTokenType,
    ids::FunctionId,
    program::{Program, StatementIdx},
};
use cairo_lang_sierra_ap_change::{
    ap_change_info::ApChangeInfo, calc_ap_changes,
    compute::calc_ap_changes as linear_calc_ap_changes, ApChangeError,
};
use cairo_lang_sierra_gas::{
    compute_postcost_info, compute_precost_info, gas_info::GasInfo, CostError,
};
use cairo_lang_utils::ordered_hash_map::OrderedHashMap;
use std::collections::BTreeMap;

/// Holds global gas info.
#[derive(Default)]
pub struct GasMetadata(pub CairoGasMetadata);

/// The gas cost associated to a determined sierra statement.
///
/// It contains the amount of tokens for each token type,
/// that a given sierra statement costs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GasCost(pub Vec<(u64, CostTokenType)>);

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

impl GasMetadata {
    pub fn new(
        sierra_program: &Program,
        config: Option<MetadataComputationConfig>,
    ) -> Result<GasMetadata, GasMetadataError> {
        let cairo_gas_metadata = if let Some(metadata_config) = config {
            calc_metadata(sierra_program, metadata_config)?
        } else {
            calc_metadata_ap_change_only(sierra_program)?
        };

        Ok(GasMetadata::from(cairo_gas_metadata))
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
                .map(|(token_type, val)| {
                    TryInto::<usize>::try_into(*val)
                        .expect("could not cast gas cost from i64 to usize")
                        * token_gas_cost(*token_type)
                })
                .sum::<usize>() as u64,
        )
    }

    pub fn initial_required_gas_for_entry_points(
        &self,
    ) -> NativeResult<BTreeMap<u64, BTreeMap<u64, u64>>> {
        self.gas_info
            .function_costs
            .iter()
            .map(|func| {
                Ok((func.0.id, {
                    let mut costs = BTreeMap::new();

                    for (token, val) in func.1.iter() {
                        let offset: u64 = match token {
                            CostTokenType::Const => 0,
                            CostTokenType::Pedersen => 1,
                            CostTokenType::Bitwise => 2,
                            CostTokenType::EcOp => 3,
                            CostTokenType::Poseidon => 4,
                            CostTokenType::AddMod => 5,
                            CostTokenType::MulMod => 6,
                            _ => native_panic!("matched an unexpected CostTokenType"),
                        };
                        costs.insert(offset, *val as u64);
                    }

                    costs
                }))
            })
            .collect()
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
        self.gas_info
            .variable_values
            .get(&(idx, cost_type))
            .copied()
            .map(|x| x.try_into().expect("gas cost couldn't be converted to u64"))
    }
}

impl From<CairoGasMetadata> for GasMetadata {
    fn from(value: CairoGasMetadata) -> Self {
        Self(value)
    }
}

impl Deref for GasMetadata {
    type Target = CairoGasMetadata;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Debug for GasMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GasMetadata")
            .field("ap_change_info", &self.ap_change_info)
            .field("gas_info", &self.gas_info)
            .finish()
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

impl From<CairoGasMetadataError> for GasMetadataError {
    fn from(value: CairoGasMetadataError) -> Self {
        match value {
            CairoGasMetadataError::ApChangeError(x) => GasMetadataError::ApChangeError(x),
            CairoGasMetadataError::CostError(x) => GasMetadataError::CostError(x),
        }
    }
}
