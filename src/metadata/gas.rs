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
    extensions::{
        circuit::CircuitInfo,
        gas::{CostTokenMap, CostTokenType},
    },
    ids::{ConcreteTypeId, FunctionId},
    program::{Program, StatementIdx},
};
use cairo_lang_sierra_ap_change::{ap_change_info::ApChangeInfo, ApChangeError};
use cairo_lang_sierra_gas::{
    core_libfunc_cost::InvocationCostInfoProvider, gas_info::GasInfo, CostError,
};
use cairo_lang_sierra_to_casm::{
    circuit::CircuitsInfo,
    environment::gas_wallet::{GasWallet as CairoGasWallet, GasWalletError},
    metadata::{
        calc_metadata, calc_metadata_ap_change_only, Metadata as CairoGasMetadata,
        MetadataComputationConfig, MetadataError as CairoGasMetadataError,
    },
};
use cairo_lang_sierra_type_size::ProgramRegistryInfo;
use cairo_lang_utils::unordered_hash_map::UnorderedHashMap;

use crate::{
    error::{panic::ToNativeAssertError, Error, Result as NativeResult},
    native_panic,
};

use std::{cell::Cell, collections::BTreeMap, fmt};

/// Holds global gas info.
#[derive(Default)]
pub struct GasMetadata {
    pub metadata: CairoGasMetadata,
    // This means that MetadataComputationConfig was provided.
    // This allows for the use of GasWallets.
    pub gas_usage_check: bool,
}

/// The gas cost associated to a determined sierra statement.
///
/// It contains the amount of tokens for each token type,
/// that a given sierra statement costs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GasCost(pub Vec<(u64, CostTokenType)>);

#[derive(Debug, Clone)]
pub struct GasWallet(pub CairoGasWallet);

impl GasWallet {
    pub fn update(&mut self, gas_changes: CostTokenMap<i64>) -> Result<(), GasWalletError> {
        self.0.update(gas_changes)?;

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CostInfoProvider {
    type_sizes: UnorderedHashMap<ConcreteTypeId, i16>,
    circuits_info: CircuitsInfo,
    pub gas_metadata: GasMetadata,
    // Current statement id.
    idx: Cell<StatementIdx>,
}

impl CostInfoProvider {
    pub fn new(
        sierra_program: &Program,
        program_info: &ProgramRegistryInfo,
        config: Option<MetadataComputationConfig>,
    ) -> NativeResult<Self> {
        let gas_metadata = GasMetadata::new(sierra_program, program_info, config)?;
        let type_sizes = program_info.type_sizes.clone();
        let circuits_info = CircuitsInfo::new(
            &program_info.registry,
            sierra_program.type_declarations.iter().map(|td| &td.id),
        )
        .to_native_assert_error("Error creating CircuitsInfo")?;

        Ok(Self {
            gas_metadata,
            circuits_info,
            type_sizes,
            idx: Cell::new(StatementIdx(0)),
        })
    }

    pub fn update_statement_id(&self, idx: StatementIdx) {
        self.idx.set(idx);
    }
}

impl InvocationCostInfoProvider for CostInfoProvider {
    fn type_size(&self, ty: &ConcreteTypeId) -> usize {
        self.type_sizes[ty] as usize
    }

    fn token_usages(&self, token_type: CostTokenType) -> usize {
        self.gas_metadata
            .metadata
            .gas_info
            .variable_values
            .get(&(self.idx.get(), token_type))
            .copied()
            .unwrap_or(0) as usize
    }

    fn ap_change_var_value(&self) -> usize {
        self.gas_metadata
            .metadata
            .ap_change_info
            .variable_values
            .get(&self.idx.get())
            .copied()
            .unwrap_or_default()
    }

    fn circuit_info(&self, ty: &ConcreteTypeId) -> &CircuitInfo {
        self.circuits_info.circuits.get(ty).unwrap()
    }
}

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
        sierra_program_info: &ProgramRegistryInfo,
        config: Option<MetadataComputationConfig>,
    ) -> Result<Self, GasMetadataError> {
        let gas_usage_check = config.is_some();
        let cairo_gas_metadata = if let Some(metadata_config) = config {
            calc_metadata(sierra_program, sierra_program_info, metadata_config)?
        } else {
            calc_metadata_ap_change_only(sierra_program, sierra_program_info)?
        };

        Ok(Self {
            metadata: cairo_gas_metadata,
            gas_usage_check,
        })
    }

    /// Returns the initial value for the gas counter.
    /// If `available_gas` is None returns 0.
    pub fn get_initial_available_gas(
        &self,
        func: &FunctionId,
        available_gas: Option<u64>,
    ) -> Result<u64, Error> {
        let Some(available_gas) = available_gas else {
            return Ok(0);
        };

        // In case we don't have any costs - it means no gas equations were solved (and we are in
        // the case of no gas checking enabled) - so the gas builtin is irrelevant, and we
        // can return any value.
        let Some(required_gas) = self.initial_required_gas(func)? else {
            return Ok(0);
        };

        available_gas
            .checked_sub(required_gas)
            .ok_or(Error::GasMetadataError(GasMetadataError::NotEnoughGas {
                gas: Box::new((required_gas, available_gas)),
            }))
    }

    pub fn initial_required_gas(&self, func: &FunctionId) -> Result<Option<u64>, Error> {
        if self.metadata.gas_info.function_costs.is_empty() {
            return Ok(None);
        }
        Ok(Some(
            self.metadata.gas_info.function_costs[func]
                .iter()
                .map(|(token_type, val)| {
                    let Ok(val) = TryInto::<usize>::try_into(*val) else {
                        native_panic!("could not cast gas cost from i64 to usize");
                    };

                    Ok(val * token_gas_cost(*token_type))
                })
                .collect::<Result<Vec<_>, _>>()?
                .iter()
                .sum::<usize>() as u64,
        ))
    }

    pub fn initial_required_gas_for_entry_points(
        &self,
    ) -> NativeResult<BTreeMap<u64, BTreeMap<u64, u64>>> {
        self.metadata
            .gas_info
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
        self.metadata
            .gas_info
            .variable_values
            .get(&(idx, cost_type))
            .copied()
            .map(|x| x.try_into().expect("gas cost couldn't be converted to u64"))
    }
}

// impl From<CairoGasMetadata> for GasMetadata {
//     fn from(value: CairoGasMetadata) -> Self {
//         Self(value)
//     }
// }

impl fmt::Debug for GasMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GasMetadata")
            .field("ap_change_info", &self.metadata.ap_change_info)
            .field("gas_info", &self.metadata.gas_info)
            .finish()
    }
}

impl Clone for GasMetadata {
    fn clone(&self) -> Self {
        Self {
            metadata: CairoGasMetadata {
                ap_change_info: ApChangeInfo {
                    variable_values: self.metadata.ap_change_info.variable_values.clone(),
                    function_ap_change: self.metadata.ap_change_info.function_ap_change.clone(),
                },
                gas_info: GasInfo {
                    variable_values: self.metadata.gas_info.variable_values.clone(),
                    function_costs: self.metadata.gas_info.function_costs.clone(),
                },
            },
            gas_usage_check: self.gas_usage_check,
        }
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
