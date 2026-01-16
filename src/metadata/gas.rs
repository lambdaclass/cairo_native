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

use crate::{
    error::{Error, Result as NativeResult},
    native_panic,
};
use cairo_lang_runner::token_gas_cost;
use cairo_lang_sierra::{
    extensions::{circuit::CircuitInfo, gas::CostTokenType},
    ids::{ConcreteTypeId, FunctionId},
    program::{GenStatement, Program, StatementIdx},
    program_registry::ProgramRegistryError,
};
use cairo_lang_sierra_ap_change::{ap_change_info::ApChangeInfo, ApChangeError};
use cairo_lang_sierra_gas::{
    core_libfunc_cost::{core_libfunc_cost, InvocationCostInfoProvider},
    gas_info::GasInfo,
    CostError,
};
use cairo_lang_sierra_to_casm::{
    circuit::CircuitsInfo,
    compiler::CompilationError,
    environment::gas_wallet::{GasWallet, GasWalletError},
    metadata::{
        calc_metadata, calc_metadata_ap_change_only, Metadata as CairoMetadata,
        MetadataComputationConfig, MetadataError as CairoMetadataError,
    },
};
use cairo_lang_sierra_type_size::{ProgramRegistryInfo, TypeSizeMap};
use cairo_lang_utils::small_ordered_map::SmallOrderedMap;
use itertools::Itertools;
use std::{collections::BTreeMap, fmt};

/// Holds global gas info.
#[derive(Default)]
pub struct GasMetadata {
    cairo_metadata: CairoMetadata,
    statement_wallets: Vec<GasWallet>,
}

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
    #[error(transparent)]
    CairoMetadataError(#[from] CairoMetadataError),
    #[error("Not enough gas to run the operation. Required: {:?}, Available: {:?}.", gas.0, gas.1)]
    NotEnoughGas { gas: Box<(u64, u64)> },
    #[error("Could not find gas wallet for statement")]
    MissingGasWallet,
    #[error("Found an inconsistent gas wallet state")]
    InconsistentGasWallet,
    #[error(transparent)]
    GasWalletError(#[from] GasWalletError),
    #[error(transparent)]
    CasmCompilationError(#[from] Box<CompilationError>),
    #[error(transparent)]
    ProgramRegistryError(#[from] Box<ProgramRegistryError>),
}

impl GasMetadata {
    pub fn new(
        sierra_program: &Program,
        sierra_program_info: &ProgramRegistryInfo,
        config: Option<MetadataComputationConfig>,
    ) -> Result<GasMetadata, GasMetadataError> {
        let cairo_metadata = if let Some(metadata_config) = config {
            calc_metadata(sierra_program, sierra_program_info, metadata_config)?
        } else {
            calc_metadata_ap_change_only(sierra_program, sierra_program_info)?
        };

        let statement_wallets =
            calculate_statement_wallets(sierra_program, sierra_program_info, &cairo_metadata)?;

        Ok(GasMetadata {
            cairo_metadata,
            statement_wallets,
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
        if self.cairo_metadata.gas_info.function_costs.is_empty() {
            return Ok(None);
        }
        Ok(Some(
            self.cairo_metadata.gas_info.function_costs[func]
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
        self.cairo_metadata
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
        self.cairo_metadata
            .gas_info
            .variable_values
            .get(&(idx, cost_type))
            .copied()
            .map(|x| x.try_into().expect("gas cost couldn't be converted to u64"))
    }

    pub fn get_gas_wallet(&self, idx: StatementIdx) -> GasWallet {
        self.statement_wallets[idx.0].clone()
    }
}

/// Calculates the gas wallet for each Sierra statement, which tracks the
/// available gas. The calculation algorithm was taken from the sierra-to-casm
/// compiler.
fn calculate_statement_wallets(
    program: &Program,
    program_info: &ProgramRegistryInfo,
    cairo_metadata: &CairoMetadata,
) -> Result<Vec<GasWallet>, GasMetadataError> {
    let mut wallets: Vec<Option<GasWallet>> = vec![None; program.statements.len()];

    // The gas wallet of a function entrypoint is defined by the cost of calling that function.
    // See https://github.com/starkware-libs/cairo/blob/v2.15.0/crates/cairo-lang-sierra-to-casm/src/annotations.rs#L181
    for function in &program.funcs {
        wallets[function.entry_point.0] = Some(
            match cairo_metadata.gas_info.function_costs.get(&function.id) {
                Some(cost) => GasWallet::Value(cost.clone()),
                None => GasWallet::Disabled,
            },
        );
    }
    let circuits_info = CircuitsInfo::new(
        &program_info.registry,
        program.type_declarations.iter().map(|td| &td.id),
    )
    .map_err(Box::new)?;

    for (statement_idx, statement) in program.statements.iter().enumerate() {
        if let GenStatement::Invocation(statement) = statement {
            let statement_idx = StatementIdx(statement_idx);
            let statement_gas_metadata = StatementCostInfo {
                metadata: cairo_metadata,
                type_sizes: &program_info.type_sizes,
                circuits_info: &circuits_info,
                idx: statement_idx,
            };
            let libfunc = program_info.registry().get_libfunc(&statement.libfunc_id)?;

            // We calculate the gas change for each branch.
            // See https://github.com/starkware-libs/cairo/blob/v2.15.0/crates/cairo-lang-sierra-to-casm/src/invocations/mod.rs#L398.
            let changes = core_libfunc_cost(
                &cairo_metadata.gas_info,
                &statement_idx,
                libfunc,
                &statement_gas_metadata,
            )
            .iter()
            .map(|change| {
                change
                    .iter()
                    .map(|(token_type, val)| (*token_type, -val))
                    .collect::<SmallOrderedMap<_, _>>()
            })
            .collect_vec();

            let src_wallet = wallets[statement_idx.0]
                .clone()
                .ok_or(GasMetadataError::MissingGasWallet)?;

            // We calculate the gas wallet of each branch's statement by
            // updating the current gas wallet with the branch's gas change.
            // See: https://github.com/starkware-libs/cairo/blob/v2.15.0/crates/cairo-lang-sierra-to-casm/src/annotations.rs#L433
            for (branch_info, gas_change) in statement.branches.iter().zip(changes) {
                let dst_statement_idx = statement_idx.next(&branch_info.target);

                let new_wallet = src_wallet.update(gas_change)?;
                let old_wallet = &mut wallets[dst_statement_idx.0];

                // Multiple different statements can branch to the same
                // statement. In all cases, the calculated gas wallet must be
                // the same.
                // See: https://github.com/starkware-libs/cairo/blob/v2.15.0/crates/cairo-lang-sierra-to-casm/src/annotations.rs#L208.
                match old_wallet {
                    Some(old_wallet) => {
                        if new_wallet != *old_wallet {
                            return Err(GasMetadataError::InconsistentGasWallet);
                        }
                    }
                    None => *old_wallet = Some(new_wallet),
                }
            }
        }
    }

    wallets
        .into_iter()
        .map(|w| w.ok_or(GasMetadataError::MissingGasWallet))
        .try_collect()
}

pub struct StatementCostInfo<'m> {
    pub metadata: &'m CairoMetadata,
    pub type_sizes: &'m TypeSizeMap,
    pub circuits_info: &'m CircuitsInfo,
    pub idx: StatementIdx,
}

impl<'m> InvocationCostInfoProvider for StatementCostInfo<'m> {
    fn type_size(&self, ty: &cairo_lang_sierra::ids::ConcreteTypeId) -> usize {
        self.type_sizes[ty] as usize
    }

    fn token_usages(&self, token_type: CostTokenType) -> usize {
        self.metadata
            .gas_info
            .variable_values
            .get(&(self.idx, token_type))
            .copied()
            .unwrap_or(0) as usize
    }

    fn ap_change_var_value(&self) -> usize {
        self.metadata
            .ap_change_info
            .variable_values
            .get(&self.idx)
            .copied()
            .unwrap_or_default()
    }

    fn circuit_info(&self, ty: &ConcreteTypeId) -> &CircuitInfo {
        self.circuits_info.circuits.get(ty).unwrap()
    }
}

impl fmt::Debug for GasMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GasMetadata")
            .field("ap_change_info", &self.cairo_metadata.ap_change_info)
            .field("gas_info", &self.cairo_metadata.gas_info)
            .field("statement_wallets", &self.statement_wallets)
            .finish()
    }
}

impl Clone for GasMetadata {
    fn clone(&self) -> Self {
        Self {
            cairo_metadata: CairoMetadata {
                ap_change_info: ApChangeInfo {
                    variable_values: self.cairo_metadata.ap_change_info.variable_values.clone(),
                    function_ap_change: self
                        .cairo_metadata
                        .ap_change_info
                        .function_ap_change
                        .clone(),
                },
                gas_info: GasInfo {
                    variable_values: self.cairo_metadata.gas_info.variable_values.clone(),
                    function_costs: self.cairo_metadata.gas_info.function_costs.clone(),
                },
            },
            statement_wallets: self.statement_wallets.clone(),
        }
    }
}
