use cairo_lang_runner::token_gas_cost;
use cairo_lang_sierra::{
    extensions::gas::CostTokenType,
    ids::FunctionId,
    program::{Program, StatementIdx},
};
use cairo_lang_sierra_ap_change::{ap_change_info::ApChangeInfo, calc_ap_changes};
use cairo_lang_sierra_ap_change::{
    compute::calc_ap_changes as linear_calc_ap_changes, ApChangeError,
};
use cairo_lang_sierra_gas::{
    compute_postcost_info, compute_precost_info, gas_info::GasInfo, CostError,
};
use cairo_lang_utils::{casts::IntoOrPanic, ordered_hash_map::OrderedHashMap};

/// Holds global gas info.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct GasMetadata {
    pub ap_change_info: ApChangeInfo,
    pub gas_info: GasInfo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GasCost(pub Option<u128>);

/// Configuration for metadata computation.
#[derive(Debug, Clone)]
pub struct MetadataComputationConfig {
    pub function_set_costs: OrderedHashMap<FunctionId, OrderedHashMap<CostTokenType, i32>>,
    // ignored, its always used
    pub linear_gas_solver: bool,
    pub linear_ap_change_solver: bool,
}

/// Error for metadata calculations.
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum GasMetadataError {
    #[error(transparent)]
    ApChangeError(#[from] ApChangeError),
    #[error(transparent)]
    CostError(#[from] CostError),
    #[error("not enough gas to run")]
    NotEnoughGas,
}

impl Default for MetadataComputationConfig {
    fn default() -> Self {
        Self {
            function_set_costs: Default::default(),
            linear_gas_solver: true,
            linear_ap_change_solver: true,
        }
    }
}

impl GasMetadata {
    pub fn new(
        sierra_program: &Program,
        config: Option<MetadataComputationConfig>,
    ) -> Result<GasMetadata, GasMetadataError> {
        if let Some(metadata_config) = config {
            calc_metadata(sierra_program, metadata_config)
        } else {
            calc_metadata_ap_change_only(sierra_program)
        }
    }

    /// Returns the initial value for the gas counter.
    /// If `available_gas` is None returns 0.
    pub fn get_initial_available_gas(
        &self,
        func: &FunctionId,
        available_gas: Option<u128>,
    ) -> Result<u128, GasMetadataError> {
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
            .ok_or(GasMetadataError::NotEnoughGas)
    }

    pub fn initial_required_gas(&self, func: &FunctionId) -> Option<u128> {
        if self.gas_info.function_costs.is_empty() {
            return None;
        }
        Some(
            self.gas_info.function_costs[func]
                .iter()
                .map(|(token_type, val)| val.into_or_panic::<usize>() * token_gas_cost(*token_type))
                .sum::<usize>() as u128,
        )
    }

    pub fn get_gas_cost_for_statement(&self, idx: StatementIdx) -> Option<u128> {
        let mut cost = None;
        for cost_type in CostTokenType::iter() {
            if let Some(amount) =
                self.get_gas_cost_for_statement_and_cost_token_type(idx, *cost_type)
            {
                *cost.get_or_insert(0) += amount * token_gas_cost(*cost_type) as u128;
            }
        }
        cost
    }

    pub fn get_gas_cost_for_statement_and_cost_token_type(
        &self,
        idx: StatementIdx,
        cost_type: CostTokenType,
    ) -> Option<u128> {
        self.gas_info
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
        Self {
            ap_change_info: ApChangeInfo {
                variable_values: self.ap_change_info.variable_values.clone(),
                function_ap_change: self.ap_change_info.function_ap_change.clone(),
            },
            gas_info: GasInfo {
                variable_values: self.gas_info.variable_values.clone(),
                function_costs: self.gas_info.function_costs.clone(),
            },
        }
    }
}

/// Methods from https://github.com/starkware-libs/cairo/blob/fbdbbe4c42a6808eccbff8436078f73d0710c772/crates/cairo-lang-sierra-to-casm/src/metadata.rs#L71

/// Calculates the metadata for a Sierra program, with ap change info only.
fn calc_metadata_ap_change_only(program: &Program) -> Result<GasMetadata, GasMetadataError> {
    Ok(GasMetadata {
        ap_change_info: calc_ap_changes(program, |_, _| 0)?,
        gas_info: GasInfo {
            variable_values: Default::default(),
            function_costs: Default::default(),
        },
    })
}

/// Calculates the metadata for a Sierra program.
///
/// `no_eq_solver` uses a linear-time algorithm for calculating the gas, instead of solving
/// equations.
fn calc_metadata(
    program: &Program,
    config: MetadataComputationConfig,
) -> Result<GasMetadata, GasMetadataError> {
    let pre_gas_info = compute_precost_info(program)?;

    let ap_change_info = if config.linear_ap_change_solver {
        linear_calc_ap_changes
    } else {
        calc_ap_changes
    }(program, |idx, token_type| {
        pre_gas_info.variable_values[&(idx, token_type)] as usize
    })?;

    let enforced_function_costs: OrderedHashMap<FunctionId, i32> = config
        .function_set_costs
        .iter()
        .map(|(func, costs)| (func.clone(), costs[&CostTokenType::Const]))
        .collect();
    let post_gas_info = compute_postcost_info(
        program,
        &|idx| {
            ap_change_info
                .variable_values
                .get(idx)
                .copied()
                .unwrap_or_default()
        },
        &pre_gas_info,
        &enforced_function_costs,
    )?;

    Ok(GasMetadata {
        ap_change_info,
        gas_info: pre_gas_info.combine(post_gas_info),
    })
}
