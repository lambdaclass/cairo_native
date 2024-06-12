//use cairo_lang_runner::token_gas_cost;
use cairo_lang_runner::token_gas_cost;
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::gas::CostTokenType,
    extensions::gas::CostTokenType,
//    ids::FunctionId,
    ids::FunctionId,
//    program::{Program, StatementIdx},
    program::{Program, StatementIdx},
//};
};
//use cairo_lang_sierra_ap_change::{ap_change_info::ApChangeInfo, calc_ap_changes};
use cairo_lang_sierra_ap_change::{ap_change_info::ApChangeInfo, calc_ap_changes};
//use cairo_lang_sierra_ap_change::{
use cairo_lang_sierra_ap_change::{
//    compute::calc_ap_changes as linear_calc_ap_changes, ApChangeError,
    compute::calc_ap_changes as linear_calc_ap_changes, ApChangeError,
//};
};
//use cairo_lang_sierra_gas::{
use cairo_lang_sierra_gas::{
//    compute_postcost_info, compute_precost_info, gas_info::GasInfo, CostError,
    compute_postcost_info, compute_precost_info, gas_info::GasInfo, CostError,
//};
};
//use cairo_lang_utils::{casts::IntoOrPanic, ordered_hash_map::OrderedHashMap};
use cairo_lang_utils::{casts::IntoOrPanic, ordered_hash_map::OrderedHashMap};
//

///// Holds global gas info.
/// Holds global gas info.
//#[derive(Debug, Default, PartialEq, Eq)]
#[derive(Debug, Default, PartialEq, Eq)]
//pub struct GasMetadata {
pub struct GasMetadata {
//    pub ap_change_info: ApChangeInfo,
    pub ap_change_info: ApChangeInfo,
//    pub gas_info: GasInfo,
    pub gas_info: GasInfo,
//}
}
//

//#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
//pub struct GasCost(pub Option<u128>);
pub struct GasCost(pub Option<u128>);
//

///// Configuration for metadata computation.
/// Configuration for metadata computation.
//#[derive(Debug, Clone)]
#[derive(Debug, Clone)]
//pub struct MetadataComputationConfig {
pub struct MetadataComputationConfig {
//    pub function_set_costs: OrderedHashMap<FunctionId, OrderedHashMap<CostTokenType, i32>>,
    pub function_set_costs: OrderedHashMap<FunctionId, OrderedHashMap<CostTokenType, i32>>,
//    // ignored, its always used
    // ignored, its always used
//    pub linear_gas_solver: bool,
    pub linear_gas_solver: bool,
//    pub linear_ap_change_solver: bool,
    pub linear_ap_change_solver: bool,
//}
}
//

///// Error for metadata calculations.
/// Error for metadata calculations.
//#[derive(Debug, thiserror::Error, Eq, PartialEq)]
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
//pub enum GasMetadataError {
pub enum GasMetadataError {
//    #[error(transparent)]
    #[error(transparent)]
//    ApChangeError(#[from] ApChangeError),
    ApChangeError(#[from] ApChangeError),
//    #[error(transparent)]
    #[error(transparent)]
//    CostError(#[from] CostError),
    CostError(#[from] CostError),
//    #[error("not enough gas to run")]
    #[error("not enough gas to run")]
//    NotEnoughGas,
    NotEnoughGas,
//}
}
//

//impl Default for MetadataComputationConfig {
impl Default for MetadataComputationConfig {
//    fn default() -> Self {
    fn default() -> Self {
//        Self {
        Self {
//            function_set_costs: Default::default(),
            function_set_costs: Default::default(),
//            linear_gas_solver: true,
            linear_gas_solver: true,
//            linear_ap_change_solver: true,
            linear_ap_change_solver: true,
//        }
        }
//    }
    }
//}
}
//

//impl GasMetadata {
impl GasMetadata {
//    pub fn new(
    pub fn new(
//        sierra_program: &Program,
        sierra_program: &Program,
//        config: Option<MetadataComputationConfig>,
        config: Option<MetadataComputationConfig>,
//    ) -> Result<GasMetadata, GasMetadataError> {
    ) -> Result<GasMetadata, GasMetadataError> {
//        if let Some(metadata_config) = config {
        if let Some(metadata_config) = config {
//            calc_metadata(sierra_program, metadata_config)
            calc_metadata(sierra_program, metadata_config)
//        } else {
        } else {
//            calc_metadata_ap_change_only(sierra_program)
            calc_metadata_ap_change_only(sierra_program)
//        }
        }
//    }
    }
//

//    /// Returns the initial value for the gas counter.
    /// Returns the initial value for the gas counter.
//    /// If `available_gas` is None returns 0.
    /// If `available_gas` is None returns 0.
//    pub fn get_initial_available_gas(
    pub fn get_initial_available_gas(
//        &self,
        &self,
//        func: &FunctionId,
        func: &FunctionId,
//        available_gas: Option<u128>,
        available_gas: Option<u128>,
//    ) -> Result<u128, GasMetadataError> {
    ) -> Result<u128, GasMetadataError> {
//        let Some(available_gas) = available_gas else {
        let Some(available_gas) = available_gas else {
//            return Ok(0);
            return Ok(0);
//        };
        };
//

//        // In case we don't have any costs - it means no gas equations were solved (and we are in
        // In case we don't have any costs - it means no gas equations were solved (and we are in
//        // the case of no gas checking enabled) - so the gas builtin is irrelevant, and we
        // the case of no gas checking enabled) - so the gas builtin is irrelevant, and we
//        // can return any value.
        // can return any value.
//        let Some(required_gas) = self.initial_required_gas(func) else {
        let Some(required_gas) = self.initial_required_gas(func) else {
//            return Ok(0);
            return Ok(0);
//        };
        };
//

//        available_gas
        available_gas
//            .checked_sub(required_gas)
            .checked_sub(required_gas)
//            .ok_or(GasMetadataError::NotEnoughGas)
            .ok_or(GasMetadataError::NotEnoughGas)
//    }
    }
//

//    pub fn initial_required_gas(&self, func: &FunctionId) -> Option<u128> {
    pub fn initial_required_gas(&self, func: &FunctionId) -> Option<u128> {
//        if self.gas_info.function_costs.is_empty() {
        if self.gas_info.function_costs.is_empty() {
//            return None;
            return None;
//        }
        }
//        Some(
        Some(
//            self.gas_info.function_costs[func]
            self.gas_info.function_costs[func]
//                .iter()
                .iter()
//                .map(|(token_type, val)| val.into_or_panic::<usize>() * token_gas_cost(*token_type))
                .map(|(token_type, val)| val.into_or_panic::<usize>() * token_gas_cost(*token_type))
//                .sum::<usize>() as u128,
                .sum::<usize>() as u128,
//        )
        )
//    }
    }
//

//    pub fn get_gas_cost_for_statement(&self, idx: StatementIdx) -> Option<u128> {
    pub fn get_gas_cost_for_statement(&self, idx: StatementIdx) -> Option<u128> {
//        let mut cost = None;
        let mut cost = None;
//        for cost_type in CostTokenType::iter_casm_tokens() {
        for cost_type in CostTokenType::iter_casm_tokens() {
//            if let Some(amount) =
            if let Some(amount) =
//                self.get_gas_cost_for_statement_and_cost_token_type(idx, *cost_type)
                self.get_gas_cost_for_statement_and_cost_token_type(idx, *cost_type)
//            {
            {
//                *cost.get_or_insert(0) += amount * token_gas_cost(*cost_type) as u128;
                *cost.get_or_insert(0) += amount * token_gas_cost(*cost_type) as u128;
//            }
            }
//        }
        }
//        cost
        cost
//    }
    }
//

//    pub fn get_gas_cost_for_statement_and_cost_token_type(
    pub fn get_gas_cost_for_statement_and_cost_token_type(
//        &self,
        &self,
//        idx: StatementIdx,
        idx: StatementIdx,
//        cost_type: CostTokenType,
        cost_type: CostTokenType,
//    ) -> Option<u128> {
    ) -> Option<u128> {
//        self.gas_info
        self.gas_info
//            .variable_values
            .variable_values
//            .get(&(idx, cost_type))
            .get(&(idx, cost_type))
//            .copied()
            .copied()
//            .map(|x| {
            .map(|x| {
//                x.try_into()
                x.try_into()
//                    .expect("gas cost couldn't be converted to u128, should never happen")
                    .expect("gas cost couldn't be converted to u128, should never happen")
//            })
            })
//    }
    }
//}
}
//

//impl Clone for GasMetadata {
impl Clone for GasMetadata {
//    fn clone(&self) -> Self {
    fn clone(&self) -> Self {
//        Self {
        Self {
//            ap_change_info: ApChangeInfo {
            ap_change_info: ApChangeInfo {
//                variable_values: self.ap_change_info.variable_values.clone(),
                variable_values: self.ap_change_info.variable_values.clone(),
//                function_ap_change: self.ap_change_info.function_ap_change.clone(),
                function_ap_change: self.ap_change_info.function_ap_change.clone(),
//            },
            },
//            gas_info: GasInfo {
            gas_info: GasInfo {
//                variable_values: self.gas_info.variable_values.clone(),
                variable_values: self.gas_info.variable_values.clone(),
//                function_costs: self.gas_info.function_costs.clone(),
                function_costs: self.gas_info.function_costs.clone(),
//            },
            },
//        }
        }
//    }
    }
//}
}
//

///// Methods from https://github.com/starkware-libs/cairo/blob/fbdbbe4c42a6808eccbff8436078f73d0710c772/crates/cairo-lang-sierra-to-casm/src/metadata.rs#L71
/// Methods from https://github.com/starkware-libs/cairo/blob/fbdbbe4c42a6808eccbff8436078f73d0710c772/crates/cairo-lang-sierra-to-casm/src/metadata.rs#L71
//

///// Calculates the metadata for a Sierra program, with ap change info only.
/// Calculates the metadata for a Sierra program, with ap change info only.
//fn calc_metadata_ap_change_only(program: &Program) -> Result<GasMetadata, GasMetadataError> {
fn calc_metadata_ap_change_only(program: &Program) -> Result<GasMetadata, GasMetadataError> {
//    Ok(GasMetadata {
    Ok(GasMetadata {
//        ap_change_info: calc_ap_changes(program, |_, _| 0)?,
        ap_change_info: calc_ap_changes(program, |_, _| 0)?,
//        gas_info: GasInfo {
        gas_info: GasInfo {
//            variable_values: Default::default(),
            variable_values: Default::default(),
//            function_costs: Default::default(),
            function_costs: Default::default(),
//        },
        },
//    })
    })
//}
}
//

///// Calculates the metadata for a Sierra program.
/// Calculates the metadata for a Sierra program.
/////
///
///// `no_eq_solver` uses a linear-time algorithm for calculating the gas, instead of solving
/// `no_eq_solver` uses a linear-time algorithm for calculating the gas, instead of solving
///// equations.
/// equations.
//fn calc_metadata(
fn calc_metadata(
//    program: &Program,
    program: &Program,
//    config: MetadataComputationConfig,
    config: MetadataComputationConfig,
//) -> Result<GasMetadata, GasMetadataError> {
) -> Result<GasMetadata, GasMetadataError> {
//    let pre_gas_info = compute_precost_info(program)?;
    let pre_gas_info = compute_precost_info(program)?;
//

//    let ap_change_info = if config.linear_ap_change_solver {
    let ap_change_info = if config.linear_ap_change_solver {
//        linear_calc_ap_changes
        linear_calc_ap_changes
//    } else {
    } else {
//        calc_ap_changes
        calc_ap_changes
//    }(program, |idx, token_type| {
    }(program, |idx, token_type| {
//        pre_gas_info.variable_values[&(idx, token_type)] as usize
        pre_gas_info.variable_values[&(idx, token_type)] as usize
//    })?;
    })?;
//

//    let enforced_function_costs: OrderedHashMap<FunctionId, i32> = config
    let enforced_function_costs: OrderedHashMap<FunctionId, i32> = config
//        .function_set_costs
        .function_set_costs
//        .iter()
        .iter()
//        .map(|(func, costs)| (func.clone(), costs[&CostTokenType::Const]))
        .map(|(func, costs)| (func.clone(), costs[&CostTokenType::Const]))
//        .collect();
        .collect();
//    let post_gas_info = compute_postcost_info(
    let post_gas_info = compute_postcost_info(
//        program,
        program,
//        &|idx| {
        &|idx| {
//            ap_change_info
            ap_change_info
//                .variable_values
                .variable_values
//                .get(idx)
                .get(idx)
//                .copied()
                .copied()
//                .unwrap_or_default()
                .unwrap_or_default()
//        },
        },
//        &pre_gas_info,
        &pre_gas_info,
//        &enforced_function_costs,
        &enforced_function_costs,
//    )?;
    )?;
//

//    Ok(GasMetadata {
    Ok(GasMetadata {
//        ap_change_info,
        ap_change_info,
//        gas_info: pre_gas_info.combine(post_gas_info),
        gas_info: pre_gas_info.combine(post_gas_info),
//    })
    })
//}
}
