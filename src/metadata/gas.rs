use cairo_lang_runner::token_gas_cost;
use cairo_lang_sierra::{
    extensions::gas::CostTokenType,
    ids::FunctionId,
    program::{Function, Program, StatementIdx},
};
use cairo_lang_sierra_ap_change::compute::calc_ap_changes as linear_calc_ap_changes;
use cairo_lang_sierra_ap_change::{ap_change_info::ApChangeInfo, calc_ap_changes};
use cairo_lang_sierra_gas::{
    calc_gas_postcost_info, calc_gas_precost_info, compute_postcost_info, compute_precost_info,
    gas_info::GasInfo,
};
use cairo_lang_utils::ordered_hash_map::OrderedHashMap;

/// Holds global gas info.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct GasMetadata {
    pub ap_change_info: ApChangeInfo,
    pub gas_info: GasInfo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GasCost(pub Option<u128>);

/// Configuration for metadata computation.
#[derive(Default, Debug, Clone)]
pub struct MetadataComputationConfig {
    pub function_set_costs: OrderedHashMap<FunctionId, OrderedHashMap<CostTokenType, i32>>,
    pub linear_gas_solver: bool,
    pub linear_ap_change_solver: bool,
}

impl GasMetadata {
    pub fn new(program: &Program, config: MetadataComputationConfig) -> GasMetadata {
        let pre_function_set_costs = config
            .function_set_costs
            .iter()
            .map(|(func, costs)| {
                (
                    func.clone(),
                    CostTokenType::iter_precost()
                        .filter_map(|token| costs.get(token).map(|v| (*token, *v)))
                        .collect(),
                )
            })
            .collect();
        let pre_gas_info_new = compute_precost_info(program).unwrap();
        let pre_gas_info_old = calc_gas_precost_info(program, pre_function_set_costs).unwrap();
        pre_gas_info_old.assert_eq_functions(&pre_gas_info_new);
        let pre_gas_info = if config.linear_gas_solver {
            pre_gas_info_new
        } else {
            pre_gas_info_old.assert_eq_variables(&pre_gas_info_new);
            pre_gas_info_old
        };

        let ap_change_info = if config.linear_ap_change_solver {
            linear_calc_ap_changes
        } else {
            calc_ap_changes
        }(program, |idx, token_type| {
            pre_gas_info.variable_values[&(idx, token_type)] as usize
        })
        .unwrap();

        let post_function_set_costs = config
            .function_set_costs
            .iter()
            .map(|(func, costs)| {
                (
                    func.clone(),
                    [CostTokenType::Const]
                        .iter()
                        .filter_map(|token| costs.get(token).map(|v| (*token, *v)))
                        .collect(),
                )
            })
            .collect();
        let mut post_gas_info =
            calc_gas_postcost_info(program, post_function_set_costs, &pre_gas_info, |idx| {
                ap_change_info
                    .variable_values
                    .get(&idx)
                    .copied()
                    .unwrap_or_default()
            })
            .unwrap();

        if config.linear_gas_solver {
            let enforced_function_costs: OrderedHashMap<FunctionId, i32> = config
                .function_set_costs
                .iter()
                .map(|(func, costs)| (func.clone(), costs[&CostTokenType::Const]))
                .collect();
            let post_gas_info2 = compute_postcost_info(
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
            )
            .unwrap();

            post_gas_info.assert_eq_functions(&post_gas_info2);

            // Replace post_gas_info with the result of the non-equation-based algorithm.
            post_gas_info = post_gas_info2;
        }

        GasMetadata {
            ap_change_info,
            gas_info: pre_gas_info.combine(post_gas_info),
        }
    }

    // Compute the initial gas required by the function.
    pub fn get_initial_required_gas(&self, func: &FunctionId) -> Option<u128> {
        // In case we don't have any costs - it means no equations were solved - so the gas builtin
        // is irrelevant, and we can return any value.
        if self.gas_info.function_costs.is_empty() {
            return None;
        }

        // Compute the initial gas required by the function.
        let required_gas: usize = self.gas_info.function_costs[func]
            .iter()
            .map(|(cost_token_type, val)| {
                let val_usize: usize = (*val)
                    .try_into()
                    .expect("gas couldn't be converted to u128, should never happen");
                val_usize * token_gas_cost(*cost_token_type)
            })
            .sum();

        Some(required_gas as u128)
    }

    pub fn get_gas_cost_for_statement(
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
