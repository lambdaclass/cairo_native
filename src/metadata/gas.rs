use cairo_lang_sierra::{
    extensions::gas::CostTokenType,
    ids::FunctionId,
    program::{Program, StatementIdx},
};
use cairo_lang_sierra_ap_change::{ap_change_info::ApChangeInfo, calc_ap_changes};
use cairo_lang_sierra_gas::{calc_gas_postcost_info, compute_precost_info, gas_info::GasInfo};
use cairo_lang_utils::ordered_hash_map::OrderedHashMap;

/// Holds global gas info.
pub struct GasMetadata {
    pub ap_change_info: ApChangeInfo,
    pub gas_info: GasInfo,
}

#[derive(Debug, Clone, Copy)]
pub struct GasCost(pub Option<u64>);

/// Configuration for metadata computation.
#[derive(Default)]
pub struct MetadataComputationConfig {
    pub function_set_costs: OrderedHashMap<FunctionId, OrderedHashMap<CostTokenType, i32>>,
}

impl GasMetadata {
    pub fn new(program: &Program, config: MetadataComputationConfig) -> GasMetadata {
        let pre_gas_info = compute_precost_info(program).unwrap();

        let ap_change_info = calc_ap_changes(program, |idx, token_type| {
            pre_gas_info.variable_values[(idx, token_type)] as usize
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
        let post_gas_info =
            calc_gas_postcost_info(program, post_function_set_costs, &pre_gas_info, |idx| {
                ap_change_info
                    .variable_values
                    .get(&idx)
                    .copied()
                    .unwrap_or_default()
            })
            .unwrap();

        GasMetadata {
            ap_change_info,
            gas_info: pre_gas_info.combine(post_gas_info),
        }
    }

    // Compute the initial gas required by the function.
    pub fn get_initial_required_gas(&self, func: &FunctionId) -> Option<u64> {
        // In case we don't have any costs - it means no equations were solved - so the gas builtin
        // is irrelevant, and we can return any value.
        if self.gas_info.function_costs.is_empty() {
            return None;
        }

        // Compute the initial gas required by the function.
        let required_gas = self.gas_info.function_costs[func.clone()]
            .iter()
            .map(|(cost_token_type, val)| {
                let val_usize: u64 = (*val).try_into().unwrap();
                let token_cost = if *cost_token_type == CostTokenType::Const {
                    1
                } else {
                    10_000 // DUMMY_BUILTIN_GAS_COST
                };
                val_usize * token_cost
            })
            .sum();

        Some(required_gas)
    }

    pub fn get_gas_cost_for_statement(
        &self,
        idx: StatementIdx,
        cost_type: CostTokenType,
    ) -> Option<u64> {
        self.gas_info
            .variable_values
            .get(&(idx, cost_type))
            .copied()
            .map(|x| x as u64)
    }
}
