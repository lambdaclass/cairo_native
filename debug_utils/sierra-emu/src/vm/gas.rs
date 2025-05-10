use super::EvalAction;
use crate::{
    gas::{BuiltinCosts, GasMetadata},
    Value,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        gas::{CostTokenType, GasConcreteLibfunc},
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program::StatementIdx,
    program_registry::ProgramRegistry,
};
use smallvec::smallvec;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &GasConcreteLibfunc,
    args: Vec<Value>,
    gas: &GasMetadata,
    statement_idx: StatementIdx,
    builtin_costs: BuiltinCosts,
) -> EvalAction {
    match selector {
        GasConcreteLibfunc::WithdrawGas(info) => {
            eval_withdraw_gas(registry, info, args, gas, statement_idx, builtin_costs)
        }
        GasConcreteLibfunc::RedepositGas(info) => {
            eval_redeposit_gas(registry, info, args, gas, statement_idx, builtin_costs)
        }
        GasConcreteLibfunc::GetAvailableGas(info) => eval_get_available_gas(registry, info, args),
        GasConcreteLibfunc::BuiltinWithdrawGas(info) => {
            eval_builtin_withdraw_gas(registry, info, args, gas, statement_idx)
        }
        GasConcreteLibfunc::GetBuiltinCosts(info) => {
            eval_get_builtin_costs(registry, info, args, builtin_costs)
        }
        GasConcreteLibfunc::GetUnspentGas(_) => todo!(),
    }
}

fn eval_builtin_withdraw_gas(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    gas_meta: &GasMetadata,
    statement_idx: StatementIdx,
) -> EvalAction {
    let [range_check @ Value::Unit, Value::U64(gas), Value::BuiltinCosts(builtin_costs)]: [Value;
        3] = args.try_into().unwrap()
    else {
        panic!()
    };

    let builtin_costs: [u64; 7] = builtin_costs.into();

    let gas_cost = gas_meta.get_gas_costs_for_statement(statement_idx);

    let mut total_gas_cost = 0;

    for (cost_count, token_type) in &gas_cost {
        if *cost_count == 0 {
            continue;
        }

        let builtin_costs_index = match token_type {
            CostTokenType::Const => 0,
            CostTokenType::Pedersen => 1,
            CostTokenType::Bitwise => 2,
            CostTokenType::EcOp => 3,
            CostTokenType::Poseidon => 4,
            CostTokenType::AddMod => 5,
            CostTokenType::MulMod => 6,
            _ => panic!(),
        };

        let cost_value = cost_count * builtin_costs[builtin_costs_index as usize];
        total_gas_cost += cost_value;
    }

    let new_gas = gas.saturating_sub(total_gas_cost);
    if gas >= total_gas_cost {
        EvalAction::NormalBranch(0, smallvec![range_check, Value::U64(new_gas)])
    } else {
        EvalAction::NormalBranch(1, smallvec![range_check, Value::U64(gas)])
    }
}

fn eval_withdraw_gas(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    gas_meta: &GasMetadata,
    statement_idx: StatementIdx,
    builtin_costs: BuiltinCosts,
) -> EvalAction {
    let [range_check @ Value::Unit, Value::U64(gas)]: [Value; 2] = args.try_into().unwrap() else {
        panic!()
    };

    let builtin_costs: [u64; 7] = builtin_costs.into();

    let gas_cost = gas_meta.get_gas_costs_for_statement(statement_idx);

    let mut total_gas_cost = 0;

    for (cost_count, token_type) in &gas_cost {
        if *cost_count == 0 {
            continue;
        }

        let builtin_costs_index = match token_type {
            CostTokenType::Const => 0,
            CostTokenType::Pedersen => 1,
            CostTokenType::Bitwise => 2,
            CostTokenType::EcOp => 3,
            CostTokenType::Poseidon => 4,
            CostTokenType::AddMod => 5,
            CostTokenType::MulMod => 6,
            _ => panic!(),
        };

        let cost_value = cost_count * builtin_costs[builtin_costs_index as usize];
        total_gas_cost += cost_value;
    }

    let new_gas = gas.saturating_sub(total_gas_cost);
    if gas >= total_gas_cost {
        EvalAction::NormalBranch(0, smallvec![range_check, Value::U64(new_gas)])
    } else {
        EvalAction::NormalBranch(1, smallvec![range_check, Value::U64(gas)])
    }
}

fn eval_redeposit_gas(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
    gas_meta: &GasMetadata,
    statement_idx: StatementIdx,
    builtin_costs: BuiltinCosts,
) -> EvalAction {
    let [Value::U64(gas)]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    let builtin_costs: [u64; 7] = builtin_costs.into();

    let gas_cost = gas_meta.get_gas_costs_for_statement(statement_idx);
    let mut total_gas_cost = 0;
    for (cost_count, token_type) in &gas_cost {
        if *cost_count == 0 {
            continue;
        }

        let builtin_costs_index = match token_type {
            CostTokenType::Const => 0,
            CostTokenType::Pedersen => 1,
            CostTokenType::Bitwise => 2,
            CostTokenType::EcOp => 3,
            CostTokenType::Poseidon => 4,
            CostTokenType::AddMod => 5,
            CostTokenType::MulMod => 6,
            _ => panic!(),
        };

        let cost_value = cost_count * builtin_costs[builtin_costs_index as usize];
        total_gas_cost += cost_value;
    }

    let new_gas = gas.saturating_add(total_gas_cost);

    EvalAction::NormalBranch(0, smallvec![Value::U64(new_gas)])
}

fn eval_get_available_gas(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [gas_val @ Value::U64(gas)]: [Value; 1] = args.try_into().unwrap() else {
        panic!();
    };

    EvalAction::NormalBranch(0, smallvec![gas_val, Value::U128(gas as u128)])
}

fn eval_get_builtin_costs(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    _args: Vec<Value>,
    builtin_costs: BuiltinCosts,
) -> EvalAction {
    EvalAction::NormalBranch(0, smallvec![Value::BuiltinCosts(builtin_costs)])
}
