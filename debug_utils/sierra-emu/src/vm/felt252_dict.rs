use super::EvalAction;
use crate::Value;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        felt252_dict::Felt252DictConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use cairo_lang_sierra_gas::core_libfunc_cost::{
    DICT_SQUASH_REPEATED_ACCESS_COST, DICT_SQUASH_UNIQUE_KEY_COST,
};
use smallvec::smallvec;
use std::collections::HashMap;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &Felt252DictConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        Felt252DictConcreteLibfunc::New(info) => eval_new(registry, info, args),
        Felt252DictConcreteLibfunc::Squash(info) => eval_squash(registry, info, args),
    }
}

pub fn eval_new(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [segment_arena @ Value::Unit]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    let type_info = registry
        .get_type(&info.signature.branch_signatures[0].vars[1].ty)
        .unwrap();
    let ty = match type_info {
        CoreTypeConcrete::Felt252Dict(info) => &info.ty,
        _ => unreachable!(),
    };

    EvalAction::NormalBranch(
        0,
        smallvec![
            segment_arena,
            Value::FeltDict {
                ty: ty.clone(),
                data: HashMap::new(),
                count: 0
            },
        ],
    )
}

pub fn eval_squash(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [mut range_check @ Value::RangeCheck(_), Value::U64(gas_builtin), segment_arena @ Value::Unit, Value::FeltDict { ty, data, count }]: [Value; 4] =
        args.try_into().unwrap()
    else {
        panic!();
    };

    // Increment builtin counter
    range_check = match range_check {
        Value::RangeCheck(n) => Value::RangeCheck(n + 1),
        _ => panic!(),
    };

    const DICT_GAS_REFUND_PER_ACCESS: u64 =
        (DICT_SQUASH_UNIQUE_KEY_COST.cost() - DICT_SQUASH_REPEATED_ACCESS_COST.cost()) as u64;

    let refund = count.saturating_sub(data.len() as u64) * DICT_GAS_REFUND_PER_ACCESS;
    let new_gas_builtin = gas_builtin + refund;

    EvalAction::NormalBranch(
        0,
        smallvec![
            range_check,
            Value::U64(new_gas_builtin),
            segment_arena,
            Value::FeltDict { ty, data, count }
        ],
    )
}
