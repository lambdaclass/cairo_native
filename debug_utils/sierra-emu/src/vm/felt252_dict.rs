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
            },
        ],
    )
}

pub fn eval_squash(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [range_check @ Value::Unit, Value::U64(gas_builtin), segment_arena @ Value::Unit, Value::FeltDict { ty, data }]: [Value; 4] =
        args.try_into().unwrap()
    else {
        panic!();
    };

    EvalAction::NormalBranch(
        0,
        smallvec![
            range_check,
            Value::U64(gas_builtin),
            segment_arena,
            Value::FeltDict { ty, data }
        ],
    )
}
