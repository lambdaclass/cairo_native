use super::EvalAction;
use crate::Value;
use cairo_lang_sierra::{
    extensions::{
        bytes31::Bytes31ConcreteLibfunc,
        consts::SignatureAndConstConcreteLibfunc,
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use num_bigint::BigInt;
use smallvec::smallvec;
use starknet_crypto::Felt;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &Bytes31ConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        Bytes31ConcreteLibfunc::Const(info) => eval_const(registry, info, args),
        Bytes31ConcreteLibfunc::ToFelt252(info) => eval_to_felt252(registry, info, args),
        Bytes31ConcreteLibfunc::TryFromFelt252(info) => eval_from_felt(registry, info, args),
    }
}

pub fn eval_const(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureAndConstConcreteLibfunc,
    _args: Vec<Value>,
) -> EvalAction {
    EvalAction::NormalBranch(0, smallvec![Value::Bytes31(info.c.clone().into())])
}

pub fn eval_from_felt(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [range_check @ Value::Unit, Value::Felt(value)]: [Value; 2] = args.try_into().unwrap()
    else {
        panic!()
    };

    let max = Felt::from(BigInt::from(2).pow(248) - 1);

    if value <= max {
        EvalAction::NormalBranch(0, smallvec![range_check, Value::Bytes31(value)])
    } else {
        EvalAction::NormalBranch(1, smallvec![range_check])
    }
}

pub fn eval_to_felt252(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Bytes31(value)]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    EvalAction::NormalBranch(0, smallvec![Value::Felt(value)])
}
