use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        pedersen::PedersenConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use smallvec::smallvec;

use crate::Value;

use super::EvalAction;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &PedersenConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        PedersenConcreteLibfunc::PedersenHash(info) => eval_pedersen_hash(registry, info, args),
    }
}

fn eval_pedersen_hash(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [pedersen @ Value::Unit, Value::Felt(lhs), Value::Felt(rhs)]: [Value; 3] =
        args.try_into().unwrap()
    else {
        panic!()
    };

    let res = starknet_crypto::pedersen_hash(&lhs, &rhs);

    EvalAction::NormalBranch(0, smallvec![pedersen, Value::Felt(res),])
}
