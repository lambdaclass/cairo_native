use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        poseidon::PoseidonConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use smallvec::smallvec;

use crate::Value;

use super::EvalAction;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &PoseidonConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        PoseidonConcreteLibfunc::HadesPermutation(info) => {
            eval_hades_permutation(registry, info, args)
        }
    }
}

fn eval_hades_permutation(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [poseidon @ Value::Unit, Value::Felt(p1), Value::Felt(p2), Value::Felt(p3)]: [Value; 4] =
        args.try_into().unwrap()
    else {
        panic!()
    };

    let mut state = [p1, p2, p3];

    starknet_crypto::poseidon_permute_comp(&mut state);

    EvalAction::NormalBranch(
        0,
        smallvec![
            poseidon,
            Value::Felt(state[0]),
            Value::Felt(state[1]),
            Value::Felt(state[2])
        ],
    )
}
