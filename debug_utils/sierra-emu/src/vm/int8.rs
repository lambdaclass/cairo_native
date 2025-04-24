use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        int::{
            signed::{Sint8Concrete, Sint8Traits},
            IntConstConcreteLibfunc,
        },
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use num_bigint::{BigInt, BigUint, ToBigInt};
use smallvec::smallvec;
use starknet_crypto::Felt;

use crate::Value;

use super::EvalAction;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &Sint8Concrete,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        Sint8Concrete::Const(info) => eval_const(registry, info, args),
        Sint8Concrete::Operation(info) => eval_operation(registry, info, args),
        Sint8Concrete::Equal(info) => eval_equal(registry, info, args),
        Sint8Concrete::ToFelt252(info) => eval_to_felt(registry, info, args),
        Sint8Concrete::FromFelt252(info) => eval_from_felt(registry, info, args),
        Sint8Concrete::Diff(info) => eval_diff(registry, info, args),
        Sint8Concrete::WideMul(info) => eval_widemul(registry, info, args),
    }
}

fn eval_const(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &IntConstConcreteLibfunc<Sint8Traits>,
    _args: Vec<Value>,
) -> EvalAction {
    EvalAction::NormalBranch(0, smallvec![Value::I8(info.c)])
}

fn eval_equal(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::I8(lhs), Value::I8(rhs)]: [Value; 2] = args.try_into().unwrap() else {
        panic!()
    };

    EvalAction::NormalBranch((lhs == rhs) as usize, smallvec![Value::I8(info.c)])
}

fn eval_to_felt(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::I8(value)]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    EvalAction::NormalBranch(0, smallvec![Value::Felt(value.into())])
}

fn eval_from_felt(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [range_check @ Value::Unit, Value::Felt(value_felt)]: [Value; 2] = args.try_into().unwrap()
    else {
        panic!()
    };
    let prime = Felt::prime();
    let half_prime = &prime / BigUint::from(2u8);

    let min = Felt::from(i8::MIN).to_bigint();
    let max = Felt::from(i8::MAX).to_bigint();

    let value = {
        if value_felt.to_biguint() > half_prime {
            (prime - value_felt.to_biguint()).to_bigint().unwrap() * BigInt::from(-1)
        } else {
            value_felt.to_bigint()
        }
    };

    if value >= min || value <= max {
        let value: i8 = value.try_into().unwrap();
        EvalAction::NormalBranch(0, smallvec![range_check, Value::I8(value)])
    } else {
        EvalAction::NormalBranch(1, smallvec![range_check])
    }
}
