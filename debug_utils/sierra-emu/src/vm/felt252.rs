use super::EvalAction;
use crate::Value;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        felt252::{
            Felt252BinaryOperationConcrete, Felt252BinaryOperator, Felt252Concrete,
            Felt252ConstConcreteLibfunc,
        },
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use smallvec::smallvec;
use starknet_crypto::Felt;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &Felt252Concrete,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        Felt252Concrete::BinaryOperation(info) => eval_operation(registry, info, args),
        Felt252Concrete::Const(info) => eval_const(registry, info, args),
        Felt252Concrete::IsZero(info) => eval_felt_is_zero(registry, info, args),
    }
}

pub fn eval_operation(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &Felt252BinaryOperationConcrete,
    args: Vec<Value>,
) -> EvalAction {
    let res = match info {
        Felt252BinaryOperationConcrete::WithVar(info) => {
            let [Value::Felt(lhs), Value::Felt(rhs)]: [Value; 2] = args.try_into().unwrap() else {
                panic!()
            };

            match info.operator {
                Felt252BinaryOperator::Add => lhs + rhs,
                Felt252BinaryOperator::Sub => lhs - rhs,
                Felt252BinaryOperator::Mul => lhs * rhs,
                Felt252BinaryOperator::Div => lhs.field_div(&rhs.try_into().unwrap()),
            }
        }
        Felt252BinaryOperationConcrete::WithConst(_info) => todo!(),
    };

    EvalAction::NormalBranch(0, smallvec![Value::Felt(res)])
}

pub fn eval_const(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &Felt252ConstConcreteLibfunc,
    _args: Vec<Value>,
) -> EvalAction {
    EvalAction::NormalBranch(0, smallvec![Value::Felt(info.c.clone().into())])
}

pub fn eval_felt_is_zero(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Felt(value)]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    if value == Felt::ZERO {
        EvalAction::NormalBranch(0, smallvec![])
    } else {
        EvalAction::NormalBranch(1, smallvec![Value::Felt(value)])
    }
}
