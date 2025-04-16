use super::EvalAction;
use crate::Value;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        int::{
            unsigned::{Uint64Concrete, Uint64Traits},
            IntConstConcreteLibfunc, IntOperationConcreteLibfunc, IntOperator,
        },
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use smallvec::smallvec;
use starknet_crypto::Felt;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &Uint64Concrete,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        Uint64Concrete::Const(info) => eval_const(registry, info, args),
        Uint64Concrete::Operation(info) => eval_operation(registry, info, args),
        Uint64Concrete::SquareRoot(_) => todo!(),
        Uint64Concrete::Equal(info) => eval_equal(registry, info, args),
        Uint64Concrete::ToFelt252(info) => eval_to_felt252(registry, info, args),
        Uint64Concrete::FromFelt252(info) => eval_from_felt(registry, info, args),
        Uint64Concrete::IsZero(info) => eval_is_zero(registry, info, args),
        Uint64Concrete::Divmod(info) => eval_divmod(registry, info, args),
        Uint64Concrete::WideMul(info) => eval_widemul(registry, info, args),
        Uint64Concrete::Bitwise(info) => eval_bitwise(registry, info, args),
    }
}

pub fn eval_divmod(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [range_check @ Value::Unit, Value::U64(x), Value::U64(y)]: [Value; 3] =
        args.try_into().unwrap()
    else {
        panic!()
    };

    let val = Value::U64(x / y);
    let rem = Value::U64(x % y);

    EvalAction::NormalBranch(0, smallvec![range_check, val, rem])
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

    let max = Felt::from(u64::MAX);

    if value <= max {
        let value: u64 = value.to_biguint().try_into().unwrap();
        EvalAction::NormalBranch(0, smallvec![range_check, Value::U64(value)])
    } else {
        EvalAction::NormalBranch(1, smallvec![range_check])
    }
}

pub fn eval_operation(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &IntOperationConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [range_check @ Value::Unit, Value::U64(lhs), Value::U64(rhs)]: [Value; 3] =
        args.try_into().unwrap()
    else {
        panic!()
    };

    let (result, has_overflow) = match info.operator {
        IntOperator::OverflowingAdd => lhs.overflowing_add(rhs),
        IntOperator::OverflowingSub => lhs.overflowing_sub(rhs),
    };

    EvalAction::NormalBranch(
        has_overflow as usize,
        smallvec![range_check, Value::U64(result)],
    )
}

pub fn eval_equal(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::U64(lhs), Value::U64(rhs)]: [Value; 2] = args.try_into().unwrap() else {
        panic!()
    };

    EvalAction::NormalBranch((lhs == rhs) as usize, smallvec![])
}

pub fn eval_bitwise(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [bitwise @ Value::Unit, Value::U64(lhs), Value::U64(rhs)]: [Value; 3] =
        args.try_into().unwrap()
    else {
        panic!()
    };

    let and = lhs & rhs;
    let or = lhs | rhs;
    let xor = lhs ^ rhs;

    EvalAction::NormalBranch(
        0,
        smallvec![bitwise, Value::U64(and), Value::U64(xor), Value::U64(or)],
    )
}

pub fn eval_is_zero(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [vm_value @ Value::U64(value)]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    if value == 0 {
        EvalAction::NormalBranch(0, smallvec![])
    } else {
        EvalAction::NormalBranch(1, smallvec![vm_value])
    }
}

pub fn eval_to_felt252(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::U64(value)]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    EvalAction::NormalBranch(0, smallvec![Value::Felt(value.into())])
}

pub fn eval_const(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &IntConstConcreteLibfunc<Uint64Traits>,
    _args: Vec<Value>,
) -> EvalAction {
    EvalAction::NormalBranch(0, smallvec![Value::U64(info.c)])
}

pub fn eval_widemul(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::U64(lhs), Value::U64(rhs)]: [Value; 2] = args.try_into().unwrap() else {
        panic!()
    };

    let result = (lhs as u128) * (rhs as u128);

    EvalAction::NormalBranch(0, smallvec![Value::U128(result)])
}
