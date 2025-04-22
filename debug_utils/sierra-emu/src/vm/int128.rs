use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        int::{signed128::Sint128Concrete, IntOperationConcreteLibfunc, IntOperator},
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
    selector: &Sint128Concrete,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        Sint128Concrete::Const(_) => todo!("int128 const"),
        Sint128Concrete::Operation(info) => eval_operation(registry, info, args),
        Sint128Concrete::Equal(info) => eval_equal(registry, info, args),
        Sint128Concrete::ToFelt252(info) => eval_to_felt(registry, info, args),
        Sint128Concrete::FromFelt252(info) => eval_from_felt(registry, info, args),
        Sint128Concrete::Diff(info) => eval_diff(registry, info, args),
    }
}

fn eval_diff(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _selector: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [range_check @ Value::Unit, Value::I128(lhs), Value::I128(rhs)]: [Value; 3] =
        args.try_into().unwrap()
    else {
        panic!()
    };

    if lhs >= rhs {
        EvalAction::NormalBranch(
            0,
            smallvec![range_check, Value::U128((lhs - rhs).try_into().unwrap())],
        )
    } else {
        EvalAction::NormalBranch(
            1,
            smallvec![
                range_check,
                Value::U128(lhs.wrapping_sub(rhs).try_into().unwrap())
            ],
        )
    }
}

fn eval_operation(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &IntOperationConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [range_check @ Value::Unit, Value::I128(lhs), Value::I128(rhs)]: [Value; 3] =
        args.try_into().unwrap()
    else {
        panic!()
    };

    let (result, overflow) = match selector.operator {
        IntOperator::OverflowingAdd => lhs.overflowing_add(rhs),
        IntOperator::OverflowingSub => lhs.overflowing_sub(rhs),
    };

    EvalAction::NormalBranch(
        overflow as usize,
        smallvec![range_check, Value::I128(result)],
    )
}

fn eval_equal(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _selector: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::I128(lhs), Value::I128(rhs)]: [Value; 2] = args.try_into().unwrap() else {
        panic!()
    };

    EvalAction::NormalBranch((lhs == rhs) as usize, smallvec![])
}

pub fn eval_to_felt(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::I128(value)]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    EvalAction::NormalBranch(0, smallvec![Value::Felt(value.into())])
}

pub fn eval_from_felt(
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

    let min = Felt::from(i128::MIN).to_bigint();
    let max = Felt::from(i128::MAX).to_bigint();

    let value = {
        if value_felt.to_biguint() > half_prime {
            (prime - value_felt.to_biguint()).to_bigint().unwrap() * BigInt::from(-1)
        } else {
            value_felt.to_bigint()
        }
    };

    if value >= min || value <= max {
        let value: i128 = value.try_into().unwrap();
        EvalAction::NormalBranch(0, smallvec![range_check, Value::I128(value)])
    } else {
        EvalAction::NormalBranch(1, smallvec![range_check])
    }
}

#[cfg(test)]
mod tests {
    use crate::{load_cairo, test_utils::run_test_program, Value};

    #[test]
    fn test_eval_diff() {
        let (_, program) = load_cairo!(
            use core::integer;

            fn main() -> u128 {
                integer::i128_diff(2, 1).unwrap()
            }
        );

        let result = run_test_program(program);

        let Value::Enum {
            self_ty: _,
            index: _,
            payload,
            ..
        } = result.last().unwrap()
        else {
            panic!("No output");
        };

        let expected = Value::Struct(vec![Value::U128(1)]);

        assert_eq!(**payload, expected);
    }

    #[test]
    fn test_eval_from_felt() {
        let (_, program) = load_cairo!(
            use core::integer;

            fn main() -> i128 {
                0x7fffffffffffffffffffffffffffffff_felt252
                    .try_into()
                    .unwrap()
            }
        );

        let result = run_test_program(program);

        let Value::Enum {
            self_ty: _,
            index: _,
            payload,
            ..
        } = result.last().unwrap()
        else {
            panic!("No output");
        };

        let expected = Value::Struct(vec![Value::I128(0x7fffffffffffffffffffffffffffffff)]);

        assert_eq!(**payload, expected);
    }
}
