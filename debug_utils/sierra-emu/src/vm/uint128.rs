use super::EvalAction;
use crate::Value;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        int::{
            unsigned128::{Uint128Concrete, Uint128Traits},
            IntConstConcreteLibfunc, IntOperationConcreteLibfunc, IntOperator,
        },
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use num_bigint::BigUint;
use smallvec::smallvec;
use starknet_crypto::Felt;
use starknet_types_core::felt::NonZeroFelt;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &Uint128Concrete,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        Uint128Concrete::Const(info) => eval_const(registry, info, args),
        Uint128Concrete::Operation(info) => eval_operation(registry, info, args),
        Uint128Concrete::SquareRoot(info) => eval_square_root(registry, info, args),
        Uint128Concrete::Equal(info) => eval_equal(registry, info, args),
        Uint128Concrete::ToFelt252(info) => eval_to_felt(registry, info, args),
        Uint128Concrete::FromFelt252(info) => eval_from_felt(registry, info, args),
        Uint128Concrete::IsZero(info) => eval_is_zero(registry, info, args),
        Uint128Concrete::Divmod(info) => eval_divmod(registry, info, args),
        Uint128Concrete::Bitwise(info) => eval_bitwise(registry, info, args),
        Uint128Concrete::GuaranteeMul(info) => eval_guarantee_mul(registry, info, args),
        Uint128Concrete::MulGuaranteeVerify(info) => eval_guarantee_verify(registry, info, args),
        Uint128Concrete::ByteReverse(info) => eval_byte_reverse(registry, info, args),
    }
}

pub fn eval_guarantee_mul(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::U128(lhs), Value::U128(rhs)]: [Value; 2] = args.try_into().unwrap() else {
        panic!()
    };

    let mask128 = BigUint::from(u128::MAX);
    let result = BigUint::from(lhs) * BigUint::from(rhs);
    let high = Value::U128((&result >> 128u32).try_into().unwrap());
    let low = Value::U128((result & mask128).try_into().unwrap());

    EvalAction::NormalBranch(0, smallvec![high, low, Value::Unit])
}

pub fn eval_square_root(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [range_check @ Value::Unit, Value::U128(value)]: [Value; 2] = args.try_into().unwrap()
    else {
        panic!()
    };

    let value_big = BigUint::from(value);

    let result: u64 = value_big.sqrt().try_into().unwrap();

    EvalAction::NormalBranch(0, smallvec![range_check, Value::U64(result)])
}

pub fn eval_guarantee_verify(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [range_check @ Value::Unit, _verify @ Value::Unit]: [Value; 2] = args.try_into().unwrap()
    else {
        panic!()
    };

    EvalAction::NormalBranch(0, smallvec![range_check])
}

pub fn eval_divmod(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [range_check @ Value::Unit, Value::U128(x), Value::U128(y)]: [Value; 3] =
        args.try_into().unwrap()
    else {
        panic!()
    };

    let val = Value::U128(x / y);
    let rem = Value::U128(x % y);

    EvalAction::NormalBranch(0, smallvec![range_check, val, rem])
}

pub fn eval_operation(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &IntOperationConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [range_check @ Value::Unit, Value::U128(lhs), Value::U128(rhs)]: [Value; 3] =
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
        smallvec![range_check, Value::U128(result)],
    )
}

pub fn eval_equal(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::U128(lhs), Value::U128(rhs)]: [Value; 2] = args.try_into().unwrap() else {
        panic!()
    };

    EvalAction::NormalBranch((lhs == rhs) as usize, smallvec![])
}

pub fn eval_is_zero(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [vm_value @ Value::U128(value)]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    if value == 0 {
        EvalAction::NormalBranch(0, smallvec![])
    } else {
        EvalAction::NormalBranch(1, smallvec![vm_value])
    }
}

pub fn eval_bitwise(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [bitwise @ Value::Unit, Value::U128(lhs), Value::U128(rhs)]: [Value; 3] =
        args.try_into().unwrap()
    else {
        panic!()
    };

    let and = lhs & rhs;
    let or = lhs | rhs;
    let xor = lhs ^ rhs;

    EvalAction::NormalBranch(
        0,
        smallvec![bitwise, Value::U128(and), Value::U128(xor), Value::U128(or)],
    )
}

pub fn eval_const(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &IntConstConcreteLibfunc<Uint128Traits>,
    _args: Vec<Value>,
) -> EvalAction {
    EvalAction::NormalBranch(0, smallvec![Value::U128(info.c)])
}

pub fn eval_to_felt(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::U128(value)]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    EvalAction::NormalBranch(0, smallvec![Value::Felt(value.into())])
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

    let bound = Felt::from(u128::MAX) + 1;

    if value < bound {
        let value: u128 = value.to_biguint().try_into().unwrap();
        EvalAction::NormalBranch(0, smallvec![range_check, Value::U128(value)])
    } else {
        let (new_value, overflow) = value.div_rem(&NonZeroFelt::try_from(bound).unwrap());

        let overflow: u128 = overflow.to_biguint().try_into().unwrap();
        let new_value: u128 = new_value.to_biguint().try_into().unwrap();
        EvalAction::NormalBranch(
            1,
            smallvec![range_check, Value::U128(new_value), Value::U128(overflow)],
        )
    }
}

pub fn eval_byte_reverse(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [bitwise @ Value::Unit, Value::U128(value)]: [Value; 2] = args.try_into().unwrap() else {
        panic!()
    };

    let value = value.swap_bytes();

    EvalAction::NormalBranch(0, smallvec![bitwise, Value::U128(value)])
}

#[cfg(test)]
mod test {
    use crate::{load_cairo, test_utils::run_test_program, Value};

    #[test]
    fn test_square_root() {
        let (_, program) = load_cairo!(
            use core::num::traits::Sqrt;
            fn main() -> u64 {
                0xffffffffffffffffffffffffffffffff_u128.sqrt()
            }
        );

        let result = run_test_program(program);

        let Value::U64(payload) = result.last().unwrap() else {
            panic!("No output");
        };

        assert_eq!(*payload, 0xffffffffffffffff);
    }
}
