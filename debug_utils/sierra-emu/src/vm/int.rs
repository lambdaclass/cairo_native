use std::fmt::Debug;

use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        int::{
            signed::{SintConcrete, SintTraits},
            signed128::Sint128Concrete,
            unsigned::{UintConcrete, UintTraits},
            unsigned128::Uint128Concrete,
            IntConstConcreteLibfunc, IntMulTraits, IntOperationConcreteLibfunc, IntOperator,
            IntTraits,
        },
        is_zero::IsZeroTraits,
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use num_bigint::{BigInt, BigUint};
use num_traits::ops::overflowing::{OverflowingAdd, OverflowingSub};
use smallvec::smallvec;
use starknet_crypto::Felt;

use crate::{
    utils::{get_numeric_args_as_bigints, get_value_from_integer, integer_range},
    Value,
};

use super::EvalAction;

fn apply_overflowing_op_for_type(
    ty: &CoreTypeConcrete,
    lhs: BigInt,
    rhs: BigInt,
    op: IntOperator,
) -> (BigInt, bool) {
    fn overflowing_op<T>(lhs: BigInt, rhs: BigInt, op: IntOperator) -> (BigInt, bool)
    where
        T: OverflowingAdd + OverflowingSub + Into<BigInt> + TryFrom<BigInt>,
        <T as TryFrom<BigInt>>::Error: Debug,
    {
        let lhs: T = lhs.try_into().unwrap();
        let rhs: T = rhs.try_into().unwrap();

        let (result, had_overflow) = match op {
            IntOperator::OverflowingAdd => OverflowingAdd::overflowing_add(&lhs, &rhs),
            IntOperator::OverflowingSub => OverflowingSub::overflowing_sub(&lhs, &rhs),
        };

        (result.into(), had_overflow)
    }

    match ty {
        CoreTypeConcrete::Sint8(_) => overflowing_op::<u8>(lhs, rhs, op),
        CoreTypeConcrete::Sint16(_) => overflowing_op::<i16>(lhs, rhs, op),
        CoreTypeConcrete::Sint32(_) => overflowing_op::<i32>(lhs, rhs, op),
        CoreTypeConcrete::Sint64(_) => overflowing_op::<i64>(lhs, rhs, op),
        CoreTypeConcrete::Sint128(_) => overflowing_op::<i128>(lhs, rhs, op),
        CoreTypeConcrete::Uint8(_) => overflowing_op::<u8>(lhs, rhs, op),
        CoreTypeConcrete::Uint16(_) => overflowing_op::<u16>(lhs, rhs, op),
        CoreTypeConcrete::Uint32(_) => overflowing_op::<u32>(lhs, rhs, op),
        CoreTypeConcrete::Uint64(_) => overflowing_op::<u64>(lhs, rhs, op),
        CoreTypeConcrete::Uint128(_) => overflowing_op::<u128>(lhs, rhs, op),
        _ => panic!("cannot apply integer operation to non-integer type"),
    }
}

pub fn eval_signed<T>(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &SintConcrete<T>,
    args: Vec<Value>,
) -> EvalAction
where
    T: IntMulTraits + SintTraits,
{
    match selector {
        SintConcrete::Const(info) => eval_const(registry, info, args),
        SintConcrete::Diff(info) => eval_diff(registry, info, args),
        SintConcrete::Equal(info) => eval_equal(registry, info, args),
        SintConcrete::FromFelt252(info) => eval_from_felt(registry, info, args),
        SintConcrete::Operation(info) => eval_operation(registry, info, args),
        SintConcrete::ToFelt252(info) => eval_to_felt(registry, info, args),
        SintConcrete::WideMul(info) => eval_widemul(registry, info, args),
    }
}

pub fn eval_i128(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &Sint128Concrete,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        Sint128Concrete::Const(info) => eval_const(registry, info, args),
        Sint128Concrete::Diff(info) => eval_diff(registry, info, args),
        Sint128Concrete::Equal(info) => eval_equal(registry, info, args),
        Sint128Concrete::FromFelt252(info) => eval_from_felt(registry, info, args),
        Sint128Concrete::Operation(info) => eval_operation(registry, info, args),
        Sint128Concrete::ToFelt252(info) => eval_to_felt(registry, info, args),
    }
}

pub fn eval_unsigned<T>(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &UintConcrete<T>,
    args: Vec<Value>,
) -> EvalAction
where
    T: IntMulTraits + IsZeroTraits + UintTraits,
{
    match selector {
        UintConcrete::Const(info) => eval_const(registry, info, args),
        UintConcrete::Bitwise(info) => eval_bitwise(registry, info, args),
        UintConcrete::Divmod(info) => eval_divmod(registry, info, args),
        UintConcrete::Equal(info) => eval_equal(registry, info, args),
        UintConcrete::FromFelt252(info) => eval_from_felt(registry, info, args),
        UintConcrete::Operation(info) => eval_operation(registry, info, args),
        UintConcrete::IsZero(info) => eval_is_zero(registry, info, args),
        UintConcrete::SquareRoot(info) => eval_square_root(registry, info, args),
        UintConcrete::ToFelt252(info) => eval_to_felt(registry, info, args),
        UintConcrete::WideMul(info) => eval_widemul(registry, info, args),
    }
}

pub fn eval_uint128(
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

fn eval_const<T: IntTraits>(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &IntConstConcreteLibfunc<T>,
    _args: Vec<Value>,
) -> EvalAction {
    let int_ty = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)
        .unwrap();

    EvalAction::NormalBranch(
        0,
        smallvec![get_value_from_integer(registry, int_ty, info.c.into())],
    )
}

fn eval_bitwise(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let bitwise @ Value::Unit: Value = args[0].clone() else {
        panic!()
    };
    let [lhs, rhs]: [BigInt; 2] = get_numeric_args_as_bigints(&args[1..]).try_into().unwrap();

    let int_ty = registry
        .get_type(&info.signature.branch_signatures[0].vars[1].ty)
        .unwrap();

    let and = get_value_from_integer(registry, int_ty, &lhs & &rhs);
    let or = get_value_from_integer(registry, int_ty, &lhs | &rhs);
    let xor = get_value_from_integer(registry, int_ty, &lhs ^ &rhs);

    EvalAction::NormalBranch(0, smallvec![bitwise, and, xor, or])
}

fn eval_diff(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let range_check @ Value::Unit: Value = args[0].clone() else {
        panic!()
    };
    let [lhs, rhs]: [BigInt; 2] = get_numeric_args_as_bigints(&args[1..]).try_into().unwrap();

    let int_ty = registry
        .get_type(&info.signature.branch_signatures[0].vars[1].ty)
        .unwrap();

    let overflow = (lhs >= rhs) as usize;
    let (res, _) = apply_overflowing_op_for_type(int_ty, lhs, rhs, IntOperator::OverflowingSub);
    let res = get_value_from_integer(registry, int_ty, res);

    EvalAction::NormalBranch(overflow, smallvec![range_check, res])
}

fn eval_divmod(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let range_check @ Value::Unit: Value = args[0].clone() else {
        panic!()
    };
    let [lhs, rhs]: [BigInt; 2] = get_numeric_args_as_bigints(&args[1..]).try_into().unwrap();

    let int_ty = registry
        .get_type(&info.signature.branch_signatures[0].vars[1].ty)
        .unwrap();

    let res = &lhs / &rhs;
    let rem = lhs % rhs;

    let res = get_value_from_integer(registry, int_ty, res);
    let rem = get_value_from_integer(registry, int_ty, rem);

    EvalAction::NormalBranch(0, smallvec![range_check, res, rem])
}

fn eval_equal(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [lhs, rhs] = args.try_into().unwrap();

    EvalAction::NormalBranch((lhs == rhs) as usize, smallvec![])
}

fn eval_from_felt(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [range_check @ Value::Unit, Value::Felt(value_felt)]: [Value; 2] = args.try_into().unwrap()
    else {
        panic!()
    };

    let int_ty = registry
        .get_type(&info.signature.branch_signatures[0].vars[1].ty)
        .unwrap();

    let range = integer_range(int_ty, registry);

    let value = value_felt.to_bigint();

    if value > range.lower && value <= range.upper {
        let value = get_value_from_integer(registry, int_ty, value);
        EvalAction::NormalBranch(0, smallvec![range_check, value])
    } else {
        EvalAction::NormalBranch(1, smallvec![range_check])
    }
}

fn eval_is_zero(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value]: [BigInt; 1] = get_numeric_args_as_bigints(&args).try_into().unwrap();

    let int_ty = registry
        .get_type(&info.signature.branch_signatures[1].vars[0].ty)
        .unwrap();

    if value == 0.into() {
        EvalAction::NormalBranch(0, smallvec![])
    } else {
        EvalAction::NormalBranch(
            1,
            smallvec![get_value_from_integer(registry, int_ty, value)],
        )
    }
}

fn eval_operation(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &IntOperationConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let range_check @ Value::Unit: Value = args[0].clone() else {
        panic!()
    };
    let [lhs, rhs]: [BigInt; 2] = get_numeric_args_as_bigints(&args[1..]).try_into().unwrap();
    let int_ty = registry
        .get_type(&info.signature.branch_signatures[0].vars[1].ty)
        .unwrap();

    let (res, had_overflow) = apply_overflowing_op_for_type(int_ty, lhs, rhs, info.operator);
    let res = get_value_from_integer(registry, int_ty, res);

    EvalAction::NormalBranch(had_overflow as usize, smallvec![range_check, res])
}

fn eval_square_root(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let range_check @ Value::Unit: Value = args[0].clone() else {
        panic!()
    };
    let [value]: [BigInt; 1] = get_numeric_args_as_bigints(&args[1..]).try_into().unwrap();

    let int_ty = registry
        .get_type(&info.signature.branch_signatures[0].vars[1].ty)
        .unwrap();

    let res = value.sqrt();

    EvalAction::NormalBranch(
        0,
        smallvec![range_check, get_value_from_integer(registry, int_ty, res)],
    )
}

fn eval_to_felt(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [val]: [BigInt; 1] = get_numeric_args_as_bigints(&args).try_into().unwrap();

    EvalAction::NormalBranch(0, smallvec![Value::Felt(Felt::from(val))])
}

fn eval_widemul(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [lhs, rhs]: [BigInt; 2] = get_numeric_args_as_bigints(&args).try_into().unwrap();

    let int_ty = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)
        .unwrap();

    let res = lhs * rhs;

    EvalAction::NormalBranch(0, smallvec![get_value_from_integer(registry, int_ty, res)])
}

fn eval_guarantee_mul(
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

fn eval_guarantee_verify(
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

fn eval_byte_reverse(
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
