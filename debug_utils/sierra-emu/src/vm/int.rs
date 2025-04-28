use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        int::{
            signed::{SintConcrete, SintTraits},
            unsigned::{UintConcrete, UintTraits},
            IntConstConcreteLibfunc, IntMulTraits, IntOperator, IntTraits,
        },
        is_zero::IsZeroTraits,
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use num_bigint::BigInt;
use num_traits::{
    ops::overflowing::{OverflowingAdd, OverflowingSub},
    ToPrimitive,
};
use smallvec::smallvec;
use starknet_crypto::Felt;

use crate::{utils::get_numberic_args_as_bigints, Value};

use super::EvalAction;

fn get_int_value_from_type(ty: &CoreTypeConcrete, value: BigInt) -> Value {
    match ty {
        CoreTypeConcrete::Sint8(_) => Value::I8(value.to_i8().unwrap()),
        CoreTypeConcrete::Sint16(_) => Value::I16(value.to_i16().unwrap()),
        CoreTypeConcrete::Sint32(_) => Value::I32(value.to_i32().unwrap()),
        CoreTypeConcrete::Sint64(_) => Value::I64(value.to_i64().unwrap()),
        CoreTypeConcrete::Sint128(_) => Value::I128(value.to_i128().unwrap()),
        CoreTypeConcrete::Uint8(_) => Value::U8(value.to_u8().unwrap()),
        CoreTypeConcrete::Uint16(_) => Value::U16(value.to_u16().unwrap()),
        CoreTypeConcrete::Uint32(_) => Value::U32(value.to_u32().unwrap()),
        CoreTypeConcrete::Uint64(_) => Value::U64(value.to_u64().unwrap()),
        CoreTypeConcrete::Uint128(_) => Value::U128(value.to_u128().unwrap()),
        _ => panic!("Found a non-numeric type"),
    }
}

fn apply_wrapping_op(
    ty: &CoreTypeConcrete,
    lhs: BigInt,
    rhs: BigInt,
    op: IntOperator,
) -> (BigInt, usize) {
    fn wrapping<T>(lhs: T, rhs: T, op: IntOperator) -> (BigInt, usize)
    where
        T: OverflowingAdd + OverflowingSub + Into<BigInt>,
    {
        let (res, ovf) = match op {
            IntOperator::OverflowingAdd => OverflowingAdd::overflowing_add(&lhs, &rhs),
            IntOperator::OverflowingSub => OverflowingSub::overflowing_sub(&lhs, &rhs),
        };

        (res.into(), ovf as usize)
    }

    match ty {
        CoreTypeConcrete::Sint8(_) => {
            wrapping::<i8>(lhs.try_into().unwrap(), rhs.try_into().unwrap(), op)
        }
        CoreTypeConcrete::Sint16(_) => {
            wrapping::<i16>(lhs.try_into().unwrap(), rhs.try_into().unwrap(), op)
        }
        CoreTypeConcrete::Sint32(_) => {
            wrapping::<i32>(lhs.try_into().unwrap(), rhs.try_into().unwrap(), op)
        }
        CoreTypeConcrete::Sint64(_) => {
            wrapping::<i64>(lhs.try_into().unwrap(), rhs.try_into().unwrap(), op)
        }
        CoreTypeConcrete::Sint128(_) => {
            wrapping::<i128>(lhs.try_into().unwrap(), rhs.try_into().unwrap(), op)
        }
        CoreTypeConcrete::Uint8(_) => {
            wrapping::<u8>(lhs.try_into().unwrap(), rhs.try_into().unwrap(), op)
        }
        CoreTypeConcrete::Uint16(_) => {
            wrapping::<u16>(lhs.try_into().unwrap(), rhs.try_into().unwrap(), op)
        }
        CoreTypeConcrete::Uint32(_) => {
            wrapping::<u32>(lhs.try_into().unwrap(), rhs.try_into().unwrap(), op)
        }
        CoreTypeConcrete::Uint64(_) => {
            wrapping::<u64>(lhs.try_into().unwrap(), rhs.try_into().unwrap(), op)
        }
        CoreTypeConcrete::Uint128(_) => {
            wrapping::<u128>(lhs.try_into().unwrap(), rhs.try_into().unwrap(), op)
        }
        _ => panic!("Found a non-numeric type"),
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
        // SintConcrete::Operation(info) => eval_operation(registry, info, args),
        SintConcrete::ToFelt252(info) => eval_to_felt(registry, info, args),
        SintConcrete::WideMul(info) => eval_widemul(registry, info, args),
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
        // UintConcrete::Operation(info) => eval_operation(registry, info, args),
        UintConcrete::IsZero(info) => eval_is_zero(registry, info, args),
        UintConcrete::SquareRoot(info) => eval_square_root(registry, info, args),
        UintConcrete::ToFelt252(info) => eval_to_felt(registry, info, args),
        UintConcrete::WideMul(info) => eval_widemul(registry, info, args),
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

    EvalAction::NormalBranch(0, smallvec![get_int_value_from_type(int_ty, info.c.into())])
}

fn eval_bitwise(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [lhs, rhs]: [BigInt; 2] = get_numberic_args_as_bigints(args).try_into().unwrap();

    let int_ty = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)
        .unwrap();

    let and = get_int_value_from_type(int_ty, &lhs & &rhs);
    let or = get_int_value_from_type(int_ty, &lhs | &rhs);
    let xor = get_int_value_from_type(int_ty, &lhs ^ &rhs);

    EvalAction::NormalBranch(
        0,
        smallvec![
            Value::Unit, // bitwise
            and,
            xor,
            or
        ],
    )
}

fn eval_diff(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [lhs, rhs]: [BigInt; 2] = get_numberic_args_as_bigints(args).try_into().unwrap();

    let int_ty = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)
        .unwrap();

    let overflow = (&lhs >= &rhs) as usize;
    let (res, _) = apply_wrapping_op(int_ty, lhs, rhs, IntOperator::OverflowingSub);
    let res = get_int_value_from_type(int_ty, res);

    EvalAction::NormalBranch(
        overflow,
        smallvec![
            Value::Unit, // range_check
            res
        ],
    )
}

fn eval_divmod(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [lhs, rhs]: [BigInt; 2] = get_numberic_args_as_bigints(args).try_into().unwrap();

    let int_ty = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)
        .unwrap();

    let res = &lhs / &rhs;
    let rem = lhs % rhs;

    let res = get_int_value_from_type(int_ty, res);
    let rem = get_int_value_from_type(int_ty, rem);

    EvalAction::NormalBranch(
        0,
        smallvec![
            Value::Unit, // range_check
            res,
            rem
        ],
    )
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
    let [Value::Felt(val)]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    let bigint = val.to_bigint();

    let int_ty = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)
        .unwrap();

    EvalAction::NormalBranch(0, smallvec![get_int_value_from_type(int_ty, bigint)])
}

fn eval_is_zero(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [lhs, rhs]: [BigInt; 2] = get_numberic_args_as_bigints(args).try_into().unwrap();

    let int_ty = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)
        .unwrap();

    let and = get_int_value_from_type(int_ty, &lhs & &rhs);
    let or = get_int_value_from_type(int_ty, &lhs | &rhs);
    let xor = get_int_value_from_type(int_ty, &lhs ^ &rhs);

    EvalAction::NormalBranch(
        0,
        smallvec![
            Value::Unit, // bitwise
            and,
            xor,
            or
        ],
    )
}

fn eval_square_root(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value]: [BigInt; 1] = get_numberic_args_as_bigints(args).try_into().unwrap();

    let int_ty = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)
        .unwrap();

    let res = value.sqrt();

    EvalAction::NormalBranch(
        0,
        smallvec![
            Value::Unit, // range_check
            get_int_value_from_type(int_ty, res)
        ],
    )
}

fn eval_to_felt(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [val]: [BigInt; 1] = get_numberic_args_as_bigints(args).try_into().unwrap();

    EvalAction::NormalBranch(0, smallvec![Value::Felt(Felt::from(val))])
}

pub fn eval_widemul(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [lhs, rhs]: [BigInt; 2] = get_numberic_args_as_bigints(args).try_into().unwrap();

    let int_ty = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)
        .unwrap();

    let res = lhs * rhs;

    EvalAction::NormalBranch(0, smallvec![get_int_value_from_type(int_ty, res)])
}
