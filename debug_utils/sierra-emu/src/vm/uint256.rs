use super::EvalAction;
use crate::Value;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        int::unsigned256::Uint256Concrete,
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use num_bigint::BigUint;
use smallvec::smallvec;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &Uint256Concrete,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        Uint256Concrete::IsZero(info) => eval_is_zero(registry, info, args),
        Uint256Concrete::Divmod(info) => eval_divmod(registry, info, args),
        Uint256Concrete::SquareRoot(info) => eval_square_root(registry, info, args),
        Uint256Concrete::InvModN(info) => eval_inv_mod_n(registry, info, args),
    }
}

fn eval_inv_mod_n(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [range_check @ Value::Unit, Value::Struct(x), Value::Struct(modulo)]: [Value; 3] =
        args.try_into().unwrap()
    else {
        panic!()
    };

    let [Value::U128(x_lo), Value::U128(x_hi)]: [Value; 2] = x.clone().try_into().unwrap() else {
        panic!()
    };

    let [Value::U128(mod_lo), Value::U128(mod_hi)]: [Value; 2] = modulo.clone().try_into().unwrap()
    else {
        panic!()
    };

    let x = u256_to_biguint(x_lo, x_hi);
    let modulo = u256_to_biguint(mod_lo, mod_hi);

    match x.modinv(&modulo) {
        None => EvalAction::NormalBranch(1, smallvec![range_check, Value::Unit, Value::Unit]),
        Some(r) if r == 0u8.into() => {
            EvalAction::NormalBranch(1, smallvec![range_check, Value::Unit, Value::Unit])
        }
        Some(r) => EvalAction::NormalBranch(
            0,
            smallvec![
                range_check,
                u256_to_value(r),
                Value::Unit,
                Value::Unit,
                Value::Unit,
                Value::Unit,
                Value::Unit,
                Value::Unit,
                Value::Unit,
                Value::Unit
            ],
        ),
    }
}

pub fn eval_is_zero(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Struct(fields)]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    let [Value::U128(lo), Value::U128(hi)]: [Value; 2] = fields.clone().try_into().unwrap() else {
        panic!()
    };

    if lo == 0 && hi == 0 {
        EvalAction::NormalBranch(0, smallvec![])
    } else {
        EvalAction::NormalBranch(1, smallvec![Value::Struct(fields)])
    }
}

#[inline]
pub fn u256_to_biguint(lo: u128, hi: u128) -> BigUint {
    BigUint::from(lo) + (BigUint::from(hi) << 128)
}

#[inline]
pub fn u256_to_value(value: BigUint) -> Value {
    let hi: u128 = (&value >> 128u32).try_into().unwrap();
    let lo: u128 = (value & BigUint::from(u128::MAX)).try_into().unwrap();
    Value::Struct(vec![Value::U128(lo), Value::U128(hi)])
}

#[inline]
pub fn u516_to_value(value: BigUint) -> Value {
    let upper_u256: BigUint = &value >> 256u32;
    let hi1: u128 = (&upper_u256 >> 128u32).try_into().unwrap();
    let lo1: u128 = (upper_u256 & BigUint::from(u128::MAX)).try_into().unwrap();
    let lower_mask = BigUint::from_bytes_le(&[0xFF; 32]);
    let lower_u256: BigUint = value & lower_mask;
    let hi: u128 = (&lower_u256 >> 128u32).try_into().unwrap();
    let lo: u128 = (lower_u256 & BigUint::from(u128::MAX)).try_into().unwrap();
    Value::Struct(vec![
        Value::U128(lo),
        Value::U128(hi),
        Value::U128(lo1),
        Value::U128(hi1),
    ])
}

pub fn eval_divmod(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [mut range_check @ Value::RangeCheck(_), Value::Struct(lhs), Value::Struct(rhs)]: [Value;
        3] = args.try_into().unwrap()
    else {
        panic!()
    };

    // Increment builtin counter
    range_check = match range_check {
        Value::RangeCheck(n) => Value::RangeCheck(n + 1),
        _ => panic!(),
    };

    let [Value::U128(lhs_lo), Value::U128(lhs_hi)]: [Value; 2] = lhs.try_into().unwrap() else {
        panic!()
    };

    let lhs = u256_to_biguint(lhs_lo, lhs_hi);

    let [Value::U128(rhs_lo), Value::U128(rhs_hi)]: [Value; 2] = rhs.try_into().unwrap() else {
        panic!()
    };

    let rhs = u256_to_biguint(rhs_lo, rhs_hi);

    let div = &lhs / &rhs;
    let modulo = lhs % rhs;

    EvalAction::NormalBranch(
        0,
        smallvec![
            range_check,
            u256_to_value(div),
            u256_to_value(modulo),
            Value::Unit
        ],
    )
}

pub fn eval_square_root(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [range_check @ Value::Unit, Value::Struct(lhs)]: [Value; 2] = args.try_into().unwrap()
    else {
        panic!()
    };

    let [Value::U128(lhs_lo), Value::U128(lhs_hi)]: [Value; 2] = lhs.try_into().unwrap() else {
        panic!()
    };

    let lhs = u256_to_biguint(lhs_lo, lhs_hi);
    let sqrt = lhs.sqrt();

    let sqrt_lo: u128 = sqrt.clone().try_into().unwrap();

    EvalAction::NormalBranch(0, smallvec![range_check, Value::U128(sqrt_lo)])
}
