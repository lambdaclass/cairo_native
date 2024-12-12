use super::EvalAction;
use crate::{
    utils::{get_numeric_args_as_bigints, get_value_from_integer},
    Value,
};
use cairo_lang_sierra::{
    extensions::{
        bounded_int::{
            BoundedIntConcreteLibfunc, BoundedIntConstrainConcreteLibfunc,
            BoundedIntDivRemConcreteLibfunc, BoundedIntTrimConcreteLibfunc,
        },
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use num_bigint::BigInt;
use smallvec::smallvec;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &BoundedIntConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        BoundedIntConcreteLibfunc::Add(info) => eval_add(registry, info, args),
        BoundedIntConcreteLibfunc::Sub(info) => eval_sub(registry, info, args),
        BoundedIntConcreteLibfunc::Mul(info) => eval_mul(registry, info, args),
        BoundedIntConcreteLibfunc::DivRem(info) => eval_div_rem(registry, info, args),
        BoundedIntConcreteLibfunc::Constrain(info) => eval_constrain(registry, info, args),
        BoundedIntConcreteLibfunc::IsZero(info) => eval_is_zero(registry, info, args),
        BoundedIntConcreteLibfunc::WrapNonZero(info) => eval_wrap_non_zero(registry, info, args),
        BoundedIntConcreteLibfunc::TrimMin(info) | BoundedIntConcreteLibfunc::TrimMax(info) => {
            eval_trim(registry, info, args)
        }
    }
}

pub fn eval_add(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [lhs, rhs]: [BigInt; 2] = get_numeric_args_as_bigints(&args).try_into().unwrap();

    let range = match registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)
        .unwrap()
    {
        CoreTypeConcrete::BoundedInt(info) => info.range.lower.clone()..info.range.upper.clone(),
        CoreTypeConcrete::NonZero(info) => match registry.get_type(&info.ty).unwrap() {
            CoreTypeConcrete::BoundedInt(info) => {
                info.range.lower.clone()..info.range.upper.clone()
            }
            _ => unreachable!(),
        },
        _ => unreachable!(),
    };
    EvalAction::NormalBranch(
        0,
        smallvec![Value::BoundedInt {
            range,
            value: lhs + rhs,
        }],
    )
}

pub fn eval_sub(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [lhs, rhs]: [BigInt; 2] = get_numeric_args_as_bigints(&args).try_into().unwrap();

    let range = match registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)
        .unwrap()
    {
        CoreTypeConcrete::BoundedInt(info) => info.range.lower.clone()..info.range.upper.clone(),
        CoreTypeConcrete::NonZero(info) => match registry.get_type(&info.ty).unwrap() {
            CoreTypeConcrete::BoundedInt(info) => {
                info.range.lower.clone()..info.range.upper.clone()
            }
            _ => unreachable!(),
        },
        _ => unreachable!(),
    };
    EvalAction::NormalBranch(
        0,
        smallvec![Value::BoundedInt {
            range,
            value: lhs - rhs,
        }],
    )
}

pub fn eval_mul(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [lhs, rhs]: [BigInt; 2] = get_numeric_args_as_bigints(&args).try_into().unwrap();

    let range = match registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)
        .unwrap()
    {
        CoreTypeConcrete::BoundedInt(info) => info.range.lower.clone()..info.range.upper.clone(),
        CoreTypeConcrete::NonZero(info) => match registry.get_type(&info.ty).unwrap() {
            CoreTypeConcrete::BoundedInt(info) => {
                info.range.lower.clone()..info.range.upper.clone()
            }
            _ => unreachable!(),
        },
        _ => unreachable!(),
    };
    EvalAction::NormalBranch(
        0,
        smallvec![Value::BoundedInt {
            range,
            value: lhs * rhs,
        }],
    )
}

pub fn eval_div_rem(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &BoundedIntDivRemConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let range_check @ Value::Unit: Value = args[0].clone() else {
        panic!()
    };
    let [lhs, rhs]: [BigInt; 2] = get_numeric_args_as_bigints(&args[1..]).try_into().unwrap();
    let quo = &lhs / &rhs;
    let rem = lhs % rhs;

    let quo_range = match registry
        .get_type(&info.branch_signatures()[0].vars[1].ty)
        .unwrap()
    {
        CoreTypeConcrete::BoundedInt(info) => info.range.lower.clone()..info.range.upper.clone(),
        _ => unreachable!(),
    };
    let rem_range = match registry
        .get_type(&info.branch_signatures()[0].vars[2].ty)
        .unwrap()
    {
        CoreTypeConcrete::BoundedInt(info) => info.range.lower.clone()..info.range.upper.clone(),
        _ => unreachable!(),
    };
    assert!(quo_range.contains(&quo));
    assert!(rem_range.contains(&rem));

    EvalAction::NormalBranch(
        0,
        smallvec![
            range_check,
            Value::BoundedInt {
                range: quo_range,
                value: quo,
            },
            Value::BoundedInt {
                range: rem_range,
                value: rem,
            },
        ],
    )
}

pub fn eval_constrain(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &BoundedIntConstrainConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let range_check @ Value::Unit: Value = args[0].clone() else {
        panic!()
    };
    let [value]: [BigInt; 1] = get_numeric_args_as_bigints(&args[1..]).try_into().unwrap();

    if value < info.boundary {
        let range = match registry
            .get_type(&info.branch_signatures()[0].vars[1].ty)
            .unwrap()
        {
            CoreTypeConcrete::BoundedInt(info) => {
                info.range.lower.clone()..info.range.upper.clone()
            }
            CoreTypeConcrete::NonZero(info) => match registry.get_type(&info.ty).unwrap() {
                CoreTypeConcrete::BoundedInt(info) => {
                    info.range.lower.clone()..info.range.upper.clone()
                }
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };

        EvalAction::NormalBranch(
            0,
            smallvec![range_check, Value::BoundedInt { range, value }],
        )
    } else {
        let range = match registry
            .get_type(&info.branch_signatures()[1].vars[1].ty)
            .unwrap()
        {
            CoreTypeConcrete::BoundedInt(info) => {
                info.range.lower.clone()..info.range.upper.clone()
            }
            CoreTypeConcrete::NonZero(info) => match registry.get_type(&info.ty).unwrap() {
                CoreTypeConcrete::BoundedInt(info) => {
                    info.range.lower.clone()..info.range.upper.clone()
                }
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };

        EvalAction::NormalBranch(
            1,
            smallvec![range_check, Value::BoundedInt { range, value }],
        )
    }
}

pub fn eval_is_zero(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value] = get_numeric_args_as_bigints(&args).try_into().unwrap();
    let is_zero = value == 0.into();

    let int_ty = &info.branch_signatures()[1].vars[0].ty;

    if is_zero {
        EvalAction::NormalBranch(0, smallvec![])
    } else {
        let value = get_value_from_integer(registry, int_ty, value);
        EvalAction::NormalBranch(1, smallvec![value])
    }
}

pub fn eval_wrap_non_zero(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value] = args.try_into().unwrap();

    EvalAction::NormalBranch(0, smallvec![value])
}

pub fn eval_trim(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &BoundedIntTrimConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value] = args.try_into().unwrap();
    let value = match value {
        Value::I8(v) => BigInt::from(v),
        Value::I16(v) => BigInt::from(v),
        Value::I32(v) => BigInt::from(v),
        Value::I64(v) => BigInt::from(v),
        Value::I128(v) => BigInt::from(v),
        Value::U8(v) => BigInt::from(v),
        Value::U16(v) => BigInt::from(v),
        Value::U32(v) => BigInt::from(v),
        Value::U64(v) => BigInt::from(v),
        Value::U128(v) => BigInt::from(v),
        _ => panic!("Not a valid integer type"),
    };
    let is_invalid = value == info.trimmed_value;
    let int_range = match registry
        .get_type(&info.branch_signatures()[1].vars[0].ty)
        .unwrap()
    {
        CoreTypeConcrete::BoundedInt(info) => info.range.clone(),
        _ => panic!("should be bounded int"),
    };

    if !is_invalid {
        let range = int_range.lower.clone()..int_range.upper.clone();
        EvalAction::NormalBranch(1, smallvec![Value::BoundedInt { range, value }])
    } else {
        EvalAction::NormalBranch(0, smallvec![])
    }
}

#[cfg(test)]
mod tests {

    use num_bigint::BigInt;

    use super::Value;

    use crate::{load_cairo, test_utils::run_test_program};

    #[test]
    fn test_bounded_int_sub() {
        let (_, program) = load_cairo!(
            #[feature("bounded-int-utils")]
            use core::internal::bounded_int::{self, SubHelper, BoundedInt};

            impl U8BISub of SubHelper<u8, u8> {
                type Result = BoundedInt<-255, 255>;
            }

            extern fn bounded_int_sub<Lhs, Rhs, impl H: SubHelper<Lhs, Rhs>>(
                lhs: Lhs, rhs: Rhs,
            ) -> H::Result nopanic;

            fn main() -> BoundedInt<-255, 255> {
                bounded_int_sub(0_u8, 255_u8)
            }
        );

        run_test_program(program);
    }

    #[test]
    fn test_trim_i8() {
        let (_, program) = load_cairo!(
            #[feature("bounded-int-utils")]
            use core::internal::bounded_int::{self, BoundedInt};
            use core::internal::OptionRev;

            fn main() -> BoundedInt<-127, 127> {
                let num = match bounded_int::trim_min::<i8>(1) {
                    OptionRev::Some(n) => n,
                    OptionRev::None => 1,
                };

                num
            }
        );

        let result = run_test_program(program);
        let result = result.last().unwrap();
        let expected = Value::BoundedInt {
            range: BigInt::from(-127)..BigInt::from(128),
            value: BigInt::from(1u8),
        };

        assert_eq!(*result, expected);
    }

    #[test]
    fn test_trim_u32() {
        let (_, program) = load_cairo!(
            #[feature("bounded-int-utils")]
            use core::internal::bounded_int::{self, BoundedInt};
            use core::internal::OptionRev;

            fn main() -> BoundedInt<0, 4294967294> {
                let num = match bounded_int::trim_max::<u32>(0xfffffffe) {
                    OptionRev::Some(n) => n,
                    OptionRev::None => 0,
                };

                num
            }
        );

        let result = run_test_program(program);
        let result = result.last().unwrap();
        let expected = Value::BoundedInt {
            range: BigInt::from(0)..BigInt::from(4294967295u32),
            value: BigInt::from(0xfffffffeu32),
        };

        assert_eq!(*result, expected);
    }

    #[test]
    fn test_trim_none() {
        let (_, program) = load_cairo!(
            #[feature("bounded-int-utils")]
            use core::internal::bounded_int::{self, BoundedInt};
            use core::internal::OptionRev;

            fn main() -> BoundedInt<-32767, 32767> {
                let num = match bounded_int::trim_min::<i16>(-0x8000) {
                    OptionRev::Some(n) => n,
                    OptionRev::None => 0,
                };

                num
            }
        );

        let result = run_test_program(program);
        let result = result.last().unwrap();
        let expected = Value::BoundedInt {
            range: BigInt::from(-32767)..BigInt::from(32768),
            value: BigInt::from(0),
        };

        assert_eq!(*result, expected);
    }
}
