use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        range::IntRangeConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use num_bigint::BigInt;
use smallvec::smallvec;

use crate::{
    utils::{get_numberic_args_as_bigints, get_value_from_integer},
    Value,
};

use super::EvalAction;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &IntRangeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        IntRangeConcreteLibfunc::TryNew(info) => eval_try_new(registry, info, args),
        IntRangeConcreteLibfunc::PopFront(info) => eval_pop_front(registry, info, args),
    }
}

fn eval_try_new(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [x, y]: [BigInt; 2] = get_numberic_args_as_bigints(args).try_into().unwrap();

    let int_ty = registry.get_type(&info.param_signatures()[1].ty).unwrap();

    // if x >= y then the range is not valid and we return [y, y) (empty range)
    let range = if x < y {
        let x = get_value_from_integer(registry, int_ty, x);
        let y = get_value_from_integer(registry, int_ty, y);

        Value::IntRange {
            x: Box::new(x),
            y: Box::new(y),
        }
    } else {
        let y = get_value_from_integer(registry, int_ty, y);

        Value::IntRange {
            x: Box::new(y.clone()),
            y: Box::new(y),
        }
    };

    EvalAction::NormalBranch(
        1,
        smallvec![
            Value::Unit, //range_check
            range
        ],
    )
}

fn eval_pop_front(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::IntRange { x, y }]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };
    let [x, y]: [BigInt; 2] = get_numberic_args_as_bigints(vec![*x, *y])
        .try_into()
        .unwrap();
    let int_ty = registry.get_type(&info.param_signatures()[1].ty).unwrap();

    if x < y {
        let x_plus_1 = get_value_from_integer(registry, int_ty, &x + 1);
        let x = get_value_from_integer(registry, int_ty, x);
        let y = get_value_from_integer(registry, int_ty, y);

        EvalAction::NormalBranch(
            0,
            smallvec![
                Value::IntRange {
                    x: Box::new(x_plus_1),
                    y: Box::new(y)
                },
                x
            ],
        )
    } else {
        EvalAction::NormalBranch(1, smallvec![])
    }
}
