use super::EvalAction;
use crate::{
    utils::{get_numberic_args_as_bigints, get_value_from_integer},
    Value,
};
use cairo_lang_sierra::{
    extensions::{
        casts::{CastConcreteLibfunc, DowncastConcreteLibfunc},
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use smallvec::smallvec;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &CastConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        CastConcreteLibfunc::Downcast(info) => eval_downcast(registry, info, args),
        CastConcreteLibfunc::Upcast(info) => eval_upcast(registry, info, args),
    }
}

fn eval_downcast(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &DowncastConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let range_check @ Value::Unit: Value = args[0].clone() else {
        panic!()
    };
    let [value] = get_numberic_args_as_bigints(&args[1..].to_vec())
        .try_into()
        .unwrap();

    let int_ty = registry.get_type(&info.to_ty).unwrap();
    let range = info.to_range.lower.clone()..info.to_range.upper.clone();
    if range.contains(&value) {
        EvalAction::NormalBranch(
            0,
            smallvec![
                range_check, // range_check
                get_value_from_integer(registry, int_ty, value)
            ],
        )
    } else {
        EvalAction::NormalBranch(1, smallvec![Value::Unit])
    }
}

fn eval_upcast(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value] = get_numberic_args_as_bigints(&args)
        .try_into()
        .unwrap();
    let int_ty = registry
        .get_type(&info.branch_signatures()[0].vars[0].ty)
        .unwrap();

    EvalAction::NormalBranch(
        0,
        smallvec![get_value_from_integer(registry, int_ty, value)],
    )
}
