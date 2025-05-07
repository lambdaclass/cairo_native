use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::{SignatureAndTypeConcreteLibfunc, SignatureOnlyConcreteLibfunc},
        nullable::NullableConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use smallvec::smallvec;

use crate::Value;

use super::EvalAction;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &NullableConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        NullableConcreteLibfunc::Null(info) => eval_null(registry, info, args),
        NullableConcreteLibfunc::NullableFromBox(info) => {
            eval_nullable_from_box(registry, info, args)
        }
        NullableConcreteLibfunc::MatchNullable(info) => eval_match_nullable(registry, info, args),
        NullableConcreteLibfunc::ForwardSnapshot(info) => {
            eval_forward_snapshot(registry, info, args)
        }
    }
}

fn eval_null(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [] = args.try_into().unwrap();

    EvalAction::NormalBranch(0, smallvec![Value::Null])
}

fn eval_nullable_from_box(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value]: [Value; 1] = args.try_into().unwrap();

    EvalAction::NormalBranch(0, smallvec![value])
}

fn eval_match_nullable(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value]: [Value; 1] = args.try_into().unwrap();

    if matches!(value, Value::Null) {
        EvalAction::NormalBranch(0, smallvec![])
    } else {
        EvalAction::NormalBranch(1, smallvec![value])
    }
}

fn eval_forward_snapshot(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value]: [Value; 1] = args.try_into().unwrap();

    EvalAction::NormalBranch(0, smallvec![value])
}
