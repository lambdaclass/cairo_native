use super::EvalAction;
use crate::Value;
use cairo_lang_sierra::{
    extensions::{
        boxing::BoxConcreteLibfunc,
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureAndTypeConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use smallvec::smallvec;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &BoxConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        BoxConcreteLibfunc::Into(info) => eval_into_box(registry, info, args),
        BoxConcreteLibfunc::LocalInto(_info) => todo!(),
        BoxConcreteLibfunc::Unbox(info) => eval_unbox(registry, info, args),
        BoxConcreteLibfunc::ForwardSnapshot(info) => eval_forward_snapshot(registry, info, args),
    }
}

pub fn eval_unbox(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value] = args.try_into().unwrap();

    EvalAction::NormalBranch(0, smallvec![value])
}

pub fn eval_into_box(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value] = args.try_into().unwrap();

    EvalAction::NormalBranch(0, smallvec![value])
}

pub fn eval_forward_snapshot(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value] = args.try_into().unwrap();

    EvalAction::NormalBranch(0, smallvec![value])
}
