use super::EvalAction;
use crate::Value;
use cairo_lang_sierra::{
    extensions::{
        ap_tracking::ApTrackingConcreteLibfunc,
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use smallvec::smallvec;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &ApTrackingConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        ApTrackingConcreteLibfunc::Revoke(info) => eval_revoke(registry, info, args),
        ApTrackingConcreteLibfunc::Enable(info) => eval_enable(registry, info, args),
        ApTrackingConcreteLibfunc::Disable(info) => eval_disable(registry, info, args),
    }
}

pub fn eval_revoke(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [] = args.try_into().unwrap();

    EvalAction::NormalBranch(0, smallvec![])
}

pub fn eval_enable(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [] = args.try_into().unwrap();

    EvalAction::NormalBranch(0, smallvec![])
}

pub fn eval_disable(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [] = args.try_into().unwrap();

    EvalAction::NormalBranch(0, smallvec![])
}
