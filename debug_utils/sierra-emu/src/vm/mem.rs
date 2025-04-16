use super::EvalAction;
use crate::Value;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::{SignatureAndTypeConcreteLibfunc, SignatureOnlyConcreteLibfunc},
        mem::MemConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use smallvec::smallvec;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &MemConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        MemConcreteLibfunc::StoreTemp(info) => eval_store_temp(registry, info, args),
        MemConcreteLibfunc::StoreLocal(info) => eval_store_local(registry, info, args),
        MemConcreteLibfunc::FinalizeLocals(info) => eval_finalize_locals(registry, info, args),
        MemConcreteLibfunc::AllocLocal(info) => eval_alloc_local(registry, info, args),
        MemConcreteLibfunc::Rename(info) => eval_rename(registry, info, args),
    }
}

pub fn eval_store_temp(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value] = args.try_into().unwrap();

    EvalAction::NormalBranch(0, smallvec![value])
}

pub fn eval_store_local(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Uninitialized { ty }, value]: [Value; 2] = args.try_into().unwrap() else {
        panic!()
    };
    assert!(value.is(registry, &ty));

    EvalAction::NormalBranch(0, smallvec![value])
}

pub fn eval_finalize_locals(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [] = args.try_into().unwrap();

    EvalAction::NormalBranch(0, smallvec![])
}

pub fn eval_alloc_local(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [] = args.try_into().unwrap();

    EvalAction::NormalBranch(
        0,
        smallvec![Value::Uninitialized {
            ty: info.ty.clone()
        }],
    )
}

pub fn eval_rename(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value] = args.try_into().unwrap();

    EvalAction::NormalBranch(0, smallvec![value])
}
