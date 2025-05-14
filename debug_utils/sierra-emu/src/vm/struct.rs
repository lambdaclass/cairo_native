use super::EvalAction;
use crate::Value;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        lib_func::SignatureOnlyConcreteLibfunc,
        structure::{StructConcreteLibfunc, StructConcreteType},
    },
    program_registry::ProgramRegistry,
};
use smallvec::smallvec;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &StructConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        StructConcreteLibfunc::Construct(info) => eval_construct(registry, info, args),
        StructConcreteLibfunc::Deconstruct(info) => eval_deconstruct(registry, info, args),
        StructConcreteLibfunc::SnapshotDeconstruct(info) => {
            eval_snapshot_deconstruct(registry, info, args)
        }
    }
}

pub fn eval_construct(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let CoreTypeConcrete::Struct(StructConcreteType { members, .. }) = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(args.len(), members.len());
    assert!(args
        .iter()
        .zip(members)
        .all(|(value, ty)| value.is(registry, ty)));

    EvalAction::NormalBranch(0, smallvec![Value::Struct(args)])
}

pub fn eval_deconstruct(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Struct(values)]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    let CoreTypeConcrete::Struct(StructConcreteType { members, .. }) = registry
        .get_type(&info.signature.param_signatures[0].ty)
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(values.len(), members.len());
    assert!(values
        .iter()
        .zip(members)
        .all(|(value, ty)| value.is(registry, ty)));

    EvalAction::NormalBranch(0, values.into())
}

pub fn eval_snapshot_deconstruct(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Struct(values)]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    let CoreTypeConcrete::Snapshot(snapshot_ty) = registry
        .get_type(&info.signature.param_signatures[0].ty)
        .unwrap()
    else {
        panic!()
    };

    let CoreTypeConcrete::Struct(StructConcreteType { members, .. }) =
        registry.get_type(&snapshot_ty.ty).unwrap()
    else {
        panic!()
    };
    assert_eq!(values.len(), members.len());
    assert!(values
        .iter()
        .zip(members)
        .all(|(value, ty)| value.is(registry, ty)));

    EvalAction::NormalBranch(0, values.into())
}
