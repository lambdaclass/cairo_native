use super::EvalAction;
use crate::Value;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        enm::{
            EnumConcreteLibfunc, EnumConcreteType, EnumFromBoundedIntConcreteLibfunc,
            EnumInitConcreteLibfunc,
        },
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use smallvec::smallvec;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &EnumConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        EnumConcreteLibfunc::Init(info) => eval_init(registry, info, args),
        EnumConcreteLibfunc::FromBoundedInt(info) => eval_from_bounded_int(registry, info, args),
        EnumConcreteLibfunc::Match(info) => eval_match(registry, info, args),
        EnumConcreteLibfunc::SnapshotMatch(info) => eval_snapshot_match(registry, info, args),
    }
}

pub fn eval_init(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &EnumInitConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value] = args.try_into().unwrap();

    let self_ty = &info.signature.branch_signatures[0].vars[0].ty;
    let CoreTypeConcrete::Enum(EnumConcreteType { variants, .. }) =
        registry.get_type(self_ty).unwrap()
    else {
        panic!()
    };
    assert_eq!(info.n_variants, variants.len());
    assert!(info.index < info.n_variants);
    assert!(value.is(registry, &variants[info.index]));

    EvalAction::NormalBranch(
        0,
        smallvec![Value::Enum {
            self_ty: self_ty.clone(),
            index: info.index,
            payload: Box::new(value),
            debug_name: self_ty.debug_name.as_ref().map(|n| n.to_string())
        }],
    )
}

pub fn eval_from_bounded_int(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &EnumFromBoundedIntConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::BoundedInt { range: _, value }]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };
    let self_ty = &info.branch_signatures()[0].vars[0].ty;

    let enm = Value::Enum {
        self_ty: self_ty.clone(),
        index: value.try_into().unwrap(),
        payload: Box::new(Value::Struct(vec![])),
        debug_name: self_ty.debug_name.as_ref().map(|n| n.to_string()),
    };

    EvalAction::NormalBranch(0, smallvec![enm])
}

pub fn eval_match(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Enum {
        self_ty,
        index,
        payload,
        ..
    }]: [Value; 1] = args.try_into().unwrap()
    else {
        panic!()
    };
    assert_eq!(self_ty, info.signature.param_signatures[0].ty);
    assert!(payload.is(
        registry,
        &info.signature.branch_signatures[index].vars[0].ty
    ));

    EvalAction::NormalBranch(index, smallvec![*payload])
}

pub fn eval_snapshot_match(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Enum {
        self_ty,
        index,
        payload,
        ..
    }]: [Value; 1] = args.try_into().unwrap()
    else {
        panic!()
    };

    let ty = registry
        .get_type(&info.signature.param_signatures[0].ty)
        .unwrap();

    if let CoreTypeConcrete::Snapshot(inner) = ty {
        assert_eq!(inner.ty, self_ty);
    } else {
        panic!("expected snapshot type")
    }

    assert!(payload.is(
        registry,
        &info.signature.branch_signatures[index].vars[0].ty
    ));

    EvalAction::NormalBranch(index, smallvec![*payload])
}
