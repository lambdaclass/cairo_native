use super::EvalAction;
use crate::Value;
use cairo_lang_sierra::{
    extensions::{
        boolean::BoolConcreteLibfunc,
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use smallvec::smallvec;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &BoolConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        BoolConcreteLibfunc::And(info) => eval_and(registry, info, args),
        BoolConcreteLibfunc::Not(info) => eval_not(registry, info, args),
        BoolConcreteLibfunc::Xor(info) => eval_xor(registry, info, args),
        BoolConcreteLibfunc::Or(info) => eval_or(registry, info, args),
        BoolConcreteLibfunc::ToFelt252(info) => eval_to_felt252(registry, info, args),
    }
}

pub fn eval_and(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Enum {
        self_ty,
        index: lhs,
        payload,
        ..
    }, Value::Enum {
        self_ty: _,
        index: rhs,
        payload: _,
        ..
    }]: [Value; 2] = args.try_into().unwrap()
    else {
        panic!()
    };

    let lhs = lhs != 0;
    let rhs = rhs != 0;

    EvalAction::NormalBranch(
        0,
        smallvec![Value::Enum {
            self_ty: self_ty.clone(),
            index: (lhs && rhs) as usize,
            payload,
        }],
    )
}

pub fn eval_not(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Enum {
        self_ty,
        index: lhs,
        payload,
        ..
    }]: [Value; 1] = args.try_into().unwrap()
    else {
        panic!()
    };

    EvalAction::NormalBranch(
        0,
        smallvec![Value::Enum {
            self_ty: self_ty.clone(),
            index: (lhs == 0) as usize,
            payload,
        }],
    )
}

pub fn eval_xor(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Enum {
        self_ty,
        index: lhs,
        payload,
        ..
    }, Value::Enum {
        self_ty: _,
        index: rhs,
        payload: _,
        ..
    }]: [Value; 2] = args.try_into().unwrap()
    else {
        panic!()
    };

    let lhs = lhs != 0;
    let rhs = rhs != 0;

    EvalAction::NormalBranch(
        0,
        smallvec![Value::Enum {
            self_ty: self_ty.clone(),
            index: (lhs ^ rhs) as usize,
            payload,
        }],
    )
}

pub fn eval_or(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Enum {
        self_ty,
        index: lhs,
        payload,
        ..
    }, Value::Enum {
        self_ty: _,
        index: rhs,
        payload: _,
        ..
    }]: [Value; 2] = args.try_into().unwrap()
    else {
        panic!()
    };

    let lhs = lhs != 0;
    let rhs = rhs != 0;

    EvalAction::NormalBranch(
        0,
        smallvec![Value::Enum {
            self_ty: self_ty.clone(),
            index: (lhs || rhs) as usize,
            payload,
        }],
    )
}

pub fn eval_to_felt252(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Enum {
        self_ty: _,
        index: lhs,
        payload: _,
        ..
    }]: [Value; 1] = args.try_into().unwrap()
    else {
        panic!()
    };

    EvalAction::NormalBranch(0, smallvec![Value::Felt(lhs.into())])
}
