use super::EvalAction;
use crate::{find_real_type, Value};
use cairo_lang_sierra::{
    extensions::{
        array::ArrayConcreteLibfunc,
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        lib_func::{SignatureAndTypeConcreteLibfunc, SignatureOnlyConcreteLibfunc},
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use smallvec::smallvec;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &ArrayConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        ArrayConcreteLibfunc::New(info) => eval_new(registry, info, args),
        ArrayConcreteLibfunc::SpanFromTuple(info) => eval_span_from_tuple(registry, info, args),
        ArrayConcreteLibfunc::TupleFromSpan(info) => eval_tuple_from_span(registry, info, args),
        ArrayConcreteLibfunc::Append(info) => eval_append(registry, info, args),
        ArrayConcreteLibfunc::PopFront(info) => eval_pop_front(registry, info, args),
        ArrayConcreteLibfunc::PopFrontConsume(info) => eval_pop_front_consume(registry, info, args),
        ArrayConcreteLibfunc::Get(info) => eval_get(registry, info, args),
        ArrayConcreteLibfunc::Slice(info) => eval_slice(registry, info, args),
        ArrayConcreteLibfunc::Len(info) => eval_len(registry, info, args),
        ArrayConcreteLibfunc::SnapshotPopFront(info) => {
            eval_snapshot_pop_front(registry, info, args)
        }
        ArrayConcreteLibfunc::SnapshotPopBack(info) => eval_snapshot_pop_back(registry, info, args),
        ArrayConcreteLibfunc::SnapshotMultiPopFront(info) => {
            eval_snapshot_multi_pop_front(registry, info, args)
        }
        ArrayConcreteLibfunc::SnapshotMultiPopBack(info) => {
            eval_snapshot_multi_pop_back(registry, info, args)
        }
    }
}

fn eval_span_from_tuple(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Struct(data)]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    let ty = &info.branch_signatures()[0].vars[0].ty;
    let ty = find_real_type(registry, ty);

    let CoreTypeConcrete::Array(info) = registry.get_type(&ty).unwrap() else {
        panic!()
    };

    let value = Value::Array {
        ty: info.ty.clone(),
        data,
    };

    EvalAction::NormalBranch(0, smallvec![value])
}

fn eval_tuple_from_span(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Array { data, .. }]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    let tuple_len = {
        let CoreTypeConcrete::Struct(param) = registry.get_type(&info.ty).unwrap() else {
            panic!()
        };

        param.members.len()
    };

    if data.len() == tuple_len {
        EvalAction::NormalBranch(0, smallvec![Value::Struct(data)])
    } else {
        EvalAction::NormalBranch(1, smallvec![])
    }
}

fn eval_new(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [] = args.try_into().unwrap();

    let type_info = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)
        .unwrap();
    let ty = match type_info {
        CoreTypeConcrete::Array(info) => &info.ty,
        _ => unreachable!(),
    };

    EvalAction::NormalBranch(
        0,
        smallvec![Value::Array {
            ty: ty.clone(),
            data: Vec::new(),
        }],
    )
}

fn eval_append(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Array { ty, mut data }, item]: [Value; 2] = args.try_into().unwrap() else {
        panic!()
    };

    assert_eq!(info.signature.param_signatures[1].ty, ty);
    assert!(item.is(registry, &ty));
    data.push(item.clone());

    EvalAction::NormalBranch(0, smallvec![Value::Array { ty, data }])
}

fn eval_get(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [mut range_check @ Value::RangeCheck(_), Value::Array { data, .. }, Value::U32(index)]: [Value; 3] =
        args.try_into().unwrap()
    else {
        panic!()
    };

    // Increment builtin counter
    range_check = match range_check {
        Value::RangeCheck(n) => Value::RangeCheck(n + 1),
        _ => panic!(),
    };

    match data.get(index as usize).cloned() {
        Some(value) => EvalAction::NormalBranch(0, smallvec![range_check, value]),
        None => EvalAction::NormalBranch(1, smallvec![range_check]),
    }
}

fn eval_slice(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [mut range_check @ Value::RangeCheck(_), Value::Array { data, ty }, Value::U32(start), Value::U32(len)]: [Value; 4] =
        args.try_into().unwrap()
    else {
        panic!()
    };

    // Increment builtin counter
    range_check = match range_check {
        Value::RangeCheck(n) => Value::RangeCheck(n + 1),
        _ => panic!(),
    };

    match data.get(start as usize..(start + len) as usize) {
        Some(value) => EvalAction::NormalBranch(
            0,
            smallvec![
                range_check,
                Value::Array {
                    data: value.to_vec(),
                    ty
                }
            ],
        ),
        None => EvalAction::NormalBranch(1, smallvec![range_check]),
    }
}

fn eval_len(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Array { data, .. }]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    let array_len = data.len().try_into().unwrap();
    EvalAction::NormalBranch(0, smallvec![Value::U32(array_len)])
}

fn eval_pop_front(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Array { mut data, ty }]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    if !data.is_empty() {
        let new_data = data.split_off(1);
        let value = data[0].clone();
        EvalAction::NormalBranch(0, smallvec![Value::Array { data: new_data, ty }, value])
    } else {
        EvalAction::NormalBranch(1, smallvec![Value::Array { data, ty }])
    }
}

fn eval_pop_front_consume(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Array { mut data, ty }]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    if !data.is_empty() {
        let new_data = data.split_off(1);
        let value = data[0].clone();
        EvalAction::NormalBranch(0, smallvec![Value::Array { data: new_data, ty }, value])
    } else {
        EvalAction::NormalBranch(1, smallvec![])
    }
}

fn eval_snapshot_pop_front(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Array { mut data, ty }]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    if !data.is_empty() {
        let new_data = data.split_off(1);
        let value = data[0].clone();
        assert!(value.is(registry, &info.ty));
        EvalAction::NormalBranch(0, smallvec![Value::Array { data: new_data, ty }, value])
    } else {
        EvalAction::NormalBranch(1, smallvec![Value::Array { data, ty }])
    }
}

fn eval_snapshot_pop_back(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Array { mut data, ty }]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    if !data.is_empty() {
        let new_data = data.split_off(data.len() - 1);
        let value = new_data[0].clone();
        assert!(value.is(registry, &info.ty));
        EvalAction::NormalBranch(0, smallvec![Value::Array { data, ty }, value])
    } else {
        EvalAction::NormalBranch(1, smallvec![Value::Array { data, ty }])
    }
}

fn eval_snapshot_multi_pop_front(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &cairo_lang_sierra::extensions::array::ConcreteMultiPopLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [rangecheck, Value::Array { mut data, ty }]: [Value; 2] = args.try_into().unwrap() else {
        panic!()
    };

    let CoreTypeConcrete::Struct(popped_cty) = registry.get_type(&info.popped_ty).unwrap() else {
        panic!()
    };

    if data.len() >= popped_cty.members.len() {
        let new_data = data.split_off(popped_cty.members.len());
        let value = Value::Struct(data);
        assert!(value.is(registry, &info.popped_ty));
        EvalAction::NormalBranch(
            0,
            smallvec![rangecheck, Value::Array { data: new_data, ty }, value],
        )
    } else {
        EvalAction::NormalBranch(1, smallvec![rangecheck, Value::Array { data, ty }])
    }
}

fn eval_snapshot_multi_pop_back(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &cairo_lang_sierra::extensions::array::ConcreteMultiPopLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [rangecheck, Value::Array { mut data, ty }]: [Value; 2] = args.try_into().unwrap() else {
        panic!()
    };

    let CoreTypeConcrete::Struct(popped_cty) = registry.get_type(&info.popped_ty).unwrap() else {
        panic!()
    };

    if data.len() >= popped_cty.members.len() {
        let popped_data = data.split_off(data.len() - popped_cty.members.len());
        let value = Value::Struct(popped_data);
        assert!(value.is(registry, &info.popped_ty));
        EvalAction::NormalBranch(0, smallvec![rangecheck, Value::Array { data, ty }, value])
    } else {
        EvalAction::NormalBranch(1, smallvec![rangecheck, Value::Array { data, ty }])
    }
}
