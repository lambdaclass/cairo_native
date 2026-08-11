use super::EvalAction;
use crate::Value;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        ec::EcConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use num_traits::identities::Zero;
use smallvec::smallvec;
use starknet_crypto::Felt;
use starknet_curve::curve_params::BETA;
use starknet_types_core::curve::{AffinePoint, ProjectivePoint};
use std::ops::Mul;
use std::ops::Neg;

// todo: verify these are correct.

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &EcConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        EcConcreteLibfunc::IsZero(info) => eval_is_zero(registry, info, args),
        EcConcreteLibfunc::Neg(info) => eval_neg(registry, info, args),
        EcConcreteLibfunc::StateAdd(info) => eval_state_add(registry, info, args),
        EcConcreteLibfunc::TryNew(info) => eval_new(registry, info, args),
        EcConcreteLibfunc::StateFinalize(info) => eval_state_finalize(registry, info, args),
        EcConcreteLibfunc::StateInit(info) => eval_state_init(registry, info, args),
        EcConcreteLibfunc::StateAddMul(info) => eval_state_add_mul(registry, info, args),
        EcConcreteLibfunc::PointFromX(info) => eval_point_from_x(registry, info, args),
        EcConcreteLibfunc::UnwrapPoint(info) => eval_unwrap_point(registry, info, args),
        EcConcreteLibfunc::Zero(info) => eval_zero(registry, info, args),
        EcConcreteLibfunc::NegNz(info) => eval_neg(registry, info, args),
    }
}

fn eval_is_zero(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value @ Value::EcPoint { x: _, y }]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };
    // To check whether `(x, y) = (0, 0)` (the zero point), it is enough to check
    // whether `y = 0`, since there is no point on the curve with y = 0.
    if y.is_zero() {
        EvalAction::NormalBranch(0, smallvec![])
    } else {
        EvalAction::NormalBranch(1, smallvec![value])
    }
}

fn eval_unwrap_point(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::EcPoint { x, y }]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };
    EvalAction::NormalBranch(0, smallvec![Value::Felt(x), Value::Felt(y)])
}

fn eval_neg(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::EcPoint { x, y }]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    let point = AffinePoint::new(x, y).unwrap().neg();

    EvalAction::NormalBranch(
        0,
        smallvec![Value::EcPoint {
            x: point.x(),
            y: point.y(),
        }],
    )
}

fn eval_new(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::Felt(x), Value::Felt(y)]: [Value; 2] = args.try_into().unwrap() else {
        panic!()
    };

    match AffinePoint::new(x, y) {
        Ok(point) => EvalAction::NormalBranch(
            0,
            smallvec![Value::EcPoint {
                x: point.x(),
                y: point.y(),
            }],
        ),
        Err(_) => EvalAction::NormalBranch(1, smallvec![]),
    }
}

fn eval_state_init(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    _args: Vec<Value>,
) -> EvalAction {
    // The canonical projective identity `[0 : 1 : 0]`.
    EvalAction::NormalBranch(
        0,
        smallvec![Value::EcState {
            x: 0.into(),
            y: 1.into(),
            z: 0.into(),
        }],
    )
}

fn eval_state_add(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::EcState {
        x: s_x,
        y: s_y,
        z: s_z,
    }, Value::EcPoint { x, y }]: [Value; 2] = args.try_into().unwrap()
    else {
        panic!()
    };

    let mut state = ProjectivePoint::new_unchecked(s_x, s_y, s_z);
    let point = ProjectivePoint::from_affine(x, y).unwrap();

    state += &point;
    EvalAction::NormalBranch(
        0,
        smallvec![Value::EcState {
            x: state.x(),
            y: state.y(),
            z: state.z(),
        }],
    )
}

fn eval_state_add_mul(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [ec @ Value::Unit, Value::EcState {
        x: s_x,
        y: s_y,
        z: s_z,
    }, Value::Felt(scalar), Value::EcPoint { x, y }]: [Value; 4] = args.try_into().unwrap()
    else {
        panic!()
    };

    let mut state = ProjectivePoint::new_unchecked(s_x, s_y, s_z);
    let point = ProjectivePoint::from_affine(x, y).unwrap();

    state += &point.mul(scalar);
    EvalAction::NormalBranch(
        0,
        smallvec![
            ec,
            Value::EcState {
                x: state.x(),
                y: state.y(),
                z: state.z(),
            }
        ],
    )
}

fn eval_state_finalize(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::EcState { x, y, z }]: [Value; 1] = args.try_into().unwrap() else {
        panic!()
    };

    // The single normalisation of the pipeline. `to_affine` fails exactly when
    // `z == 0`, so it doubles as the point-at-infinity test.
    match ProjectivePoint::new_unchecked(x, y, z).to_affine() {
        Ok(point) => EvalAction::NormalBranch(
            0,
            smallvec![Value::EcPoint {
                x: point.x(),
                y: point.y(),
            }],
        ),
        Err(_) => EvalAction::NormalBranch(1, smallvec![]),
    }
}

fn eval_point_from_x(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [range_check @ Value::Unit, Value::Felt(x)]: [Value; 2] = args.try_into().unwrap() else {
        panic!()
    };

    // https://github.com/starkware-libs/cairo/blob/aaad921bba52e729dc24ece07fab2edf09ccfa15/crates/cairo-lang-sierra-to-casm/src/invocations/ec.rs#L63

    let x2 = x * x;
    let x3 = x2 * x;
    let alpha_x_plus_beta = x + BETA;
    let rhs = x3 + alpha_x_plus_beta;
    let y = rhs.sqrt().unwrap_or_else(|| Felt::from(3) * rhs);

    match AffinePoint::new(x, y) {
        Ok(point) => EvalAction::NormalBranch(
            0,
            smallvec![
                range_check,
                Value::EcPoint {
                    x: point.x(),
                    y: point.y(),
                }
            ],
        ),
        Err(_) => EvalAction::NormalBranch(1, smallvec![range_check]),
    }
}

fn eval_zero(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    _args: Vec<Value>,
) -> EvalAction {
    EvalAction::NormalBranch(
        0,
        smallvec![Value::EcPoint {
            x: 0.into(),
            y: 0.into()
        }],
    )
}
