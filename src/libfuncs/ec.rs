//! # Elliptic curve libfuncs
//!
//! TODO

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::MetadataStorage,
    types::TypeBuilder,
};
use cairo_lang_sierra::{
    extensions::{
        ec::EcConcreteLibfunc, lib_func::SignatureOnlyConcreteLibfunc, ConcreteLibfunc,
        GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        llvm,
    },
    ir::{
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        r#type::IntegerType,
        Block, Location,
    },
    Context,
};

pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &EcConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        EcConcreteLibfunc::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, metadata, info)
        }
        EcConcreteLibfunc::Neg(_) => todo!(),
        EcConcreteLibfunc::PointFromX(_) => todo!(),
        EcConcreteLibfunc::StateAdd(_) => todo!(),
        EcConcreteLibfunc::StateAddMul(_) => todo!(),
        EcConcreteLibfunc::StateFinalize(_) => todo!(),
        EcConcreteLibfunc::StateInit(_) => todo!(),
        EcConcreteLibfunc::TryNew(_) => todo!(),
        EcConcreteLibfunc::UnwrapPoint(info) => {
            build_unwrap_point(context, registry, entry, location, helper, metadata, info)
        }
        EcConcreteLibfunc::Zero(info) => {
            build_zero(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_is_zero<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let x = entry
        .append_operation(llvm::extract_value(
            context,
            entry.argument(0)?.into(),
            DenseI64ArrayAttribute::new(context, &[0]),
            IntegerType::new(context, 252).into(),
            location,
        ))
        .result(0)?
        .into();
    let y = entry
        .append_operation(llvm::extract_value(
            context,
            entry.argument(0)?.into(),
            DenseI64ArrayAttribute::new(context, &[1]),
            IntegerType::new(context, 252).into(),
            location,
        ))
        .result(0)?
        .into();

    let k0 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(0, IntegerType::new(context, 252).into()).into(),
            location,
        ))
        .result(0)?
        .into();

    let x_is_zero = entry
        .append_operation(arith::cmpi(context, CmpiPredicate::Eq, x, k0, location))
        .result(0)?
        .into();
    let y_is_zero = entry
        .append_operation(arith::cmpi(context, CmpiPredicate::Eq, y, k0, location))
        .result(0)?
        .into();

    let point_is_zero = entry
        .append_operation(arith::andi(x_is_zero, y_is_zero, location))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(
        point_is_zero,
        [0, 1],
        [&[], &[entry.argument(0)?.into()]],
        location,
    ));
    Ok(())
}

pub fn build_unwrap_point<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let x = entry
        .append_operation(llvm::extract_value(
            context,
            entry.argument(0)?.into(),
            DenseI64ArrayAttribute::new(context, &[0]),
            registry
                .get_type(&info.branch_signatures()[0].vars[0].ty)?
                .build(context, helper, registry, metadata)?,
            location,
        ))
        .result(0)?
        .into();
    let y = entry
        .append_operation(llvm::extract_value(
            context,
            entry.argument(0)?.into(),
            DenseI64ArrayAttribute::new(context, &[1]),
            registry
                .get_type(&info.branch_signatures()[0].vars[1].ty)?
                .build(context, helper, registry, metadata)?,
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[x, y], location));
    Ok(())
}

pub fn build_zero<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let ec_point_ty = registry
        .get_type(&info.branch_signatures()[0].vars[0].ty)?
        .build(context, helper, registry, metadata)?;

    let point = entry
        .append_operation(llvm::undef(ec_point_ty, location))
        .result(0)?
        .into();

    let k0 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(0, IntegerType::new(context, 252).into()).into(),
            location,
        ))
        .result(0)?
        .into();
    let point = entry
        .append_operation(llvm::insert_value(
            context,
            point,
            DenseI64ArrayAttribute::new(context, &[0]),
            k0,
            location,
        ))
        .result(0)?
        .into();
    let point = entry
        .append_operation(llvm::insert_value(
            context,
            point,
            DenseI64ArrayAttribute::new(context, &[1]),
            k0,
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[point], location));
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::{felt, load_cairo, run_program};
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;
    use serde_json::json;

    lazy_static! {
        static ref EC_POINT_IS_ZERO: (String, Program) = load_cairo! {
            use core::{ec::{ec_point_is_zero, EcPoint}, zeroable::IsZeroResult};

            fn run_test(point: EcPoint) -> IsZeroResult<EcPoint> {
                ec_point_is_zero(point)
            }
        };
        static ref EC_POINT_UNWRAP: (String, Program) = load_cairo! {
            use core::{ec::{ec_point_unwrap, EcPoint}, zeroable::NonZero};

            fn run_test(point: NonZero<EcPoint>) -> (felt252, felt252) {
                ec_point_unwrap(point)
            }
        };
        static ref EC_POINT_ZERO: (String, Program) = load_cairo! {
            use core::ec::{ec_point_zero, EcPoint};

            fn run_test() -> EcPoint {
                ec_point_zero()
            }
        };
    }

    #[test]
    fn ec_point_is_zero() {
        let r = |x, y| run_program(&EC_POINT_IS_ZERO, "run_test", json!([[x, y]]));

        assert_eq!(r(felt("0"), felt("0")), json!([[0, []]]));
        assert_eq!(
            r(felt("0"), felt("1")),
            json!([[1, [felt("0"), felt("1")]]])
        );
        assert_eq!(
            r(felt("1"), felt("0")),
            json!([[1, [felt("1"), felt("0")]]])
        );
        assert_eq!(
            r(felt("1"), felt("1")),
            json!([[1, [felt("1"), felt("1")]]])
        );
    }

    #[test]
    fn ec_point_unwrap() {
        let r = |x, y| run_program(&EC_POINT_UNWRAP, "run_test", json!([[x, y]]));

        assert_eq!(r(felt("0"), felt("0")), json!([[felt("0"), felt("0")]]));
        assert_eq!(r(felt("0"), felt("1")), json!([[felt("0"), felt("1")]]));
        assert_eq!(r(felt("0"), felt("-1")), json!([[felt("0"), felt("-1")]]));
        assert_eq!(r(felt("1"), felt("0")), json!([[felt("1"), felt("0")]]));
        assert_eq!(r(felt("1"), felt("1")), json!([[felt("1"), felt("1")]]));
        assert_eq!(r(felt("1"), felt("-1")), json!([[felt("1"), felt("-1")]]));
        assert_eq!(r(felt("-1"), felt("0")), json!([[felt("-1"), felt("0")]]));
        assert_eq!(r(felt("-1"), felt("1")), json!([[felt("-1"), felt("1")]]));
        assert_eq!(r(felt("-1"), felt("-1")), json!([[felt("-1"), felt("-1")]]));
    }

    #[test]
    fn ec_point_zero() {
        assert_eq!(
            run_program(&EC_POINT_ZERO, "run_test", json!([])),
            json!([[felt("0"), felt("0")]]),
        );
    }
}
