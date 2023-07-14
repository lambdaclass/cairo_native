//! # Elliptic curve libfuncs
//!
//! TODO

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::{
        prime_modulo::PrimeModuloMeta, runtime_bindings::RuntimeBindingsMeta, MetadataStorage,
    },
    types::{
        felt252::{register_prime_modulo_meta, Felt252},
        TypeBuilder,
    },
    utils::get_integer_layout,
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
        llvm::{self, LoadStoreOptions},
    },
    ir::{
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        operation::OperationBuilder,
        r#type::IntegerType,
        Attribute, Block, Identifier, Location,
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
        EcConcreteLibfunc::Neg(info) => {
            build_neg(context, registry, entry, location, helper, metadata, info)
        }
        EcConcreteLibfunc::PointFromX(info) => {
            build_point_from_x(context, registry, entry, location, helper, metadata, info)
        }
        EcConcreteLibfunc::StateAdd(_) => todo!(),
        EcConcreteLibfunc::StateAddMul(_) => todo!(),
        EcConcreteLibfunc::StateFinalize(_) => todo!(),
        EcConcreteLibfunc::StateInit(info) => {
            build_state_init(context, registry, entry, location, helper, metadata, info)
        }
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

pub fn build_neg<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
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

    let k_prime = entry
        .append_operation(arith::constant(
            context,
            Attribute::parse(
                context,
                &format!(
                    "{} : i252",
                    match metadata.get::<PrimeModuloMeta<Felt252>>() {
                        Some(x) => x.prime(),
                        None => {
                            // Since the `EcPoint` type is external, there is no guarantee that
                            // `PrimeModuloMeta<Felt252>` will be available.
                            register_prime_modulo_meta(metadata).prime()
                        }
                    }
                ),
            )
            .unwrap(),
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
    let y_is_zero = entry
        .append_operation(arith::cmpi(context, CmpiPredicate::Eq, y, k0, location))
        .result(0)?
        .into();

    let y_neg = entry
        .append_operation(arith::subi(k_prime, y, location))
        .result(0)?
        .into();
    let y_neg = entry
        .append_operation(
            OperationBuilder::new("arith.select", location)
                .add_operands(&[y_is_zero, k0, y_neg])
                .add_results(&[IntegerType::new(context, 252).into()])
                .build(),
        )
        .result(0)?
        .into();

    let result = entry
        .append_operation(llvm::insert_value(
            context,
            entry.argument(0)?.into(),
            DenseI64ArrayAttribute::new(context, &[1]),
            y_neg,
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[result], location));
    Ok(())
}

pub fn build_point_from_x<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let ec_point_ty = llvm::r#type::r#struct(
        context,
        &[
            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
        ],
        false,
    );

    let k1 = helper
        .init_block()
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
            location,
        ))
        .result(0)?
        .into();
    let point_ptr = entry
        .append_operation(
            OperationBuilder::new("llvm.alloca", location)
                .add_attributes(&[(
                    Identifier::new(context, "alignment"),
                    IntegerAttribute::new(
                        get_integer_layout(252).align().try_into()?,
                        IntegerType::new(context, 64).into(),
                    )
                    .into(),
                )])
                .add_operands(&[k1])
                .add_results(&[llvm::r#type::pointer(ec_point_ty, 0)])
                .build(),
        )
        .result(0)?
        .into();

    let point = entry
        .append_operation(llvm::undef(ec_point_ty, location))
        .result(0)?
        .into();
    let point = entry
        .append_operation(llvm::insert_value(
            context,
            point,
            DenseI64ArrayAttribute::new(context, &[0]),
            entry.argument(1)?.into(),
            location,
        ))
        .result(0)?
        .into();
    entry.append_operation(llvm::store(
        context,
        point,
        point_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    let result = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .unwrap()
        .libfunc_ec_point_from_x_nz(context, helper, entry, point_ptr, location)?
        .result(0)?
        .into();

    let point = entry
        .append_operation(llvm::load(
            context,
            point_ptr,
            ec_point_ty,
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(
        result,
        [0, 1],
        [
            &[entry.argument(0)?.into(), point],
            &[entry.argument(0)?.into()],
        ],
        location,
    ));
    Ok(())
}

pub fn build_state_init<'ctx, 'this, TType, TLibfunc>(
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
    let ec_state_ty = llvm::r#type::r#struct(
        context,
        &[
            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
        ],
        false,
    );

    let point = entry
        .append_operation(llvm::undef(ec_state_ty, location))
        .result(0)?
        .into();

    let x = entry
        .append_operation(arith::constant(
            context,
            Attribute::parse(context, "3151312365169595090315724863753927489909436624354740709748557281394568342450 : i252").unwrap(),
            location,
        ))
        .result(0)?
        .into();
    let y = entry
        .append_operation(arith::constant(
            context,
            Attribute::parse(context, "2835232394579952276045648147338966184268723952674536708929458753792035266179 : i252").unwrap(),
            location,
        ))
        .result(0)?
        .into();

    let point = entry
        .append_operation(llvm::insert_value(
            context,
            point,
            DenseI64ArrayAttribute::new(context, &[0]),
            x,
            location,
        ))
        .result(0)?
        .into();
    let point = entry
        .append_operation(llvm::insert_value(
            context,
            point,
            DenseI64ArrayAttribute::new(context, &[1]),
            y,
            location,
        ))
        .result(0)?
        .into();
    let point = entry
        .append_operation(llvm::insert_value(
            context,
            point,
            DenseI64ArrayAttribute::new(context, &[2]),
            x,
            location,
        ))
        .result(0)?
        .into();
    let point = entry
        .append_operation(llvm::insert_value(
            context,
            point,
            DenseI64ArrayAttribute::new(context, &[3]),
            y,
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[point], location));
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
        static ref EC_NEG: (String, Program) = load_cairo! {
            use core::ec::{ec_neg, EcPoint};

            fn run_test(point: EcPoint) -> EcPoint {
                ec_neg(point)
            }
        };
        static ref EC_POINT_FROM_X_NZ: (String, Program) = load_cairo! {
            use core::ec::{ec_point_from_x, EcPoint};

            fn run_test(x: felt252) -> Option<EcPoint> {
                ec_point_from_x(x)
            }
        };
        static ref EC_STATE_INIT: (String, Program) = load_cairo! {
            use core::ec::{ec_state_init, EcState};

            fn run_test() -> EcState {
                ec_state_init()
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
    fn ec_neg() {
        let r = |x, y| run_program(&EC_NEG, "run_test", json!([[x, y]]));

        assert_eq!(r(felt("0"), felt("0")), json!([[felt("0"), felt("0")]]));
        assert_eq!(r(felt("0"), felt("1")), json!([[felt("0"), felt("-1")]]));
        assert_eq!(r(felt("1"), felt("0")), json!([[felt("1"), felt("0")]]));
        assert_eq!(r(felt("1"), felt("1")), json!([[felt("1"), felt("-1")]]));
    }

    #[test]
    fn ec_point_from_x() {
        let r = |x| run_program(&EC_POINT_FROM_X_NZ, "run_test", json!([(), x]));

        assert_eq!(r(felt("0")), json!([(), [1, []]]));
        assert_eq!(
            r(felt("1234")),
            json!([(), [0, [felt("1234"), felt("1301976514684871091717790968549291947487646995000837413367950573852273027507")]]])
        );
    }

    #[test]
    fn ec_state_init() {
        assert_eq!(
            run_program(&EC_STATE_INIT, "run_test", json!([])),
            json!([[
                felt(
                    "3151312365169595090315724863753927489909436624354740709748557281394568342450"
                ),
                felt(
                    "2835232394579952276045648147338966184268723952674536708929458753792035266179"
                ),
                felt(
                    "3151312365169595090315724863753927489909436624354740709748557281394568342450"
                ),
                felt(
                    "2835232394579952276045648147338966184268723952674536708929458753792035266179"
                )
            ]]),
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
