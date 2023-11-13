//! # Gas management libfuncs
//!
//! TODO

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::{gas::GasCost, MetadataStorage},
    types::TypeBuilder,
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        gas::GasConcreteLibfunc, lib_func::SignatureOnlyConcreteLibfunc, ConcreteLibfunc,
        GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        llvm,
    },
    ir::{attribute::StringAttribute, r#type::IntegerType, Attribute, Block, Location, ValueLike},
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &GasConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        GasConcreteLibfunc::WithdrawGas(info) => {
            build_withdraw_gas(context, registry, entry, location, helper, metadata, info)
        }
        GasConcreteLibfunc::RedepositGas(_) => todo!(),
        GasConcreteLibfunc::GetAvailableGas(_) => todo!(),
        GasConcreteLibfunc::BuiltinWithdrawGas(info) => {
            build_builtin_withdraw_gas(context, registry, entry, location, helper, metadata, info)
        }
        GasConcreteLibfunc::GetBuiltinCosts(info) => {
            build_get_builtin_costs(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `withdraw_gas` libfunc.
pub fn build_withdraw_gas<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let range_check = entry.argument(0)?.into();
    let current_gas = entry.argument(1)?.into();

    let cost = metadata.get::<GasCost>().and_then(|x| x.0);

    let u128_type: melior::ir::Type = IntegerType::new(context, 128).into();
    let gas_cost_val = entry
        .append_operation(arith::constant(
            context,
            Attribute::parse(context, &format!("{} : {}", cost.unwrap_or(0), u128_type)).unwrap(),
            location,
        ))
        .result(0)?
        .into();

    let is_enough = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Uge,
            current_gas,
            gas_cost_val,
            location,
        ))
        .result(0)?
        .into();

    let resulting_gas = entry
        .append_operation(llvm::call_intrinsic(
            context,
            StringAttribute::new(context, "llvm.usub.sat"),
            &[current_gas, gas_cost_val],
            &[gas_cost_val.r#type()],
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(
        is_enough,
        [0, 1],
        [&[range_check, resulting_gas]; 2],
        location,
    ));

    Ok(())
}

/// Generate MLIR operations for the `withdraw_gas_all` libfunc.
pub fn build_builtin_withdraw_gas<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let range_check = entry.argument(0)?.into();
    let current_gas = entry.argument(1)?.into();

    let cost = metadata.get::<GasCost>().and_then(|x| x.0);

    let u128_type: melior::ir::Type = IntegerType::new(context, 128).into();
    let gas_cost_val = entry
        .append_operation(arith::constant(
            context,
            Attribute::parse(context, &format!("{} : {}", cost.unwrap_or(0), u128_type)).unwrap(),
            location,
        ))
        .result(0)?
        .into();

    let is_enough = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Uge,
            current_gas,
            gas_cost_val,
            location,
        ))
        .result(0)?
        .into();

    let resulting_gas = entry
        .append_operation(llvm::call_intrinsic(
            context,
            StringAttribute::new(context, "llvm.usub.sat"),
            &[current_gas, gas_cost_val],
            &[gas_cost_val.r#type()],
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(
        is_enough,
        [0, 1],
        [&[range_check, resulting_gas]; 2],
        location,
    ));

    Ok(())
}

/// Generate MLIR operations for the `get_builtin_costs` libfunc.
pub fn build_get_builtin_costs<'ctx, 'this, TType, TLibfunc>(
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
    let builtin_costs_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;

    // TODO: Implement libfunc.
    let op0 = entry.append_operation(llvm::undef(builtin_costs_ty, location));

    entry.append_operation(helper.br(0, &[op0.result(0)?.into()], location));

    Ok(())
}

#[cfg(test)]
mod test {
    /* TODO: fix tests
    use crate::utils::test::{load_cairo, run_program};
    use serde_json::json;

    #[test]
    fn run_withdraw_gas() {
        #[rustfmt::skip]
        let program = load_cairo!(
            use gas::withdraw_gas;

            fn run_test() {
                let mut i = 10;

                loop {
                    if i == 0 {
                        break;
                    }

                    match withdraw_gas() {
                        Option::Some(()) => {
                            i = i - 1;
                        },
                        Option::None(()) => {
                            break;
                        }
                    };
                    i = i - 1;
                }
            }
        );

        let result = run_program(&program, "run_test", json!([null, 60000]));
        assert_eq!(result, json!([null, 44260, [0, [[]]]]));
    }

    */
}
