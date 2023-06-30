//! # Gas management libfuncs
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
        gas::GasConcreteLibfunc, lib_func::SignatureOnlyConcreteLibfunc, ConcreteLibfunc,
        GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith, llvm},
    ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, Location},
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
    <TType as GenericType>::Concrete: TypeBuilder<Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<Error = Error>,
{
    match selector {
        GasConcreteLibfunc::WithdrawGas(_) => todo!(),
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

/// Generate MLIR operations for the `withdraw_gas_all` libfunc.
pub fn build_builtin_withdraw_gas<'ctx, 'this, TType, TLibfunc>(
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
    <TType as GenericType>::Concrete: TypeBuilder<Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<Error = Error>,
{
    // TODO: Implement libfunc.
    let op0 = entry.append_operation(arith::constant(
        context,
        IntegerAttribute::new(1, IntegerType::new(context, 1).into()).into(),
        location,
    ));

    entry.append_operation(helper.cond_br(
        op0.result(0)?.into(),
        [0, 1],
        [&[entry.argument(0)?.into(), entry.argument(1)?.into()]; 2],
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
    <TType as GenericType>::Concrete: TypeBuilder<Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<Error = Error>,
{
    let builtin_costs_ty = registry
        .get_type(&info.branch_signatures()[0].vars[0].ty)?
        .build(context, helper, registry, metadata)?;

    // TODO: Implement libfunc.
    let op0 = entry.append_operation(llvm::undef(builtin_costs_ty, location));

    entry.append_operation(helper.br(0, &[op0.result(0)?.into()], location));

    Ok(())
}
