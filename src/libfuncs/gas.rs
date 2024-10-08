//! # Gas management libfuncs

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::{gas::GasCost, MetadataStorage},
    utils::{BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        gas::GasConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        llvm::r#type::pointer,
        ods,
    },
    ir::{attribute::FlatSymbolRefAttribute, r#type::IntegerType, Block, Location},
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &GasConcreteLibfunc,
) -> Result<()> {
    match selector {
        GasConcreteLibfunc::WithdrawGas(info) => {
            build_withdraw_gas(context, registry, entry, location, helper, metadata, info)
        }
        GasConcreteLibfunc::RedepositGas(_) => todo!("implement redeposit gas libfunc"),
        GasConcreteLibfunc::GetAvailableGas(info) => {
            build_get_available_gas(context, registry, entry, location, helper, metadata, info)
        }
        GasConcreteLibfunc::BuiltinWithdrawGas(info) => {
            build_builtin_withdraw_gas(context, registry, entry, location, helper, metadata, info)
        }
        GasConcreteLibfunc::GetBuiltinCosts(info) => {
            build_get_builtin_costs(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `get_builtin_costs` libfunc.
pub fn build_get_available_gas<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    entry.append_operation(helper.br(
        0,
        &[entry.argument(0)?.into(), entry.argument(0)?.into()],
        location,
    ));
    Ok(())
}

/// Generate MLIR operations for the `withdraw_gas` libfunc.
pub fn build_withdraw_gas<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
    let current_gas = entry.argument(1)?.into();

    let (cost, token_type) = metadata
        .get::<GasCost>()
        .and_then(|x| x.0)
        .unwrap_or((0, 0));

    let u128_type: melior::ir::Type = IntegerType::new(context, 128).into();
    let u64_type: melior::ir::Type = IntegerType::new(context, 64).into();

    let gas_cost_val = entry.const_int_from_type(context, location, cost, u128_type)?;
    let token_type_val = entry.const_int_from_type(context, location, token_type, u64_type)?;

    let is_enough = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Uge,
        current_gas,
        gas_cost_val,
        location,
    ))?;

    let resulting_gas = entry.append_op_result(
        ods::llvm::intr_usub_sat(context, current_gas, gas_cost_val, location).into(),
    )?;

    entry.append_operation(helper.cond_br(
        context,
        is_enough,
        [0, 1],
        [&[range_check, resulting_gas]; 2],
        location,
    ));

    Ok(())
}

/// Generate MLIR operations for the `withdraw_gas_all` libfunc.
pub fn build_builtin_withdraw_gas<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
    let current_gas = entry.argument(1)?.into();

    let cost = metadata.get::<GasCost>().and_then(|x| x.0);

    let u128_type: melior::ir::Type = IntegerType::new(context, 128).into();
    let gas_cost_val =
        entry.const_int_from_type(context, location, cost.unwrap_or(0), u128_type)?;

    let is_enough = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Uge,
        current_gas,
        gas_cost_val,
        location,
    ))?;

    let resulting_gas = entry.append_op_result(
        ods::llvm::intr_usub_sat(context, current_gas, gas_cost_val, location).into(),
    )?;

    entry.append_operation(helper.cond_br(
        context,
        is_enough,
        [0, 1],
        [&[range_check, resulting_gas]; 2],
        location,
    ));

    Ok(())
}

/// Generate MLIR operations for the `get_builtin_costs` libfunc.
pub fn build_get_builtin_costs<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let builtin_costs_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;

    // Get the ptr to the global, holding a ptr to the list.
    let builtin_ptr_ptr = entry.append_op_result(
        ods::llvm::mlir_addressof(
            context,
            pointer(context, 0),
            FlatSymbolRefAttribute::new(context, "builtin_costs"),
            location,
        )
        .into(),
    )?;

    let builtin_ptr = entry.load(context, location, builtin_ptr_ptr, builtin_costs_ty)?;

    entry.append_operation(helper.br(0, &[builtin_ptr], location));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::{load_cairo, run_program};

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

        let result = run_program(&program, "run_test", &[]);
        assert_eq!(
            result.remaining_gas,
            Some(340282366920938463463374607431768205035),
        );
    }
}
