//! # Gas management libfuncs

use super::LibfuncHelper;
use crate::{
    error::{panic::ToNativeAssertError, Error, Result},
    metadata::{gas::GasCost, runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    utils::{BlockExt, BuiltinCosts, GepIndex},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        gas::GasConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith::CmpiPredicate, ods},
    ir::{r#type::IntegerType, Block, BlockLike, Location, Value},
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
        GasConcreteLibfunc::RedepositGas(info) => {
            build_redeposit_gas(context, registry, entry, location, helper, metadata, info)
        }
        GasConcreteLibfunc::GetAvailableGas(info) => {
            build_get_available_gas(context, registry, entry, location, helper, metadata, info)
        }
        GasConcreteLibfunc::BuiltinWithdrawGas(info) => {
            build_builtin_withdraw_gas(context, registry, entry, location, helper, metadata, info)
        }
        GasConcreteLibfunc::GetBuiltinCosts(info) => {
            build_get_builtin_costs(context, registry, entry, location, helper, metadata, info)
        }
        GasConcreteLibfunc::GetUnspentGas(info) => {
            build_get_unspent_gas(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `get_available_gas` libfunc.
pub fn build_get_available_gas<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let gas = entry.arg(0)?;
    let gas_u128 = entry.extui(gas, IntegerType::new(context, 128).into(), location)?;
    // The gas is returned as u128 on the second arg.
    entry.append_operation(helper.br(0, &[entry.arg(0)?, gas_u128], location));
    Ok(())
}

/// Generate MLIR operations for the `withdraw_gas` libfunc.
pub fn build_withdraw_gas<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;
    let current_gas = entry.arg(1)?;
    let builtin_ptr = {
        let runtime = metadata
            .get_mut::<RuntimeBindingsMeta>()
            .ok_or(Error::MissingMetadata)?;
        runtime
            .get_gas_builtin(context, helper, entry, location)?
            .result(0)?
            .into()
    };

    let gas_cost = metadata
        .get::<GasCost>()
        .to_native_assert_error("withdraw_gas should always have a gas cost")?
        .clone();

    let total_gas_cost_value =
        build_actual_gas_cost(context, entry, location, gas_cost, builtin_ptr)?;

    let is_enough = entry.cmpi(
        context,
        CmpiPredicate::Uge,
        current_gas,
        total_gas_cost_value,
        location,
    )?;

    let resulting_gas = entry.append_op_result(
        ods::llvm::intr_usub_sat(context, current_gas, total_gas_cost_value, location).into(),
    )?;

    entry.append_operation(helper.cond_br(
        context,
        is_enough,
        [0, 1],
        [&[range_check, resulting_gas], &[range_check, current_gas]],
        location,
    ));

    Ok(())
}

/// Returns the unused gas to the remaining
///
/// ```cairo
/// extern fn redeposit_gas() implicits(GasBuiltin) nopanic;
/// ```
pub fn build_redeposit_gas<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let current_gas = entry.arg(0)?;

    let gas_cost = metadata
        .get::<GasCost>()
        .to_native_assert_error("redeposit_gas should always have a gas cost")?
        .clone();

    let builtin_ptr = {
        let runtime = metadata
            .get_mut::<RuntimeBindingsMeta>()
            .ok_or(Error::MissingMetadata)?;
        runtime
            .get_gas_builtin(context, helper, entry, location)?
            .result(0)?
            .into()
    };

    let total_gas_cost_value =
        build_actual_gas_cost(context, entry, location, gas_cost, builtin_ptr)?;

    let resulting_gas = entry.append_op_result(
        ods::llvm::intr_uadd_sat(context, current_gas, total_gas_cost_value, location).into(),
    )?;

    entry.append_operation(helper.br(0, &[resulting_gas], location));

    Ok(())
}

/// Generate MLIR operations for the `withdraw_gas_all` libfunc.
pub fn build_builtin_withdraw_gas<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;
    let current_gas = entry.arg(1)?;
    let builtin_ptr = entry.arg(2)?;

    let gas_cost = metadata
        .get::<GasCost>()
        .to_native_assert_error("builtin_withdraw_gas should always have a gas cost")?
        .clone();

    let total_gas_cost_value =
        build_actual_gas_cost(context, entry, location, gas_cost, builtin_ptr)?;

    let is_enough = entry.cmpi(
        context,
        CmpiPredicate::Uge,
        current_gas,
        total_gas_cost_value,
        location,
    )?;

    let resulting_gas = entry.append_op_result(
        ods::llvm::intr_usub_sat(context, current_gas, total_gas_cost_value, location).into(),
    )?;

    entry.append_operation(helper.cond_br(
        context,
        is_enough,
        [0, 1],
        [&[range_check, resulting_gas], &[range_check, current_gas]],
        location,
    ));

    Ok(())
}

/// Generate MLIR operations for the `get_builtin_costs` libfunc.
pub fn build_get_builtin_costs<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Get the ptr to the global, holding a ptr to the list.
    let builtin_ptr = {
        let runtime = metadata
            .get_mut::<RuntimeBindingsMeta>()
            .ok_or(Error::MissingMetadata)?;
        runtime
            .get_gas_builtin(context, helper, entry, location)?
            .result(0)?
            .into()
    };

    entry.append_operation(helper.br(0, &[builtin_ptr], location));

    Ok(())
}

/// Returns the amount of gas available in the `GasBuiltin`, as well as the
/// amount of gas unused in the local wallet.
///
/// # Cairo Signature
///
/// ```cairo
/// extern fn get_unspent_gas() -> u128 implicits(GasBuiltin) nopanic;
/// ```
#[allow(unused_variables)]
pub fn build_get_unspent_gas<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let current_gas = entry.arg(0)?;

    let gas_cost = metadata
        .get::<GasCost>()
        .to_native_assert_error("redeposit_gas should always have a gas cost")?
        .clone();

    let builtin_ptr = {
        let runtime = metadata
            .get_mut::<RuntimeBindingsMeta>()
            .ok_or(Error::MissingMetadata)?;
        runtime
            .get_gas_builtin(context, helper, entry, location)?
            .result(0)?
            .into()
    };

    let current_gas_u128 =
        entry.extui(current_gas, IntegerType::new(context, 128).into(), location)?;

    let gas_cost = build_actual_gas_cost(context, entry, location, gas_cost, builtin_ptr)?;
    let gas_cost_u128 = entry.extui(gas_cost, IntegerType::new(context, 128).into(), location)?;

    let unspent_gas = entry.append_op_result(
        ods::llvm::intr_uadd_sat(context, current_gas_u128, gas_cost_u128, location).into(),
    )?;

    entry.append_operation(helper.br(0, &[current_gas, unspent_gas], location));

    Ok(())
}

/// Calculate the actual gas cost, given the constant `GasCost` configuration,
/// and the current `BuiltinCosts` pointer.
pub fn build_actual_gas_cost<'c, 'b>(
    context: &'c Context,
    block: &'b Block<'c>,
    location: Location<'c>,
    gas_cost: GasCost,
    builtin_ptr: Value<'c, 'b>,
) -> Result<Value<'c, 'b>> {
    let u64_type: melior::ir::Type = IntegerType::new(context, 64).into();

    let mut total_gas_cost = block.const_int_from_type(context, location, 0, u64_type)?;

    // For each gas cost entry
    for (token_count, token_type) in &gas_cost.0 {
        if *token_count == 0 {
            continue;
        }

        let token_count = block.const_int_from_type(context, location, *token_count, u64_type)?;

        // Calculate the index of the token type in the builtin costs array
        let token_costs_index = block.const_int_from_type(
            context,
            location,
            BuiltinCosts::index_for_token_type(token_type)?,
            u64_type,
        )?;

        // Index the builtin costs array
        let token_cost_ptr = block.gep(
            context,
            location,
            builtin_ptr,
            &[GepIndex::Value(token_costs_index)],
            u64_type,
        )?;
        let token_cost = block.load(context, location, token_cost_ptr, u64_type)?;

        // Multiply the number of tokens by the cost of each token
        let gas_cost = block.muli(token_count, token_cost, location)?;

        total_gas_cost = block.addi(total_gas_cost, gas_cost, location)?;
    }

    Ok(total_gas_cost)
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{jit_enum, jit_struct, load_cairo, run_program},
        Value,
    };
    use pretty_assertions_sorted::assert_eq;

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
        assert_eq!(result.remaining_gas, Some(18446744073709545265));
    }

    #[test]
    fn run_get_unspent_gas() {
        let program = load_cairo!(
            pub extern fn get_unspent_gas() -> u128 implicits(GasBuiltin) nopanic;

            fn run_test() -> u128 {
                let n: u32 = 10;

                let one = 1;
                let two = 2;

                let prev = get_unspent_gas();
                for _ in 0..n { let _three = one + two; }
                let after = get_unspent_gas();

                return prev - after;
            }
        );

        let result = run_program(&program, "run_test", &[]);

        assert_eq!(
            result.return_value,
            jit_enum!(0, jit_struct!(Value::Felt252(27160.into())))
        )
    }
}
