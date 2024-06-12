////! # Gas management libfuncs
//! # Gas management libfuncs
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{
use crate::{
//    block_ext::BlockExt,
    block_ext::BlockExt,
//    error::Result,
    error::Result,
//    metadata::{gas::GasCost, MetadataStorage},
    metadata::{gas::GasCost, MetadataStorage},
//    utils::ProgramRegistryExt,
    utils::ProgramRegistryExt,
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        gas::GasConcreteLibfunc,
        gas::GasConcreteLibfunc,
//        lib_func::SignatureOnlyConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
//        ConcreteLibfunc,
        ConcreteLibfunc,
//    },
    },
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    dialect::{
    dialect::{
//        arith::{self, CmpiPredicate},
        arith::{self, CmpiPredicate},
//        llvm, ods,
        llvm, ods,
//    },
    },
//    ir::{r#type::IntegerType, Block, Location},
    ir::{r#type::IntegerType, Block, Location},
//    Context,
    Context,
//};
};
//

///// Select and call the correct libfunc builder function from the selector.
/// Select and call the correct libfunc builder function from the selector.
//pub fn build<'ctx, 'this>(
pub fn build<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    selector: &GasConcreteLibfunc,
    selector: &GasConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        GasConcreteLibfunc::WithdrawGas(info) => {
        GasConcreteLibfunc::WithdrawGas(info) => {
//            build_withdraw_gas(context, registry, entry, location, helper, metadata, info)
            build_withdraw_gas(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        GasConcreteLibfunc::RedepositGas(_) => todo!("implement redeposit gas libfunc"),
        GasConcreteLibfunc::RedepositGas(_) => todo!("implement redeposit gas libfunc"),
//        GasConcreteLibfunc::GetAvailableGas(info) => {
        GasConcreteLibfunc::GetAvailableGas(info) => {
//            build_get_available_gas(context, registry, entry, location, helper, metadata, info)
            build_get_available_gas(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        GasConcreteLibfunc::BuiltinWithdrawGas(info) => {
        GasConcreteLibfunc::BuiltinWithdrawGas(info) => {
//            build_builtin_withdraw_gas(context, registry, entry, location, helper, metadata, info)
            build_builtin_withdraw_gas(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        GasConcreteLibfunc::GetBuiltinCosts(info) => {
        GasConcreteLibfunc::GetBuiltinCosts(info) => {
//            build_get_builtin_costs(context, registry, entry, location, helper, metadata, info)
            build_get_builtin_costs(context, registry, entry, location, helper, metadata, info)
//        }
        }
//    }
    }
//}
}
//

///// Generate MLIR operations for the `get_builtin_costs` libfunc.
/// Generate MLIR operations for the `get_builtin_costs` libfunc.
//pub fn build_get_available_gas<'ctx, 'this>(
pub fn build_get_available_gas<'ctx, 'this>(
//    _context: &'ctx Context,
    _context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    _metadata: &mut MetadataStorage,
    _metadata: &mut MetadataStorage,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    entry.append_operation(helper.br(
    entry.append_operation(helper.br(
//        0,
        0,
//        &[entry.argument(0)?.into(), entry.argument(0)?.into()],
        &[entry.argument(0)?.into(), entry.argument(0)?.into()],
//        location,
        location,
//    ));
    ));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `withdraw_gas` libfunc.
/// Generate MLIR operations for the `withdraw_gas` libfunc.
//pub fn build_withdraw_gas<'ctx, 'this>(
pub fn build_withdraw_gas<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &MetadataStorage,
    metadata: &MetadataStorage,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let range_check =
    let range_check =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//    let current_gas = entry.argument(1)?.into();
    let current_gas = entry.argument(1)?.into();
//

//    let cost = metadata.get::<GasCost>().and_then(|x| x.0);
    let cost = metadata.get::<GasCost>().and_then(|x| x.0);
//

//    let u128_type: melior::ir::Type = IntegerType::new(context, 128).into();
    let u128_type: melior::ir::Type = IntegerType::new(context, 128).into();
//    let gas_cost_val =
    let gas_cost_val =
//        entry.const_int_from_type(context, location, cost.unwrap_or(0), u128_type)?;
        entry.const_int_from_type(context, location, cost.unwrap_or(0), u128_type)?;
//

//    let is_enough = entry.append_op_result(arith::cmpi(
    let is_enough = entry.append_op_result(arith::cmpi(
//        context,
        context,
//        CmpiPredicate::Uge,
        CmpiPredicate::Uge,
//        current_gas,
        current_gas,
//        gas_cost_val,
        gas_cost_val,
//        location,
        location,
//    ))?;
    ))?;
//

//    let resulting_gas = entry.append_op_result(
    let resulting_gas = entry.append_op_result(
//        ods::llvm::intr_usub_sat(context, current_gas, gas_cost_val, location).into(),
        ods::llvm::intr_usub_sat(context, current_gas, gas_cost_val, location).into(),
//    )?;
    )?;
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        is_enough,
        is_enough,
//        [0, 1],
        [0, 1],
//        [&[range_check, resulting_gas]; 2],
        [&[range_check, resulting_gas]; 2],
//        location,
        location,
//    ));
    ));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `withdraw_gas_all` libfunc.
/// Generate MLIR operations for the `withdraw_gas_all` libfunc.
//pub fn build_builtin_withdraw_gas<'ctx, 'this>(
pub fn build_builtin_withdraw_gas<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &MetadataStorage,
    metadata: &MetadataStorage,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let range_check =
    let range_check =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//    let current_gas = entry.argument(1)?.into();
    let current_gas = entry.argument(1)?.into();
//

//    let cost = metadata.get::<GasCost>().and_then(|x| x.0);
    let cost = metadata.get::<GasCost>().and_then(|x| x.0);
//

//    let u128_type: melior::ir::Type = IntegerType::new(context, 128).into();
    let u128_type: melior::ir::Type = IntegerType::new(context, 128).into();
//    let gas_cost_val =
    let gas_cost_val =
//        entry.const_int_from_type(context, location, cost.unwrap_or(0), u128_type)?;
        entry.const_int_from_type(context, location, cost.unwrap_or(0), u128_type)?;
//

//    let is_enough = entry.append_op_result(arith::cmpi(
    let is_enough = entry.append_op_result(arith::cmpi(
//        context,
        context,
//        CmpiPredicate::Uge,
        CmpiPredicate::Uge,
//        current_gas,
        current_gas,
//        gas_cost_val,
        gas_cost_val,
//        location,
        location,
//    ))?;
    ))?;
//

//    let resulting_gas = entry.append_op_result(
    let resulting_gas = entry.append_op_result(
//        ods::llvm::intr_usub_sat(context, current_gas, gas_cost_val, location).into(),
        ods::llvm::intr_usub_sat(context, current_gas, gas_cost_val, location).into(),
//    )?;
    )?;
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        is_enough,
        is_enough,
//        [0, 1],
        [0, 1],
//        [&[range_check, resulting_gas]; 2],
        [&[range_check, resulting_gas]; 2],
//        location,
        location,
//    ));
    ));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `get_builtin_costs` libfunc.
/// Generate MLIR operations for the `get_builtin_costs` libfunc.
//pub fn build_get_builtin_costs<'ctx, 'this>(
pub fn build_get_builtin_costs<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: &SignatureOnlyConcreteLibfunc,
    info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let builtin_costs_ty = registry.build_type(
    let builtin_costs_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.branch_signatures()[0].vars[0].ty,
        &info.branch_signatures()[0].vars[0].ty,
//    )?;
    )?;
//

//    // TODO: Implement libfunc.
    // TODO: Implement libfunc.
//    let op0 = entry.append_op_result(llvm::undef(builtin_costs_ty, location))?;
    let op0 = entry.append_op_result(llvm::undef(builtin_costs_ty, location))?;
//

//    entry.append_operation(helper.br(0, &[op0], location));
    entry.append_operation(helper.br(0, &[op0], location));
//

//    Ok(())
    Ok(())
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod test {
mod test {
//    use crate::utils::test::{load_cairo, run_program};
    use crate::utils::test::{load_cairo, run_program};
//

//    #[test]
    #[test]
//    fn run_withdraw_gas() {
    fn run_withdraw_gas() {
//        #[rustfmt::skip]
        #[rustfmt::skip]
//        let program = load_cairo!(
        let program = load_cairo!(
//            use gas::withdraw_gas;
            use gas::withdraw_gas;
//

//            fn run_test() {
            fn run_test() {
//                let mut i = 10;
                let mut i = 10;
//

//                loop {
                loop {
//                    if i == 0 {
                    if i == 0 {
//                        break;
                        break;
//                    }
                    }
//

//                    match withdraw_gas() {
                    match withdraw_gas() {
//                        Option::Some(()) => {
                        Option::Some(()) => {
//                            i = i - 1;
                            i = i - 1;
//                        },
                        },
//                        Option::None(()) => {
                        Option::None(()) => {
//                            break;
                            break;
//                        }
                        }
//                    };
                    };
//                    i = i - 1;
                    i = i - 1;
//                }
                }
//            }
            }
//        );
        );
//

//        let result = run_program(&program, "run_test", &[]);
        let result = run_program(&program, "run_test", &[]);
//        assert_eq!(
        assert_eq!(
//            result.remaining_gas,
            result.remaining_gas,
//            Some(340282366920938463463374607431768204835),
            Some(340282366920938463463374607431768204835),
//        );
        );
//    }
    }
//}
}
