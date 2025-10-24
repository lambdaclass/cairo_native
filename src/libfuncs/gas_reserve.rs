use crate::libfuncs::LibfuncHelper;
use crate::{error::Result, metadata::MetadataStorage};
use cairo_lang_sierra::extensions::lib_func::SignatureOnlyConcreteLibfunc;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        gas_reserve::GasReserveConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::dialect::arith::{self, CmpiPredicate};
use melior::helpers::{ArithBlockExt, BuiltinBlockExt};
use melior::ir::r#type::IntegerType;
use melior::{
    ir::{Block, Location},
    Context,
};

pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &GasReserveConcreteLibfunc,
) -> Result<()> {
    match selector {
        GasReserveConcreteLibfunc::Create(info) => {
            build_gas_reserve_create(context, registry, entry, location, helper, metadata, info)
        }
        GasReserveConcreteLibfunc::Utilize(info) => {
            build_gas_reserve_utilize(context, registry, entry, location, helper, metadata, info)
        }
    }
}

fn build_gas_reserve_create<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;
    let current_gas = entry.arg(1)?; // u64
    let amount = entry.arg(2)?; // u128

    let amount_ty = IntegerType::new(context, 128).into();
    let current_gas_128 = entry.append_op_result(arith::extui(current_gas, amount_ty, location))?;
    let enough_gas = entry.cmpi(
        context,
        CmpiPredicate::Uge,
        current_gas_128,
        amount,
        location,
    )?;

    // TODO: Check if this trunci is affecting results
    let gas_builtin_ty = IntegerType::new(context, 64).into();
    let spare_gas = entry.append_op_result(arith::subi(current_gas_128, amount, location))?;
    let spare_gas = entry.append_op_result(arith::trunci(spare_gas, gas_builtin_ty, location))?;

    helper.cond_br(
        context,
        entry,
        enough_gas,
        [0, 1],
        [
            &[range_check, spare_gas, amount],
            &[range_check, current_gas],
        ],
        location,
    )
}

fn build_gas_reserve_utilize<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let current_gas = entry.arg(0)?;
    let gas_reserve = entry.arg(1)?;

    let trunc_reserve = entry.append_op_result(arith::trunci(
        gas_reserve,
        IntegerType::new(context, 64).into(),
        location,
    ))?;
    let updated_gas = entry.append_op_result(arith::addi(current_gas, trunc_reserve, location))?;

    helper.br(entry, 0, &[updated_gas], location)
}

#[cfg(test)]
mod test {
    use crate::{load_cairo, utils::testing::run_program, Value};

    #[test]
    fn run_gas_reserve_create() {
        let program = load_cairo!(
            use core::gas::{GasReserve, gas_reserve_create};

            fn run_test() -> Option<GasReserve> {
                gas_reserve_create(100)
            }
        );

        let result = run_program(&program, "run_test", &[Value::Uint128(1000)]).return_value;
        if let Value::Enum { tag, value, .. } = result {
            assert_eq!(tag, 0);
            assert_eq!(value, Box::new(Value::Sint128(100))) // TODO: Should it return a Sint128 or a Uint128?
        }
    }

    #[test]
    fn run_gas_reserve_utilize() {
        let program = load_cairo!(
            use core::gas::{GasReserve, gas_reserve_create, gas_reserve_utilize};

            fn run_test(gas_quant: u128) -> (u128, u128) {
                let initial_gas = core::testing::get_available_gas();
                let reserve = gas_reserve_create(gas_quant).unwrap();
                gas_reserve_utilize(reserve);
                let final_gas = core::testing::get_available_gas();

                (initial_gas, final_gas)
            }
        );

        let gas_quant = 10;
        let result = run_program(&program, "run_test", &[Value::Uint128(gas_quant)]).return_value;
        assert_eq!(result, Value::Null);
        if let Value::Enum { value, .. } = result {
            // TODO: Do this nicer
            if let Value::Struct { fields, .. } = *value {
                let initial_gas = &fields[0];
                let final_gas = &fields[1];
                if let (Value::Uint128(initial_gas), Value::Uint128(final_gas)) =
                    (initial_gas, final_gas)
                {
                    assert_eq!(initial_gas - gas_quant, *final_gas)
                }
            }
        }
    }
}
