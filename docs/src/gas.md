# Gas Calculations

Gas costs are calculated using methods provided by the `cairo_lang_sierra_gas` crate.

If the generated sierra program doesn't end up using any libfunc which requires the `GasBuiltin`, cairo-runner won't return the remaining gas.

Gas is only spent:
- Upfront (before the program executes) by getting the entry point and using the crate methods to get the needed gas.
- In the withdraw_gas, withdraw_gas_all, redeposit_gas libfuncs, the cost is known and is located on the GasInfo structure.
