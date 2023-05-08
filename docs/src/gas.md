# Gas Calculations

Gas costs are calculated using methods provided by the `cairo_lang_sierra_gas` crate.

## Findings

- The entry point gas cost is spent from the available gas before executing.
- withdraw_gas is the only runtime libfunc that spends the gas
