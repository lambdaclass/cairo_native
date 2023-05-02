use cairo_lang_sierra::{extensions::gas::CostTokenType, program::StatementIdx};
use color_eyre::Result;
use melior_next::ir::Block;

use crate::compiler::Compiler;

/*
   Here are the non control flow libfuncs implemented inline,
   meaning that they are implemented in place where they are called.


*/

impl<'ctx> Compiler<'ctx> {
    /// Must be implemented inline due to it depending on the statement idx to get the gas count.
    pub fn inline_branch_align(&'ctx self, statement_idx: usize, block: &Block) -> Result<()> {
        if let Some(gas) = &self.gas {
            // branch_align equalizes gas usage across branches.

            // get the requested amount of gas.
            let requested_gas_count: i64 = gas
                .gas_info
                .variable_values
                .get(&(StatementIdx(statement_idx), CostTokenType::Const))
                .copied()
                .unwrap();

            if requested_gas_count != 0 {
                let requested_gas_op =
                    self.op_const(block, &requested_gas_count.to_string(), self.u128_type());
                let gas_value = requested_gas_op.result(0)?.into();
                self.call_decrease_gas_counter(block, gas_value)?;
            }
        }

        Ok(())
    }
}
