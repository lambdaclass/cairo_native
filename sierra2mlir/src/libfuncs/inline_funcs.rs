use std::collections::{BTreeMap, HashMap};

use cairo_lang_sierra::program::{GenBranchTarget, Invocation};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::Block;

use crate::{
    compiler::{CmpOp, Compiler},
    statements::{BlockInfo, Variable},
};

/*
   Here are the libfuncs implemented inline,
   meaning that they are implemented in place where they are called.

   Mostly control flow related libfuncs.
*/

impl<'ctx> Compiler<'ctx> {
    pub fn inline_jump(
        &'ctx self,
        invocation: &Invocation,
        block: &Block<'ctx>,
        variables: &mut HashMap<u64, Variable>,
        blocks: &BTreeMap<usize, BlockInfo<'ctx>>,
    ) -> Result<()> {
        let target_block_info = match &invocation.branches[0].target {
            GenBranchTarget::Fallthrough => {
                unreachable!("jump should never be fallthrough")
            }
            GenBranchTarget::Statement(id) => blocks.get(&(id.0)).unwrap(),
        };
        let mut operand_indices = target_block_info.variables_at_start.keys().collect_vec();
        operand_indices.sort_unstable();
        let operand_values =
            operand_indices.iter().map(|id| variables.get(id).unwrap().get_value()).collect_vec();
        self.op_br(block, &target_block_info.block, &operand_values);

        Ok(())
    }

    pub fn inline_felt252_is_zero(
        &'ctx self,
        invocation: &Invocation,
        block: &Block<'ctx>,
        variables: &mut HashMap<u64, Variable>,
        blocks: &BTreeMap<usize, BlockInfo<'ctx>>,
        statement_idx: usize,
    ) -> Result<()> {
        let felt_op_zero = self.op_felt_const(block, "0");
        let zero = felt_op_zero.result(0)?.into();

        let input = variables
            .get(&invocation.args[0].id)
            .expect("Variable should be registered before use")
            .get_value();
        let eq_op = self.op_cmp(block, CmpOp::Equal, input, zero);
        let eq = eq_op.result(0)?;

        // felt_is_zero forwards its argument to the non-zero branch
        // Since no processing is done, we can simply assign to the variable here
        variables.insert(
            invocation.branches[1].results[0].id,
            *variables.get(&invocation.args[0].id).unwrap(),
        );
        let target_blocks = invocation
            .branches
            .iter()
            .map(|branch| match branch.target {
                GenBranchTarget::Fallthrough => statement_idx + 1,
                GenBranchTarget::Statement(idx) => idx.0,
            })
            .map(|idx| {
                let target_block_info = blocks.get(&idx).unwrap();
                let mut operand_indices = target_block_info.variables_at_start.keys().collect_vec();
                operand_indices.sort_unstable();
                let operand_values = operand_indices
                    .iter()
                    .map(|id| variables.get(id).unwrap().get_value())
                    .collect_vec();
                (&target_block_info.block, operand_values)
            })
            .collect_vec();

        let (zero_block, zero_vars) = &target_blocks[0];
        let (nonzero_block, nonzero_vars) = &target_blocks[1];

        self.op_cond_br(block, eq.into(), zero_block, nonzero_block, zero_vars, nonzero_vars)?;

        Ok(())
    }

    pub fn inline_enum_match(
        &'ctx self,
        invocation: &Invocation,
        block: &Block<'ctx>,
        variables: &mut HashMap<u64, Variable>,
        blocks: &BTreeMap<usize, BlockInfo<'ctx>>,
        statement_idx: usize,
    ) -> Result<()> {
        todo!();

        Ok(())
    }
}
