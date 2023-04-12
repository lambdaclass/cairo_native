use std::collections::{BTreeMap, HashMap};

use cairo_lang_sierra::program::{GenBranchTarget, Invocation};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{Block, BlockRef, Region};

use crate::{
    compiler::{CmpOp, Compiler, SierraType, Storage},
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

    pub fn inline_int_is_zero(
        &'ctx self,
        name: &str,
        invocation: &Invocation,
        block: &Block<'ctx>,
        variables: &mut HashMap<u64, Variable>,
        blocks: &BTreeMap<usize, BlockInfo<'ctx>>,
        statement_idx: usize,
    ) -> Result<()> {
        let op_zero = match name {
            "u8_is_zero" => self.op_u8_const(block, "0"),
            "u16_is_zero" => self.op_u16_const(block, "0"),
            "u32_is_zero" => self.op_u32_const(block, "0"),
            "u64_is_zero" => self.op_u64_const(block, "0"),
            "u128_is_zero" => self.op_u128_const(block, "0"),
            "felt252_is_zero" => self.op_felt_const(block, "0"),
            _ => panic!("Unexpected is_zero libfunc name {}", name),
        };
        let zero = op_zero.result(0)?.into();

        let input = variables
            .get(&invocation.args[0].id)
            .expect("Variable should be registered before use")
            .get_value();
        let eq_op = self.op_cmp(block, CmpOp::Equal, input, zero);
        let eq = eq_op.result(0)?;

        // X_is_zero forwards its argument to the non-zero branch
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

    #[allow(clippy::too_many_arguments)]
    pub fn inline_enum_match(
        &self,
        id: &str,
        statement_idx: usize,
        region: &Region,
        block: &Block,
        blocks: &BTreeMap<usize, BlockInfo>,
        invocation: &Invocation,
        variables: &HashMap<u64, Variable>,
        storage: &Storage,
    ) -> Result<()> {
        let args = storage.libfuncs.get(id).unwrap().get_args();

        let (tag_type, storage_type, variants_types) = match &args[0].ty {
            SierraType::Enum { tag_type, storage_type, variants_types, .. } => {
                (tag_type, storage_type, variants_types)
            }
            _ => {
                panic!("Argument of enum match should be an enum")
            }
        };

        // Get the argument- the enum to case split upon
        let enum_value = variables.get(&invocation.args[0].id).unwrap().get_value();

        // get the tag
        let tag_value_op = self.op_llvm_extractvalue(block, 0, enum_value, *tag_type)?;
        let tag_value = tag_value_op.result(0)?.into();

        // put the enum's data on the stack and get a pointer to its value
        let data_op = self.op_llvm_extractvalue(block, 1, enum_value, *storage_type)?;
        let data = data_op.result(0)?.into();
        let data_ptr_op = self.op_llvm_alloca(block, *storage_type, 1)?;
        let data_ptr = data_ptr_op.result(0)?.into();
        self.op_llvm_store(block, data, data_ptr)?;

        // Blocks for the switch statement to jump to. Each extract's the appropriate value and forwards it on
        let variant_blocks: Vec<BlockRef> = variants_types
            .iter()
            .enumerate()
            .map(|(variant_index, variant_type)| {
                // Intermediary block in which to extract the correct data from the enum
                let variant_block = region.append_block(Block::new(&[]));
                // Target block to jump to as the result of the match
                let target_block_info = blocks
                    .get(&match invocation.branches[variant_index].target {
                        GenBranchTarget::Fallthrough => statement_idx + 1,
                        GenBranchTarget::Statement(idx) => idx.0,
                    })
                    .unwrap();

                let mut args_to_target_block = vec![];
                for var_idx in target_block_info.variables_at_start.keys().sorted() {
                    if *var_idx == invocation.branches[variant_index].results[0].id {
                        let load_op =
                            self.op_llvm_load(&variant_block, data_ptr, variant_type.get_type())?;
                        args_to_target_block.push(Variable::Local { op: load_op, result_idx: 0 });
                    } else {
                        args_to_target_block.push(*variables.get(var_idx).unwrap());
                    }
                }
                let args_to_target_block =
                    args_to_target_block.iter().map(Variable::get_value).collect_vec();

                self.op_br(&variant_block, &target_block_info.block, &args_to_target_block);

                Ok(variant_block)
            })
            .collect::<Result<Vec<BlockRef>>>()?;

        let case_values = (0..variants_types.len()).map(|x| x.to_string()).collect_vec();
        // The default block is unreachable
        // NOTE To truly guarantee this, we'll need guards on external inputs once we take them
        let default_block = region.append_block(Block::new(&[]));
        self.op_unreachable(&default_block);
        self.op_switch(block, &case_values, tag_value, default_block, &variant_blocks)?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn inline_array_get(
        &self,
        _id: &str,
        _statement_idx: usize,
        _region: &Region,
        _block: &Block,
        _blocks: &BTreeMap<usize, BlockInfo>,
        _invocation: &Invocation,
        _variables: &HashMap<u64, Variable>,
        _storage: &Storage,
    ) -> Result<()> {
        // let _libfuncdef = storage.libfuncs.get(id).unwrap().as_lib_func_def();

        // arg 0 is range check, can ignore
        // arg 1 is the array
        // arg 2 is the index

        todo!();

        // fallthrough if array out of bounds
        // jump if ok

        // Ok(())
    }
}
