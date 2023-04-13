use std::collections::{BTreeMap, HashMap};

use cairo_lang_sierra::program::{GenBranchTarget, Invocation};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{Block, BlockRef, Region, Value};

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
        let operand_values = target_block_info
            .variables_at_start
            .keys()
            .map(|id| variables.get(id).unwrap().get_value())
            .collect_vec();
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
                let operand_values = target_block_info
                    .variables_at_start
                    .keys()
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
                for var_idx in target_block_info.variables_at_start.keys() {
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
        id: &str,
        statement_idx: usize,
        region: &Region,
        block: &Block,
        blocks: &BTreeMap<usize, BlockInfo>,
        invocation: &Invocation,
        variables: &HashMap<u64, Variable>,
        storage: &Storage,
    ) -> Result<()> {
        let libfunc = storage.libfuncs.get(id).unwrap();
        dbg!(&libfunc);
        let array_arg = &libfunc.get_args()[0];
        let index_arg = &libfunc.get_args()[1];

        // fallthrough if ok
        // jump if panic

        // 0 = ok block, 1 = panic block
        let target_blocks = invocation
            .branches
            .iter()
            .map(|branch| match branch.target {
                GenBranchTarget::Fallthrough => statement_idx + 1,
                GenBranchTarget::Statement(idx) => idx.0,
            })
            .map(|idx| {
                let target_block_info = blocks.get(&idx).unwrap();
                target_block_info
            })
            .collect_vec();

        let target_block_info = target_blocks[0];
        let panic_block_info = target_blocks[1];

        if let SierraType::Array { ty: _, len_type, element_type } = &array_arg.ty {
            // arg 0 is range check, can ignore
            // arg 1 is the array
            // arg 2 is the index

            let array_var = variables
                .get(&invocation.args[array_arg.loc].id)
                .expect("variable array should exist");
            let index_var = variables
                .get(&invocation.args[index_arg.loc].id)
                .expect("variable index should exist");

            let array_value = array_var.get_value();

            // get the current length
            let length_op = self.op_llvm_extractvalue(block, 0, array_value, *len_type)?;
            let length: Value = length_op.result(0)?.into();

            // check if index is out of bounds
            let cmp_op = self.op_cmp(block, CmpOp::UnsignedLess, index_var.get_value(), length);
            let cmp = cmp_op.result(0)?.into();

            let block_get_idx = region.append_block(Block::new(&[]));

            // collect args to the panic block
            let mut args_to_panic_block = vec![];
            for var_idx in panic_block_info.variables_at_start.keys().sorted() {
                args_to_panic_block.push(*variables.get(var_idx).unwrap());
            }
            let args_to_panic_block =
                args_to_panic_block.iter().map(Variable::get_value).collect_vec();

            self.op_cond_br(
                block,
                cmp,
                &block_get_idx,
                &panic_block_info.block,
                &[],
                &args_to_panic_block,
            )?;

            // get the value at index

            let data_ptr_op =
                self.op_llvm_extractvalue(&block_get_idx, 2, array_value, self.llvm_ptr_type())?;
            let data_ptr: Value = data_ptr_op.result(0)?.into();
            // get the pointer to the data index
            let value_ptr_op = self.op_llvm_gep_dynamic(
                &block_get_idx,
                &[index_var.get_value()],
                data_ptr,
                element_type.get_type(),
            )?;
            let value_ptr = value_ptr_op.result(0)?.into();

            let target_value_var_id = invocation.branches[0].results[1].id;

            // get the args to the target block (fallthrough here)
            let mut args_to_target_block = vec![];
            for var_idx in target_block_info.variables_at_start.keys().sorted() {
                if *var_idx == target_value_var_id {
                    let value_load_op =
                        self.op_llvm_load(&block_get_idx, value_ptr, element_type.get_type())?;
                    args_to_target_block.push(Variable::Local { op: value_load_op, result_idx: 0 });
                } else {
                    args_to_target_block.push(*variables.get(var_idx).unwrap());
                }
            }
            let args_to_target_block =
                args_to_target_block.iter().map(Variable::get_value).collect_vec();

            self.op_br(&block_get_idx, &target_block_info.block, &args_to_target_block);

            Ok(())
        } else {
            panic!("argument should be array type");
        }
    }
}
