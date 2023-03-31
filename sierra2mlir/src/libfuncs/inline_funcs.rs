use std::{
    collections::{BTreeMap, HashMap},
    ops::Deref,
};

use cairo_lang_sierra::program::{GenBranchTarget, Invocation};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{Block, Region, Value, ValueLike};

use crate::{
    compiler::{CmpOp, Compiler, SierraType, Storage},
    libfuncs::lib_func_def::SierraLibFunc,
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
        region: &Region,
        block: &Block<'ctx>,
        storage: &Storage,
        variables: &mut HashMap<u64, Variable<'ctx>>,
        blocks: &BTreeMap<usize, BlockInfo<'ctx>>,
        statement_idx: usize,
    ) -> Result<()> {
        dbg!(invocation);
        let libfunc_id = invocation.libfunc_id.debug_name.as_ref().unwrap().as_str();
        let libfunc = storage.libfuncs.get(libfunc_id).unwrap();

        if let SierraLibFunc::Function(libfunc) = libfunc {
            let enum_type = &libfunc.args[0].ty;

            if let SierraType::Enum {
                ty,
                tag_type,
                storage_bytes_len,
                storage_type,
                variants_types,
            } = enum_type
            {
                dbg!(libfunc);

                // this block should never be reached, only if for some reason the enum tag is invalid.
                // it will call abort()
                let default_block = region.append_block(Block::new(&[]));
                // todo: call abort

                let target_blocks = invocation
                    .branches
                    .iter()
                    .map(|branch| match branch.target {
                        GenBranchTarget::Fallthrough => statement_idx + 1,
                        GenBranchTarget::Statement(idx) => idx.0,
                    })
                    .map(|idx| {
                        let target_block_info = blocks.get(&idx).unwrap();
                        let mut operand_indices =
                            target_block_info.variables_at_start.keys().collect_vec();
                        operand_indices.sort_unstable();
                        let operand_values = operand_indices
                            .iter()
                            .map(|id| variables.get(id).cloned().unwrap())
                            .collect_vec();
                        (&target_block_info.block, operand_values)
                    })
                    .collect_vec();

                let var = variables
                    .get(&invocation.args[0].id)
                    .cloned()
                    .expect("Variable should be registered before use");
                let input_enum = var.get_value();

                // get the tag
                let tag_value_op = self.op_llvm_extractvalue(&block, 0, input_enum, *tag_type)?;
                let tag_value = tag_value_op.result(0)?.into();

                // put the enum on the stack and get a pointer to its value
                let enum_alloca_op = self.op_llvm_alloca(&block, *ty, 1)?;
                let enum_ptr = enum_alloca_op.result(0)?.into();

                self.op_llvm_store(&block, input_enum, enum_ptr)?;

                let value_ptr_op = self.op_llvm_gep(&block, 1, enum_ptr, *ty)?;
                let value_ptr: Value = value_ptr_op.result(0)?.into();

                let case_values =
                    variants_types.iter().enumerate().map(|x| x.0.to_string()).collect_vec();

                self.op_switch(
                    &block,
                    &case_values,
                    tag_value,
                    (default_block.deref(), &[]),
                    //blockrefs.into_iter().map(|x| (x, [].as_slice())).collect_vec().as_slice(),
                    target_blocks
                        .iter()
                        .map(|(b, v)| {
                            (*b, v.iter().map(|x| x.get_value()).collect_vec().as_slice())
                        })
                        .collect_vec()
                        .as_slice(),
                )?;

                for (i, (block, var_ty)) in target_blocks.iter().enumerate() {
                    let value_op =
                        self.op_llvm_load(block, value_ptr, var_ty[0].get_value().r#type())?;
                    variables.insert(
                        invocation.branches[i].results[0].id,
                        Variable::Local { op: value_op, result_idx: 0 },
                    );
                }

                Ok(())
            } else {
                panic!("should be a enum")
            }
        } else {
            panic!("should be a function")
        }
    }
}
