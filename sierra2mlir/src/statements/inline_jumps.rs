use std::{
    collections::{BTreeMap, HashMap},
    ops::{Deref, Shl},
};

use cairo_lang_sierra::{
    extensions::gas::CostTokenType,
    program::{GenBranchTarget, Invocation, StatementIdx},
};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::{
    dialect::cf,
    ir::{operation, Block, BlockRef, Location, NamedAttribute, Region, Type, Value, ValueLike},
};
use num_bigint::BigUint;
use num_traits::FromPrimitive;

use crate::compiler::mlir_ops::CmpOp;
use crate::{
    compiler::{Compiler, Storage},
    sierra_type::SierraType,
    statements::{BlockInfo, Variable},
};

/*
   Here are the control flow libfuncs implemented inline,
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

        self.op_cond_br(block, eq.into(), zero_block, nonzero_block, zero_vars, nonzero_vars);

        Ok(())
    }

    // eq, le, lt
    pub fn inline_int_cmpop(
        &'ctx self,
        id: &str,
        invocation: &Invocation,
        block: &Block<'ctx>,
        variables: &HashMap<u64, Variable>,
        blocks: &BTreeMap<usize, BlockInfo<'ctx>>,
        statement_idx: usize,
        storage: &Storage,
        cmpop: CmpOp,
    ) -> Result<()> {
        let libfunc = storage.libfuncs.get(id).unwrap();
        let pos_arg_1 = &libfunc.get_args()[0];
        let pos_arg_2 = &libfunc.get_args()[1];

        let arg1 = variables
            .get(&invocation.args[pos_arg_1.loc].id)
            .expect("Variable should be registered before use")
            .get_value();

        let arg2 = variables
            .get(&invocation.args[pos_arg_2.loc].id)
            .expect("Variable should be registered before use")
            .get_value();

        let eq_op = self.op_cmp(block, cmpop, arg1, arg2);
        let eq = eq_op.result(0)?;

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

        let (true_block, true_vars) = &target_blocks[1];
        let (false_block, false_vars) = &target_blocks[0];

        self.op_cond_br(block, eq.into(), true_block, false_block, true_vars, false_vars);

        Ok(())
    }

    // https://github.com/starkware-libs/cairo/blob/main/crates/cairo-lang-sierra/src/extensions/modules/uint.rs#L339
    pub fn inline_int_overflowing_op(
        &'ctx self,
        id: &str,
        invocation: &Invocation,
        block: &Block<'ctx>,
        variables: &HashMap<u64, Variable>,
        blocks: &BTreeMap<usize, BlockInfo<'ctx>>,
        statement_idx: usize,
        storage: &Storage,
        // true = add, false = sub
        is_add: bool,
    ) -> Result<()> {
        let libfunc = storage.libfuncs.get(id).unwrap();
        let pos_arg_1 = &libfunc.get_args()[0];
        let pos_arg_2 = &libfunc.get_args()[1];

        let arg_type = pos_arg_1.ty.clone();

        let arg1 = variables
            .get(&invocation.args[pos_arg_1.loc].id)
            .expect("Variable should be registered before use")
            .get_value();

        let arg2 = variables
            .get(&invocation.args[pos_arg_2.loc].id)
            .expect("Variable should be registered before use")
            .get_value();

        let overflow_result_op = block.append_operation(
            operation::Builder::new(
                if is_add {
                    "llvm.intr.uadd.with.overflow"
                } else {
                    "llvm.intr.usub.with.overflow"
                },
                Location::unknown(&self.context),
            )
            .add_operands(&[arg1, arg2])
            .add_results(&[self.llvm_struct_type(&[arg1.r#type(), self.bool_type()], false)])
            .build(),
        );

        // {iN, i1}
        let overflow_result = overflow_result_op.result(0)?.into();
        let result_op =
            self.op_llvm_extractvalue(block, 0, overflow_result, arg_type.get_type())?;
        let overflow_op = self.op_llvm_extractvalue(block, 1, overflow_result, self.bool_type())?;

        let result = result_op.result(0)?.into();

        let overflow = overflow_op.result(0)?;

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
                    .map(|id| {
                        if *id == invocation.branches[0].results[1].id
                            || *id == invocation.branches[1].results[1].id
                        {
                            result
                        } else {
                            variables.get(id).unwrap().get_value()
                        }
                    })
                    .collect_vec();
                (&target_block_info.block, operand_values)
            })
            .collect_vec();

        let (true_block, true_vars) = &target_blocks[1];
        let (false_block, false_vars) = &target_blocks[0];

        self.op_cond_br(block, overflow.into(), true_block, false_block, true_vars, false_vars);

        Ok(())
    }

    pub fn inline_try_from_felt252(
        &'ctx self,
        id: &str,
        invocation: &Invocation,
        region: &Region,
        block: &Block<'ctx>,
        variables: &HashMap<u64, Variable>,
        blocks: &BTreeMap<usize, BlockInfo<'ctx>>,
        statement_idx: usize,
        storage: &Storage,
    ) -> Result<()> {
        let libfunc = storage.libfuncs.get(id).expect("should find libfunc");
        let pos_arg_1 = &libfunc.get_args()[0]; // always felt type
        let ret_pos_type = &libfunc.get_return_types()[0][0];
        let ret_type = &ret_pos_type.ty;

        let arg_type = &pos_arg_1.ty;
        let arg = variables
            .get(&invocation.args[pos_arg_1.loc].id)
            .expect("Variable should be registered before use")
            .get_value();

        let max_value = BigUint::from_i32(1).unwrap().shl(ret_type.get_width());
        let max_val = self.op_const(block, &max_value.to_string(), arg_type.get_type());

        let cmp_op = self.op_cmp(block, CmpOp::UnsignedLessThan, arg, max_val.result(0)?.into());
        let cmp = cmp_op.result(0)?.into();

        let trunc_block = region.append_block(Block::new(&[]));

        // truncate block
        let trunc_op = self.op_trunc(&trunc_block, arg, ret_type.get_type());
        let trunc_res = trunc_op.result(0)?.into();

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
                    .map(|id| {
                        if *id == invocation.branches[0].results[1].id {
                            trunc_res
                        } else {
                            variables.get(id).unwrap().get_value()
                        }
                    })
                    .collect_vec();
                (&target_block_info.block, operand_values)
            })
            .collect_vec();

        let (true_block, true_vars) = &target_blocks[0];
        let (false_block, false_vars) = &target_blocks[1];

        self.op_cond_br(block, cmp, &trunc_block, false_block, &[], false_vars);
        self.op_br(&trunc_block, true_block, true_vars);

        Ok(())
    }

    pub fn inline_enum_match(
        &'ctx self,
        id: &str,
        statement_idx: usize,
        region: &Region,
        block: &Block,
        blocks: &BTreeMap<usize, BlockInfo>,
        invocation: &Invocation,
        variables: &HashMap<u64, Variable>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let args = storage.libfuncs.get(id).unwrap().get_args();

        let enum_type = &args[0].ty;
        let enum_name = id.strip_prefix("enum_match<").unwrap().strip_suffix('>').unwrap();

        let variant_count = match enum_type {
            SierraType::Enum { variants_types, .. } => variants_types.len(),
            _ => panic!("Argument of enum match should be an enum"),
        };

        // Get the argument- the enum to case split upon
        let enum_value = variables.get(&invocation.args[0].id).unwrap().get_value();

        // get the tag
        let tag_op = self.call_enum_get_tag(block, enum_value, enum_type, storage)?;
        let tag = tag_op.result(0)?.into();

        // Blocks for the switch statement to jump to. Each extract's the appropriate value and forwards it on
        let variant_blocks: Vec<BlockRef> = (0..variant_count)
            .map(|variant_index| {
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
                        let get_data_op = self.call_enum_get_data_as_variant_type(
                            &variant_block,
                            enum_name,
                            enum_value,
                            enum_type,
                            variant_index,
                            storage,
                        )?;
                        args_to_target_block
                            .push(Variable::Local { op: get_data_op, result_idx: 0 });
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

        let case_values = (0..variant_count).map(|x| x.to_string()).collect_vec();
        // The default block is unreachable
        // NOTE To truly guarantee this, we'll need guards on external inputs once we take them
        let default_block = region.append_block(Block::new(&[]));
        self.op_unreachable(&default_block);

        let variant_blocks_with_ops =
            variant_blocks.iter().map(|x| (x.deref(), [].as_slice())).collect_vec();
        block.append_operation(cf::switch(
            &self.context,
            &case_values,
            tag,
            (&default_block, &[]),
            variant_blocks_with_ops.as_slice(),
            Location::unknown(&self.context),
        ));

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn inline_array_get(
        &'ctx self,
        id: &str,
        statement_idx: usize,
        region: &Region,
        block: &Block,
        blocks: &BTreeMap<usize, BlockInfo>,
        invocation: &Invocation,
        variables: &HashMap<u64, Variable>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let libfunc = storage.libfuncs.get(id).unwrap();
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

        // Our implementation of array_get takes two values: the array and the index (sierra's array_get also takes RangeCheck)

        let array_value = variables
            .get(&invocation.args[array_arg.loc].id)
            .expect("variable array should exist")
            .get_value();
        let index_value = variables
            .get(&invocation.args[index_arg.loc].id)
            .expect("variable index should exist")
            .get_value();

        // get the current length
        let length_op = self.call_array_len_impl(block, array_value, &array_arg.ty, storage)?;
        let length: Value = length_op.result(0)?.into();

        // check if index is out of bounds
        let in_bounds_op = self.op_cmp(block, CmpOp::UnsignedLessThan, index_value, length);
        let in_bounds = in_bounds_op.result(0)?.into();

        // Create a block in which to get the element once we know its index is valid
        let in_bounds_block = region.append_block(Block::new(&[]));

        // collect args to the panic block
        let args_to_panic_block = panic_block_info
            .variables_at_start
            .keys()
            .map(|var_idx| variables.get(var_idx).unwrap().get_value())
            .collect_vec();

        // Jump to the in_bounds_block if the index is valid, or to the panic_block if not
        self.op_cond_br(
            block,
            in_bounds,
            &in_bounds_block,
            &panic_block_info.block,
            &[],
            &args_to_panic_block,
        );

        // get the value at index

        let element_op = self.call_array_get_unchecked(
            &in_bounds_block,
            array_value,
            index_value,
            &array_arg.ty,
            storage,
        )?;

        let target_value_var_id = invocation.branches[0].results[1].id;

        // get the args to the target block (fallthrough here)
        let args_to_target_block = target_block_info
            .variables_at_start
            .keys()
            .map(|var_idx| {
                if *var_idx == target_value_var_id {
                    Variable::Local { op: element_op, result_idx: 0 }
                } else {
                    *variables.get(var_idx).unwrap()
                }
            })
            .collect_vec();
        let args_to_target_block =
            args_to_target_block.iter().map(Variable::get_value).collect_vec();

        self.op_br(&in_bounds_block, &target_block_info.block, &args_to_target_block);

        Ok(())
    }

    pub fn inline_array_pop_front(
        &'ctx self,
        id: &str,
        statement_idx: usize,
        region: &Region,
        block: &Block,
        blocks: &BTreeMap<usize, BlockInfo>,
        invocation: &Invocation,
        variables: &HashMap<u64, Variable>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let libfunc = storage.libfuncs.get(id).unwrap();
        let array_arg = &libfunc.get_args()[0];

        // fallthrough if there is a element to pop
        // jump otherwise

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

        let some_block_info = target_blocks[0];
        let none_block_info = target_blocks[1];

        let (array_type, element_type) = match &array_arg.ty {
            SierraType::Array { ty, len_type: _, element_type } => (*ty, element_type),
            _ => panic!("argument should be array type"),
        };

        let array_var =
            variables.get(&invocation.args[array_arg.loc].id).expect("variable array should exist");

        let array_value = array_var.get_value();

        let const_1_op = self.op_u32_const(block, "1");
        let const_1 = const_1_op.result(0)?.into();
        let const_0_op = self.op_u32_const(block, "0");
        let const_0 = const_0_op.result(0)?.into();

        // get the current length
        let length_op = self.call_array_len_impl(block, array_value, &array_arg.ty, storage)?;
        let length: Value = length_op.result(0)?.into();

        // check if there is something to pop
        let cmp_op = self.op_cmp(block, CmpOp::UnsignedGreaterThanEqual, length, const_1);
        let cmp = cmp_op.result(0)?.into();

        let block_pop_idx = region.append_block(Block::new(&[]));

        // collect args to the none block
        let args_to_none_block = none_block_info
            .variables_at_start
            .keys()
            .map(|var_idx| {
                if *var_idx == invocation.branches[1].results[0].id {
                    *array_var
                } else {
                    *variables.get(var_idx).unwrap()
                }
            })
            .collect_vec();

        let args_to_none_block = args_to_none_block.iter().map(Variable::get_value).collect_vec();

        self.op_cond_br(
            block,
            cmp,
            &block_pop_idx,
            &none_block_info.block,
            &[],
            &args_to_none_block,
        );

        // get the element to return
        let first_element_op = self.call_array_get_unchecked(
            &block_pop_idx,
            array_value,
            const_0,
            &array_arg.ty,
            storage,
        )?;

        // decrement the length
        let new_length_op = self.op_sub(&block_pop_idx, length, const_1);
        let new_length = new_length_op.result(0)?.into();

        // get the current data ptr, and the pointer to the second element
        let data_ptr_op =
            self.op_llvm_extractvalue(&block_pop_idx, 2, array_value, self.llvm_ptr_type())?;
        let data_ptr: Value = data_ptr_op.result(0)?.into();
        let src_ptr_op = self.op_llvm_gep_dynamic(
            &block_pop_idx,
            &[const_1],
            data_ptr,
            element_type.get_type(),
        )?;
        let src_ptr = src_ptr_op.result(0)?.into();

        // its safe if new_length is 0
        let new_length_zext_op = self.op_zext(&block_pop_idx, new_length, self.u64_type());
        let new_length_zext = new_length_zext_op.result(0)?.into();

        let element_size_bytes = (element_type.get_width() + 7) / 8;
        let const_element_size_bytes =
            self.op_const(&block_pop_idx, &element_size_bytes.to_string(), self.u64_type());

        let new_length_bytes_op = self.op_mul(
            &block_pop_idx,
            new_length_zext,
            const_element_size_bytes.result(0)?.into(),
        );
        let new_length_bytes = new_length_bytes_op.result(0)?.into();

        let dst_ptr_op =
            self.call_memmove(&block_pop_idx, data_ptr, src_ptr, new_length_bytes, storage)?;
        let dst_ptr: Value = dst_ptr_op.result(0)?.into();

        // insert new length
        let insert_op =
            self.op_llvm_insertvalue(&block_pop_idx, 0, array_value, new_length, array_type)?;
        let array_value: Value = insert_op.result(0)?.into();

        // insert new ptr
        let insert_array_op =
            self.op_llvm_insertvalue(&block_pop_idx, 2, array_value, dst_ptr, array_type)?;

        // get the args to the target block (fallthrough here)
        let args_to_target_block = some_block_info
            .variables_at_start
            .keys()
            .map(|var_idx| {
                let array_value_id = invocation.branches[0].results[0].id;
                let popped_value_id = invocation.branches[0].results[1].id;
                match *var_idx {
                    // popped value
                    var_idx if var_idx == popped_value_id => {
                        Variable::Local { op: first_element_op, result_idx: 0 }
                    }
                    // updated array
                    var_idx if var_idx == array_value_id => {
                        Variable::Local { op: insert_array_op, result_idx: 0 }
                    }
                    var_idx => *variables.get(&var_idx).unwrap(),
                }
            })
            .collect_vec();
        let args_to_target_block =
            args_to_target_block.iter().map(Variable::get_value).collect_vec();

        self.op_br(&block_pop_idx, &some_block_info.block, &args_to_target_block);

        Ok(())
    }

    pub fn inline_downcast(
        &'ctx self,
        id: &str,
        invocation: &Invocation,
        region: &Region,
        block: &Block<'ctx>,
        variables: &HashMap<u64, Variable>,
        blocks: &BTreeMap<usize, BlockInfo<'ctx>>,
        statement_idx: usize,
        storage: &Storage,
    ) -> Result<()> {
        let libfunc = storage.libfuncs.get(id).expect("should find libfunc");
        let pos_arg_1 = &libfunc.get_args()[0];
        let ret_pos_type = &libfunc.get_return_types()[0][0];
        let ret_type = &ret_pos_type.ty;

        let arg_type = &pos_arg_1.ty;
        let arg = variables
            .get(&invocation.args[pos_arg_1.loc].id)
            .expect("Variable should be registered before use")
            .get_value();

        let cmp_op = if arg_type.get_width() == ret_type.get_width() {
            self.op_const(block, "1", self.bool_type())
        } else {
            let max_val = self.op_const(
                block,
                &(BigUint::from_i32(1).unwrap().shl(ret_type.get_width())).to_string(),
                arg_type.get_type(),
            );

            let cmp_op =
                self.op_cmp(block, CmpOp::UnsignedLessThan, arg, max_val.result(0)?.into());
            cmp_op
        };
        let cmp = cmp_op.result(0)?.into();

        let trunc_block = region.append_block(Block::new(&[]));

        // truncate block
        let trunc_op = self.op_trunc(&trunc_block, arg, ret_type.get_type());
        let trunc_res = trunc_op.result(0)?.into();

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
                    .map(|id| {
                        if *id == invocation.branches[0].results[1].id {
                            trunc_res
                        } else {
                            variables.get(id).unwrap().get_value()
                        }
                    })
                    .collect_vec();
                (&target_block_info.block, operand_values)
            })
            .collect_vec();

        let (true_block, true_vars) = &target_blocks[0];
        let (false_block, false_vars) = &target_blocks[1];

        self.op_cond_br(block, cmp, &trunc_block, false_block, &[], false_vars);
        self.op_br(&trunc_block, true_block, true_vars);

        Ok(())
    }

    pub fn inline_match_nullable(
        &'ctx self,
        id: &str,
        statement_idx: usize,
        block: &Block,
        blocks: &BTreeMap<usize, BlockInfo>,
        invocation: &Invocation,
        variables: &HashMap<u64, Variable>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let libfunc = storage.libfuncs.get(id).unwrap();
        let nullable_arg = &libfunc.get_args()[0];

        // fallthrough if null
        // jump otherwise

        let target_blocks = invocation
            .branches
            .iter()
            .map(|branch| match &branch.target {
                GenBranchTarget::Fallthrough => statement_idx + 1,
                GenBranchTarget::Statement(idx) => idx.0,
            })
            .map(|idx| {
                let target_block_info = blocks.get(&idx).unwrap();
                target_block_info
            })
            .collect_vec();

        let fallthrough_block = target_blocks[0];
        let notnull_block = target_blocks[1];

        let nullable_value_type = nullable_arg.ty.get_field_types().unwrap()[0];

        let nullable_value =
            variables.get(&invocation.args[nullable_arg.loc].id).expect("variable should exist");

        let is_null_op =
            self.op_llvm_extractvalue(block, 1, nullable_value.get_value(), self.bool_type())?;
        let is_null = is_null_op.result(0)?.into();

        // collect args to the none block
        let args_to_fallthrough = fallthrough_block
            .variables_at_start
            .keys()
            .map(|var_idx| *variables.get(var_idx).unwrap())
            .collect_vec();

        let get_value_op =
            self.op_llvm_extractvalue(block, 0, nullable_value.get_value(), nullable_value_type)?;

        // collect args to the none block
        let args_to_notnull = notnull_block
            .variables_at_start
            .keys()
            .map(|var_idx| {
                if *var_idx == invocation.branches[1].results[0].id {
                    Variable::Local { op: get_value_op, result_idx: 0 }
                } else {
                    *variables.get(var_idx).unwrap()
                }
            })
            .collect_vec();

        let args_to_fallthrough = args_to_fallthrough.iter().map(Variable::get_value).collect_vec();
        let args_to_notnull = args_to_notnull.iter().map(Variable::get_value).collect_vec();

        self.op_cond_br(
            block,
            is_null,
            &notnull_block.block,
            &fallthrough_block.block,
            &args_to_notnull,
            &args_to_fallthrough,
        );

        Ok(())
    }

    pub fn inline_withdraw_gas(
        &'ctx self,
        statement_idx: usize,
        block: &Block,
        blocks: &BTreeMap<usize, BlockInfo>,
        invocation: &Invocation,
        variables: &HashMap<u64, Variable>,
    ) -> Result<()> {
        let gas = self.gas.as_ref().expect("gas should exist on a gas related libfunc");

        // fallthrough success
        // jump on failure

        let target_blocks = invocation
            .branches
            .iter()
            .map(|branch| match &branch.target {
                GenBranchTarget::Fallthrough => statement_idx + 1,
                GenBranchTarget::Statement(idx) => idx.0,
            })
            .map(|idx| {
                let target_block_info = blocks.get(&idx).unwrap();
                target_block_info
            })
            .collect_vec();

        let success_block = target_blocks[1];
        let failure_block = target_blocks[0];

        // get the requested amount of gas.
        let requested_gas_count: i64 = gas
            .gas_info
            .variable_values
            .get(&(StatementIdx(statement_idx), CostTokenType::Const))
            .copied()
            .unwrap();

        // check if there is enough.
        let requested_gas_op =
            self.op_const(block, &requested_gas_count.to_string(), self.u128_type());
        let requested_gas_value = requested_gas_op.result(0)?.into();
        let has_enough_op = self.call_has_enough_gas(block, requested_gas_value)?;

        // jump condition
        let is_success = has_enough_op.result(0)?.into();

        // collect args to the none block
        let args_to_success_block = success_block
            .variables_at_start
            .keys()
            .map(|var_idx| *variables.get(var_idx).unwrap())
            .collect_vec();

        // collect args to the none block
        let args_to_failure_block = failure_block
            .variables_at_start
            .keys()
            .map(|var_idx| *variables.get(var_idx).unwrap())
            .collect_vec();

        let args_to_success_block =
            args_to_success_block.iter().map(Variable::get_value).collect_vec();
        let args_to_failure_block =
            args_to_failure_block.iter().map(Variable::get_value).collect_vec();

        self.op_cond_br(
            block,
            is_success,
            &failure_block.block,
            &success_block.block,
            &args_to_failure_block,
            &args_to_success_block,
        );

        Ok(())
    }

    pub fn inline_ec_point_try_new_nz(
        &'ctx self,
        id: &str,
        invocation: &Invocation,
        region: &Region,
        block: &Block<'ctx>,
        variables: &HashMap<u64, Variable>,
        blocks: &BTreeMap<usize, BlockInfo<'ctx>>,
        statement_idx: usize,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        self.create_utils(storage)?;

        let libfunc = storage.libfuncs.get(id).expect("should find libfunc");
        let arg0 = &libfunc.get_args()[0];
        let arg1 = &libfunc.get_args()[1];

        let branch_block_idx = match invocation.branches[1].target {
            GenBranchTarget::Fallthrough => statement_idx + 1,
            GenBranchTarget::Statement(x) => x.0,
        };
        let branch_block = &blocks[&branch_block_idx].block;
        let cont_block = region.append_block(Block::new(&[]));

        let arg0 = variables
            .get(&invocation.args[arg0.loc].id)
            .expect("variable should be registered before use")
            .get_value();
        let arg1 = variables
            .get(&invocation.args[arg1.loc].id)
            .expect("variable should be registered before use")
            .get_value();

        let op0 = block.append_operation(
            operation::Builder::new("memref.alloca", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "operand_segment_sizes",
                    "array<i32: 0, 0>",
                )
                .unwrap()])
                .add_results(&[Type::parse(&self.context, "memref<32xi8>").unwrap()])
                .build(),
        );
        let op1 = block.append_operation(
            operation::Builder::new("memref.alloca", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "operand_segment_sizes",
                    "array<i32: 0, 0>",
                )
                .unwrap()])
                .add_results(&[Type::parse(&self.context, "memref<32xi8>").unwrap()])
                .build(),
        );

        let op2 = block.append_operation(
            operation::Builder::new("index.constant", Location::unknown(&self.context))
                .add_attributes(&[
                    NamedAttribute::new_parsed(&self.context, "value", "0 : index").unwrap()
                ])
                .add_results(&[Type::index(&self.context)])
                .build(),
        );
        let op3 = block.append_operation(
            operation::Builder::new("memref.view", Location::unknown(&self.context))
                .add_operands(&[op0.result(0)?.into(), op2.result(0)?.into()])
                .add_results(&[Type::parse(&self.context, "memref<i256>").unwrap()])
                .build(),
        );
        let op4 = block.append_operation(
            operation::Builder::new("memref.view", Location::unknown(&self.context))
                .add_operands(&[op1.result(0)?.into(), op2.result(0)?.into()])
                .add_results(&[Type::parse(&self.context, "memref<i256>").unwrap()])
                .build(),
        );

        let op5 = block.append_operation(
            operation::Builder::new("llvm.call_intrinsic", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "intrin",
                    "\"llvm.bswap.i256\"",
                )
                .unwrap()])
                .add_operands(&[arg0])
                .add_results(&[self.felt_type()])
                .build(),
        );
        let op6 = block.append_operation(
            operation::Builder::new("llvm.call_intrinsic", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "intrin",
                    "\"llvm.bswap.i256\"",
                )
                .unwrap()])
                .add_operands(&[arg1])
                .add_results(&[self.felt_type()])
                .build(),
        );
        block.append_operation(
            operation::Builder::new("memref.store", Location::unknown(&self.context))
                .add_operands(&[op5.result(0)?.into(), op3.result(0)?.into()])
                .build(),
        );
        block.append_operation(
            operation::Builder::new("memref.store", Location::unknown(&self.context))
                .add_operands(&[op6.result(0)?.into(), op4.result(0)?.into()])
                .build(),
        );

        let op7 = block.append_operation(
            operation::Builder::new(
                "memref.extract_aligned_pointer_as_index",
                Location::unknown(&self.context),
            )
            .add_operands(&[op0.result(0)?.into()])
            .add_results(&[Type::index(&self.context)])
            .build(),
        );
        let op8 = block.append_operation(
            operation::Builder::new(
                "memref.extract_aligned_pointer_as_index",
                Location::unknown(&self.context),
            )
            .add_operands(&[op1.result(0)?.into()])
            .add_results(&[Type::index(&self.context)])
            .build(),
        );
        let op9 = block.append_operation(
            operation::Builder::new("index.castu", Location::unknown(&self.context))
                .add_operands(&[op7.result(0)?.into()])
                .add_results(&[Type::integer(&self.context, 64)])
                .build(),
        );
        let op10 = block.append_operation(
            operation::Builder::new("index.castu", Location::unknown(&self.context))
                .add_operands(&[op8.result(0)?.into()])
                .add_results(&[Type::integer(&self.context, 64)])
                .build(),
        );
        let op11 = block.append_operation(
            operation::Builder::new("llvm.inttoptr", Location::unknown(&self.context))
                .add_operands(&[op9.result(0)?.into()])
                .add_results(&[self.llvm_ptr_type()])
                .build(),
        );
        let op12 = block.append_operation(
            operation::Builder::new("llvm.inttoptr", Location::unknown(&self.context))
                .add_operands(&[op10.result(0)?.into()])
                .add_results(&[self.llvm_ptr_type()])
                .build(),
        );
        let op13 = block.append_operation(
            operation::Builder::new("func.call", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "callee",
                    "@sierra2mlir_util_ec_point_try_new_nz",
                )?])
                .add_operands(&[op11.result(0)?.into(), op12.result(0)?.into()])
                .add_results(&[self.i32_type()])
                .build(),
        );

        let op14 = self.op_u32_const(block, "0");
        let op15 = self.op_cmp(block, CmpOp::Equal, op13.result(0)?.into(), op14.result(0)?.into());
        self.op_cond_br(
            block,
            op15.result(0)?.into(),
            &cont_block,
            branch_block,
            &[],
            &blocks[&branch_block_idx]
                .variables_at_start
                .keys()
                .map(|x| variables[x].get_value())
                .collect::<Vec<_>>(),
        );

        let op16 = cont_block.append_operation(
            operation::Builder::new("llvm.mlir.undef", Location::unknown(&self.context))
                .add_results(&[
                    Type::parse(&self.context, "!llvm.struct<packed (i256, i256, i1)>").unwrap()
                ])
                .build(),
        );
        let op17 = cont_block.append_operation(
            operation::Builder::new("llvm.insertvalue", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "position",
                    "array<i64: 0>",
                )?])
                .add_operands(&[op16.result(0)?.into(), arg0])
                .add_results(&[
                    Type::parse(&self.context, "!llvm.struct<packed (i256, i256, i1)>").unwrap()
                ])
                .build(),
        );
        let op18 = cont_block.append_operation(
            operation::Builder::new("llvm.insertvalue", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "position",
                    "array<i64: 1>",
                )?])
                .add_operands(&[op17.result(0)?.into(), arg1])
                .add_results(&[
                    Type::parse(&self.context, "!llvm.struct<packed (i256, i256, i1)>").unwrap()
                ])
                .build(),
        );
        let op19 = cont_block.append_operation(
            operation::Builder::new("llvm.insertvalue", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "position",
                    "array<i64: 2>",
                )?])
                .add_operands(&[op18.result(0)?.into(), op15.result(0)?.into()])
                .add_results(&[
                    Type::parse(&self.context, "!llvm.struct<packed (i256, i256, i1)>").unwrap()
                ])
                .build(),
        );

        let next_block_idx = match invocation.branches[0].target {
            GenBranchTarget::Fallthrough => statement_idx + 1,
            GenBranchTarget::Statement(x) => x.0,
        };
        let next_block = &blocks[&next_block_idx].block;
        let next_block_args = blocks[&next_block_idx]
            .variables_at_start
            .keys()
            .map(|id| {
                if *id == invocation.branches[0].results[0].id {
                    op19.result(0).unwrap().into()
                } else {
                    variables.get(id).unwrap().get_value()
                }
            })
            .collect::<Vec<_>>();

        self.op_br(&cont_block, next_block, &next_block_args);

        Ok(())
    }

    pub fn inline_ec_point_from_x_nz(
        &'ctx self,
        id: &str,
        invocation: &Invocation,
        region: &Region,
        block: &Block<'ctx>,
        variables: &HashMap<u64, Variable>,
        blocks: &BTreeMap<usize, BlockInfo<'ctx>>,
        statement_idx: usize,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        self.create_utils(storage)?;

        let libfunc = storage.libfuncs.get(id).expect("should find libfunc");
        let arg = &libfunc.get_args()[0];

        let branch_block_idx = match invocation.branches[1].target {
            GenBranchTarget::Fallthrough => statement_idx + 1,
            GenBranchTarget::Statement(x) => x.0,
        };
        let branch_block = &blocks[&branch_block_idx].block;
        let cont_block = region.append_block(Block::new(&[]));

        let arg = variables
            .get(&invocation.args[arg.loc].id)
            .expect("Variable should be registered before use")
            .get_value();

        let op = self.op_felt_const(block, "0");
        let op = self.op_cmp(block, CmpOp::Equal, arg, op.result(0)?.into());
        self.op_cond_br(
            block,
            op.result(0)?.into(),
            branch_block,
            &cont_block,
            &blocks[&branch_block_idx]
                .variables_at_start
                .keys()
                .map(|x| variables[x].get_value())
                .collect::<Vec<_>>(),
            &[],
        );

        let op0 = cont_block.append_operation(
            operation::Builder::new("memref.alloca", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "operand_segment_sizes",
                    "array<i32: 0, 0>",
                )
                .unwrap()])
                .add_results(&[Type::parse(&self.context, "memref<32xi8>").unwrap()])
                .build(),
        );
        let op1 = cont_block.append_operation(
            operation::Builder::new("memref.alloca", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "operand_segment_sizes",
                    "array<i32: 0, 0>",
                )
                .unwrap()])
                .add_results(&[Type::parse(&self.context, "memref<32xi8>").unwrap()])
                .build(),
        );
        let op2 = cont_block.append_operation(
            operation::Builder::new("memref.alloca", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "operand_segment_sizes",
                    "array<i32: 0, 0>",
                )
                .unwrap()])
                .add_results(&[Type::parse(&self.context, "memref<i8>").unwrap()])
                .build(),
        );

        let k0 = cont_block.append_operation(
            operation::Builder::new("index.constant", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(&self.context, "value", "0 : index")?])
                .add_results(&[Type::index(&self.context)])
                .build(),
        );
        let op10 = cont_block.append_operation(
            operation::Builder::new("memref.view", Location::unknown(&self.context))
                .add_operands(&[op0.result(0)?.into(), k0.result(0)?.into()])
                .add_results(&[Type::parse(&self.context, "memref<i256>").unwrap()])
                .build(),
        );
        let u0 = cont_block.append_operation(
            operation::Builder::new("llvm.call_intrinsic", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "intrin",
                    "\"llvm.bswap.i256\"",
                )?])
                .add_operands(&[arg])
                .add_results(&[self.felt_type()])
                .build(),
        );
        cont_block.append_operation(
            operation::Builder::new("memref.store", Location::unknown(&self.context))
                .add_operands(&[u0.result(0)?.into(), op10.result(0)?.into()])
                .build(),
        );

        let op3 = cont_block.append_operation(
            operation::Builder::new(
                "memref.extract_aligned_pointer_as_index",
                Location::unknown(&self.context),
            )
            .add_operands(&[op0.result(0)?.into()])
            .add_results(&[Type::index(&self.context)])
            .build(),
        );
        let op4 = cont_block.append_operation(
            operation::Builder::new(
                "memref.extract_aligned_pointer_as_index",
                Location::unknown(&self.context),
            )
            .add_operands(&[op1.result(0)?.into()])
            .add_results(&[Type::index(&self.context)])
            .build(),
        );
        let op5 = cont_block.append_operation(
            operation::Builder::new(
                "memref.extract_aligned_pointer_as_index",
                Location::unknown(&self.context),
            )
            .add_operands(&[op2.result(0)?.into()])
            .add_results(&[Type::index(&self.context)])
            .build(),
        );

        let op6 = cont_block.append_operation(
            operation::Builder::new("index.castu", Location::unknown(&self.context))
                .add_operands(&[op3.result(0)?.into()])
                .add_results(&[Type::integer(&self.context, 64)])
                .build(),
        );
        let op7 = cont_block.append_operation(
            operation::Builder::new("index.castu", Location::unknown(&self.context))
                .add_operands(&[op4.result(0)?.into()])
                .add_results(&[Type::integer(&self.context, 64)])
                .build(),
        );
        let op8 = cont_block.append_operation(
            operation::Builder::new("index.castu", Location::unknown(&self.context))
                .add_operands(&[op5.result(0)?.into()])
                .add_results(&[Type::integer(&self.context, 64)])
                .build(),
        );

        let ptr0 = cont_block.append_operation(
            operation::Builder::new("llvm.inttoptr", Location::unknown(&self.context))
                .add_operands(&[op6.result(0)?.into()])
                .add_results(&[self.llvm_ptr_type()])
                .build(),
        );
        let ptr1 = cont_block.append_operation(
            operation::Builder::new("llvm.inttoptr", Location::unknown(&self.context))
                .add_operands(&[op7.result(0)?.into()])
                .add_results(&[self.llvm_ptr_type()])
                .build(),
        );
        let ptr2 = cont_block.append_operation(
            operation::Builder::new("llvm.inttoptr", Location::unknown(&self.context))
                .add_operands(&[op8.result(0)?.into()])
                .add_results(&[self.llvm_ptr_type()])
                .build(),
        );

        cont_block.append_operation(
            operation::Builder::new("func.call", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "callee",
                    "@sierra2mlir_util_ec_point_from_x_nz",
                )?])
                .add_operands(&[
                    ptr0.result(0)?.into(),
                    ptr1.result(0)?.into(),
                    ptr2.result(0)?.into(),
                ])
                .build(),
        );

        let op11 = cont_block.append_operation(
            operation::Builder::new("memref.view", Location::unknown(&self.context))
                .add_operands(&[op1.result(0)?.into(), k0.result(0)?.into()])
                .add_results(&[Type::parse(&self.context, "memref<i256>").unwrap()])
                .build(),
        );

        let op12 = cont_block.append_operation(
            operation::Builder::new("memref.load", Location::unknown(&self.context))
                .add_operands(&[op10.result(0)?.into()])
                .add_results(&[self.felt_type()])
                .build(),
        );
        let op13 = cont_block.append_operation(
            operation::Builder::new("memref.load", Location::unknown(&self.context))
                .add_operands(&[op11.result(0)?.into()])
                .add_results(&[self.felt_type()])
                .build(),
        );
        let op14 = cont_block.append_operation(
            operation::Builder::new("memref.load", Location::unknown(&self.context))
                .add_operands(&[op2.result(0)?.into()])
                .add_results(&[self.u8_type()])
                .build(),
        );
        let t0 = cont_block.append_operation(
            operation::Builder::new("arith.trunci", Location::unknown(&self.context))
                .add_operands(&[op14.result(0)?.into()])
                .add_results(&[self.bool_type()])
                .build(),
        );

        let v0 = cont_block.append_operation(
            operation::Builder::new("llvm.call_intrinsic", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "intrin",
                    "\"llvm.bswap.i256\"",
                )?])
                .add_operands(&[op12.result(0)?.into()])
                .add_results(&[self.felt_type()])
                .build(),
        );
        let v1 = cont_block.append_operation(
            operation::Builder::new("llvm.call_intrinsic", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "intrin",
                    "\"llvm.bswap.i256\"",
                )?])
                .add_operands(&[op13.result(0)?.into()])
                .add_results(&[self.felt_type()])
                .build(),
        );

        let op15 = cont_block.append_operation(
            operation::Builder::new("llvm.mlir.undef", Location::unknown(&self.context))
                .add_results(&[
                    Type::parse(&self.context, "!llvm.struct<packed (i256, i256, i1)>").unwrap()
                ])
                .build(),
        );
        let op16 = cont_block.append_operation(
            operation::Builder::new("llvm.insertvalue", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "position",
                    "array<i64: 0>",
                )
                .unwrap()])
                .add_operands(&[op15.result(0)?.into(), v0.result(0)?.into()])
                .add_results(&[
                    Type::parse(&self.context, "!llvm.struct<packed (i256, i256, i1)>").unwrap()
                ])
                .build(),
        );
        let op17 = cont_block.append_operation(
            operation::Builder::new("llvm.insertvalue", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "position",
                    "array<i64: 1>",
                )
                .unwrap()])
                .add_operands(&[op16.result(0)?.into(), v1.result(0)?.into()])
                .add_results(&[
                    Type::parse(&self.context, "!llvm.struct<packed (i256, i256, i1)>").unwrap()
                ])
                .build(),
        );
        let op18 = cont_block.append_operation(
            operation::Builder::new("llvm.insertvalue", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "position",
                    "array<i64: 2>",
                )
                .unwrap()])
                .add_operands(&[op17.result(0)?.into(), t0.result(0)?.into()])
                .add_results(&[
                    Type::parse(&self.context, "!llvm.struct<packed (i256, i256, i1)>").unwrap()
                ])
                .build(),
        );

        let next_block_idx = match invocation.branches[0].target {
            GenBranchTarget::Fallthrough => statement_idx + 1,
            GenBranchTarget::Statement(x) => x.0,
        };
        let next_block = &blocks[&next_block_idx].block;
        let next_block_args = blocks[&next_block_idx]
            .variables_at_start
            .keys()
            .map(|id| {
                if *id == invocation.branches[0].results[1].id {
                    op18.result(0).unwrap().into()
                } else {
                    variables.get(id).unwrap().get_value()
                }
            })
            .collect::<Vec<_>>();

        self.op_br(&cont_block, next_block, &next_block_args);

        Ok(())
    }

    pub fn inline_ec_point_is_zero(
        &'ctx self,
        id: &str,
        invocation: &Invocation,
        block: &Block<'ctx>,
        variables: &HashMap<u64, Variable>,
        blocks: &BTreeMap<usize, BlockInfo<'ctx>>,
        statement_idx: usize,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let libfunc = storage.libfuncs.get(id).expect("should find libfunc");
        let arg = &libfunc.get_args()[0];

        // let branch_block_idx = match invocation.branches[1].target {
        //     GenBranchTarget::Fallthrough => statement_idx + 1,
        //     GenBranchTarget::Statement(x) => x.0,
        // };
        // let branch_block = &blocks[&branch_block_idx].block;

        let arg = variables
            .get(&invocation.args[arg.loc].id)
            .expect("variable should be registered before use")
            .get_value();

        let op0 = self.op_llvm_extractvalue(block, 0, arg, self.ec_point_type())?;
        let op1 = self.op_llvm_extractvalue(block, 1, arg, self.ec_point_type())?;
        let op2 = self.op_llvm_extractvalue(block, 2, arg, self.ec_point_type())?;

        let op3 = self.op_const(block, "0", self.felt_type());
        let op4 = self.op_const(block, "0", self.bool_type());

        let op5 = self.op_cmp(block, CmpOp::Equal, op0.result(0)?.into(), op3.result(0)?.into());
        let op6 = self.op_cmp(block, CmpOp::Equal, op1.result(0)?.into(), op3.result(0)?.into());
        let op7 = self.op_cmp(block, CmpOp::Equal, op2.result(0)?.into(), op4.result(0)?.into());

        let op8 = self.op_or(block, op5.result(0)?.into(), op6.result(0)?.into(), self.bool_type());
        let op9 = self.op_or(block, op8.result(0)?.into(), op7.result(0)?.into(), self.bool_type());

        let b0_block_idx = match invocation.branches[0].target {
            GenBranchTarget::Fallthrough => statement_idx + 1,
            GenBranchTarget::Statement(x) => x.0,
        };
        let b1_block_idx = match invocation.branches[1].target {
            GenBranchTarget::Fallthrough => statement_idx + 1,
            GenBranchTarget::Statement(x) => x.0,
        };
        self.op_cond_br(
            block,
            op9.result(0)?.into(),
            &blocks[&b0_block_idx].block,
            &blocks[&b1_block_idx].block,
            &blocks[&b0_block_idx]
                .variables_at_start
                .keys()
                .map(|x| variables[x].get_value())
                .collect::<Vec<_>>(),
            &blocks[&b1_block_idx]
                .variables_at_start
                .keys()
                .map(|id| {
                    if *id == invocation.branches[0].results[1].id {
                        arg
                    } else {
                        variables[id].get_value()
                    }
                })
                .collect::<Vec<_>>(),
        );

        Ok(())
    }

    pub fn inline_ec_state_try_finalize_nz(
        &'ctx self,
        id: &str,
        invocation: &Invocation,
        block: &Block<'ctx>,
        variables: &HashMap<u64, Variable>,
        blocks: &BTreeMap<usize, BlockInfo<'ctx>>,
        statement_idx: usize,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        self.create_utils(storage)?;

        let libfunc = storage.libfuncs.get(id).expect("should find libfunc");
        let arg = &libfunc.get_args()[0];

        let next_success = &blocks[&match invocation.branches[0].target {
            GenBranchTarget::Fallthrough => statement_idx + 1,
            GenBranchTarget::Statement(x) => x.0,
        }];
        let next_failure = &blocks[&match invocation.branches[1].target {
            GenBranchTarget::Fallthrough => statement_idx + 1,
            GenBranchTarget::Statement(x) => x.0,
        }];

        let arg = variables
            .get(&invocation.args[arg.loc].id)
            .expect("variable should be registered before use")
            .get_value();

        let op0 = self.op_llvm_extractvalue(block, 0, arg, self.felt_type())?;
        let op1 = self.op_llvm_extractvalue(block, 1, arg, self.felt_type())?;
        let op2 = self.op_llvm_extractvalue(block, 2, arg, self.felt_type())?;
        let op3 = self.op_llvm_extractvalue(block, 3, arg, self.felt_type())?;

        let op4 = block.append_operation(
            operation::Builder::new("memref.alloca", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "operand_segment_sizes",
                    "array<i32: 0, 0>",
                )
                .unwrap()])
                .add_results(&[Type::parse(&self.context, "memref<32xi8>").unwrap()])
                .build(),
        );
        let op5 = block.append_operation(
            operation::Builder::new("memref.alloca", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "operand_segment_sizes",
                    "array<i32: 0, 0>",
                )
                .unwrap()])
                .add_results(&[Type::parse(&self.context, "memref<32xi8>").unwrap()])
                .build(),
        );
        let op6 = block.append_operation(
            operation::Builder::new("memref.alloca", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "operand_segment_sizes",
                    "array<i32: 0, 0>",
                )
                .unwrap()])
                .add_results(&[Type::parse(&self.context, "memref<32xi8>").unwrap()])
                .build(),
        );
        let op7 = block.append_operation(
            operation::Builder::new("memref.alloca", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "operand_segment_sizes",
                    "array<i32: 0, 0>",
                )
                .unwrap()])
                .add_results(&[Type::parse(&self.context, "memref<32xi8>").unwrap()])
                .build(),
        );

        let op8 = block.append_operation(
            operation::Builder::new("llvm.call_intrinsic", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "intrin",
                    "\"llvm.bswap.i256\"",
                )
                .unwrap()])
                .add_operands(&[op0.result(0)?.into()])
                .add_results(&[self.felt_type()])
                .build(),
        );
        let op9 = block.append_operation(
            operation::Builder::new("llvm.call_intrinsic", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "intrin",
                    "\"llvm.bswap.i256\"",
                )
                .unwrap()])
                .add_operands(&[op1.result(0)?.into()])
                .add_results(&[self.felt_type()])
                .build(),
        );
        let op10 = block.append_operation(
            operation::Builder::new("llvm.call_intrinsic", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "intrin",
                    "\"llvm.bswap.i256\"",
                )
                .unwrap()])
                .add_operands(&[op2.result(0)?.into()])
                .add_results(&[self.felt_type()])
                .build(),
        );
        let op11 = block.append_operation(
            operation::Builder::new("llvm.call_intrinsic", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "intrin",
                    "\"llvm.bswap.i256\"",
                )
                .unwrap()])
                .add_operands(&[op3.result(0)?.into()])
                .add_results(&[self.felt_type()])
                .build(),
        );

        let k0 = block.append_operation(
            operation::Builder::new("index.constant", Location::unknown(&self.context))
                .add_attributes(&[
                    NamedAttribute::new_parsed(&self.context, "value", "0 : index").unwrap()
                ])
                .add_results(&[Type::index(&self.context)])
                .build(),
        );
        let op12 = block.append_operation(
            operation::Builder::new("memref.view", Location::unknown(&self.context))
                .add_operands(&[op4.result(0)?.into(), k0.result(0)?.into()])
                .add_results(&[Type::parse(&self.context, "memref<i256>").unwrap()])
                .build(),
        );
        let op13 = block.append_operation(
            operation::Builder::new("memref.view", Location::unknown(&self.context))
                .add_operands(&[op5.result(0)?.into(), k0.result(0)?.into()])
                .add_results(&[Type::parse(&self.context, "memref<i256>").unwrap()])
                .build(),
        );
        let op14 = block.append_operation(
            operation::Builder::new("memref.view", Location::unknown(&self.context))
                .add_operands(&[op6.result(0)?.into(), k0.result(0)?.into()])
                .add_results(&[Type::parse(&self.context, "memref<i256>").unwrap()])
                .build(),
        );
        let op15 = block.append_operation(
            operation::Builder::new("memref.view", Location::unknown(&self.context))
                .add_operands(&[op7.result(0)?.into(), k0.result(0)?.into()])
                .add_results(&[Type::parse(&self.context, "memref<i256>").unwrap()])
                .build(),
        );

        block.append_operation(
            operation::Builder::new("memref.store", Location::unknown(&self.context))
                .add_operands(&[op8.result(0)?.into(), op12.result(0)?.into()])
                .build(),
        );
        block.append_operation(
            operation::Builder::new("memref.store", Location::unknown(&self.context))
                .add_operands(&[op9.result(0)?.into(), op13.result(0)?.into()])
                .build(),
        );
        block.append_operation(
            operation::Builder::new("memref.store", Location::unknown(&self.context))
                .add_operands(&[op10.result(0)?.into(), op14.result(0)?.into()])
                .build(),
        );
        block.append_operation(
            operation::Builder::new("memref.store", Location::unknown(&self.context))
                .add_operands(&[op11.result(0)?.into(), op15.result(0)?.into()])
                .build(),
        );

        let op16 = block.append_operation(
            operation::Builder::new(
                "memref.extract_aligned_pointer_as_index",
                Location::unknown(&self.context),
            )
            .add_operands(&[op4.result(0)?.into()])
            .add_results(&[Type::index(&self.context)])
            .build(),
        );
        let op17 = block.append_operation(
            operation::Builder::new(
                "memref.extract_aligned_pointer_as_index",
                Location::unknown(&self.context),
            )
            .add_operands(&[op5.result(0)?.into()])
            .add_results(&[Type::index(&self.context)])
            .build(),
        );
        let op18 = block.append_operation(
            operation::Builder::new(
                "memref.extract_aligned_pointer_as_index",
                Location::unknown(&self.context),
            )
            .add_operands(&[op6.result(0)?.into()])
            .add_results(&[Type::index(&self.context)])
            .build(),
        );
        let op19 = block.append_operation(
            operation::Builder::new(
                "memref.extract_aligned_pointer_as_index",
                Location::unknown(&self.context),
            )
            .add_operands(&[op7.result(0)?.into()])
            .add_results(&[Type::index(&self.context)])
            .build(),
        );

        let op20 = block.append_operation(
            operation::Builder::new("index.castu", Location::unknown(&self.context))
                .add_operands(&[op16.result(0)?.into()])
                .add_results(&[Type::integer(&self.context, 64)])
                .build(),
        );
        let op21 = block.append_operation(
            operation::Builder::new("index.castu", Location::unknown(&self.context))
                .add_operands(&[op17.result(0)?.into()])
                .add_results(&[Type::integer(&self.context, 64)])
                .build(),
        );
        let op22 = block.append_operation(
            operation::Builder::new("index.castu", Location::unknown(&self.context))
                .add_operands(&[op18.result(0)?.into()])
                .add_results(&[Type::integer(&self.context, 64)])
                .build(),
        );
        let op23 = block.append_operation(
            operation::Builder::new("index.castu", Location::unknown(&self.context))
                .add_operands(&[op19.result(0)?.into()])
                .add_results(&[Type::integer(&self.context, 64)])
                .build(),
        );

        let op24 = block.append_operation(
            operation::Builder::new("llvm.inttoptr", Location::unknown(&self.context))
                .add_operands(&[op20.result(0)?.into()])
                .add_results(&[Type::parse(&self.context, "!llvm.ptr").unwrap()])
                .build(),
        );
        let op25 = block.append_operation(
            operation::Builder::new("llvm.inttoptr", Location::unknown(&self.context))
                .add_operands(&[op21.result(0)?.into()])
                .add_results(&[Type::parse(&self.context, "!llvm.ptr").unwrap()])
                .build(),
        );
        let op26 = block.append_operation(
            operation::Builder::new("llvm.inttoptr", Location::unknown(&self.context))
                .add_operands(&[op22.result(0)?.into()])
                .add_results(&[Type::parse(&self.context, "!llvm.ptr").unwrap()])
                .build(),
        );
        let op27 = block.append_operation(
            operation::Builder::new("llvm.inttoptr", Location::unknown(&self.context))
                .add_operands(&[op23.result(0)?.into()])
                .add_results(&[Type::parse(&self.context, "!llvm.ptr").unwrap()])
                .build(),
        );

        let op28 = block.append_operation(
            operation::Builder::new("func.call", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "callee",
                    "@sierra2mlir_util_ec_state_try_finalize_nz",
                )?])
                .add_operands(&[
                    op24.result(0)?.into(),
                    op25.result(0)?.into(),
                    op26.result(0)?.into(),
                    op27.result(0)?.into(),
                ])
                .add_results(&[self.i32_type()])
                .build(),
        );

        let t0 = block.append_operation(
            operation::Builder::new("memref.load", Location::unknown(&self.context))
                .add_operands(&[op12.result(0)?.into()])
                .add_results(&[self.felt_type()])
                .build(),
        );
        let t1 = block.append_operation(
            operation::Builder::new("memref.load", Location::unknown(&self.context))
                .add_operands(&[op13.result(0)?.into()])
                .add_results(&[self.felt_type()])
                .build(),
        );
        let t2 = block.append_operation(
            operation::Builder::new("llvm.call_intrinsic", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "intrin",
                    "\"llvm.bswap.i256\"",
                )
                .unwrap()])
                .add_operands(&[t0.result(0)?.into()])
                .add_results(&[self.felt_type()])
                .build(),
        );
        let t3 = block.append_operation(
            operation::Builder::new("llvm.call_intrinsic", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "intrin",
                    "\"llvm.bswap.i256\"",
                )
                .unwrap()])
                .add_operands(&[t1.result(0)?.into()])
                .add_results(&[self.felt_type()])
                .build(),
        );

        let op29 = self.op_const(block, "0", self.bool_type());
        let op30 = self.op_llvm_undef(block, self.ec_point_type());
        let op31 = self.op_llvm_insertvalue(
            block,
            0,
            op30.result(0)?.into(),
            t2.result(0)?.into(),
            self.ec_point_type(),
        )?;
        let op32 = self.op_llvm_insertvalue(
            block,
            1,
            op31.result(0)?.into(),
            t3.result(0)?.into(),
            self.ec_point_type(),
        )?;
        let op33 = self.op_llvm_insertvalue(
            block,
            2,
            op32.result(0)?.into(),
            op29.result(0)?.into(),
            self.ec_point_type(),
        )?;

        let op34 = self.op_u32_const(block, "0");
        let op35 = self.op_cmp(block, CmpOp::Equal, op28.result(0)?.into(), op34.result(0)?.into());
        self.op_cond_br(
            block,
            op35.result(0)?.into(),
            &next_success.block,
            &next_failure.block,
            &next_success
                .variables_at_start
                .keys()
                .map(|id| {
                    if *id == invocation.branches[0].results[0].id {
                        op33.result(0).unwrap().into()
                    } else {
                        variables[id].get_value()
                    }
                })
                .collect::<Vec<_>>(),
            &next_failure
                .variables_at_start
                .keys()
                .map(|x| variables[x].get_value())
                .collect::<Vec<_>>(),
        );

        Ok(())
    }
}
