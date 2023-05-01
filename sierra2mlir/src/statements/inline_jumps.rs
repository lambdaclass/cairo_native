use std::{
    collections::{BTreeMap, HashMap},
};

use cairo_lang_sierra::program::{GenBranchTarget, Invocation};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::{
    dialect::cf,
    ir::{Block, Location, Region, Value, ValueLike},
};

use crate::{
    compiler::mlir_ops::CmpOp,
    libfuncs::lib_func_def::{BranchProcessing, BranchSelector, SierraLibFunc},
};
use crate::{
    compiler::{Compiler, Storage},
    sierra_type::SierraType,
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

    pub fn inline_branching_libfunc(
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
        println!("Processing {}", id);

        let libfunc = storage.libfuncs.get(id).unwrap();

        let (selector, branch_processing) =
            if let SierraLibFunc::Branching { selector, branch_processing } = libfunc {
                (selector, branch_processing)
            } else {
                panic!()
            };

        let selector_var = match selector {
            BranchSelector::Arg(arg) => *variables.get(&invocation.args[arg.loc].id).unwrap(),
            BranchSelector::Call { name, args, return_type, return_pos } => {
                let resolved_args = args
                    .iter()
                    .map(|arg| variables.get(&invocation.args[arg.loc].id).unwrap().get_value())
                    .collect_vec();
                let selector_op =
                    self.op_func_call(block, name, &resolved_args, &[*return_type])?;
                Variable::Local { op: selector_op, result_idx: *return_pos }
            }
        };

        let processing_blocks = branch_processing.iter().enumerate().map(|(branch_idx, processing)| {
            let branch_info = &invocation.branches[branch_idx];
            let block = self.new_block(&[]);

            // A map of the variables produced by this libfunc
            let branch_vars = match processing {
                BranchProcessing::Args(args) => {
                    args.iter().map(|(arg, res_idx)| (branch_info.results[*res_idx].id, *variables.get(&invocation.args[arg.loc].id).unwrap())).collect::<BTreeMap<_, _>>()
                },
                BranchProcessing::Call { name, args, return_types } => {
                    let resolved_args = args.iter().map(|arg| variables.get(&invocation.args[arg.loc].id).unwrap().get_value()).collect_vec();
                    let resolved_ret_types = return_types.iter().map(SierraType::get_type).collect_vec();
                    let op = self.op_func_call(&block, name, &resolved_args, &resolved_ret_types)?;
                    (0..branch_info.results.len()).map(|idx| (branch_info.results[idx].id, Variable::Local { op: op, result_idx: idx })).collect::<BTreeMap<_, _>>()
                },
                BranchProcessing::SelectorResult(results) => {
                    if let Variable::Local { op, .. } = selector_var {
                        results.iter().map(|(arg, res_idx)| (branch_info.results[*res_idx].id, Variable::Local { op, result_idx: arg.loc })).collect::<BTreeMap<_, _>>()
                    } else {
                        panic!("BranchProcessing::SelectorResult can only be used with BranchSelector::Call")
                    }
                },
            };

            let target_block_info = blocks.get(&match branch_info.target {
                GenBranchTarget::Fallthrough => statement_idx + 1,
                GenBranchTarget::Statement(idx) => idx.0,
            }).unwrap();

            let args_to_target_block = target_block_info.variables_at_start.keys().map(|id| {
                branch_vars.get(id).unwrap_or_else(|| variables.get(id).unwrap()).get_value()
            }).collect_vec();

            self.op_br(&block, &target_block_info.block, &args_to_target_block);

            Ok(block)
        }).collect::<Result<Vec<_>>>()?;

        if processing_blocks.len() == 2 {
            let zero_op = self.op_const(&block, "0", self.bool_type());
            let zero = zero_op.result(0)?.into();
            let selector_eq_zero_op = self.op_cmp(&block, CmpOp::Equal, selector_var.get_value(), zero);
            let selector_eq_zero = selector_eq_zero_op.result(0)?.into();
            self.op_cond_br(
                &block,
                selector_eq_zero,
                &processing_blocks[0],
                &processing_blocks[1],
                &[],
                &[],
            );
        } else {
            let case_values = (0..processing_blocks.len() - 1).map(|x| x.to_string()).collect_vec();
            let (default_block, case_blocks) = processing_blocks.split_last().unwrap();

            block.append_operation(cf::switch(
                &self.context,
                &case_values,
                selector_var.get_value(),
                (&default_block, &[]),
                &case_blocks.iter().map(|b| (b, &[] as &[Value])).collect_vec(),
                Location::unknown(&self.context),
            ));
        }

        for b in processing_blocks {
            region.append_block(b);
        }

        Ok(())

        // let target_function_args = args
        //     .iter()
        //     .map(|arg| variables.get(&invocation.args[arg.loc].id).unwrap().get_value())
        //     .collect_vec();

        // let selector_call = match selector {
        //     BranchSelector::Call(name) => {
        //         Some(self.op_func_call(block, name, &target_function_args, &[index_type])?)
        //     }
        //     BranchSelector::Arg(_) => None,
        // };

        // let selector_value = match selector {
        //     BranchSelector::Call(_) => selector_call.as_ref().unwrap().result(0)?.into(),
        //     BranchSelector::Arg(arg) => {
        //         variables.get(&invocation.args[arg.loc].id).unwrap().get_value()
        //     }
        // };

        // let branch_blocks = branch_functions
        //     .iter()
        //     .enumerate()
        //     .map(|(idx, branch_function)| {
        //         let branch_info = &invocation.branches[idx];
        //         let block = self.new_block(&[]);

        //         let branch_vars: HashMap<u64, Variable> =
        //             if let Some(BranchFunction { name, args, return_types }) = branch_function {
        //                 let branch_function_args = args
        //                     .iter()
        //                     .map(|arg| {
        //                         variables.get(&invocation.args[arg.loc].id).unwrap().get_value()
        //                     })
        //                     .collect_vec();
        //                 let branch_function_ret_types =
        //                     return_types.iter().map(|ret_ty| ret_ty.ty.get_type()).collect_vec();
        //                 let branch_data_op = self.op_func_call(
        //                     &block,
        //                     name,
        //                     &branch_function_args,
        //                     &branch_function_ret_types,
        //                 )?;
        //                 return_types
        //                     .iter()
        //                     .enumerate()
        //                     .map(|(result_idx, ret_type)| {
        //                         let id = branch_info.results[ret_type.loc].id;
        //                         (id, Variable::Local { op: branch_data_op, result_idx })
        //                     })
        //                     .collect()
        //             } else {
        //                 HashMap::new()
        //             };

        //         let target_block_info = blocks
        //             .get(&match branch_info.target {
        //                 GenBranchTarget::Fallthrough => statement_idx + 1,
        //                 GenBranchTarget::Statement(idx) => idx.0,
        //             })
        //             .unwrap();

        //         let args_to_target_block = target_block_info
        //             .variables_at_start
        //             .keys()
        //             .map(|id| {
        //                 branch_vars
        //                     .get(id)
        //                     .unwrap_or_else(|| variables.get(id).unwrap())
        //                     .get_value()
        //             })
        //             .collect_vec();

        //         self.op_br(&block, &target_block_info.block, &args_to_target_block);

        //         Ok(block)
        //     })
        //     .collect::<Result<Vec<_>>>()?;

        // let case_values = (0..branch_blocks.len() - 1).map(|x| x.to_string()).collect_vec();
        // let (default_block, case_blocks) = branch_blocks.split_last().unwrap();

        // block.append_operation(cf::switch(
        //     &self.context,
        //     &case_values,
        //     selector_value,
        //     (&default_block, &[]),
        //     &case_blocks.iter().map(|b| (b, &[] as &[Value])).collect_vec(),
        //     Location::unknown(&self.context),
        // ));

        // for b in branch_blocks {
        //     region.append_block(b);
        // }

        // Ok(())
    }

    // fn inline_branching_libfunc_branches(
    //     &'ctx self,
    //     id: &str,
    //     statement_idx: usize,
    //     region: &Region,
    //     block: &Block,
    //     blocks: &BTreeMap<usize, BlockInfo>,
    //     invocation: &Invocation,
    //     variables: &HashMap<u64, Variable>,
    //     selector: Value,
    //     storage: &mut Storage<'ctx>,
    // ) -> Result<()> {
    //     let libfunc = storage.libfuncs.get(id).unwrap();

    //     let branch_processing = if let SierraLibFunc::Branching { branch_processing, .. } = libfunc
    //     {
    //         branch_processing
    //     } else {
    //         panic!()
    //     };

    //     let mut processing_blocks = vec![];

    //     for (branch_idx, processing) in branch_processing.iter().enumerate() {
    //         let invocation_branch = &invocation.branches[branch_idx];
    //         let generated_vars = match processing {
    //             BranchProcessing::Args(args) => {
    //                 args.iter().zip_eq(invocation_branch.results.iter()).map(|(arg, res)| {
    //                     (res.id, variables.get(&invocation.args[arg.loc].id).unwrap().get_value())
    //                 }).collect_vec()
    //             }
    //             BranchProcessing::Call { name, args, return_types } => {
    //                 let block = self.new_block(&[]);
    //                 let resolved_args = args.iter().map(|arg| variables.get(&invocation.args[arg.loc].id).unwrap().get_value()).collect_vec();
    //                 let resolved_ret_types = return_types.iter().map(SierraType::get_type).collect_vec();
    //                 let call_op = self.op_func_call(&block, name, &resolved_args, &resolved_ret_types)?;
    //                 invocation_branch.results.iter().enumerate().map(|(i, res)| {
    //                     (res.id, call_op.result(i).unwrap().into())
    //                 }).collect_vec()
    //             }
    //             BranchProcessing::SelectorResult(results) => {
    //                 todo!()
    //             }
    //         };
    //     }

    //     Ok(())
    // }
}
