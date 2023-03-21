use std::{cell::RefCell, collections::HashMap, rc::Rc};

use cairo_lang_sierra::{
    ids::VarId,
    program::{GenBranchTarget, GenStatement},
};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{Block, Location, OperationRef, Region, Type, Value};
use tracing::{debug, error};

use crate::compiler::{Compiler, SierraType, Storage};

impl<'ctx> Compiler<'ctx> {
    #[allow(clippy::cognitive_complexity)]
    pub fn process_functions(&self, storage: Rc<RefCell<Storage<'ctx>>>) -> Result<()> {
        for func in &self.program.funcs {
            debug!(?func, "processing func");

            let name = Self::normalize_func_name(func.id.debug_name.as_ref().unwrap().as_str())
                .to_string();
            let entry = func.entry_point.0;
            let mut params = vec![];
            let mut return_types = vec![];

            let storage = storage.borrow();

            for param in &func.params {
                let ty = storage
                    .types
                    .get(&param.ty.id.to_string())
                    .expect("type for param should exist");

                let ty = match ty {
                    SierraType::Simple(ty) => ty,
                    SierraType::Struct { ty, field_types: _ } => ty,
                };
                params.push((*ty, Location::unknown(&self.context)));
            }

            for ret in &func.signature.ret_types {
                let ty = storage
                    .types
                    .get(&ret.id.to_string())
                    .expect("type for param should exist");

                let ty = match ty {
                    SierraType::Simple(ty) => ty,
                    SierraType::Struct { ty, field_types: _ } => ty,
                };
                return_types.push((*ty, Location::unknown(&self.context)));
            }

            // The varid -> operation ref which holds the variable value in the result at index.
            let mut variables: HashMap<&VarId, (OperationRef, usize)> = HashMap::new();
            let mut param_values: HashMap<&VarId, Value> = HashMap::new();

            let region = Region::new();
            let function_block = region.append_block(Block::new(&params));

            for (i, param) in func.params.iter().enumerate() {
                let value = function_block.argument(i)?;
                param_values.insert(&param.id, value.into());
            }

            let statements_entry = self.program.statements.iter().enumerate().skip(entry);

            // create blocks at each jump destination
            let mut jump_dests = HashMap::new();

            for (i, s) in statements_entry.clone() {
                match s {
                    GenStatement::Invocation(inv) => {
                        let mut had_jump = false;
                        let mut had_fall_through = false;
                        for branch in &inv.branches {
                            match branch.target {
                                GenBranchTarget::Fallthrough => {
                                    had_fall_through = true;
                                }
                                GenBranchTarget::Statement(statement_id) => {
                                    debug!(from = i, to = statement_id.0, "added jump");
                                    jump_dests.insert(
                                        statement_id.0,
                                        region.append_block(Block::new(&[])),
                                    );
                                    had_jump = true;
                                }
                            }
                        }

                        // next statement from the jump needs a block, because this is a conditional jump
                        // where one condition simply follows through
                        if had_jump && had_fall_through {
                            let name = inv.libfunc_id.debug_name.as_ref().unwrap().as_str();
                            let name_without_generics = name.split('<').next().unwrap();

                            if name_without_generics == "felt_is_zero" {
                                jump_dests.insert(i + 1, region.append_block(Block::new(&[])));
                            }
                        }
                    }
                    GenStatement::Return(_) => break,
                }
            }

            let mut current_statements_iter = statements_entry.clone();
            let mut current_block = &function_block;

            // Use a loop so we can change the iterator in the middle of it in case of jumps.
            loop {
                if let Some((statement_id, statement)) = current_statements_iter.next() {
                    if let Some(block) = jump_dests.get(&statement_id) {
                        debug!(statement_id, "changed current block");
                        current_block = block;
                    }

                    // We need to add a terminator in case there is a block on the next statement and
                    // we didn't branch jump on this statement.
                    let mut had_jump = false;

                    match statement {
                        GenStatement::Invocation(inv) => {
                            let name = inv.libfunc_id.debug_name.as_ref().unwrap().as_str();
                            let id = Self::normalize_func_name(
                                inv.libfunc_id.debug_name.as_ref().unwrap().as_str(),
                            )
                            .to_string();
                            debug!(name, "processing statement: invocation");

                            let name_without_generics = name.split('<').next().unwrap();

                            match name_without_generics {
                                "disable_ap_tracking" | "drop" | "branch_align" => continue,
                                "felt_const" => {
                                    let felt_const = storage
                                        .felt_consts
                                        .get(&id)
                                        .expect("constant should exist");
                                    let op = self.op_felt_const(current_block, felt_const);
                                    let var_id = &inv.branches[0].results[0];
                                    variables.insert(var_id, (op, 0));
                                }
                                "jump" => {
                                    let target_block = match &inv.branches[0].target {
                                        GenBranchTarget::Fallthrough => {
                                            unreachable!("jump should never be fallthrough")
                                        }
                                        GenBranchTarget::Statement(id) => {
                                            jump_dests.get(&(id.0)).unwrap()
                                        }
                                    };
                                    self.op_br(current_block, target_block);
                                    had_jump = true;
                                }
                                name if inv.branches.len() > 1 => {
                                    match name {
                                        "felt_is_zero" => {
                                            let felt_op_zero =
                                                self.op_felt_const(current_block, "0");
                                            let zero = felt_op_zero.result(0)?.into();

                                            let var = &inv.args[0];
                                            let felt_val =
                                                if let Some((var, i)) = variables.get(var) {
                                                    let res = var.result(*i)?;
                                                    res.into()
                                                } else {
                                                    let res = param_values
                                                        .get(var)
                                                        .expect("couldn't find variable");
                                                    *res
                                                };
                                            let eq_op = self.op_eq(current_block, felt_val, zero);
                                            let eq = eq_op.result(0)?;

                                            let next_block = jump_dests
                                                .get(&(statement_id + 1))
                                                .expect(
                                                "there should be a block next to this statement",
                                            );
                                            let true_block = match &inv.branches[0].target {
                                                GenBranchTarget::Fallthrough => next_block,
                                                GenBranchTarget::Statement(statement_id) => {
                                                    jump_dests.get(&statement_id.0).unwrap()
                                                }
                                            };

                                            let false_block = match &inv.branches[1].target {
                                                GenBranchTarget::Fallthrough => next_block,
                                                GenBranchTarget::Statement(statement_id) => {
                                                    jump_dests.get(&statement_id.0).unwrap()
                                                }
                                            };

                                            self.op_cond_br(
                                                current_block,
                                                eq.into(),
                                                true_block,
                                                false_block,
                                            )?;

                                            had_jump = true;
                                        }
                                        _ => todo!(),
                                    };
                                }
                                name => {
                                    let func_def =
                                        if let Some(func_def) = storage.functions.get(&id) {
                                            func_def
                                        } else {
                                            error!(
                                                id,
                                                name, "encountered undefined libfunc invocation"
                                            );
                                            continue;
                                        };
                                    let mut args = vec![];

                                    for var in &inv.args {
                                        let res = if let Some((var, i)) = variables.get(&var) {
                                            let res = var.result(*i)?;
                                            res.into()
                                        } else {
                                            let res = param_values
                                                .get(var)
                                                .expect("couldn't find variable");
                                            *res
                                        };
                                        args.push(res);
                                    }

                                    debug!(id, "creating func call");
                                    let op = self.op_func_call(
                                        current_block,
                                        &id,
                                        &args,
                                        &func_def.return_types,
                                    )?;
                                    debug!("created");

                                    for (i, var_id) in inv.branches[0].results.iter().enumerate() {
                                        variables.insert(var_id, (op, i));
                                    }

                                    // debug!(name, "unimplemented");
                                }
                            }
                        }
                        GenStatement::Return(ret) => {
                            let mut ret_values: Vec<Value> = vec![];

                            for var in ret {
                                let val = if let Some((op, i)) = variables.get(&var) {
                                    let val = op.result(*i)?;
                                    val.into()
                                } else {
                                    let res =
                                        param_values.get(var).expect("couldn't find variable");
                                    *res
                                };
                                ret_values.push(val);
                            }

                            self.op_return(current_block, &ret_values);
                            debug!(?ret, "processing statement: return");
                            break;
                        }
                    }

                    // All blocks need a terminator.
                    if !had_jump {
                        if let Some(next_block) = jump_dests.get(&(statement_id + 1)) {
                            self.op_br(current_block, next_block);
                        }
                    }
                }
            }

            let function_type = self.create_fn_signature(&params, &return_types);

            let op = self.op_func(&name, &function_type, vec![region], false, true)?;

            self.module.body().append_operation(op);
        }
        Ok(())
    }

    pub fn create_fn_signature(
        &'ctx self,
        params: &[(Type, Location)],
        return_types: &[(Type, Location)],
    ) -> String {
        format!(
            "({}) -> {}",
            params.iter().map(|x| x.0.to_string()).join(", "),
            &format!(
                "({})",
                return_types.iter().map(|x| x.0.to_string()).join(", ")
            ),
        )
    }

    /*
    #[allow(clippy::only_used_in_recursion)] // false positive, the lifetime of self is needed.
    pub fn collect_values(
        &self,
        data: &mut Vec<Value<'ctx>>,
        var: &'ctx VarInfo<'ctx>,
    ) -> Result<()> {
        match var {
            VarInfo::Value { op, result_index } => {
                let res = op.result(*result_index)?;
                data.push(res.into());
            }
            VarInfo::Struct(vars) => {
                for var in vars {
                    self.collect_values(data, var)?;
                }
            }
        };
        Ok(())
    }
     */
}
