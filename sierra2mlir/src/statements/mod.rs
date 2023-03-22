use std::{cell::RefCell, collections::HashMap, rc::Rc};

use cairo_lang_sierra::program::{GenBranchTarget, GenStatement};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{Block, BlockRef, Location, OperationRef, Region, Type, Value};
use tracing::{debug, error};

use crate::compiler::{Compiler, SierraType, Storage};

#[derive(Debug, Clone, Copy)]
enum VariableValue<'c> {
    Local {
        op: OperationRef<'c>,
        result_idx: usize,
    },
    Param {
        argument_idx: usize,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct Variable<'c> {
    value: VariableValue<'c>,
    block: BlockRef<'c>,
}

impl<'c> Variable<'c> {
    pub const fn local(op: OperationRef<'c>, result_idx: usize, block: BlockRef<'c>) -> Self {
        Self {
            value: VariableValue::Local { op, result_idx },
            block,
        }
    }

    pub const fn param(argument_idx: usize, block: BlockRef<'c>) -> Self {
        Self {
            value: VariableValue::Param { argument_idx },
            block,
        }
    }

    pub fn get_value(&self) -> Value {
        match &self.value {
            VariableValue::Local { op, result_idx } => {
                let res = op.result(*result_idx).unwrap();
                res.into()
            }
            VariableValue::Param { argument_idx } => self
                .block
                .argument(*argument_idx)
                .expect("couldn't get argument")
                .into(),
        }
    }
}

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
            let mut variables: HashMap<u64, Variable> = HashMap::new();

            let region = Region::new();
            let function_block = region.append_block(Block::new(&params));

            for (i, param) in func.params.iter().enumerate() {
                variables.insert(param.id.id, Variable::param(i, function_block));
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
                            jump_dests.insert(i + 1, region.append_block(Block::new(&[])));
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
                                "felt252_const" => {
                                    let felt_const = storage
                                        .felt_consts
                                        .get(&id)
                                        .expect("constant should exist");
                                    let op = self.op_felt_const(current_block, felt_const);
                                    let var_id = &inv.branches[0].results[0];
                                    variables
                                        .insert(var_id.id, Variable::local(op, 0, *current_block));
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
                                "u8_const" => {
                                    let value = storage
                                        .u8_consts
                                        .get(&id)
                                        .expect("constant value not found");
                                    let op = self.op_u8_const(current_block, value);
                                    let var_id = &inv.branches[0].results[0];
                                    variables
                                        .insert(var_id.id, Variable::local(op, 0, *current_block));
                                }
                                "u16_const" => {
                                    let value = storage
                                        .u16_consts
                                        .get(&id)
                                        .expect("constant value not found");
                                    let op = self.op_u16_const(current_block, value);
                                    let var_id = &inv.branches[0].results[0];
                                    variables
                                        .insert(var_id.id, Variable::local(op, 0, *current_block));
                                }
                                "u32_const" => {
                                    let value = storage
                                        .u32_consts
                                        .get(&id)
                                        .expect("constant value not found");
                                    let op = self.op_u32_const(current_block, value);
                                    let var_id = &inv.branches[0].results[0];
                                    variables
                                        .insert(var_id.id, Variable::local(op, 0, *current_block));
                                }
                                "u64_const" => {
                                    let value = storage
                                        .u64_consts
                                        .get(&id)
                                        .expect("constant value not found");
                                    let op = self.op_u64_const(current_block, value);
                                    let var_id = &inv.branches[0].results[0];
                                    variables
                                        .insert(var_id.id, Variable::local(op, 0, *current_block));
                                }
                                "u128_const" => {
                                    let value = storage
                                        .u128_consts
                                        .get(&id)
                                        .expect("constant value not found");
                                    let op = self.op_u128_const(current_block, value);
                                    let var_id = &inv.branches[0].results[0];
                                    variables
                                        .insert(var_id.id, Variable::local(op, 0, *current_block));
                                }
                                name if inv.branches.len() > 1 => {
                                    match name {
                                        "felt_is_zero" => {
                                            let felt_op_zero =
                                                self.op_felt_const(current_block, "0");
                                            let zero = felt_op_zero.result(0)?.into();

                                            let var = &inv.args[0];
                                            let felt_val = variables
                                                .get(&var.id)
                                                .expect("couldn't find variable")
                                                .get_value();
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
                                        let res = variables
                                            .get(&var.id)
                                            .expect("couldn't find variable")
                                            .get_value();
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
                                        variables.insert(
                                            var_id.id,
                                            Variable::local(op, i, *current_block),
                                        );
                                    }
                                }
                            }
                        }
                        GenStatement::Return(ret) => {
                            let mut ret_values: Vec<Value> = vec![];

                            for var in ret {
                                let val = variables
                                    .get(&var.id)
                                    .expect("couldn't find variable")
                                    .get_value();
                                ret_values.push(val);
                            }

                            if name.ends_with("main") && self.main_print {
                                for val in &ret_values {
                                    todo!()
                                }
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

            let op = self.op_func(&name, &function_type, vec![region], true, true)?;

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
}
