use std::{
    cell::{Ref, RefCell},
    collections::HashMap,
    rc::Rc,
};

use cairo_lang_sierra::{
    ids::ConcreteTypeId,
    program::{GenBranchTarget, GenStatement, GenericArg, Program, TypeDeclaration},
};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{Block, BlockRef, Location, OperationRef, Region, Type, Value};
use tracing::{debug, error};

use crate::compiler::{CmpOp, Compiler, Storage};

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
    pub fn process_functions(&self, storage_cell: Rc<RefCell<Storage<'ctx>>>) -> Result<()> {
        for func in &self.program.funcs {
            debug!(?func, "processing func");

            let raw_func_name = func.id.debug_name.as_ref().unwrap().as_str();

            let should_create_wrapper = self.main_print && should_create_wrapper(raw_func_name);
            let name = Self::normalize_func_name(raw_func_name).to_string();

            let entry = func.entry_point.0;
            let mut param_types = vec![];
            let mut return_types = vec![];
            let mut return_sierra_types = vec![];

            let storage = storage_cell.borrow();

            for param in &func.params {
                let ty = storage
                    .types
                    .get(&param.ty.id.to_string())
                    .expect("type for param should exist");

                param_types.push(ty.get_type());
            }
            let param_types = param_types;

            for ret in &func.signature.ret_types {
                let ty = storage
                    .types
                    .get(&ret.id.to_string())
                    .expect("type for param should exist")
                    .get_type();

                return_types.push(ty);
                return_sierra_types.push((ty, ret.clone()));
            }
            let return_sierra_types = return_sierra_types;
            let return_types = return_types;

            // The varid -> operation ref which holds the variable value in the result at index.
            let mut variables: HashMap<u64, Variable> = HashMap::new();

            let region = Region::new();
            let param_types_with_locations = param_types
                .iter()
                .map(|t| (*t, Location::unknown(&self.context)))
                .collect::<Vec<_>>();
            let function_block = region.append_block(Block::new(&param_types_with_locations));

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
                                        "felt252_is_zero" => {
                                            let felt_op_zero =
                                                self.op_felt_const(current_block, "0");
                                            let zero = felt_op_zero.result(0)?.into();

                                            let var = &inv.args[0];
                                            let felt_val = variables
                                                .get(&var.id)
                                                .expect("couldn't find variable")
                                                .get_value();
                                            let eq_op = self.op_cmp(
                                                current_block,
                                                CmpOp::Equal,
                                                felt_val,
                                                zero,
                                            );
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
                                                &[],
                                                &[],
                                            )?;

                                            had_jump = true;
                                        }
                                        _ => {
                                            todo!("Branching function {} not implemented yet", name)
                                        }
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

                                    let return_types = func_def
                                        .return_types
                                        .iter()
                                        .map(|x| x.get_type())
                                        .collect_vec();

                                    debug!(id, "creating func call");
                                    let op = self.op_func_call(
                                        current_block,
                                        &id,
                                        &args,
                                        &return_types,
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

            let function_type = create_fn_signature(&param_types, &return_types);

            let op = self.op_func(&name, &function_type, vec![region], true, true)?;

            self.module.body().append_operation(op);

            if should_create_wrapper {
                self.create_felt_representation_wrapper(
                    &name,
                    &param_types,
                    &return_sierra_types,
                    storage_cell.borrow(),
                )?;
            }
        }
        Ok(())
    }

    pub fn create_felt_representation_wrapper(
        &'ctx self,
        wrapped_func_name: &str,
        arg_types: &[Type],
        ret_types: &[(Type, ConcreteTypeId)],
        storage: Ref<Storage<'ctx>>,
    ) -> Result<()> {
        // We need to collect the sierra type declarations to know how to convert the mlir types to their felt representation
        // This is especially important for enums
        let ret_type_declarations = ret_types
            .iter()
            .map(|(mlir_type, type_id)| {
                (
                    mlir_type,
                    self.program
                        .type_declarations
                        .iter()
                        .find(|decl| decl.id == *type_id)
                        .unwrap(),
                )
            })
            .collect_vec();

        // Create a list of types for which to generate print functions, with no duplicates
        // For complex types, their components types must be added to the list before them
        let types_to_print = get_all_types_to_print(
            &ret_type_declarations
                .iter()
                .map(|(_t, decl)| (*decl).clone())
                .collect_vec(),
            &self.program,
        );

        for type_decl in types_to_print {
            let type_category = type_decl.long_id.generic_id.0.as_str();
            match type_category {
                "felt252" => self.create_print_felt()?,
                "NonZero" => todo!("Print box felt representation"),
                "Box" => todo!("Print box felt representation"),
                "Struct" => {
                    let arg_type = storage
                        .types
                        .get(&type_decl.id.id.to_string())
                        .expect("Type should be registered");
                    self.create_print_struct(arg_type, type_decl.clone())?
                }
                "Enum" => todo!("Print enum felt representation"),
                _ => todo!("Felt representation for {}", type_category),
            }
        }

        let region = Region::new();
        let arg_types_with_locations = arg_types
            .iter()
            .map(|t| (*t, Location::unknown(&self.context)))
            .collect::<Vec<_>>();
        let block = Block::new(&arg_types_with_locations);

        let mut arg_values: Vec<_> = vec![];
        for i in 0..block.argument_count() {
            let arg = block.argument(i)?;
            arg_values.push(arg.into());
        }
        let arg_values = arg_values;
        let mlir_ret_types = ret_types.iter().map(|(t, _id)| *t).collect_vec();

        let raw_res = self.op_func_call(&block, wrapped_func_name, &arg_values, &mlir_ret_types)?;
        for (position, (_, type_decl)) in ret_type_declarations.iter().enumerate() {
            let result_val = raw_res.result(position)?;
            self.op_func_call(
                &block,
                &format!(
                    "print_{}",
                    type_decl.id.debug_name.as_ref().unwrap().as_str()
                ),
                &[result_val.into()],
                &[],
            )?;
        }

        self.op_return(&block, &[]);

        region.append_block(block);

        // Currently this wrapper is only put on the entrypoint, so we call it main
        // We might want to change this in the future
        let op = self.op_func(
            "main",
            create_fn_signature(arg_types, &[]).as_str(),
            vec![region],
            true,
            true,
        )?;

        self.module.body().append_operation(op);

        Ok(())
    }
}

pub fn create_fn_signature(params: &[Type], return_types: &[Type]) -> String {
    format!(
        "({}) -> {}",
        params.iter().map(|x| x.to_string()).join(", "),
        &format!(
            "({})",
            return_types.iter().map(|x| x.to_string()).join(", ")
        ),
    )
}

// Currently this checks whether the function in question is the main function,
// but it could be switched for anything later
fn should_create_wrapper(raw_func_name: &str) -> bool {
    let parts = raw_func_name.split("::").collect::<Vec<_>>();
    parts.len() == 3 && parts[0] == parts[1] && parts[2] == "main"
}

// Produces an ordered list of all types and component types
fn get_all_types_to_print(
    type_declarations: &[TypeDeclaration],
    program: &Program,
) -> Vec<TypeDeclaration> {
    let mut types_to_print = vec![];
    for type_decl in type_declarations {
        let type_category = type_decl.long_id.generic_id.0.as_str();
        match type_category {
            "felt252" => {
                if !types_to_print.contains(type_decl) {
                    types_to_print.push(type_decl.clone());
                }
            }
            "NonZero" => todo!("Print box felt representation"),
            "Box" => todo!("Print box felt representation"),
            "Struct" => {
                let field_type_declarations = type_decl.long_id.generic_args[1..].iter().map(|member_type| match member_type {
                    GenericArg::Type(type_id) => type_id,
                    _ => panic!("Struct type declaration arguments after the first should all be resolved"),
                }).map(|member_type_id| program.type_declarations.iter().find(|decl| decl.id == *member_type_id).unwrap())
                .map(|component_type_decl| get_all_types_to_print(&[component_type_decl.clone()], program));

                for type_decls in field_type_declarations {
                    for type_decl in type_decls {
                        if !types_to_print.contains(&type_decl) {
                            types_to_print.push(type_decl);
                        }
                    }
                }
                if !types_to_print.contains(type_decl) {
                    types_to_print.push(type_decl.clone());
                }
            }
            "Enum" => todo!("Print enum felt representation"),
            _ => todo!("Felt representation for {}", type_category),
        }
    }
    types_to_print
}
