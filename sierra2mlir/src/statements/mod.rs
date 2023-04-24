use std::collections::{BTreeMap, HashMap, HashSet};

use std::ops::Bound::Included;

use cairo_lang_sierra::program::{
    GenBranchTarget, GenFunction, GenStatement, Program, StatementIdx,
};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{block::Argument, Block, Location, OperationRef, Region, Value};
use regex::Regex;

use crate::compiler::{fn_attributes::FnAttributes, mlir_ops::CmpOp};
use crate::sierra_type::SierraType;
use crate::{
    compiler::{Compiler, Storage},
    utility::create_fn_signature,
};

mod function_call;
mod general_libfunc_implementations;
mod inline_jumps;

#[derive(Debug)]
pub struct BlockInfo<'ctx> {
    pub variables_at_start: BTreeMap<u64, SierraType<'ctx>>,
    // pub(crate) variables_available_at_end: HashSet<u64>,
    pub block: Block<'ctx>,
    pub end: usize,
}

struct BlockFlow {
    pub start: usize,
    pub end: usize,
    pub successors: HashSet<usize>,
    pub predecessors: HashSet<usize>,
}

#[derive(Debug, Clone, Copy)]
pub enum Variable<'c> {
    Local { op: OperationRef<'c>, result_idx: usize },
    Param { argument: Argument<'c> },
}

impl<'c> Variable<'c> {
    pub fn get_value(&self) -> Value {
        match &self {
            Variable::Local { op, result_idx } => {
                let res =
                    op.result(*result_idx).expect("Failed to get result from Variable::Local");
                res.into()
            }
            Variable::Param { argument } => (*argument).into(),
        }
    }
}

// The information about dataflow needed to process statements
struct DataFlow<'ctx> {
    input_variables: BTreeMap<u64, SierraType<'ctx>>,
    block_flow: BlockFlow,
}

// The data required to calculate dataflow
struct DataFlowInfo<'ctx> {
    required_args: BTreeMap<u64, SierraType<'ctx>>,
    variables_created: HashSet<u64>,
    block_flow: BlockFlow,
}

impl<'ctx> Compiler<'ctx> {
    // Process the statements of the sierra program by breaking flow up into basic blocks and processing one at a time
    pub fn process_statements(&'ctx self, storage: &mut Storage<'ctx>) -> Result<()> {
        // Calculate the basic block structure in each function
        let block_ranges_per_function = calculate_block_ranges_per_function(self.program);

        // Process the blocks for each function
        for (func_start, (func, block_flows)) in block_ranges_per_function {
            self.process_statements_for_function(func, func_start, block_flows, storage)?;
        }

        Ok(())
    }

    fn process_statements_for_function(
        &'ctx self,
        func: &GenFunction<StatementIdx>,
        func_start: usize,
        block_flows: BTreeMap<usize, BlockFlow>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let region = Region::new();
        let user_func_name = func.id.debug_name.as_ref().unwrap().to_string();

        let blocks = self.get_blocks_with_mapped_inputs(func, block_flows, storage);
        self.create_function_entry_block(
            &region,
            user_func_name.as_str(),
            func_start,
            func,
            &blocks,
            storage,
        )?;

        // We process statements one block at a time
        for (block_start, block_info) in blocks.iter() {
            let block = &block_info.block;

            // Variables holds the most recent value associated with a variable id as we progress through the block
            // Initially it only holds the blocks arguments
            let mut variables: HashMap<u64, Variable> = HashMap::new();
            for (arg_position, var_id) in block_info.variables_at_start.keys().enumerate() {
                let argument = block.argument(arg_position)?;
                variables.insert(*var_id, Variable::Param { argument });
            }

            for statement_idx in *block_start..block_info.end {
                match &self.program.statements[statement_idx] {
                    GenStatement::Invocation(invocation) => {
                        let name = invocation.libfunc_id.debug_name.as_ref().unwrap().as_str();
                        let id = invocation.libfunc_id.debug_name.as_ref().unwrap().to_string();
                        let name_without_generics = name.split('<').next().unwrap();
                        let mut jump_processed = false;
                        match name_without_generics {
                            "jump" => {
                                self.inline_jump(invocation, block, &mut variables, &blocks)?;
                                jump_processed = true;
                            }
                            name_without_generics
                                if is_int_is_zero_libfunc(name_without_generics) =>
                            {
                                self.inline_int_is_zero(
                                    name_without_generics,
                                    invocation,
                                    block,
                                    &mut variables,
                                    &blocks,
                                    statement_idx,
                                )?;

                                jump_processed = true;
                            }
                            // is_eq,lt,le
                            name_without_generics if is_int_cmp_libfunc(name_without_generics) => {
                                let cmpop = if name_without_generics.ends_with("eq") {
                                    CmpOp::Equal
                                } else if name_without_generics.ends_with("lt") {
                                    CmpOp::UnsignedLessThan
                                } else if name_without_generics.ends_with("le") {
                                    CmpOp::UnsignedLessThanEqual
                                } else {
                                    panic!("unknown cmp op")
                                };

                                self.inline_int_cmpop(
                                    name_without_generics,
                                    invocation,
                                    block,
                                    &variables,
                                    &blocks,
                                    statement_idx,
                                    storage,
                                    cmpop,
                                )?;

                                jump_processed = true;
                            }
                            name_without_generics
                                if is_uint_overflow_libfunc(name_without_generics) =>
                            {
                                self.inline_int_overflowing_op(
                                    name_without_generics,
                                    invocation,
                                    block,
                                    &variables,
                                    &blocks,
                                    statement_idx,
                                    storage,
                                    name_without_generics.ends_with("add"),
                                )?;

                                jump_processed = true;
                            }
                            name_without_generics
                                if is_uint_try_from_libfunc(name_without_generics) =>
                            {
                                self.inline_try_from_felt252(
                                    &id,
                                    invocation,
                                    &region,
                                    block,
                                    &variables,
                                    &blocks,
                                    statement_idx,
                                    storage,
                                )?;

                                jump_processed = true;
                            }
                            "enum_match" => {
                                self.inline_enum_match(
                                    &id,
                                    statement_idx,
                                    &region,
                                    block,
                                    &blocks,
                                    invocation,
                                    &variables,
                                    storage,
                                )?;
                                jump_processed = true;
                            }
                            "array_get" => {
                                self.inline_array_get(
                                    &id,
                                    statement_idx,
                                    &region,
                                    block,
                                    &blocks,
                                    invocation,
                                    &variables,
                                    storage,
                                )?;
                                jump_processed = true;
                            }
                            "array_pop_front" => {
                                self.inline_array_pop_front(
                                    &id,
                                    statement_idx,
                                    &region,
                                    block,
                                    &blocks,
                                    invocation,
                                    &variables,
                                    storage,
                                )?;
                                jump_processed = true;
                            }
                            "downcast" => {
                                self.inline_downcast(
                                    &id,
                                    invocation,
                                    &region,
                                    block,
                                    &variables,
                                    &blocks,
                                    statement_idx,
                                    storage,
                                )?;
                                jump_processed = true;
                            }
                            "function_call" => self.process_function_call(
                                &id,
                                invocation,
                                block,
                                &mut variables,
                                storage,
                            )?,
                            _ => self.process_general_libfunc(
                                &id,
                                invocation,
                                block,
                                &mut variables,
                                storage,
                            )?,
                        }

                        if statement_idx == block_info.end - 1 && !jump_processed {
                            let target = blocks
                                .get(&(statement_idx + 1))
                                .expect("Block should be registered for fallthrough successor");

                            let operand_values = target
                                .variables_at_start
                                .keys()
                                .map(|id| variables.get(id).unwrap().get_value())
                                .collect_vec();
                            self.op_br(block, &target.block, &operand_values);
                        }
                    }
                    GenStatement::Return(ret_args) => {
                        let userfunc_def = storage.userfuncs.get(&user_func_name).unwrap();
                        let ret_values = userfunc_def
                            .return_types
                            .iter()
                            .map(|id| {
                                variables
                                    .get(&ret_args[id.loc].id)
                                    .expect("Variable should be registered before return")
                                    .get_value()
                            })
                            .collect_vec();

                        self.op_return(block, &ret_values);
                    }
                }
            }
        }

        for (_block_start, block_info) in blocks {
            region.append_block(block_info.block);
        }

        let user_func_def = storage.userfuncs.get(user_func_name.as_str()).unwrap();
        let function_type = create_fn_signature(
            &user_func_def.args.iter().map(|t| t.ty.get_type()).collect_vec(),
            &user_func_def.return_types.iter().map(|t| t.ty.get_type()).collect_vec(),
        );
        let func = self.op_func(
            &user_func_name,
            &function_type,
            vec![region],
            FnAttributes {
                public: true,
                emit_c_interface: true,
                local: true,
                inline: false,
                norecurse: false,
                nounwind: false,
            },
        )?;
        self.module.body().append_operation(func);
        Ok(())
    }

    fn get_blocks_with_mapped_inputs(
        &'ctx self,
        func: &GenFunction<StatementIdx>,
        block_flows: BTreeMap<usize, BlockFlow>,
        storage: &mut Storage<'ctx>,
    ) -> BTreeMap<usize, BlockInfo<'ctx>> {
        self.calculate_dataflow_per_function(func, block_flows, storage)
            .iter()
            .map(|(block_start, data_flow)| {
                let block = Block::new(&[]);
                (
                    *block_start,
                    BlockInfo {
                        variables_at_start: data_flow.input_variables.clone(),
                        block,
                        end: data_flow.block_flow.end,
                    },
                )
            })
            .map(|(block_start, block_info)| {
                for var_type in block_info.variables_at_start.values() {
                    block_info
                        .block
                        .add_argument(var_type.get_type(), Location::unknown(&self.context));
                }
                (block_start, block_info)
            })
            .collect::<BTreeMap<_, _>>()
    }

    fn create_function_entry_block(
        &'ctx self,
        region: &Region,
        user_func_name: &str,
        func_start: usize,
        func: &GenFunction<StatementIdx>,
        blocks: &BTreeMap<usize, BlockInfo>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let arg_types = &storage.userfuncs.get(user_func_name).unwrap().args;
        let entry_block = Block::new(
            &arg_types
                .iter()
                .map(|t| (t.ty.get_type(), Location::unknown(&self.context)))
                .collect_vec(),
        );
        let block_info = &blocks.get(&func_start).unwrap();

        let mut args_to_pass = vec![];
        for (position, arg) in arg_types.iter().enumerate() {
            let param = &func.params[arg.loc];
            if block_info.variables_at_start.contains_key(&param.id.id) {
                let arg = entry_block.argument(position)?;
                args_to_pass.push(arg.into());
            }
        }
        self.op_br(&entry_block, &block_info.block, &args_to_pass);
        region.append_block(entry_block);
        Ok(())
    }

    fn calculate_dataflow_per_function(
        &'ctx self,
        func: &GenFunction<StatementIdx>,
        block_flows: BTreeMap<usize, BlockFlow>,
        storage: &mut Storage<'ctx>,
    ) -> BTreeMap<usize, DataFlow<'ctx>> {
        let user_func_name = func.id.debug_name.as_ref().unwrap().to_string();

        let mut block_infos: BTreeMap<usize, DataFlowInfo<'ctx>> = BTreeMap::new();

        // Collect the variables required by the invocations in each block, and those produced by the invocations
        for (block_start, block_flow) in block_flows {
            let mut required_args = BTreeMap::new();
            let mut variables_created = HashSet::new();

            for statement in self.program.statements[block_start..block_flow.end].iter() {
                let vars_used = match statement {
                    GenStatement::Invocation(invocation) => {
                        let id = invocation.libfunc_id.debug_name.as_ref().unwrap().to_string();

                        let name_without_generics = id.split('<').next().unwrap();

                        if name_without_generics == "function_call" {
                            let callee_name = id
                                .strip_prefix("function_call<user@")
                                .unwrap()
                                .strip_suffix('>')
                                .unwrap();
                            let arg_types = &storage
                                .userfuncs
                                .get(callee_name)
                                .cloned()
                                .expect("UserFunc should have been registered")
                                .args;
                            let arg_indices = &invocation.args;
                            arg_types
                                .iter()
                                .map(|arg| (&arg_indices[arg.loc], arg.ty.clone()))
                                .collect_vec()
                        } else {
                            let libfunc = storage.libfuncs.get(&id).cloned().unwrap_or_else(|| {
                                panic!("LibFunc {id} should have been registered")
                            });
                            let libfunc_args = libfunc.get_args();
                            libfunc_args
                                .into_iter()
                                .map(|arg| (&invocation.args[arg.loc], arg.ty.clone()))
                                .collect_vec()
                        }
                    }
                    GenStatement::Return(ret) => {
                        let func_ret_types =
                            &storage.userfuncs.get(&user_func_name).unwrap().return_types;
                        func_ret_types
                            .iter()
                            .map(|arg| (&ret[arg.loc], arg.ty.clone()))
                            .collect_vec()
                    }
                };

                for (var_id, var_type) in vars_used {
                    if !variables_created.contains(&var_id.id) {
                        required_args.insert(var_id.id, var_type.clone());
                    }
                }

                if let GenStatement::Invocation(invocation) = statement {
                    variables_created.extend(
                        invocation
                            .branches
                            .iter()
                            .flat_map(|branch| branch.results.iter().map(|v| v.id)),
                    );
                }
            }

            block_infos
                .insert(block_start, DataFlowInfo { required_args, variables_created, block_flow });
        }

        let block_infos = propagate_required_vars(block_infos);

        block_infos
            .into_iter()
            .map(|(block_start, data_flow)| {
                (
                    block_start,
                    DataFlow {
                        input_variables: data_flow.required_args,
                        block_flow: data_flow.block_flow,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>()
    }
}

fn is_int_is_zero_libfunc(name_without_generics: &str) -> bool {
    name_without_generics == "u8_is_zero"
        || name_without_generics == "u16_is_zero"
        || name_without_generics == "u32_is_zero"
        || name_without_generics == "u64_is_zero"
        || name_without_generics == "u128_is_zero"
        || name_without_generics == "felt252_is_zero"
}

fn is_int_cmp_libfunc(name_without_generics: &str) -> bool {
    let is_cmp: Regex = Regex::new(r#"u\d{1,3}_(eq|le|lt)"#).unwrap();
    is_cmp.is_match(name_without_generics)
}

fn is_uint_overflow_libfunc(name_without_generics: &str) -> bool {
    let is_reg: Regex = Regex::new(r#"u\d{1,3}_overflowing_(add|sub)"#).unwrap();
    is_reg.is_match(name_without_generics)
}

fn is_uint_try_from_libfunc(name_without_generics: &str) -> bool {
    let is_reg: Regex = Regex::new(r#"u(8|16|32|64|128)_try_from_felt252"#).unwrap();
    is_reg.is_match(name_without_generics)
}

fn calculate_block_ranges_per_function(
    program: &Program,
) -> BTreeMap<usize, (&GenFunction<StatementIdx>, BTreeMap<usize, BlockFlow>)> {
    let block_flows = get_block_flow(program);

    // Collecting the functions into a btreemap by entry points means we can easily sort the blocks into their containing functions
    let mut funcs_by_entry = program
        .funcs
        .iter()
        .map(|func| (func.entry_point.0, (func, BTreeMap::new())))
        .collect::<BTreeMap<_, _>>();

    for block_flow in block_flows {
        let (_, (_, blocks)) = funcs_by_entry
            .range_mut((Included(0), Included(block_flow.start)))
            .next_back()
            .unwrap();

        blocks.insert(block_flow.start, block_flow);
    }

    funcs_by_entry
}

fn get_block_flow(program: &Program) -> Vec<BlockFlow> {
    // Initially we can say that all function entry points are block entry points
    let func_starts = program.funcs.iter().map(|func| func.entry_point.0);

    // We can also say that all jump targets are block entry points
    let block_starts = func_starts.chain(get_jump_targets(program).into_iter()).collect_vec();

    // Next, scan forward from each block entry point for anything that would terminate a block
    // This produces block flows without predecessors filled in
    let mut block_flows = block_starts
        .iter()
        .map(|start| {
            let (end, successors) = program
                .statements
                .iter()
                .enumerate()
                .skip(*start)
                .find_map(|(statement_id, statement)| {
                    match statement {
                        GenStatement::Invocation(invocation) => {
                            let targets = invocation
                                .branches
                                .iter()
                                .map(|branch| match branch.target {
                                    GenBranchTarget::Fallthrough => statement_id + 1,
                                    GenBranchTarget::Statement(target_id) => target_id.0,
                                })
                                .collect_vec();

                            if targets.is_empty() {
                                unreachable!("invocations always have at least one target")
                            } else if targets.len() > 1
                                || block_starts.contains(&targets[0])
                                // Special case added for enum match for safety.
                                // The cairo compiler optimises away single-value enums,
                                // however just in case manual sierra is written with one this is used so that enum_match is always a block terminator, simplifying the implementation
                                || invocation
                                    .libfunc_id
                                    .debug_name
                                    .as_ref()
                                    .unwrap()
                                    .starts_with("enum_match<")
                            {
                                Some((statement_id + 1, HashSet::from_iter(targets.into_iter())))
                            } else {
                                None
                            }
                        }
                        // Returns terminate a block with no successors
                        GenStatement::Return(_) => Some((statement_id + 1, HashSet::new())),
                    }
                })
                .unwrap();

            BlockFlow { start: *start, end, successors, predecessors: HashSet::new() }
        })
        .collect_vec();

    // Finally, fill in the predecessors
    let mut preds: HashMap<usize, HashSet<usize>> = HashMap::new();
    for block_flow in block_flows.iter_mut() {
        for succ in block_flow.successors.iter() {
            if let Some(set) = preds.get_mut(succ) {
                set.insert(block_flow.start);
            } else {
                preds.insert(*succ, HashSet::from([block_flow.start]));
            }
        }
    }
    for block_flow in block_flows.iter_mut() {
        if let Some(block_preds) = preds.get(&block_flow.start) {
            block_flow.predecessors = block_preds.clone();
        }
    }

    block_flows
}

fn get_jump_targets(program: &Program) -> HashSet<usize> {
    program
        .statements
        .iter()
        .enumerate()
        // Filter out returns since they aren't relevant to dataflow within a function
        .filter_map(|(id, statement)| match statement {
            GenStatement::Invocation(invocation) => Some((id, invocation)),
            GenStatement::Return(_) => None,
        })
        // Filter out single target fallthroughs only. Fallthroughs from multi-target invocations must remain
        .filter(|(_id, invocation)| {
            invocation.branches.len() > 1
                || invocation.branches[0].target != GenBranchTarget::Fallthrough
        })
        // Get the target index for each jump
        .flat_map(|(id, invocation)| {
            invocation
                .branches
                .iter()
                .map(move |branch| match branch.target {
                    GenBranchTarget::Fallthrough => id + 1,
                    GenBranchTarget::Statement(statement_id) => statement_id.0,
                })
        })
        .collect::<HashSet<_>>()
}

// Loop through the block flows, propagating forward required variables
// If a block's successor requires variable 3, but the block does not produce variable 3, then the block also requires variable 3
// This process is repeated until all transitive requirements are satisfied
fn propagate_required_vars(
    mut block_infos: BTreeMap<usize, DataFlowInfo>,
) -> BTreeMap<usize, DataFlowInfo> {
    // Since this algorithm uses a loop clause, we calculate a simple upper bound on the number of steps and assert that it should never exceed this
    let iteration_limit =
        block_infos.values().map(|x| x.required_args.len()).max().unwrap_or(1) * block_infos.len();

    loop {
        let mut iteration_count = 0;
        let mut variables_to_add: HashMap<usize, HashMap<u64, SierraType>> = HashMap::new();
        for (block_start, DataFlowInfo { required_args, variables_created, block_flow }) in
            block_infos.iter()
        {
            let variables_required_by_successors = block_flow
                .successors
                .iter()
                .flat_map(|succ| block_infos[succ].required_args.clone())
                .filter(|(var_id, _var_type)| {
                    !variables_created.contains(var_id) && !required_args.contains_key(var_id)
                })
                .collect::<HashMap<_, _>>();

            if !variables_required_by_successors.is_empty() {
                variables_to_add.insert(*block_start, variables_required_by_successors);
            }
        }

        // Once every block's predeccesors provide all the variables it needs we can stop iterating
        if variables_to_add.is_empty() {
            break;
        } else {
            for (block_start, vars) in variables_to_add {
                block_infos.get_mut(&block_start).unwrap().required_args.extend(vars.into_iter());
            }
        }

        iteration_count += 1;
        if iteration_count > iteration_limit {
            panic!("Bug found in dataflow propagation algorithm, iteration limit exceeded");
        }
    }

    block_infos
}
