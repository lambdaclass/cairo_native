use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet},
    rc::Rc,
};

use std::ops::Bound::Included;

use cairo_lang_sierra::program::{
    GenBranchTarget, GenFunction, GenStatement, Program, StatementIdx,
};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{block::Argument, Block, Location, OperationRef, Region, Value};

use crate::{
    compiler::{CmpOp, Compiler, SierraType, Storage},
    utility::create_fn_signature,
};

pub struct BlockInfo<'ctx> {
    pub variables_at_start: HashMap<u64, SierraType<'ctx>>,
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
                let res = op.result(*result_idx).unwrap();
                res.into()
            }
            Variable::Param { argument } => (*argument).into(),
        }
    }
}

// The information about dataflow needed to process statements
struct DataFlow<'ctx> {
    input_variables: HashMap<u64, SierraType<'ctx>>,
    block_flow: BlockFlow,
}

// The data required to calculate dataflow
struct DataFlowInfo<'ctx> {
    required_args: HashMap<u64, SierraType<'ctx>>,
    variables_created: HashSet<u64>,
    block_flow: BlockFlow,
}

impl<'ctx> Compiler<'ctx> {
    // Process the statements of the sierra program by breaking flow up into basic blocks and processing one at a time
    pub fn process_statements(&'ctx self, storage: Rc<RefCell<Storage<'ctx>>>) -> Result<()> {
        // Calculate the basic block structure in each function
        let block_ranges_per_function = calculate_block_ranges_per_function(&self.program);

        // Process the blocks for each function
        for (func_start, (func, block_flows)) in block_ranges_per_function {
            self.process_statements_for_function(func, func_start, block_flows, storage.clone())?;
        }

        Ok(())
    }

    fn process_statements_for_function(
        &'ctx self,
        func: &GenFunction<StatementIdx>,
        func_start: usize,
        block_flows: BTreeMap<usize, BlockFlow>,
        storage: Rc<RefCell<Storage<'ctx>>>,
    ) -> Result<()> {
        let region = Region::new();
        let user_func_name =
            Self::normalize_func_name(func.id.debug_name.as_ref().unwrap().as_str()).to_string();

        let blocks = self.get_blocks_with_mapped_inputs(func, block_flows, storage.clone());
        self.create_function_entry_block(
            &region,
            user_func_name.as_str(),
            func_start,
            func,
            &blocks,
            storage.clone(),
        )?;

        // We process statements one block at a time
        for (block_start, block_info) in blocks.iter() {
            let block = &block_info.block;

            // Variables holds the most recent value associated with a variable id as we progress through the block
            // Initially it only holds the blocks arguments
            let mut variables: HashMap<u64, Variable> = HashMap::new();
            for (arg_position, var_id) in block_info.variables_at_start.keys().sorted().enumerate()
            {
                let argument = block.argument(arg_position)?;
                variables.insert(*var_id, Variable::Param { argument });
            }

            for statement_idx in *block_start..block_info.end {
                let storage = storage.borrow();
                match &self.program.statements[statement_idx] {
                    GenStatement::Invocation(invocation) => {
                        let name = invocation.libfunc_id.debug_name.as_ref().unwrap().as_str();
                        let id = Self::normalize_func_name(
                            invocation.libfunc_id.debug_name.as_ref().unwrap().as_str(),
                        )
                        .to_string();
                        let name_without_generics = name.split('<').next().unwrap();
                        let mut jump_processed = false;
                        match name_without_generics {
                            "disable_ap_tracking" | "drop" | "branch_align" => continue,
                            "felt252_const" => {
                                let felt_const =
                                    storage.felt_consts.get(&id).expect("constant should exist");
                                let op = self.op_felt_const(block, felt_const);
                                let var_id = &invocation.branches[0].results[0];
                                variables.insert(var_id.id, Variable::Local { op, result_idx: 0 });
                            }
                            "jump" => {
                                let target_block_info = match &invocation.branches[0].target {
                                    GenBranchTarget::Fallthrough => {
                                        unreachable!("jump should never be fallthrough")
                                    }
                                    GenBranchTarget::Statement(id) => blocks.get(&(id.0)).unwrap(),
                                };
                                let mut operand_indices =
                                    target_block_info.variables_at_start.keys().collect_vec();
                                operand_indices.sort_unstable();
                                let operand_values = operand_indices
                                    .iter()
                                    .map(|id| variables.get(id).unwrap().get_value())
                                    .collect_vec();
                                self.op_br(block, &target_block_info.block, &operand_values);
                                jump_processed = true;
                            }
                            "u8_const" => {
                                let value =
                                    storage.u8_consts.get(&id).expect("constant value not found");
                                let op = self.op_u8_const(block, value);
                                let var_id = &invocation.branches[0].results[0];
                                variables.insert(var_id.id, Variable::Local { op, result_idx: 0 });
                            }
                            "u16_const" => {
                                let value =
                                    storage.u16_consts.get(&id).expect("constant value not found");
                                let op = self.op_u16_const(block, value);
                                let var_id = &invocation.branches[0].results[0];
                                variables.insert(var_id.id, Variable::Local { op, result_idx: 0 });
                            }
                            "u32_const" => {
                                let value =
                                    storage.u32_consts.get(&id).expect("constant value not found");
                                let op = self.op_u32_const(block, value);
                                let var_id = &invocation.branches[0].results[0];
                                variables.insert(var_id.id, Variable::Local { op, result_idx: 0 });
                            }
                            "u64_const" => {
                                let value =
                                    storage.u64_consts.get(&id).expect("constant value not found");
                                let op = self.op_u64_const(block, value);
                                let var_id = &invocation.branches[0].results[0];
                                variables.insert(var_id.id, Variable::Local { op, result_idx: 0 });
                            }
                            "u128_const" => {
                                let value =
                                    storage.u128_consts.get(&id).expect("constant value not found");
                                let op = self.op_u128_const(block, value);
                                let var_id = &invocation.branches[0].results[0];
                                variables.insert(var_id.id, Variable::Local { op, result_idx: 0 });
                            }
                            "felt252_is_zero" => {
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
                                        let mut operand_indices = target_block_info
                                            .variables_at_start
                                            .keys()
                                            .collect_vec();
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

                                self.op_cond_br(
                                    block,
                                    eq.into(),
                                    zero_block,
                                    nonzero_block,
                                    zero_vars,
                                    nonzero_vars,
                                )?;

                                jump_processed = true;
                            }
                            "function_call" => {
                                let callee_name = id
                                    .strip_prefix("function_call<user@")
                                    .unwrap()
                                    .strip_suffix('>')
                                    .unwrap();
                                let callee_def = storage
                                    .userfuncs
                                    .get(callee_name)
                                    .unwrap_or_else(|| panic!("Unhandled libfunc {name}"));
                                let args = invocation
                                    .args
                                    .iter()
                                    .map(|id| variables.get(&id.id).unwrap().get_value())
                                    .collect_vec();
                                let return_types = callee_def
                                    .return_types
                                    .iter()
                                    .map(|t| t.get_type())
                                    .collect_vec();
                                let op =
                                    self.op_func_call(block, callee_name, &args, &return_types)?;
                                variables.extend(
                                    invocation.branches[0].results.iter().enumerate().map(
                                        |(result_pos, var_id)| {
                                            (
                                                var_id.id,
                                                Variable::Local { op, result_idx: result_pos },
                                            )
                                        },
                                    ),
                                );
                            }
                            _ => {
                                let libfunc_def = storage
                                    .libfuncs
                                    .get(&id)
                                    .unwrap_or_else(|| panic!("Unhandled libfunc {name}"));
                                let args = invocation
                                    .args
                                    .iter()
                                    .map(|id| variables.get(&id.id).unwrap().get_value())
                                    .collect_vec();
                                let return_types = libfunc_def
                                    .return_types
                                    .iter()
                                    .map(|t| t.get_type())
                                    .collect_vec();
                                let op = self.op_func_call(block, &id, &args, &return_types)?;
                                variables.extend(
                                    invocation.branches[0].results.iter().enumerate().map(
                                        |(result_pos, var_id)| {
                                            (
                                                var_id.id,
                                                Variable::Local { op, result_idx: result_pos },
                                            )
                                        },
                                    ),
                                );
                            }
                        }

                        if statement_idx == block_info.end - 1 && !jump_processed {
                            let target = blocks
                                .get(&(statement_idx + 1))
                                .expect("Block should be registered for fallthrough successor");

                            let mut operand_indices =
                                target.variables_at_start.keys().collect_vec();
                            operand_indices.sort_unstable();
                            let operand_values = operand_indices
                                .iter()
                                .map(|id| variables.get(id).unwrap().get_value())
                                .collect_vec();
                            self.op_br(block, &target.block, &operand_values);
                        }
                    }
                    GenStatement::Return(ret_args) => {
                        let ret_values = ret_args
                            .iter()
                            .map(|id| {
                                variables
                                    .get(&id.id)
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

        let storage = storage.borrow();
        let user_func_def = storage.userfuncs.get(user_func_name.as_str()).unwrap();
        let function_type = create_fn_signature(
            &user_func_def.args.iter().map(|t| t.get_type()).collect_vec(),
            &user_func_def.return_types.iter().map(|t| t.get_type()).collect_vec(),
        );
        let func = self.op_func(&user_func_name, &function_type, vec![region], false, false)?;
        self.module.body().append_operation(func);
        Ok(())
    }

    fn get_blocks_with_mapped_inputs(
        &'ctx self,
        func: &GenFunction<StatementIdx>,
        block_flows: BTreeMap<usize, BlockFlow>,
        storage: Rc<RefCell<Storage<'ctx>>>,
    ) -> BTreeMap<usize, BlockInfo<'ctx>> {
        self.calculate_dataflow_per_function(func, block_flows, storage.clone())
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
                for (_, var_type) in block_info.variables_at_start.iter() {
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
        storage: Rc<RefCell<Storage<'ctx>>>,
    ) -> Result<()> {
        let storage = storage.borrow();
        let arg_types = &storage.userfuncs.get(user_func_name).unwrap().args;
        let entry_block = Block::new(
            &arg_types
                .iter()
                .map(|t| (t.get_type(), Location::unknown(&self.context)))
                .collect_vec(),
        );
        let block_info = &blocks.get(&func_start).unwrap();

        let mut args_to_pass = vec![];
        for (position, param) in func.params.iter().enumerate() {
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
        storage: Rc<RefCell<Storage<'ctx>>>,
    ) -> BTreeMap<usize, DataFlow> {
        let user_func_name =
            Self::normalize_func_name(func.id.debug_name.as_ref().unwrap().as_str()).to_string();

        let mut block_infos: BTreeMap<usize, DataFlowInfo> = BTreeMap::new();

        // Collect the variables required by the invocations in each block, and those produced by the invocations
        for (block_start, block_flow) in block_flows {
            let mut required_args = HashMap::new();
            let mut variables_created = HashSet::new();

            for statement in self.program.statements[block_start..block_flow.end].iter() {
                let storage = storage.borrow();
                let vars_used = match statement {
                    GenStatement::Invocation(invocation) => {
                        let id = Self::normalize_func_name(
                            invocation.libfunc_id.debug_name.as_ref().unwrap().as_str(),
                        )
                        .to_string();

                        let name_without_generics = id.split('<').next().unwrap();

                        // We ignore drop functions, even though they take arguments in sierra, since they are ignored during statement processing
                        if name_without_generics == "drop" {
                            continue;
                        } else if name_without_generics == "function_call" {
                            let callee_name = id
                                .strip_prefix("function_call<user@")
                                .unwrap()
                                .strip_suffix('>')
                                .unwrap();
                            let arg_types = &storage
                                .userfuncs
                                .get(callee_name)
                                .expect("UserFunc should have been registered")
                                .args;
                            let arg_indices = &invocation.args;
                            arg_indices.iter().zip_eq(arg_types.iter()).collect_vec()
                        } else {
                            let arg_types = &storage
                                .libfuncs
                                .get(&id)
                                .expect("LibFunc should have been registered")
                                .args;
                            let arg_indices = &invocation.args;
                            arg_indices.iter().zip_eq(arg_types.iter()).collect_vec()
                        }
                    }
                    GenStatement::Return(ret) => {
                        let func_ret_types =
                            &storage.userfuncs.get(&user_func_name).unwrap().return_types;
                        ret.iter().zip_eq(func_ret_types.iter()).collect_vec()
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
                            } else if (targets.len() == 1 && block_starts.contains(&targets[0]))
                                || targets.len() > 1
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
