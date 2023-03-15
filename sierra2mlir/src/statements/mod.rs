use std::collections::HashMap;

use cairo_lang_sierra::{ids::VarId, program::GenStatement};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{Block, Location, OperationRef, Region, Type, Value};
use tracing::debug;

use crate::compiler::{Compiler, SierraType, Storage};

impl<'ctx> Compiler<'ctx> {
    pub fn process_statements(&self, storage: Storage<'ctx>) -> Result<Storage<'ctx>> {
        let loc = Location::unknown(&self.context);

        for func in &self.program.funcs {
            debug!(?func, "processing func");

            let name = func.id.id.to_string();
            let entry = func.entry_point.0;
            let mut params = vec![];
            let mut return_types = vec![];

            for param in &func.params {
                let ty = storage
                    .types
                    .get(&param.ty.id.to_string())
                    .expect("type for param should exist");

                self.collect_types(&mut params, ty);
            }

            dbg!(&storage.types);
            for ret in &func.signature.ret_types {
                let ty = storage
                    .types
                    .get(&ret.id.to_string())
                    .expect("type for param should exist");
                self.collect_types(&mut return_types, ty);
            }

            // The varid -> operation ref which holds the results, usize is the index into the results.
            let mut variables: HashMap<&VarId, (OperationRef, usize)> = HashMap::new();
            let mut param_values: HashMap<&VarId, Value> = HashMap::new();

            let region = Region::new();
            let block = Block::new(&params);

            for (i, param) in func.params.iter().enumerate() {
                let value = block.argument(i)?;
                param_values.insert(&param.id, value.into());
            }

            let statements_entry = self.program.statements.iter().skip(entry);
            let mut current_statements_iter = statements_entry.clone();

            // Use a loop so we can change the iterator in the middle of it in case of jumps.
            loop {
                if let Some(statement) = current_statements_iter.next() {
                    match statement {
                        GenStatement::Invocation(inv) => {
                            let name = inv.libfunc_id.debug_name.as_ref().unwrap().as_str();
                            let id = inv.libfunc_id.id.to_string();
                            debug!(name, "processing statement: invocation");

                            let name_without_generics = name.split('<').next().unwrap();

                            match name_without_generics {
                                "felt_const" => {
                                    let felt_const = storage
                                        .felt_consts
                                        .get(&inv.libfunc_id.id.to_string())
                                        .expect("constant should exist");
                                    let op = self.op_felt_const(&block, felt_const);
                                    let var_id = &inv.branches[0].results[0];
                                    variables.insert(var_id, (op, 0));
                                }
                                "store_temp" => continue,
                                "jump" => todo!(),
                                _name if inv.branches.len() > 1 => {
                                    todo!(
                                        "invocations with multiple branches need to be implemented: {}", name
                                    )
                                }
                                name => {
                                    let func_def = storage
                                        .functions
                                        .get(&id)
                                        .expect("should find the libfunc def");
                                    let mut args = vec![];

                                    for var in &inv.args {
                                        if let Some((op_res, result_index)) = variables.get(&var) {
                                            let res = op_res.result(*result_index)?;
                                            args.push(res.into());
                                        } else {
                                            let res = param_values
                                                .get(var)
                                                .expect("couldn't find variable");
                                            args.push(*res);
                                        };
                                    }

                                    let op = self.op_func_call(
                                        &block,
                                        name,
                                        &args,
                                        &func_def.return_types,
                                    )?;

                                    let results = &inv.branches[0].results;

                                    assert_eq!(results.len(), op.result_count());

                                    for (i, res) in results.iter().enumerate() {
                                        variables.insert(res, (op, i));
                                    }

                                    // debug!(name, "unimplemented");
                                }
                            }
                        }
                        GenStatement::Return(ret) => {
                            let mut ret_values: Vec<Value> = vec![];

                            for var in ret {
                                if let Some((op_res, result_index)) = variables.get(&var) {
                                    let res = op_res.result(*result_index)?;
                                    ret_values.push(res.into());
                                } else {
                                    let res =
                                        param_values.get(var).expect("couldn't find variable");
                                    ret_values.push(*res);
                                };
                            }

                            self.op_return(&block, &ret_values);
                            debug!(?ret, "processing statement: return");
                            break;
                        }
                    }
                }
            }

            region.append_block(block);

            let function_type = self.create_fn_signature(&params, &return_types);

            dbg!(&function_type);

            let op = self.op_func(&name, &function_type, vec![region], false)?;

            self.module.body().append_operation(op);
        }
        Ok(storage)
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

    pub fn collect_types(
        &'ctx self,
        data: &mut Vec<(Type<'ctx>, Location<'ctx>)>,
        ty: &'ctx SierraType,
    ) {
        match ty {
            SierraType::Simple(ty) => {
                data.push((*ty, Location::unknown(&self.context)));
            }
            SierraType::Struct(types) => {
                for ty in types {
                    self.collect_types(data, ty);
                }
            }
        }
    }
}
