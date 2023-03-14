use std::collections::HashMap;

use cairo_lang_sierra::program::GenStatement;
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{Block, Location, Region, Type};
use tracing::debug;

use crate::compiler::{Compiler, SierraType, Storage};

impl<'ctx> Compiler<'ctx> {
    pub fn process_statements(&'ctx self, storage: &mut Storage<'ctx>) -> Result<()> {
        let loc = Location::unknown(&self.context);

        for func in &self.program.funcs {
            continue;
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

            let region = Region::new();
            let block = Block::new(&params);

            let statements_entry = self.program.statements.iter().skip(entry);
            let statements = statements_entry.clone();

            //let mut variables = HashMap::new();

            for statement in statements {
                match statement {
                    GenStatement::Invocation(inv) => {
                        debug!(name = ?inv.libfunc_id.debug_name, "processing statement: invocation");
                    }
                    GenStatement::Return(ret) => {
                        self.op_return(&block, &[]);
                        debug!(?ret, "processing statement: return");
                        break;
                    }
                }
            }

            region.append_block(block);

            let function_type = self.create_fn_signature(&params, &return_types);

            dbg!(&function_type);

            let op = self.op_func(&name, &function_type, vec![region], false)?;

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
