use cairo_lang_sierra::{
    ids::GenericTypeId,
    program::{GenStatement, GenericArg},
};
use color_eyre::Result;
use melior_next::ir::{Block, Location, Operation, OperationRef, Region, Type};
use tracing::debug;

use crate::compiler::{Compiler, Storage};

impl<'ctx> Compiler<'ctx> {
    pub fn process_statements(&'ctx self, storage: &mut Storage<'ctx>) -> Result<()> {
        let loc = Location::unknown(&self.context);

        for func in &self.program.funcs {
            debug!(?func, "processing func");

            let entry = func.entry_point.0;
            let mut params = vec![];

            for param in &func.params {
                let ty = storage
                    .types
                    .get(&param.ty.id)
                    .expect("type for param should exist");
                params.push((*ty, loc));
            }

            let region = Region::new();
            let block = Block::new(&params);

            let statements_entry = self.program.statements.iter().skip(entry);
            let mut statements = statements_entry.clone();

            for statement in statements {
                match statement {
                    GenStatement::Invocation(inv) => {
                        debug!(name = ?inv.libfunc_id.debug_name, "processing statement: invocation");
                    }
                    GenStatement::Return(ret) => {
                        debug!(?ret, "processing statement: return");
                    }
                }
            }
        }
        Ok(())
    }
}
