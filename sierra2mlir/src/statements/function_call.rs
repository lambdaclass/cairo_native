use std::collections::HashMap;

use cairo_lang_sierra::program::{GenInvocation, StatementIdx};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::Block;

use crate::compiler::{Compiler, Storage};

use super::Variable;

impl<'block, 'ctx> Compiler<'ctx> {
    pub fn process_function_call(
        &self,
        id: &str,
        invocation: &GenInvocation<StatementIdx>,
        block: &'block Block,
        variables: &mut HashMap<u64, Variable<'block>>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let callee_name =
            id.strip_prefix("function_call<user@").unwrap().strip_suffix('>').unwrap();

        // The callee definition determines which of the sierra function's arguments and returns we actually use
        // For example, the callee definition will not include reference to builtin arguments that we ignore
        let callee_def =
            storage.userfuncs.get(callee_name).unwrap_or_else(|| panic!("Unhandled libfunc {id}"));

        // Get the values to pass as the arguments
        let args = callee_def
            .args
            .iter()
            .map(|arg| variables.get(&invocation.args[arg.loc].id).unwrap().get_value())
            .collect_vec();

        // Get the return types so as to be able to construct the call
        let return_types = callee_def.return_types.iter().map(|t| t.ty.get_type()).collect_vec();

        let op = self.op_func_call(block, callee_name, &args, &return_types)?;

        // Save the results into the variables map, taking care to use those based on callee_def rather than the invocation itself
        variables.extend(callee_def.return_types.iter().enumerate().map(
            |(result_idx, ret_arg)| {
                (invocation.branches[0].results[ret_arg.loc].id, Variable::Local { op, result_idx })
            },
        ));
        Ok(())
    }
}
