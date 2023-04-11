use std::collections::HashMap;

use cairo_lang_sierra::program::{GenInvocation, StatementIdx};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::Block;

use crate::compiler::{Compiler, Storage};

use super::Variable;

impl<'a, 'ctx> Compiler<'ctx> {
    pub fn process_function_call(
        &self,
        id: &str,
        invocation: &GenInvocation<StatementIdx>,
        block: &'a Block,
        variables: &mut HashMap<u64, Variable<'a>>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let callee_name =
            id.strip_prefix("function_call<user@").unwrap().strip_suffix('>').unwrap();
        let callee_def =
            storage.userfuncs.get(callee_name).unwrap_or_else(|| panic!("Unhandled libfunc {id}"));
        let args = invocation
            .args
            .iter()
            .map(|id| variables.get(&id.id).unwrap().get_value())
            .collect_vec();
        let return_types = callee_def.return_types.iter().map(|t| t.get_type()).collect_vec();
        let op = self.op_func_call(block, callee_name, &args, &return_types)?;
        variables.extend(invocation.branches[0].results.iter().enumerate().map(
            |(result_pos, var_id)| (var_id.id, Variable::Local { op, result_idx: result_pos }),
        ));
        Ok(())
    }
}
