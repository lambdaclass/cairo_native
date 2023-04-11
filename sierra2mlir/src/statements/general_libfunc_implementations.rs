use std::collections::HashMap;

use cairo_lang_sierra::program::{GenInvocation, StatementIdx};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::Block;

use crate::{
    compiler::{Compiler, SierraType, Storage},
    libfuncs::lib_func_def::{ConstantLibFunc, LibFuncArg, LibFuncDef, SierraLibFunc},
};

use super::Variable;

impl<'block, 'ctx> Compiler<'ctx> {
    pub fn process_general_libfunc(
        &self,
        id: &str,
        invocation: &GenInvocation<StatementIdx>,
        block: &'block Block,
        variables: &mut HashMap<u64, Variable<'block>>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let libfunc_def =
            storage.libfuncs.get(id).unwrap_or_else(|| panic!("Unhandled libfunc {id}"));

        if libfunc_def.naively_skippable() {
            return Ok(());
        }

        match libfunc_def {
            SierraLibFunc::Function(LibFuncDef { args, return_types }) => self
                .process_libfunc_as_function_call(
                    id,
                    args,
                    return_types,
                    invocation,
                    block,
                    variables,
                ),
            SierraLibFunc::Constant(ConstantLibFunc { ty, value }) => {
                self.process_constant_libfunc(value, ty, invocation, block, variables);
                Ok(())
            }
            SierraLibFunc::InlineDataflow(args_forwarded) => {
                self.process_dataflow_libfunc(args_forwarded, invocation, variables);
                Ok(())
            }
        }
    }

    fn process_libfunc_as_function_call(
        &self,
        id: &str,
        args: &[LibFuncArg],
        return_types: &Vec<Vec<SierraType>>,
        invocation: &GenInvocation<StatementIdx>,
        block: &'block Block,
        variables: &mut HashMap<u64, Variable<'block>>,
    ) -> Result<()> {
        let arg_values = args
            .iter()
            .map(|a| variables.get(&invocation.args[a.loc].id).unwrap().get_value())
            .collect_vec();
        assert_eq!(
            return_types.len(),
            1,
            "Libfunc with abnormal number of returns not handled properly"
        );
        let return_types = return_types[0].iter().map(SierraType::get_type).collect_vec();
        let op = self.op_func_call(block, id, &arg_values, &return_types)?;
        variables.extend(invocation.branches[0].results.iter().enumerate().map(
            |(result_pos, var_id)| (var_id.id, Variable::Local { op, result_idx: result_pos }),
        ));

        Ok(())
    }

    fn process_constant_libfunc(
        &self,
        value: &str,
        ty: &SierraType,
        invocation: &GenInvocation<StatementIdx>,
        block: &'block Block,
        variables: &mut HashMap<u64, Variable<'block>>,
    ) {
        let op = self.op_const(block, value, ty.get_type());
        let var_id = &invocation.branches[0].results[0];
        variables.insert(var_id.id, Variable::Local { op, result_idx: 0 });
    }

    fn process_dataflow_libfunc(
        &self,
        args_forwarded: &[LibFuncArg],
        invocation: &GenInvocation<StatementIdx>,
        variables: &mut HashMap<u64, Variable<'block>>,
    ) {
        for (res_idx, arg) in args_forwarded.iter().enumerate() {
            let arg_id = invocation.args[arg.loc].id;
            let arg = variables.get(&arg_id).unwrap();
            variables.insert(invocation.branches[0].results[res_idx].id, *arg);
        }
    }
}
