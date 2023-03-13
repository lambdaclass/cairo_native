use cairo_lang_sierra::{ids::GenericTypeId, program::GenericArg};
use color_eyre::Result;
use melior_next::ir::{Block, Location, Operation, OperationRef, Region, Type};
use tracing::debug;

use crate::compiler::{Compiler, Storage};

impl<'ctx> Compiler<'ctx> {
    pub fn process_libfuncs(&'ctx self, storage: &mut Storage<'ctx>) -> Result<()> {
        for func_decl in &self.program.libfunc_declarations {
            let id = func_decl.id.id;
            let name = func_decl.long_id.generic_id.0.as_str();
            debug!(name, "processing libfunc decl");

            match name {
                // no-ops
                "revoke_ap_tracking" => continue,
                name if name.starts_with("rename") => continue,
                name if name.starts_with("drop") => continue,
                name if name.starts_with("store_temp") => continue,
                // handled directly, no need for a function
                name if name.starts_with("felt_const") => continue,
                "felt_add" => {
                    let func = self.felt_add_create()?;
                    self.module.body().append_operation(func);
                }
                "felt_sub" => {
                    let func = self.felt_sub_create()?;
                    self.module.body().append_operation(func);
                }
                _ => debug!(?func_decl, "unhandled libfunc"),
            }
        }

        debug!(types = ?storage.types, "processed");
        Ok(())
    }

    pub fn felt_add_create(&'ctx self) -> Result<Operation<'ctx>> {
        let felt_type = self.felt_type();
        let loc = Location::unknown(&self.context);

        let region = Region::new();
        let block = Block::new(&[(felt_type, loc), (felt_type, loc)]);

        let lhs_arg = block.argument(0)?;
        let rhs_arg = block.argument(0)?;

        let res = self.op_add(&block, lhs_arg.into(), rhs_arg.into());
        let res_result = res.result(0)?;

        // todo: modulo

        self.op_return(&block, &[res_result.into()]);

        region.append_block(block);

        let func = self.op_func(
            "felt_add",
            &format!("({felt_type}, {felt_type}) -> {felt_type}"),
            vec![region],
        );

        Ok(func)
    }

    pub fn felt_sub_create(&'ctx self) -> Result<Operation<'ctx>> {
        let felt_type = self.felt_type();
        let loc = Location::unknown(&self.context);

        let region = Region::new();
        let block = Block::new(&[(felt_type, loc), (felt_type, loc)]);

        let lhs_arg = block.argument(0)?;
        let rhs_arg = block.argument(0)?;

        let res = self.op_felt_sub(&block, lhs_arg.into(), rhs_arg.into());
        let res_result = res.result(0)?;

        // todo: modulo

        self.op_return(&block, &[res_result.into()]);

        region.append_block(block);

        let func = self.op_func(
            "felt_sub",
            &format!("({felt_type}, {felt_type}) -> {felt_type}"),
            vec![region],
        );

        Ok(func)
    }
}
