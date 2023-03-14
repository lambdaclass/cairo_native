use cairo_lang_sierra::program::{GenericArg, LibfuncDeclaration};
use color_eyre::Result;
use melior_next::ir::{Block, Location, Operation, Region, Value};
use tracing::debug;

use crate::compiler::{Compiler, Storage};

impl<'ctx> Compiler<'ctx> {
    pub fn process_libfuncs(&'ctx self, storage: &mut Storage<'ctx>) -> Result<()> {
        for func_decl in &self.program.libfunc_declarations {
            let _id = func_decl.id.id;
            let name = func_decl.long_id.generic_id.0.as_str();
            debug!(name, "processing libfunc decl");

            match name {
                // no-ops
                "revoke_ap_tracking" => continue,
                "disable_ap_tracking" => continue,
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
                "felt_mul" => {
                    let func = self.felt_mul_create()?;
                    self.module.body().append_operation(func);
                }
                "dup" => {
                    let func = self.felt_add_dup(func_decl, storage)?;
                    self.module.body().append_operation(func);
                }
                _ => debug!(?func_decl, "unhandled libfunc"),
            }
        }

        debug!(types = ?storage.types, "processed");
        Ok(())
    }

    pub fn felt_add_dup(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &Storage<'ctx>,
    ) -> Result<Operation<'ctx>> {
        let mut args = vec![];

        for arg in &func_decl.long_id.generic_args {
            match arg {
                GenericArg::UserType(_) => todo!(),
                GenericArg::Type(type_id) => {
                    let ty = storage
                        .types
                        .get(&type_id.id.to_string())
                        .expect("type to exist");
                    self.collect_types(&mut args, ty);
                }
                GenericArg::Value(_) => todo!(),
                GenericArg::UserFunc(_) => todo!(),
                GenericArg::Libfunc(_) => todo!(),
            }
        }

        let region = Region::new();

        let block = Block::new(&args);

        // Return the results, 2 times.
        let mut results: Vec<Value> = vec![];

        for i in 0..block.argument_count() {
            let arg = block.argument(i)?;
            results.push(arg.into());
        }

        // 2 times, duplicate.
        for i in 0..block.argument_count() {
            let arg = block.argument(i)?;
            results.push(arg.into());
        }

        self.op_return(&block, &results);

        region.append_block(block);

        let mut return_types = Vec::with_capacity(args.len() * 2);
        return_types.extend_from_slice(&args);
        return_types.extend_from_slice(&args);

        let function_type = self.create_fn_signature(&args, &return_types);

        let func = self.op_func("dup", &function_type, vec![region], false)?;

        Ok(func)
    }

    pub fn felt_add_create(&'ctx self) -> Result<Operation<'ctx>> {
        let felt_type = self.felt_type();
        let loc = Location::unknown(&self.context);

        let region = Region::new();
        let block = Block::new(&[(felt_type, loc), (felt_type, loc)]);

        let lhs_arg = block.argument(0)?;
        let rhs_arg = block.argument(0)?;

        let lhs_ext = self.op_sext(&block, lhs_arg.into(), self.double_felt_type());
        let lhs = lhs_ext.result(0)?;

        let rhs_ext = self.op_sext(&block, rhs_arg.into(), self.double_felt_type());
        let rhs = rhs_ext.result(0)?;

        let res = self.op_add(&block, lhs.into(), rhs.into());
        let res_result = res.result(0)?;

        let res = self.op_felt_modulo(&block, res_result.into())?;
        let res_result = res.result(0)?;

        self.op_return(&block, &[res_result.into()]);

        region.append_block(block);

        let func = self.op_func(
            "felt_add",
            &format!("({felt_type}, {felt_type}) -> {felt_type}"),
            vec![region],
            false,
        )?;

        Ok(func)
    }

    pub fn felt_sub_create(&'ctx self) -> Result<Operation<'ctx>> {
        let felt_type = self.felt_type();
        let loc = Location::unknown(&self.context);

        let region = Region::new();
        let block = Block::new(&[(felt_type, loc), (felt_type, loc)]);

        let lhs_arg = block.argument(0)?;
        let rhs_arg = block.argument(0)?;

        let lhs_ext = self.op_sext(&block, lhs_arg.into(), self.double_felt_type());
        let lhs = lhs_ext.result(0)?;

        let rhs_ext = self.op_sext(&block, rhs_arg.into(), self.double_felt_type());
        let rhs = rhs_ext.result(0)?;

        let res = self.op_sub(&block, lhs.into(), rhs.into());
        let res_result = res.result(0)?;

        let res = self.op_felt_modulo(&block, res_result.into())?;
        let res_result = res.result(0)?;

        self.op_return(&block, &[res_result.into()]);

        region.append_block(block);

        let func = self.op_func(
            "felt_sub",
            &format!("({felt_type}, {felt_type}) -> {felt_type}"),
            vec![region],
            false,
        )?;

        Ok(func)
    }

    pub fn felt_mul_create(&'ctx self) -> Result<Operation<'ctx>> {
        let felt_type = self.felt_type();
        let loc = Location::unknown(&self.context);

        let region = Region::new();
        let block = Block::new(&[(felt_type, loc), (felt_type, loc)]);

        let lhs_arg = block.argument(0)?;
        let rhs_arg = block.argument(0)?;

        let lhs_ext = self.op_sext(&block, lhs_arg.into(), self.double_felt_type());
        let lhs = lhs_ext.result(0)?;

        let rhs_ext = self.op_sext(&block, rhs_arg.into(), self.double_felt_type());
        let rhs = rhs_ext.result(0)?;

        let res = self.op_mul(&block, lhs.into(), rhs.into());
        let res_result = res.result(0)?;

        let res = self.op_felt_modulo(&block, res_result.into())?;
        let res_result = res.result(0)?;

        self.op_return(&block, &[res_result.into()]);

        region.append_block(block);

        let func = self.op_func(
            "felt_mul",
            &format!("({felt_type}, {felt_type}) -> {felt_type}"),
            vec![region],
            false,
        )?;

        Ok(func)
    }
}
