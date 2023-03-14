use cairo_lang_sierra::program::{GenericArg, LibfuncDeclaration};
use color_eyre::Result;
use melior_next::ir::{Block, Location, Operation, Region, Value, ValueLike};
use tracing::debug;

use crate::compiler::{Compiler, FunctionDef, Storage};

impl<'ctx> Compiler<'ctx> {
    pub fn process_libfuncs(&'ctx self, storage: &'ctx mut Storage<'ctx>) -> Result<()> {
        for func_decl in &self.program.libfunc_declarations {
            let _id = func_decl.id.id;
            let name = func_decl.long_id.generic_id.0.as_str();
            debug!(name, "processing libfunc decl");

            match name {
                // no-ops
                "revoke_ap_tracking" => continue,
                "disable_ap_tracking" => continue,
                "rename" | "drop" | "store_temp" => continue,
                "felt_const" => {
                    self.create_libfunc_felt_const(func_decl, storage);
                }
                "felt_add" => {
                    let func = self.create_libfunc_felt_add(func_decl, storage)?;
                    self.module.body().append_operation(func);
                }
                "felt_sub" => {
                    let func = self.create_libfunc_felt_sub(func_decl, storage)?;
                    self.module.body().append_operation(func);
                }
                "felt_mul" => {
                    let func = self.create_libfunc_felt_mul(func_decl, storage)?;
                    self.module.body().append_operation(func);
                }
                "dup" => {
                    let func = self.create_libfunc_dup(func_decl, storage)?;
                    self.module.body().append_operation(func);
                }
                _ => debug!(?func_decl, "unhandled libfunc"),
            }
        }

        debug!(types = ?storage.types, "processed");
        Ok(())
    }

    pub fn create_libfunc_felt_const(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) {
        let arg = match &func_decl.long_id.generic_args[0] {
            GenericArg::Value(value) => value.to_string(),
            _ => unimplemented!("should always be value"),
        };

        storage.felt_consts.insert(func_decl.id.id.to_string(), arg);
    }

    pub fn create_libfunc_dup(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<Operation<'ctx>> {
        let id = func_decl.id.id.to_string();
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

        let func = self.op_func(&id, &function_type, vec![region], false)?;

        storage.functions.insert(
            id,
            FunctionDef {
                args: args.iter().map(|x| x.0).collect(),
                return_types: return_types.iter().map(|x| x.0).collect(),
            },
        );

        Ok(func)
    }

    pub fn create_libfunc_felt_add(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<Operation<'ctx>> {
        let id = func_decl.id.id.to_string();
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
            &id,
            &format!("({felt_type}, {felt_type}) -> {felt_type}"),
            vec![region],
            false,
        )?;

        storage.functions.insert(
            id,
            FunctionDef {
                args: vec![felt_type, felt_type],
                return_types: vec![felt_type],
            },
        );

        Ok(func)
    }

    pub fn create_libfunc_felt_sub(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<Operation<'ctx>> {
        let id = func_decl.id.id.to_string();
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
            &id,
            &format!("({felt_type}, {felt_type}) -> {felt_type}"),
            vec![region],
            false,
        )?;

        storage.functions.insert(
            id,
            FunctionDef {
                args: vec![felt_type, felt_type],
                return_types: vec![felt_type],
            },
        );

        Ok(func)
    }

    pub fn create_libfunc_felt_mul(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<Operation<'ctx>> {
        let id = func_decl.id.id.to_string();
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
            &id,
            &format!("({felt_type}, {felt_type}) -> {felt_type}"),
            vec![region],
            false,
        )?;

        storage.functions.insert(
            id,
            FunctionDef {
                args: vec![felt_type, felt_type],
                return_types: vec![felt_type],
            },
        );

        Ok(func)
    }
}
