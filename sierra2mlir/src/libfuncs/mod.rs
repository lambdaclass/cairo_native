use cairo_lang_sierra::program::{GenericArg, LibfuncDeclaration};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{
    Block, BlockRef, Location, Operation, OperationRef, Region, Type, Value, ValueLike,
};
use tracing::debug;

use crate::compiler::{Compiler, FunctionDef, Storage};

impl<'ctx> Compiler<'ctx> {
    pub fn process_libfuncs<'b: 'ctx>(
        &'b self,
        mut storage: Storage<'ctx>,
    ) -> Result<Storage<'ctx>> {
        for func_decl in &self.program.libfunc_declarations {
            let _id = func_decl.id.id;
            let name = func_decl.long_id.generic_id.0.as_str();
            debug!(name, "processing libfunc decl");

            let parent_block = self.module.body();

            match name {
                // no-ops
                "revoke_ap_tracking" => continue,
                "disable_ap_tracking" => continue,
                "rename" | "drop" | "store_temp" => continue,
                "felt_const" => {
                    self.create_libfunc_felt_const(func_decl, &mut storage);
                }
                "felt_add" => {
                    self.create_libfunc_felt_add(func_decl, &parent_block, &mut storage)?;
                }
                "felt_sub" => {
                    self.create_libfunc_felt_sub(func_decl, &parent_block, &mut storage)?;
                }
                "felt_mul" => {
                    self.create_libfunc_felt_mul(func_decl, &parent_block, &mut storage)?;
                }
                "dup" => {
                    self.create_libfunc_dup(func_decl, &parent_block, &mut storage)?;
                }
                _ => debug!(?func_decl, "unhandled libfunc"),
            }
        }

        debug!(types = ?storage.types, "processed");
        Ok(storage)
    }

    pub fn create_libfunc_felt_const(
        &self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) {
        let arg = match &func_decl.long_id.generic_args[0] {
            GenericArg::Value(value) => value.to_string(),
            _ => unimplemented!("should always be value"),
        };

        storage.felt_consts.insert(func_decl.id.id.to_string(), arg);
    }

    pub fn create_libfunc_dup<'b: 'ctx>(
        &'b self,
        func_decl: &LibfuncDeclaration,
        parent_block: &'b BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'ctx>> {
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

        let args: Vec<Type<'b>> = args.iter().map(|x| x.0).collect();

        storage.functions.insert(
            id,
            FunctionDef {
                args: args.iter().map(|x| x.0).collect_vec(),
                return_types: return_types.iter().map(|x| x.0).collect(),
            },
        );

        Ok(parent_block.append_operation(func))
    }

    pub fn create_libfunc_felt_add<'b: 'ctx>(
        &'b self,
        func_decl: &LibfuncDeclaration,
        parent_block: &'b BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'ctx>> {
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

        Ok(parent_block.append_operation(func))
    }

    pub fn create_libfunc_felt_sub<'b: 'ctx>(
        &'b self,
        func_decl: &LibfuncDeclaration,
        parent_block: &'b BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'ctx>> {
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

        Ok(parent_block.append_operation(func))
    }

    pub fn create_libfunc_felt_mul<'b: 'ctx>(
        &'b self,
        func_decl: &LibfuncDeclaration,
        parent_block: &'b BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'ctx>> {
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

        Ok(parent_block.append_operation(func))
    }
}
