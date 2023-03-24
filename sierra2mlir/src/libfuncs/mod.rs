use std::{cell::RefCell, rc::Rc};

use cairo_lang_sierra::program::{GenericArg, LibfuncDeclaration};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{Block, BlockRef, Location, Region, Type, Value};
use tracing::debug;

use crate::{
    compiler::{Compiler, FunctionDef, SierraType, Storage},
    statements::create_fn_signature,
};

impl<'ctx> Compiler<'ctx> {
    pub fn process_libfuncs(&'ctx self, storage: Rc<RefCell<Storage<'ctx>>>) -> Result<()> {
        for func_decl in &self.program.libfunc_declarations {
            let id = func_decl.id.id;
            let name = func_decl.long_id.generic_id.0.as_str();
            debug!(name, id, "processing libfunc decl");

            let parent_block = self.module.body();

            match name {
                // no-ops
                "revoke_ap_tracking" => continue,
                "disable_ap_tracking" => continue,
                "drop" => continue,
                "felt252_const" => {
                    self.create_libfunc_felt_const(func_decl, &mut storage.borrow_mut());
                }
                "felt252_add" => {
                    self.create_libfunc_felt_add(func_decl, parent_block, storage.clone())?;
                }
                "felt252_sub" => {
                    self.create_libfunc_felt_sub(func_decl, parent_block, storage.clone())?;
                }
                "felt252_mul" => {
                    self.create_libfunc_felt_mul(func_decl, parent_block, storage.clone())?;
                }
                "dup" => {
                    self.create_libfunc_dup(func_decl, parent_block, storage.clone())?;
                }
                "struct_construct" => {
                    self.create_libfunc_struct_construct(func_decl, parent_block, storage.clone())?;
                }
                "store_temp" | "rename" => {
                    self.create_libfunc_store_temp(func_decl, parent_block, storage.clone())?;
                }
                "u8_const" => {
                    self.create_libfunc_u8_const(func_decl, &mut storage.borrow_mut());
                }
                "u16_const" => {
                    self.create_libfunc_u16_const(func_decl, &mut storage.borrow_mut());
                }
                "u32_const" => {
                    self.create_libfunc_u32_const(func_decl, &mut storage.borrow_mut());
                }
                "u64_const" => {
                    self.create_libfunc_u64_const(func_decl, &mut storage.borrow_mut());
                }
                "u128_const" => {
                    self.create_libfunc_u128_const(func_decl, &mut storage.borrow_mut());
                }
                "bitwise" => {
                    self.create_libfunc_bitwise(
                        func_decl,
                        parent_block,
                        &mut storage.borrow_mut(),
                    )?;
                }
                _ => debug!(?func_decl, "unhandled libfunc"),
            }
        }

        debug!(types = ?RefCell::borrow(&*storage).types, "processed");
        Ok(())
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

        storage.felt_consts.insert(
            Self::normalize_func_name(func_decl.id.debug_name.as_ref().unwrap().as_str())
                .to_string(),
            arg,
        );
    }

    pub fn create_libfunc_struct_construct(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: Rc<RefCell<Storage<'ctx>>>,
    ) -> Result<()> {
        let id = Self::normalize_func_name(func_decl.id.debug_name.as_ref().unwrap().as_str())
            .to_string();
        let mut arg_types_with_locations = vec![];

        for arg in &func_decl.long_id.generic_args {
            let storage = RefCell::borrow(&*storage);
            match arg {
                GenericArg::UserType(_) => todo!(),
                GenericArg::Type(type_id) => {
                    let ty = storage
                        .types
                        .get(&type_id.id.to_string())
                        .expect("type to exist");

                    let field_types = match ty {
                        SierraType::Simple(_ty) => {
                            unreachable!("struct construct shouldnt be called for simple types")
                        }
                        SierraType::Struct { ty: _, field_types } => field_types,
                    };

                    for ty in field_types {
                        arg_types_with_locations.push((*ty, Location::unknown(&self.context)));
                    }
                }
                GenericArg::Value(_) => todo!(),
                GenericArg::UserFunc(_) => todo!(),
                GenericArg::Libfunc(_) => todo!(),
            }
        }

        let region = Region::new();

        let block = Block::new(&arg_types_with_locations);

        let arg_types = arg_types_with_locations.iter().map(|x| x.0).collect_vec();
        let struct_llvm_type = self.struct_type_string(&arg_types);
        let mut struct_type_op = self.op_llvm_struct(&block, &arg_types);
        //let mut struct_value: Value = struct_type_op.result(0)?.into();

        for i in 0..block.argument_count() {
            let arg = block.argument(i)?;
            let struct_value = struct_type_op.result(0)?.into();
            struct_type_op =
                self.op_llvm_insertvalue(&block, i, struct_value, arg.into(), &struct_llvm_type)?;
        }

        let struct_value: Value = struct_type_op.result(0)?.into();
        self.op_return(&block, &[struct_value]);

        let return_type = Type::parse(&self.context, &struct_llvm_type).unwrap();
        let function_type = create_fn_signature(&arg_types, &[return_type]);

        region.append_block(block);

        let func = self.op_func(&id, &function_type, vec![region], false, false)?;

        {
            let mut storage = storage.borrow_mut();
            storage.functions.insert(
                id,
                FunctionDef {
                    args: arg_types,
                    return_types: vec![return_type],
                },
            );
        }

        parent_block.append_operation(func);

        Ok(())
    }

    /// Returns the given value, needed so its handled nicely when processing statements
    /// and the variable id gets assigned to the returned value.
    pub fn create_libfunc_store_temp(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: Rc<RefCell<Storage<'ctx>>>,
    ) -> Result<()> {
        let id = Self::normalize_func_name(func_decl.id.debug_name.as_ref().unwrap().as_str())
            .to_string();
        let mut arg_types_with_locations = vec![];

        for arg in &func_decl.long_id.generic_args {
            let storage = RefCell::borrow(&*storage);
            match arg {
                GenericArg::UserType(_) => todo!(),
                GenericArg::Type(type_id) => {
                    let ty = storage
                        .types
                        .get(&type_id.id.to_string())
                        .expect("type to exist");

                    arg_types_with_locations
                        .push((*ty.get_type(), Location::unknown(&self.context)));
                }
                GenericArg::Value(_) => todo!(),
                GenericArg::UserFunc(_) => todo!(),
                GenericArg::Libfunc(_) => todo!(),
            }
        }

        let region = Region::new();

        let block = Block::new(&arg_types_with_locations);

        let mut results: Vec<Value> = vec![];

        for i in 0..block.argument_count() {
            let arg = block.argument(i)?;
            results.push(arg.into());
        }

        self.op_return(&block, &results);

        region.append_block(block);

        let arg_types = arg_types_with_locations
            .iter()
            .map(|(t, _)| *t)
            .collect_vec();

        let function_type = create_fn_signature(&arg_types, &arg_types);

        let func = self.op_func(&id, &function_type, vec![region], false, false)?;

        {
            let mut storage = storage.borrow_mut();
            storage.functions.insert(
                id,
                FunctionDef {
                    args: arg_types.clone(),
                    return_types: arg_types,
                },
            );
        }

        parent_block.append_operation(func);

        Ok(())
    }

    pub fn create_libfunc_dup(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: Rc<RefCell<Storage<'ctx>>>,
    ) -> Result<()> {
        let id = Self::normalize_func_name(func_decl.id.debug_name.as_ref().unwrap().as_str())
            .to_string();
        let mut arg_types_with_locations = vec![];

        for arg in &func_decl.long_id.generic_args {
            let storage = RefCell::borrow(&*storage);
            match arg {
                GenericArg::UserType(_) => todo!(),
                GenericArg::Type(type_id) => {
                    let ty = storage
                        .types
                        .get(&type_id.id.to_string())
                        .expect("type to exist");

                    arg_types_with_locations
                        .push((*ty.get_type(), Location::unknown(&self.context)));
                }
                GenericArg::Value(_) => todo!(),
                GenericArg::UserFunc(_) => todo!(),
                GenericArg::Libfunc(_) => todo!(),
            }
        }

        let region = Region::new();

        let block = Block::new(&arg_types_with_locations);

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

        let arg_types = arg_types_with_locations
            .iter()
            .map(|(t, _)| *t)
            .collect::<Vec<_>>();

        let mut return_types = Vec::with_capacity(arg_types.len() * 2);
        return_types.extend_from_slice(&arg_types);
        return_types.extend_from_slice(&arg_types);

        let function_type = create_fn_signature(&arg_types, &return_types);

        let func = self.op_func(&id, &function_type, vec![region], false, false)?;

        {
            let mut storage = storage.borrow_mut();
            storage.functions.insert(
                id,
                FunctionDef {
                    args: arg_types,
                    return_types,
                },
            );
        }

        parent_block.append_operation(func);

        Ok(())
    }

    pub fn create_libfunc_felt_add(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: Rc<RefCell<Storage<'ctx>>>,
    ) -> Result<()> {
        let id = Self::normalize_func_name(func_decl.id.debug_name.as_ref().unwrap().as_str())
            .to_string();
        let felt_type = self.felt_type();
        let loc = Location::unknown(&self.context);

        let region = Region::new();
        let block = Block::new(&[(felt_type, loc), (felt_type, loc)]);

        let lhs_arg = block.argument(0)?;
        let rhs_arg = block.argument(1)?;

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
            false,
        )?;

        storage.borrow_mut().functions.insert(
            id,
            FunctionDef {
                args: vec![felt_type, felt_type],
                return_types: vec![felt_type],
            },
        );

        parent_block.append_operation(func);
        Ok(())
    }

    pub fn create_libfunc_felt_sub(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: Rc<RefCell<Storage<'ctx>>>,
    ) -> Result<()> {
        let id = Self::normalize_func_name(func_decl.id.debug_name.as_ref().unwrap().as_str())
            .to_string();
        let felt_type = self.felt_type();
        let loc = Location::unknown(&self.context);

        let region = Region::new();
        let block = Block::new(&[(felt_type, loc), (felt_type, loc)]);

        let lhs_arg = block.argument(0)?;
        let rhs_arg = block.argument(1)?;

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
            false,
        )?;

        storage.borrow_mut().functions.insert(
            id,
            FunctionDef {
                args: vec![felt_type, felt_type],
                return_types: vec![felt_type],
            },
        );

        parent_block.append_operation(func);
        Ok(())
    }

    pub fn create_libfunc_felt_mul(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: Rc<RefCell<Storage<'ctx>>>,
    ) -> Result<()> {
        let id = Self::normalize_func_name(func_decl.id.debug_name.as_ref().unwrap().as_str())
            .to_string();
        let felt_type = self.felt_type();
        let loc = Location::unknown(&self.context);

        let region = Region::new();
        let block = Block::new(&[(felt_type, loc), (felt_type, loc)]);

        let lhs_arg = block.argument(0)?;
        let rhs_arg = block.argument(1)?;

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
            false,
        )?;

        storage.borrow_mut().functions.insert(
            id,
            FunctionDef {
                args: vec![felt_type, felt_type],
                return_types: vec![felt_type],
            },
        );

        parent_block.append_operation(func);
        Ok(())
    }

    pub fn create_libfunc_u8_const(
        &self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) {
        let arg = match func_decl.long_id.generic_args.as_slice() {
            [GenericArg::Value(value)] => value.to_string(),
            _ => todo!(),
        };

        storage.u8_consts.insert(
            Self::normalize_func_name(func_decl.id.debug_name.as_deref().unwrap()).into_owned(),
            arg,
        );
    }

    pub fn create_libfunc_u16_const(
        &self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) {
        let arg = match func_decl.long_id.generic_args.as_slice() {
            [GenericArg::Value(value)] => value.to_string(),
            _ => todo!(),
        };

        storage.u16_consts.insert(
            Self::normalize_func_name(func_decl.id.debug_name.as_deref().unwrap()).into_owned(),
            arg,
        );
    }

    pub fn create_libfunc_u32_const(
        &self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) {
        let arg = match func_decl.long_id.generic_args.as_slice() {
            [GenericArg::Value(value)] => value.to_string(),
            _ => todo!(),
        };

        storage.u32_consts.insert(
            Self::normalize_func_name(func_decl.id.debug_name.as_deref().unwrap()).into_owned(),
            arg,
        );
    }

    pub fn create_libfunc_u64_const(
        &self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) {
        let arg = match func_decl.long_id.generic_args.as_slice() {
            [GenericArg::Value(value)] => value.to_string(),
            _ => todo!(),
        };

        storage.u64_consts.insert(
            Self::normalize_func_name(func_decl.id.debug_name.as_deref().unwrap()).into_owned(),
            arg,
        );
    }

    pub fn create_libfunc_u128_const(
        &self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) {
        let arg = match func_decl.long_id.generic_args.as_slice() {
            [GenericArg::Value(value)] => value.to_string(),
            _ => todo!(),
        };

        storage.u128_consts.insert(
            Self::normalize_func_name(func_decl.id.debug_name.as_deref().unwrap()).into_owned(),
            arg,
        );
    }

    pub fn create_libfunc_bitwise(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let data_in = &[
            (self.bitwise_type(), Location::unknown(&self.context)),
            (self.u128_type(), Location::unknown(&self.context)),
            (self.u128_type(), Location::unknown(&self.context)),
        ];
        let data_out = &[
            (self.bitwise_type(), Location::unknown(&self.context)),
            (self.u128_type(), Location::unknown(&self.context)),
            (self.u128_type(), Location::unknown(&self.context)),
            (self.u128_type(), Location::unknown(&self.context)),
        ];

        let region = Region::new();
        region.append_block({
            let block = Block::new(data_in);

            let lhs = block.argument(0)?;
            let rhs = block.argument(1)?;
            let to = self.u128_type();

            let and_ref = self.op_and(&block, lhs.into(), rhs.into(), to);
            let xor_ref = self.op_xor(&block, lhs.into(), rhs.into(), to);
            let or_ref = self.op_or(&block, lhs.into(), rhs.into(), to);

            self.op_return(
                &block,
                &[
                    and_ref.result(0)?.into(),
                    xor_ref.result(0)?.into(),
                    or_ref.result(0)?.into(),
                ],
            );

            block
        });

        let fn_id = Self::normalize_func_name(func_decl.id.debug_name.as_deref().unwrap());
        let fn_ty = self.create_fn_signature(data_in, data_out);
        let fn_op = self.op_func(&fn_id, &fn_ty, vec![region], false, false)?;

        storage.functions.insert(
            fn_id.into_owned(),
            FunctionDef {
                args: data_in.iter().map(|(x, _)| *x).collect(),
                return_types: data_out.iter().map(|(x, _)| *x).collect(),
            },
        );

        parent_block.append_operation(fn_op);
        Ok(())
    }
}
