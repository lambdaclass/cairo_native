use std::{cell::RefCell, cmp::Ordering, rc::Rc};

use cairo_lang_sierra::program::{GenericArg, LibfuncDeclaration};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{Block, BlockRef, Location, Region, Type, TypeLike, Value};
use tracing::debug;

use crate::{
    compiler::{CmpOp, Compiler, FunctionDef, SierraType, Storage},
    utility::create_fn_signature,
};

pub mod sierra_enum;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl<'ctx> Compiler<'ctx> {
    pub fn process_libfuncs(&'ctx self, storage: Rc<RefCell<Storage<'ctx>>>) -> Result<()> {
        for func_decl in &self.program.libfunc_declarations {
            let id = func_decl.id.id;
            let name = func_decl.long_id.generic_id.0.as_str();
            debug!(name, id, "processing libfunc decl");

            let parent_block = self.module.body();

            match name {
                // no-ops
                // NOTE jump stops being a nop if return types are stored
                "branch_align"
                | "revoke_ap_tracking"
                | "disable_ap_tracking"
                | "drop"
                | "jump"
                | "alloc_local"
                | "finalize_locals" => self.register_nop(func_decl, storage.clone()),
                "function_call" => continue, // Skip function call because it works differently than all the others
                "felt252_const" => {
                    self.create_libfunc_felt_const(func_decl, &mut storage.borrow_mut());
                }
                "felt252_add" => {
                    self.create_libfunc_felt_binary_op(
                        func_decl,
                        parent_block,
                        storage.clone(),
                        BinaryOp::Add,
                    )?;
                }
                "felt252_sub" => {
                    self.create_libfunc_felt_binary_op(
                        func_decl,
                        parent_block,
                        storage.clone(),
                        BinaryOp::Sub,
                    )?;
                }
                "felt252_mul" => {
                    self.create_libfunc_felt_binary_op(
                        func_decl,
                        parent_block,
                        storage.clone(),
                        BinaryOp::Mul,
                    )?;
                }
                "felt252_is_zero" => {
                    // Note no actual function is created here, however types are registered
                    self.register_libfunc_felt252_is_zero(func_decl, storage.clone());
                }
                "dup" => {
                    self.create_libfunc_dup(func_decl, parent_block, storage.clone())?;
                }
                "enum_init" => {
                    self.create_libfunc_enum_init(func_decl, parent_block, storage.clone())?;
                }
                "struct_construct" => {
                    self.create_libfunc_struct_construct(func_decl, parent_block, storage.clone())?;
                }
                "struct_deconstruct" => {
                    self.create_libfunc_struct_deconstruct(
                        func_decl,
                        parent_block,
                        storage.clone(),
                    )?;
                }
                "store_temp" | "rename" => {
                    self.create_identity_function(func_decl, parent_block, storage.clone())?;
                }
                "u8_const" => {
                    self.create_libfunc_u8_const(func_decl, storage.clone());
                }
                "u16_const" => {
                    self.create_libfunc_u16_const(func_decl, storage.clone());
                }
                "u32_const" => {
                    self.create_libfunc_u32_const(func_decl, storage.clone());
                }
                "u64_const" => {
                    self.create_libfunc_u64_const(func_decl, storage.clone());
                }
                "u128_const" => {
                    self.create_libfunc_u128_const(func_decl, storage.clone());
                }
                "u8_to_felt252" => {
                    self.create_libfunc_uint_to_felt252(
                        func_decl,
                        parent_block,
                        &mut storage.borrow_mut(),
                        self.u8_type(),
                    )?;
                }
                "u16_to_felt252" => {
                    self.create_libfunc_uint_to_felt252(
                        func_decl,
                        parent_block,
                        &mut storage.borrow_mut(),
                        self.u16_type(),
                    )?;
                }
                "u32_to_felt252" => {
                    self.create_libfunc_uint_to_felt252(
                        func_decl,
                        parent_block,
                        &mut storage.borrow_mut(),
                        self.u32_type(),
                    )?;
                }
                "u64_to_felt252" => {
                    self.create_libfunc_uint_to_felt252(
                        func_decl,
                        parent_block,
                        &mut storage.borrow_mut(),
                        self.u64_type(),
                    )?;
                }
                "u128_to_felt252" => {
                    self.create_libfunc_uint_to_felt252(
                        func_decl,
                        parent_block,
                        &mut storage.borrow_mut(),
                        self.u128_type(),
                    )?;
                }
                "u8_wide_mul" => {
                    self.create_libfunc_uint_wide_mul(
                        func_decl,
                        parent_block,
                        &mut storage.borrow_mut(),
                        self.u8_type(),
                        self.u16_type(),
                    )?;
                }
                "u16_wide_mul" => {
                    self.create_libfunc_uint_wide_mul(
                        func_decl,
                        parent_block,
                        &mut storage.borrow_mut(),
                        self.u16_type(),
                        self.u32_type(),
                    )?;
                }
                "u32_wide_mul" => {
                    self.create_libfunc_uint_wide_mul(
                        func_decl,
                        parent_block,
                        &mut storage.borrow_mut(),
                        self.u32_type(),
                        self.u64_type(),
                    )?;
                }
                "u64_wide_mul" => {
                    self.create_libfunc_uint_wide_mul(
                        func_decl,
                        parent_block,
                        &mut storage.borrow_mut(),
                        self.u64_type(),
                        self.u128_type(),
                    )?;
                }
                "u128_wide_mul" => {
                    self.create_libfunc_u128_wide_mul(
                        func_decl,
                        parent_block,
                        &mut storage.borrow_mut(),
                    )?;
                }
                "bitwise" => {
                    self.create_libfunc_bitwise(
                        func_decl,
                        parent_block,
                        &mut storage.borrow_mut(),
                    )?;
                }
                "upcast" => {
                    self.create_libfunc_upcast(func_decl, parent_block, storage.clone())?;
                }
                _ => debug!(?func_decl, "unhandled libfunc"),
            }
        }

        debug!(types = ?RefCell::borrow(&*storage).types, "processed");
        Ok(())
    }

    fn register_nop(&self, func_decl: &LibfuncDeclaration, storage: Rc<RefCell<Storage<'ctx>>>) {
        let id = Self::normalize_func_name(func_decl.id.debug_name.as_ref().unwrap().as_str())
            .to_string();
        storage
            .borrow_mut()
            .libfuncs
            .insert(id, FunctionDef { args: vec![], return_types: vec![] });
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

        let normalized_name =
            Self::normalize_func_name(func_decl.id.debug_name.as_ref().unwrap().as_str())
                .to_string();

        storage.felt_consts.insert(normalized_name.clone(), arg);

        storage.libfuncs.insert(
            normalized_name,
            FunctionDef { args: vec![], return_types: vec![SierraType::Simple(self.felt_type())] },
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
        let arg_type = match &func_decl.long_id.generic_args[0] {
            GenericArg::UserType(_) => todo!(),
            GenericArg::Type(type_id) => {
                let storage = RefCell::borrow(&*storage);
                let ty =
                    storage.types.get(&type_id.id.to_string()).cloned().expect("type to exist");

                ty
            }
            GenericArg::Value(_) => todo!(),
            GenericArg::UserFunc(_) => todo!(),
            GenericArg::Libfunc(_) => todo!(),
        };

        let args =
            arg_type.get_field_types().expect("arg should be a struct type and have field types");
        let args_with_location =
            args.iter().map(|x| (*x, Location::unknown(&self.context))).collect_vec();

        let region = Region::new();

        let block = Block::new(&args_with_location);

        let mut struct_type_op = self.op_llvm_struct(&block, &args);

        for i in 0..block.argument_count() {
            let arg = block.argument(i)?;
            let struct_value = struct_type_op.result(0)?.into();
            struct_type_op =
                self.op_llvm_insertvalue(&block, i, struct_value, arg.into(), arg_type.get_type())?;
        }

        let struct_value: Value = struct_type_op.result(0)?.into();
        self.op_return(&block, &[struct_value]);

        let function_type = create_fn_signature(&args, &[arg_type.get_type()]);

        region.append_block(block);

        let func = self.op_func(&id, &function_type, vec![region], false, false)?;

        {
            let mut storage = storage.borrow_mut();
            storage.libfuncs.insert(
                id,
                FunctionDef {
                    args: arg_type.get_field_sierra_types().unwrap().to_vec(),
                    return_types: vec![arg_type],
                },
            );
        }

        parent_block.append_operation(func);

        Ok(())
    }

    /// Extract (destructure) each struct member (in order) into variables.
    pub fn create_libfunc_struct_deconstruct(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: Rc<RefCell<Storage<'ctx>>>,
    ) -> Result<()> {
        let mut storage = storage.borrow_mut();

        let struct_type = storage
            .types
            .get(&match &func_decl.long_id.generic_args[0] {
                GenericArg::Type(x) => x.id.to_string(),
                _ => todo!("handler other types (error?)"),
            })
            .expect("struct type not found");
        let (struct_ty, field_types) = match struct_type {
            SierraType::Struct { ty, field_types } => (*ty, field_types.as_slice()),
            _ => todo!("handle non-struct types (error)"),
        };

        let region = Region::new();
        region.append_block({
            let block = Block::new(&[(struct_ty, Location::unknown(&self.context))]);

            let struct_value = block.argument(0)?;

            let mut result_ops = Vec::with_capacity(field_types.len());
            for (i, arg_ty) in field_types.iter().enumerate() {
                let op_ref =
                    self.op_llvm_extractvalue(&block, i, struct_value.into(), arg_ty.get_type())?;
                result_ops.push(op_ref);
            }

            let result_values: Vec<_> =
                result_ops.iter().map(|x| x.result(0).map(Into::into)).try_collect()?;
            self.op_return(&block, &result_values);

            block
        });

        let fn_id = Self::normalize_func_name(func_decl.id.debug_name.as_deref().unwrap());
        let fn_ty = create_fn_signature(
            &[struct_ty],
            field_types.iter().map(|x| x.get_type()).collect::<Vec<_>>().as_slice(),
        );
        let fn_op = self.op_func(&fn_id, &fn_ty, vec![region], false, false)?;

        let return_types = field_types.to_vec();
        let struct_type = struct_type.clone();
        storage
            .libfuncs
            .insert(fn_id.into_owned(), FunctionDef { args: vec![struct_type], return_types });

        parent_block.append_operation(fn_op);
        Ok(())
    }

    /// Returns the given value, needed so its handled nicely when processing statements
    /// and the variable id gets assigned to the returned value.
    pub fn create_identity_function(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: Rc<RefCell<Storage<'ctx>>>,
    ) -> Result<()> {
        let id = Self::normalize_func_name(func_decl.id.debug_name.as_ref().unwrap().as_str())
            .to_string();

        let arg_type = match &func_decl.long_id.generic_args[0] {
            GenericArg::UserType(_) => todo!(),
            GenericArg::Type(type_id) => {
                let storage = RefCell::borrow(&*storage);
                let ty = storage.types.get(&type_id.id.to_string()).expect("type to exist");

                ty.clone()
            }
            GenericArg::Value(_) => todo!(),
            GenericArg::UserFunc(_) => todo!(),
            GenericArg::Libfunc(_) => todo!(),
        };

        let region = Region::new();

        let args = &[arg_type.get_type()];
        let args_with_location = &[arg_type.get_type_location(&self.context)];

        let block = Block::new(args_with_location);

        let mut results: Vec<Value> = vec![];

        for i in 0..block.argument_count() {
            let arg = block.argument(i)?;
            results.push(arg.into());
        }

        self.op_return(&block, &results);

        region.append_block(block);

        let function_type = create_fn_signature(args, args);

        let func = self.op_func(&id, &function_type, vec![region], false, false)?;

        {
            let mut storage = storage.borrow_mut();
            storage.libfuncs.insert(
                id,
                FunctionDef { args: vec![arg_type.clone()], return_types: vec![arg_type] },
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
        let arg_type = match &func_decl.long_id.generic_args[0] {
            GenericArg::UserType(_) => todo!(),
            GenericArg::Type(type_id) => {
                let storage = RefCell::borrow(&*storage);
                let ty = storage.types.get(&type_id.id.to_string()).expect("type to exist");

                ty.clone()
            }
            GenericArg::Value(_) => todo!(),
            GenericArg::UserFunc(_) => todo!(),
            GenericArg::Libfunc(_) => todo!(),
        };

        let region = Region::new();

        let args = &[arg_type.get_type()];
        let args_with_location = &[arg_type.get_type_location(&self.context)];

        let block = Block::new(args_with_location);

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
        return_types.extend_from_slice(args);
        return_types.extend_from_slice(args);

        let function_type = create_fn_signature(args, &return_types);

        let func = self.op_func(&id, &function_type, vec![region], false, false)?;

        {
            let mut storage = storage.borrow_mut();
            storage.libfuncs.insert(
                id,
                FunctionDef {
                    args: vec![arg_type.clone()],
                    return_types: vec![arg_type.clone(), arg_type],
                },
            );
        }

        parent_block.append_operation(func);

        Ok(())
    }

    pub fn create_libfunc_felt_binary_op(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: Rc<RefCell<Storage<'ctx>>>,
        binary_op: BinaryOp,
    ) -> Result<()> {
        let id = Self::normalize_func_name(func_decl.id.debug_name.as_ref().unwrap().as_str())
            .to_string();
        let sierra_felt_type = SierraType::Simple(self.felt_type());
        let felt_type = sierra_felt_type.get_type();
        let felt_type_location = sierra_felt_type.get_type_location(&self.context);

        let region = Region::new();
        let entry_block = Block::new(&[felt_type_location, felt_type_location]);
        let entry_block = region.append_block(entry_block);

        let lhs = entry_block.argument(0)?;
        let rhs = entry_block.argument(1)?;

        let res = match binary_op {
            BinaryOp::Add => self.op_add(&entry_block, lhs.into(), rhs.into()),
            BinaryOp::Sub => self.op_sub(&entry_block, lhs.into(), rhs.into()),
            BinaryOp::Mul => {
                let lhs_zext = self.op_zext(&entry_block, lhs.into(), self.double_felt_type());
                let rhs_zext = self.op_zext(&entry_block, rhs.into(), self.double_felt_type());
                self.op_mul(&entry_block, lhs_zext.result(0)?.into(), rhs_zext.result(0)?.into())
            }
            BinaryOp::Div => todo!(),
        };
        let res_result = res.result(0)?;

        let end_block = region.append_block({
            let block = Block::new(&[(felt_type, Location::unknown(&self.context))]);

            self.op_return(&block, &[block.argument(0)?.into()]);
            block
        });

        match binary_op {
            BinaryOp::Add => {
                let prime = self.prime_constant(&entry_block);
                let prime_value = prime.result(0)?;

                let cmp_op = self.op_cmp(
                    &entry_block,
                    CmpOp::UnsignedGreaterEqual,
                    res_result.into(),
                    prime_value.into(),
                );
                let cmp_op_value = cmp_op.result(0)?;

                let mod_block = region.append_block({
                    let block = Block::new(&[]);

                    let res = self.op_sub(&block, res_result.into(), prime_value.into());
                    let res_value = res.result(0)?;

                    self.op_br(&block, &end_block, &[res_value.into()]);
                    block
                });

                self.op_cond_br(
                    &entry_block,
                    cmp_op_value.into(),
                    &mod_block,
                    &end_block,
                    &[],
                    &[res_result.into()],
                )?;
            }
            BinaryOp::Sub => {
                let prime = self.prime_constant(&entry_block);
                let prime_value = prime.result(0)?;

                let cmp_op = self.op_cmp(&entry_block, CmpOp::UnsignedLess, lhs.into(), rhs.into());
                let cmp_op_value = cmp_op.result(0)?;

                let mod_block = region.append_block({
                    let block = Block::new(&[]);

                    let res = self.op_sub(&block, res_result.into(), prime_value.into());
                    let res_value = res.result(0)?;

                    self.op_br(&block, &end_block, &[res_value.into()]);
                    block
                });

                self.op_cond_br(
                    &entry_block,
                    cmp_op_value.into(),
                    &mod_block,
                    &end_block,
                    &[],
                    &[res_result.into()],
                )?;
            }
            _ => {
                let res = self.op_felt_modulo(&entry_block, res_result.into())?;

                // Truncate to i256 after a multiplication.
                let res = match binary_op {
                    BinaryOp::Mul => {
                        self.op_trunc(&entry_block, res.result(0)?.into(), self.felt_type())
                    }
                    _ => res,
                };

                self.op_br(&entry_block, &end_block, &[res.result(0)?.into()]);
            }
        };

        let func = self.op_func(
            &id,
            &create_fn_signature(&[felt_type, felt_type], &[felt_type]),
            vec![region],
            false,
            false,
        )?;

        storage.borrow_mut().libfuncs.insert(
            id,
            FunctionDef {
                args: vec![sierra_felt_type.clone(), sierra_felt_type.clone()],
                return_types: vec![sierra_felt_type],
            },
        );

        parent_block.append_operation(func);
        Ok(())
    }

    pub fn register_libfunc_felt252_is_zero(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: Rc<RefCell<Storage<'ctx>>>,
    ) {
        let id = Self::normalize_func_name(func_decl.id.debug_name.as_ref().unwrap().as_str())
            .to_string();
        storage.borrow_mut().libfuncs.insert(
            id,
            FunctionDef {
                args: vec![SierraType::Simple(self.felt_type())],
                return_types: vec![], //TODO branching return types for libfuncs
            },
        );
    }

    pub fn create_libfunc_u8_const(
        &self,
        func_decl: &LibfuncDeclaration,
        storage: Rc<RefCell<Storage<'ctx>>>,
    ) {
        self.register_nop(func_decl, storage.clone());

        let arg = match func_decl.long_id.generic_args.as_slice() {
            [GenericArg::Value(value)] => value.to_string(),
            _ => todo!(),
        };

        storage.borrow_mut().u8_consts.insert(
            Self::normalize_func_name(func_decl.id.debug_name.as_deref().unwrap()).into_owned(),
            arg,
        );
    }

    pub fn create_libfunc_u16_const(
        &self,
        func_decl: &LibfuncDeclaration,
        storage: Rc<RefCell<Storage<'ctx>>>,
    ) {
        self.register_nop(func_decl, storage.clone());

        let arg = match func_decl.long_id.generic_args.as_slice() {
            [GenericArg::Value(value)] => value.to_string(),
            _ => todo!(),
        };

        storage.borrow_mut().u16_consts.insert(
            Self::normalize_func_name(func_decl.id.debug_name.as_deref().unwrap()).into_owned(),
            arg,
        );
    }

    pub fn create_libfunc_u32_const(
        &self,
        func_decl: &LibfuncDeclaration,
        storage: Rc<RefCell<Storage<'ctx>>>,
    ) {
        self.register_nop(func_decl, storage.clone());

        let arg = match func_decl.long_id.generic_args.as_slice() {
            [GenericArg::Value(value)] => value.to_string(),
            _ => todo!(),
        };

        storage.borrow_mut().u32_consts.insert(
            Self::normalize_func_name(func_decl.id.debug_name.as_deref().unwrap()).into_owned(),
            arg,
        );
    }

    pub fn create_libfunc_u64_const(
        &self,
        func_decl: &LibfuncDeclaration,
        storage: Rc<RefCell<Storage<'ctx>>>,
    ) {
        self.register_nop(func_decl, storage.clone());

        let arg = match func_decl.long_id.generic_args.as_slice() {
            [GenericArg::Value(value)] => value.to_string(),
            _ => todo!(),
        };

        storage.borrow_mut().u64_consts.insert(
            Self::normalize_func_name(func_decl.id.debug_name.as_deref().unwrap()).into_owned(),
            arg,
        );
    }

    pub fn create_libfunc_u128_const(
        &self,
        func_decl: &LibfuncDeclaration,
        storage: Rc<RefCell<Storage<'ctx>>>,
    ) {
        self.register_nop(func_decl, storage.clone());

        let arg = match func_decl.long_id.generic_args.as_slice() {
            [GenericArg::Value(value)] => value.to_string(),
            _ => todo!(),
        };

        storage.borrow_mut().u128_consts.insert(
            Self::normalize_func_name(func_decl.id.debug_name.as_deref().unwrap()).into_owned(),
            arg,
        );
    }

    pub fn create_libfunc_uint_to_felt252(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
        src_type: Type<'ctx>,
    ) -> Result<()> {
        let region = Region::new();
        let block =
            region.append_block(Block::new(&[(src_type, Location::unknown(&self.context))]));

        let op_zext = self.op_zext(&block, block.argument(0)?.into(), self.felt_type());
        self.op_return(&block, &[op_zext.result(0)?.into()]);

        let id =
            Self::normalize_func_name(func_decl.id.debug_name.as_deref().unwrap()).into_owned();
        let func = self.op_func(
            &id,
            &create_fn_signature(&[src_type], &[self.felt_type()]),
            vec![region],
            false,
            false,
        )?;

        storage.libfuncs.insert(
            id,
            FunctionDef {
                args: vec![SierraType::Simple(src_type)],
                return_types: vec![SierraType::Simple(self.felt_type())],
            },
        );
        parent_block.append_operation(func);

        Ok(())
    }

    pub fn create_libfunc_uint_wide_mul(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
        src_type: Type<'ctx>,
        dst_type: Type<'ctx>,
    ) -> Result<()> {
        let region = Region::new();
        let block = region.append_block(Block::new(&[
            (src_type, Location::unknown(&self.context)),
            (src_type, Location::unknown(&self.context)),
        ]));

        let op_zext_lhs = self.op_zext(&block, block.argument(0)?.into(), dst_type);
        let op_zext_rhs = self.op_zext(&block, block.argument(1)?.into(), dst_type);

        let op_mul =
            self.op_mul(&block, op_zext_lhs.result(0)?.into(), op_zext_rhs.result(0)?.into());
        self.op_return(&block, &[op_mul.result(0)?.into()]);

        let id =
            Self::normalize_func_name(func_decl.id.debug_name.as_deref().unwrap()).into_owned();
        let func = self.op_func(
            &id,
            &create_fn_signature(&[src_type, src_type], &[dst_type]),
            vec![region],
            false,
            false,
        )?;

        storage.libfuncs.insert(
            id,
            FunctionDef {
                args: vec![SierraType::Simple(src_type), SierraType::Simple(src_type)],
                return_types: vec![SierraType::Simple(dst_type)],
            },
        );
        parent_block.append_operation(func);

        Ok(())
    }

    pub fn create_libfunc_u128_wide_mul(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let region = Region::new();
        let block = region.append_block(Block::new(&[
            (self.range_check_type(), Location::unknown(&self.context)),
            (self.u128_type(), Location::unknown(&self.context)),
            (self.u128_type(), Location::unknown(&self.context)),
        ]));

        let op_zext_lhs = self.op_zext(&block, block.argument(1)?.into(), self.u256_type());
        let op_zext_rhs = self.op_zext(&block, block.argument(2)?.into(), self.u256_type());

        let op_mul =
            self.op_mul(&block, op_zext_lhs.result(0)?.into(), op_zext_rhs.result(0)?.into());

        let op_mul_hi = self.op_trunc(&block, op_mul.result(0)?.into(), self.u128_type());
        let op_mul_lo = {
            let op_const = self.op_const(&block, "128", self.u256_type());
            let op_shru =
                self.op_shru(&block, op_mul.result(0)?.into(), op_const.result(0)?.into());
            self.op_trunc(&block, op_shru.result(0)?.into(), self.u128_type())
        };

        self.op_return(
            &block,
            &[block.argument(0)?.into(), op_mul_hi.result(0)?.into(), op_mul_lo.result(0)?.into()],
        );

        let id =
            Self::normalize_func_name(func_decl.id.debug_name.as_deref().unwrap()).into_owned();
        let func = self.op_func(
            &id,
            &create_fn_signature(
                &[self.range_check_type(), self.u128_type(), self.u128_type()],
                &[self.range_check_type(), self.u128_type(), self.u128_type()],
            ),
            vec![region],
            false,
            false,
        )?;

        storage.libfuncs.insert(
            id,
            FunctionDef {
                args: vec![
                    SierraType::Simple(self.range_check_type()),
                    SierraType::Simple(self.u128_type()),
                    SierraType::Simple(self.u128_type()),
                ],
                return_types: vec![
                    SierraType::Simple(self.range_check_type()),
                    SierraType::Simple(self.u128_type()),
                    SierraType::Simple(self.u128_type()),
                ],
            },
        );
        parent_block.append_operation(func);

        Ok(())
    }

    pub fn create_libfunc_bitwise(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let data_in = &[self.bitwise_type(), self.u128_type(), self.u128_type()];
        let data_out = &[self.bitwise_type(), self.u128_type(), self.u128_type(), self.u128_type()];

        let region = Region::new();
        region.append_block({
            let block = Block::new(&[
                (data_in[0], Location::unknown(&self.context)),
                (data_in[1], Location::unknown(&self.context)),
                (data_in[2], Location::unknown(&self.context)),
            ]);

            let lhs = block.argument(0)?;
            let rhs = block.argument(1)?;
            let to = self.u128_type();

            let and_ref = self.op_and(&block, lhs.into(), rhs.into(), to);
            let xor_ref = self.op_xor(&block, lhs.into(), rhs.into(), to);
            let or_ref = self.op_or(&block, lhs.into(), rhs.into(), to);

            self.op_return(
                &block,
                &[and_ref.result(0)?.into(), xor_ref.result(0)?.into(), or_ref.result(0)?.into()],
            );

            block
        });

        let fn_id = Self::normalize_func_name(func_decl.id.debug_name.as_deref().unwrap());
        let fn_ty = create_fn_signature(data_in, data_out);
        let fn_op = self.op_func(&fn_id, &fn_ty, vec![region], false, false)?;

        storage.libfuncs.insert(
            fn_id.into_owned(),
            FunctionDef {
                args: data_in.iter().copied().map(SierraType::Simple).collect(),
                return_types: data_out.iter().copied().map(SierraType::Simple).collect(),
            },
        );

        parent_block.append_operation(fn_op);
        Ok(())
    }

    pub fn create_libfunc_upcast(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: Rc<RefCell<Storage<'ctx>>>,
    ) -> Result<()> {
        let id = Self::normalize_func_name(func_decl.id.debug_name.as_deref().unwrap()).to_string();

        let src_sierra_type = storage
            .borrow()
            .types
            .get(&match &func_decl.long_id.generic_args[0] {
                GenericArg::Type(x) => x.id.to_string(),
                _ => todo!("invalid generic kind"),
            })
            .expect("type to exist")
            .clone();
        let dst_sierra_type = storage
            .borrow()
            .types
            .get(&match &func_decl.long_id.generic_args[1] {
                GenericArg::Type(x) => x.id.to_string(),
                _ => todo!("invalid generic kind"),
            })
            .expect("type to exist")
            .clone();

        let src_type = src_sierra_type.get_type();
        let dst_type = dst_sierra_type.get_type();

        match src_type.get_width().unwrap().cmp(&dst_type.get_width().unwrap()) {
            Ordering::Less => {
                let region = Region::new();
                let block = Block::new(&[(src_type, Location::unknown(&self.context))]);

                let op_ref = self.op_zext(&block, block.argument(0)?.into(), dst_type);

                self.op_return(&block, &[op_ref.result(0)?.into()]);
                region.append_block(block);

                let func = self.op_func(
                    &id,
                    &create_fn_signature(&[src_type], &[dst_type]),
                    vec![region],
                    false,
                    false,
                )?;

                storage.borrow_mut().libfuncs.insert(
                    id,
                    FunctionDef {
                        args: vec![src_sierra_type],
                        return_types: vec![dst_sierra_type],
                    },
                );

                parent_block.append_operation(func);
            }
            Ordering::Equal => {
                // Similar to store_local and rename, create an identity function for ease of dataflow processing, under the assumption the optimiser will optimise it out
                self.create_identity_function(func_decl, parent_block, storage.clone())?;
            }
            Ordering::Greater => todo!("invalid generics for libfunc `upcast`"),
        }

        Ok(())
    }
}
