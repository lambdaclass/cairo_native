use std::cmp::Ordering;

use cairo_lang_sierra::program::{GenericArg, LibfuncDeclaration};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{Block, BlockRef, Location, Region, Type, TypeLike, Value};
use num_bigint::BigInt;
use num_traits::Signed;
use tracing::debug;

use crate::{
    compiler::{CmpOp, Compiler, SierraType, Storage},
    types::DEFAULT_PRIME,
    utility::create_fn_signature,
};

use self::lib_func_def::{LibFuncArg, LibFuncDef, SierraLibFunc};

pub mod inline_funcs;
pub mod lib_func_def;
pub mod sierra_enum;

impl<'ctx> Compiler<'ctx> {
    pub fn process_libfuncs(&'ctx self, storage: &mut Storage<'ctx>) -> Result<()> {
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
                | "finalize_locals" => self.register_nop(func_decl, storage),
                "function_call" => continue, // Skip function call because it works differently than all the others
                "felt252_const" => {
                    self.create_libfunc_felt_const(func_decl, self.felt_type(), storage)?;
                }
                "felt252_add" => {
                    self.create_libfunc_felt_add(func_decl, parent_block, storage)?;
                }
                "felt252_sub" => {
                    self.create_libfunc_felt_sub(func_decl, parent_block, storage)?;
                }
                "felt252_mul" => {
                    self.create_libfunc_felt_mul(func_decl, parent_block, storage)?;
                }
                "felt252_div" => {
                    self.create_libfunc_felt_div(func_decl, parent_block, storage)?;
                }
                "felt252_is_zero" => {
                    // Note no actual function is created here, however types are registered
                    self.register_libfunc_felt252_is_zero(func_decl, storage);
                }
                "dup" => {
                    self.create_libfunc_dup(func_decl, parent_block, storage)?;
                }
                "enum_init" => {
                    self.create_libfunc_enum_init(func_decl, parent_block, storage)?;
                }
                "enum_match" => {
                    // Note no actual function is created here, however types are registered
                    self.register_libfunc_enum_match(func_decl, storage);
                }
                "struct_construct" => {
                    self.create_libfunc_struct_construct(func_decl, parent_block, storage)?;
                }
                "struct_deconstruct" => {
                    self.create_libfunc_struct_deconstruct(func_decl, parent_block, storage)?;
                }
                "store_temp" | "rename" => {
                    self.create_identity_function(func_decl, parent_block, storage)?;
                }
                "u8_const" => {
                    self.create_libfunc_uint_const(func_decl, self.u8_type(), storage);
                }
                "u16_const" => {
                    self.create_libfunc_uint_const(func_decl, self.u16_type(), storage);
                }
                "u32_const" => {
                    self.create_libfunc_uint_const(func_decl, self.u32_type(), storage);
                }
                "u64_const" => {
                    self.create_libfunc_uint_const(func_decl, self.u64_type(), storage);
                }
                "u128_const" => {
                    self.create_libfunc_uint_const(func_decl, self.u128_type(), storage);
                }
                "u8_to_felt252" => {
                    self.create_libfunc_uint_to_felt252(
                        func_decl,
                        parent_block,
                        storage,
                        self.u8_type(),
                    )?;
                }
                "u16_to_felt252" => {
                    self.create_libfunc_uint_to_felt252(
                        func_decl,
                        parent_block,
                        storage,
                        self.u16_type(),
                    )?;
                }
                "u32_to_felt252" => {
                    self.create_libfunc_uint_to_felt252(
                        func_decl,
                        parent_block,
                        storage,
                        self.u32_type(),
                    )?;
                }
                "u64_to_felt252" => {
                    self.create_libfunc_uint_to_felt252(
                        func_decl,
                        parent_block,
                        storage,
                        self.u64_type(),
                    )?;
                }
                "u128_to_felt252" => {
                    self.create_libfunc_uint_to_felt252(
                        func_decl,
                        parent_block,
                        storage,
                        self.u128_type(),
                    )?;
                }
                "u8_wide_mul" => {
                    self.create_libfunc_uint_wide_mul(
                        func_decl,
                        parent_block,
                        storage,
                        self.u8_type(),
                        self.u16_type(),
                    )?;
                }
                "u16_wide_mul" => {
                    self.create_libfunc_uint_wide_mul(
                        func_decl,
                        parent_block,
                        storage,
                        self.u16_type(),
                        self.u32_type(),
                    )?;
                }
                "u32_wide_mul" => {
                    self.create_libfunc_uint_wide_mul(
                        func_decl,
                        parent_block,
                        storage,
                        self.u32_type(),
                        self.u64_type(),
                    )?;
                }
                "u64_wide_mul" => {
                    self.create_libfunc_uint_wide_mul(
                        func_decl,
                        parent_block,
                        storage,
                        self.u64_type(),
                        self.u128_type(),
                    )?;
                }
                "u128_wide_mul" => {
                    self.create_libfunc_u128_wide_mul(func_decl, parent_block, storage)?;
                }
                "u8_safe_divmod" => {
                    self.create_libfunc_uint_safe_divmod(
                        func_decl,
                        parent_block,
                        storage,
                        self.u8_type(),
                    )?;
                }
                "u16_safe_divmod" => {
                    self.create_libfunc_uint_safe_divmod(
                        func_decl,
                        parent_block,
                        storage,
                        self.u16_type(),
                    )?;
                }
                "u32_safe_divmod" => {
                    self.create_libfunc_uint_safe_divmod(
                        func_decl,
                        parent_block,
                        storage,
                        self.u32_type(),
                    )?;
                }
                "u64_safe_divmod" => {
                    self.create_libfunc_uint_safe_divmod(
                        func_decl,
                        parent_block,
                        storage,
                        self.u64_type(),
                    )?;
                }
                "u128_safe_divmod" => {
                    self.create_libfunc_uint_safe_divmod(
                        func_decl,
                        parent_block,
                        storage,
                        self.u128_type(),
                    )?;
                }
                "bitwise" => {
                    self.create_libfunc_bitwise(func_decl, parent_block, storage)?;
                }
                "upcast" => {
                    self.create_libfunc_upcast(func_decl, parent_block, storage)?;
                }
                _ => todo!(
                    "unhandled libfunc: {:?}",
                    func_decl.id.debug_name.as_ref().unwrap().as_str()
                ),
            }
        }

        debug!(types = ?storage.types, "processed");
        Ok(())
    }

    fn register_nop(&self, func_decl: &LibfuncDeclaration, storage: &mut Storage<'ctx>) {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        storage.libfuncs.insert(id, SierraLibFunc::create_simple(vec![], vec![]));
    }

    pub fn create_libfunc_felt_const(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        ty: Type<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let arg = match func_decl.long_id.generic_args.as_slice() {
            [GenericArg::Value(value)] => value.to_string(),
            _ => unreachable!("Expected generic arg of const creation function to be a Value"),
        };

        let arg_value = arg.parse::<BigInt>()?;
        let wrapped_arg_value = if arg_value.is_negative() {
            DEFAULT_PRIME.parse::<BigInt>()? + arg_value
        } else {
            arg_value
        };

        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        storage.libfuncs.insert(
            id,
            SierraLibFunc::create_constant(SierraType::Simple(ty), wrapped_arg_value.to_string()),
        );

        Ok(())
    }

    pub fn create_libfunc_uint_const(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        ty: Type<'ctx>,
        storage: &mut Storage<'ctx>,
    ) {
        let arg = match func_decl.long_id.generic_args.as_slice() {
            [GenericArg::Value(value)] => value.to_string(),
            _ => unreachable!("Expected generic arg of const creation function to be a Value"),
        };

        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        storage.libfuncs.insert(id, SierraLibFunc::create_constant(SierraType::Simple(ty), arg));
    }

    pub fn create_libfunc_struct_construct(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let arg_type = match &func_decl.long_id.generic_args[0] {
            GenericArg::UserType(_) => todo!(),
            GenericArg::Type(type_id) => {
                storage.types.get(&type_id.id.to_string()).cloned().expect("type to exist")
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

        storage.libfuncs.insert(
            id,
            SierraLibFunc::create_simple(
                arg_type.get_field_sierra_types().unwrap().to_vec(),
                vec![arg_type],
            ),
        );

        parent_block.append_operation(func);

        Ok(())
    }

    /// Extract (destructure) each struct member (in order) into variables.
    pub fn create_libfunc_struct_deconstruct(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
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

        let fn_id = func_decl.id.debug_name.as_deref().unwrap().to_string();
        let fn_ty = create_fn_signature(
            &[struct_ty],
            field_types.iter().map(|x| x.get_type()).collect::<Vec<_>>().as_slice(),
        );
        let fn_op = self.op_func(&fn_id, &fn_ty, vec![region], false, false)?;

        let return_types = field_types.to_vec();
        let struct_type = struct_type.clone();
        storage
            .libfuncs
            .insert(fn_id, SierraLibFunc::create_simple(vec![struct_type], return_types));

        parent_block.append_operation(fn_op);
        Ok(())
    }

    /// Returns the given value, needed so its handled nicely when processing statements
    /// and the variable id gets assigned to the returned value.
    pub fn create_identity_function(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        let arg_type = match &func_decl.long_id.generic_args[0] {
            GenericArg::UserType(_) => todo!(),
            GenericArg::Type(type_id) => {
                storage.types.get(&type_id.id.to_string()).expect("type to exist").clone()
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

        storage
            .libfuncs
            .insert(id, SierraLibFunc::create_simple(vec![arg_type.clone()], vec![arg_type]));

        parent_block.append_operation(func);

        Ok(())
    }

    pub fn create_libfunc_dup(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let arg_type = match &func_decl.long_id.generic_args[0] {
            GenericArg::UserType(_) => todo!(),
            GenericArg::Type(type_id) => {
                storage.types.get(&type_id.id.to_string()).expect("type to exist").clone()
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

        storage.libfuncs.insert(
            id,
            SierraLibFunc::create_simple(vec![arg_type.clone()], vec![arg_type.clone(), arg_type]),
        );

        parent_block.append_operation(func);

        Ok(())
    }

    pub fn create_libfunc_felt_add(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let sierra_felt_type = SierraType::Simple(self.felt_type());
        let felt_type = sierra_felt_type.get_type();
        let felt_type_location = sierra_felt_type.get_type_location(&self.context);

        let region = Region::new();
        // Block in which the calculation occurs
        let entry_block = Block::new(&[felt_type_location, felt_type_location]);
        // Block for wrapping values >= PRIME
        let gte_prime_block = Block::new(&[]);
        // Block for returning values < PRIME
        let in_range_block = Block::new(&[]);

        // res = lhs + rhs
        let lhs = entry_block.argument(0)?.into();
        let rhs = entry_block.argument(1)?.into();
        let res_op = self.op_add(&entry_block, lhs, rhs);
        let res = res_op.result(0)?.into();

        // gt_prime <=> res_result >= PRIME
        let prime_op = self.prime_constant(&entry_block);
        let prime = prime_op.result(0)?.into();
        let gte_prime_op = self.op_cmp(&entry_block, CmpOp::UnsignedGreaterEqual, res, prime);
        let gte_prime = gte_prime_op.result(0)?.into();

        // if gt_prime
        self.op_cond_br(&entry_block, gte_prime, &gte_prime_block, &in_range_block, &[], &[])?;

        //gt prime block
        let wrapped_res_op = self.op_sub(&gte_prime_block, res, prime);
        let wrapped_res = wrapped_res_op.result(0)?.into();
        self.op_return(&gte_prime_block, &[wrapped_res]);

        //in range block
        self.op_return(&in_range_block, &[res]);

        region.append_block(entry_block);
        region.append_block(in_range_block);
        region.append_block(gte_prime_block);
        let func = self.op_func(
            &id,
            &create_fn_signature(&[felt_type, felt_type], &[felt_type]),
            vec![region],
            false,
            false,
        )?;

        parent_block.append_operation(func);
        storage.libfuncs.insert(
            id,
            SierraLibFunc::create_simple(
                vec![sierra_felt_type.clone(), sierra_felt_type.clone()],
                vec![sierra_felt_type],
            ),
        );
        Ok(())
    }

    pub fn create_libfunc_felt_sub(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let sierra_felt_type = SierraType::Simple(self.felt_type());
        let felt_type = sierra_felt_type.get_type();
        let felt_type_location = sierra_felt_type.get_type_location(&self.context);

        let region = Region::new();
        // Block in which the calculation occurs
        let entry_block = Block::new(&[felt_type_location, felt_type_location]);
        // Block for wrapping values < 0
        let lt_zero_block = Block::new(&[]);
        // Block for returning values >= 0
        let in_range_block = Block::new(&[]);

        // res = lhs - rhs
        let lhs = entry_block.argument(0)?.into();
        let rhs = entry_block.argument(1)?.into();
        let res_op = self.op_sub(&entry_block, lhs, rhs);
        let res = res_op.result(0)?.into();

        // lt_zero <=> res_result < 0
        let zero_op = self.op_const(&entry_block, "0", felt_type);
        let zero = zero_op.result(0)?.into();
        let lt_zero_op = self.op_cmp(&entry_block, CmpOp::SignedLessThan, res, zero);
        let lt_zero = lt_zero_op.result(0)?.into();

        // if gt_prime
        self.op_cond_br(&entry_block, lt_zero, &lt_zero_block, &in_range_block, &[], &[])?;

        //lt zero block
        let prime_op = self.prime_constant(&lt_zero_block);
        let prime = prime_op.result(0)?.into();
        let wrapped_res_op = self.op_add(&lt_zero_block, res, prime);
        let wrapped_res = wrapped_res_op.result(0)?.into();
        self.op_return(&lt_zero_block, &[wrapped_res]);

        //in range block
        self.op_return(&in_range_block, &[res]);

        region.append_block(entry_block);
        region.append_block(in_range_block);
        region.append_block(lt_zero_block);
        let func = self.op_func(
            &id,
            &create_fn_signature(&[felt_type, felt_type], &[felt_type]),
            vec![region],
            false,
            false,
        )?;

        parent_block.append_operation(func);
        storage.libfuncs.insert(
            id,
            SierraLibFunc::create_simple(
                vec![sierra_felt_type.clone(), sierra_felt_type.clone()],
                vec![sierra_felt_type],
            ),
        );
        Ok(())
    }

    pub fn create_libfunc_felt_mul(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let sierra_felt_type = SierraType::Simple(self.felt_type());
        let felt_type = sierra_felt_type.get_type();
        let felt_type_location = sierra_felt_type.get_type_location(&self.context);

        let region = Region::new();
        let block = Block::new(&[felt_type_location, felt_type_location]);

        // Need to first widen arguments so we can accurately calculate the non-modular product before wrapping it back into range
        let wide_type = self.double_felt_type();
        let lhs = block.argument(0)?.into();
        let rhs = block.argument(1)?.into();
        let lhs_wide_op = self.op_zext(&block, lhs, wide_type);
        let rhs_wide_op = self.op_zext(&block, rhs, wide_type);
        let lhs_wide = lhs_wide_op.result(0)?.into();
        let rhs_wide = rhs_wide_op.result(0)?.into();

        // res_wide = lhs_wide * rhs_wide
        let res_wide_op = self.op_mul(&block, lhs_wide, rhs_wide);
        let res_wide = res_wide_op.result(0)?.into();

        //res = res_wide mod PRIME
        let in_range_op = self.op_felt_modulo(&block, res_wide)?;
        let in_range = in_range_op.result(0)?.into();
        let res_op = self.op_trunc(&block, in_range, felt_type);
        let res = res_op.result(0)?.into();

        self.op_return(&block, &[res]);

        region.append_block(block);
        let func = self.op_func(
            &id,
            &create_fn_signature(&[felt_type, felt_type], &[felt_type]),
            vec![region],
            false,
            false,
        )?;
        parent_block.append_operation(func);

        storage.libfuncs.insert(
            id,
            SierraLibFunc::create_simple(
                vec![sierra_felt_type.clone(), sierra_felt_type.clone()],
                vec![sierra_felt_type],
            ),
        );
        Ok(())
    }

    pub fn create_libfunc_felt_div(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let sierra_felt_type = SierraType::Simple(self.felt_type());
        let felt_type = sierra_felt_type.get_type();
        let felt_type_location = sierra_felt_type.get_type_location(&self.context);

        let region = Region::new();
        let block = Block::new(&[felt_type_location, felt_type_location]);

        // res = lhs / rhs (where / is modular division)
        let lhs = block.argument(0)?.into();
        let rhs = block.argument(1)?.into();
        let res_op = self.op_felt_div(&region, &block, lhs, rhs)?;
        let res = res_op.result(0)?.into();

        self.op_return(&block, &[res]);

        region.append_block(block);
        let func = self.op_func(
            &id,
            &create_fn_signature(&[felt_type, felt_type], &[felt_type]),
            vec![region],
            false,
            false,
        )?;
        parent_block.append_operation(func);

        storage.libfuncs.insert(
            id,
            SierraLibFunc::create_simple(
                vec![sierra_felt_type.clone(), sierra_felt_type.clone()],
                vec![sierra_felt_type],
            ),
        );
        Ok(())
    }

    pub fn register_libfunc_felt252_is_zero(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        storage.libfuncs.insert(
            id,
            SierraLibFunc::Function(LibFuncDef {
                args: vec![LibFuncArg { loc: 0, ty: SierraType::Simple(self.felt_type()) }],
                return_types: vec![vec![], vec![SierraType::Simple(self.felt_type())]],
            }),
        );
    }

    pub fn register_libfunc_enum_match(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        let arg = if let GenericArg::Type(x) = &func_decl.long_id.generic_args[0] {
            x
        } else {
            unreachable!("enum_match argument should be a type")
        };

        let arg_type = storage.types.get(&arg.id.to_string()).cloned().expect("type should exist");

        if let SierraType::Enum {
            ty: _,
            tag_type: _,
            storage_bytes_len: _,
            storage_type: _,
            variants_types,
        } = arg_type.clone()
        {
            storage.libfuncs.insert(
                id,
                SierraLibFunc::Function(LibFuncDef {
                    args: vec![LibFuncArg { loc: 0, ty: arg_type }],
                    return_types: variants_types.into_iter().map(|x| [x].to_vec()).collect_vec(),
                }),
            );
        } else {
            panic!("enum_match arg_type should be a enum")
        }
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

        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let func = self.op_func(
            &id,
            &create_fn_signature(&[src_type], &[self.felt_type()]),
            vec![region],
            false,
            false,
        )?;

        storage.libfuncs.insert(
            id,
            SierraLibFunc::create_simple(
                vec![SierraType::Simple(src_type)],
                vec![SierraType::Simple(self.felt_type())],
            ),
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

        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let func = self.op_func(
            &id,
            &create_fn_signature(&[src_type, src_type], &[dst_type]),
            vec![region],
            false,
            false,
        )?;

        storage.libfuncs.insert(
            id,
            SierraLibFunc::create_simple(
                vec![SierraType::Simple(src_type), SierraType::Simple(src_type)],
                vec![SierraType::Simple(dst_type)],
            ),
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

        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
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
            SierraLibFunc::create_simple(
                vec![
                    SierraType::Simple(self.range_check_type()),
                    SierraType::Simple(self.u128_type()),
                    SierraType::Simple(self.u128_type()),
                ],
                vec![
                    SierraType::Simple(self.range_check_type()),
                    SierraType::Simple(self.u128_type()),
                    SierraType::Simple(self.u128_type()),
                ],
            ),
        );
        parent_block.append_operation(func);

        Ok(())
    }

    pub fn create_libfunc_uint_safe_divmod(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
        src_type: Type<'ctx>,
    ) -> Result<()> {
        let region = Region::new();
        let block = region.append_block(Block::new(&[
            (self.range_check_type(), Location::unknown(&self.context)),
            (src_type, Location::unknown(&self.context)),
            (src_type, Location::unknown(&self.context)),
        ]));

        let op_div = self.op_div(&block, block.argument(1)?.into(), block.argument(2)?.into());
        let op_rem = self.op_rem(&block, block.argument(1)?.into(), block.argument(2)?.into());

        self.op_return(
            &block,
            &[block.argument(0)?.into(), op_div.result(0)?.into(), op_rem.result(0)?.into()],
        );

        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let func = self.op_func(
            &id,
            &create_fn_signature(
                &[self.range_check_type(), src_type, src_type],
                &[self.range_check_type(), src_type, src_type],
            ),
            vec![region],
            false,
            false,
        )?;

        storage.libfuncs.insert(
            id,
            SierraLibFunc::create_simple(
                vec![
                    SierraType::Simple(self.range_check_type()),
                    SierraType::Simple(src_type),
                    SierraType::Simple(src_type),
                ],
                vec![
                    SierraType::Simple(self.range_check_type()),
                    SierraType::Simple(src_type),
                    SierraType::Simple(src_type),
                ],
            ),
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
        let data_in = &[self.u128_type(), self.u128_type()];
        let data_out = &[self.u128_type(), self.u128_type(), self.u128_type()];

        let region = Region::new();
        region.append_block({
            let block = self.new_block(data_in);

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

        let fn_id = func_decl.id.debug_name.as_deref().unwrap().to_string();
        let fn_ty = create_fn_signature(data_in, data_out);
        let fn_op = self.op_func(&fn_id, &fn_ty, vec![region], false, false)?;

        storage.libfuncs.insert(
            fn_id,
            SierraLibFunc::Function(LibFuncDef {
                args: vec![
                    LibFuncArg { loc: 1, ty: SierraType::Simple(data_in[0]) },
                    LibFuncArg { loc: 2, ty: SierraType::Simple(data_in[1]) },
                ],
                return_types: vec![data_out.iter().copied().map(SierraType::Simple).collect_vec()],
            }),
        );

        parent_block.append_operation(fn_op);
        Ok(())
    }

    pub fn create_libfunc_upcast(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        let src_sierra_type = storage
            .types
            .get(&match &func_decl.long_id.generic_args[0] {
                GenericArg::Type(x) => x.id.to_string(),
                _ => todo!("invalid generic kind"),
            })
            .expect("type to exist")
            .clone();
        let dst_sierra_type = storage
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

                storage.libfuncs.insert(
                    id,
                    SierraLibFunc::create_simple(vec![src_sierra_type], vec![dst_sierra_type]),
                );

                parent_block.append_operation(func);
            }
            Ordering::Equal => {
                // Similar to store_local and rename, create an identity function for ease of dataflow processing, under the assumption the optimiser will optimise it out
                self.create_identity_function(func_decl, parent_block, storage)?;
            }
            Ordering::Greater => todo!("invalid generics for libfunc `upcast`"),
        }

        Ok(())
    }
}
