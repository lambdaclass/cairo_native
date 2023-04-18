use std::cmp::Ordering;

use cairo_lang_sierra::program::{GenericArg, LibfuncDeclaration};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{Block, BlockRef, Location, Region, Type, TypeLike, Value, ValueLike};
use num_bigint::BigInt;
use num_traits::Signed;
use tracing::debug;

use crate::{
    compiler::{CmpOp, Compiler, FnAttributes, SierraType, Storage},
    types::{is_omitted_builtin_type, DEFAULT_PRIME},
    utility::create_fn_signature,
};

use self::lib_func_def::{PositionalArg, SierraLibFunc};
use melior_asm::mlir_asm;

pub mod lib_func_def;
pub mod sierra_enum;

#[derive(Debug, Clone, Copy)]
pub enum BoolBinaryOp {
    And,
    Xor,
    Or,
}

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
                "store_local" => self.register_store_local(func_decl, storage)?,
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
                    self.register_libfunc_int_is_zero(func_decl, self.felt_type(), storage);
                }
                "dup" => {
                    self.register_libfunc_dup(func_decl, storage)?;
                }
                "snapshot_take" => {
                    self.register_libfunc_snapshot_take(func_decl, storage)?;
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
                "store_temp" | "rename" | "unbox" => {
                    self.register_identity_function(func_decl, storage)?;
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
                "u8_is_zero" => {
                    self.register_libfunc_int_is_zero(func_decl, self.u8_type(), storage);
                }
                "u16_is_zero" => {
                    self.register_libfunc_int_is_zero(func_decl, self.u16_type(), storage);
                }
                "u32_is_zero" => {
                    self.register_libfunc_int_is_zero(func_decl, self.u32_type(), storage);
                }
                "u64_is_zero" => {
                    self.register_libfunc_int_is_zero(func_decl, self.u64_type(), storage);
                }
                "u128_is_zero" => {
                    self.register_libfunc_int_is_zero(func_decl, self.u128_type(), storage);
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
                "u8_eq" => {
                    self.register_libfunc_int_eq(func_decl, self.u8_type(), storage);
                }
                "u16_eq" => {
                    self.register_libfunc_int_eq(func_decl, self.u16_type(), storage);
                }
                "u32_eq" => {
                    self.register_libfunc_int_eq(func_decl, self.u32_type(), storage);
                }
                "u64_eq" => {
                    self.register_libfunc_int_eq(func_decl, self.u64_type(), storage);
                }
                "u128_eq" => {
                    self.register_libfunc_int_eq(func_decl, self.u128_type(), storage);
                }
                "u8_le" => {
                    self.register_libfunc_int_le(func_decl, self.u8_type(), storage);
                }
                "u16_le" => {
                    self.register_libfunc_int_le(func_decl, self.u16_type(), storage);
                }
                "u32_le" => {
                    self.register_libfunc_int_le(func_decl, self.u32_type(), storage);
                }
                "u64_le" => {
                    self.register_libfunc_int_le(func_decl, self.u64_type(), storage);
                }
                "u128_le" => {
                    self.register_libfunc_int_le(func_decl, self.u128_type(), storage);
                }
                "u8_lt" => {
                    self.register_libfunc_int_lt(func_decl, self.u8_type(), storage);
                }
                "u16_lt" => {
                    self.register_libfunc_int_lt(func_decl, self.u16_type(), storage);
                }
                "u32_lt" => {
                    self.register_libfunc_int_lt(func_decl, self.u32_type(), storage);
                }
                "u64_lt" => {
                    self.register_libfunc_int_lt(func_decl, self.u64_type(), storage);
                }
                "u128_lt" => {
                    self.register_libfunc_int_lt(func_decl, self.u128_type(), storage);
                }
                "u8_overflowing_add" | "u8_overflowing_sub" => {
                    self.register_libfunc_uint_overflowing_op(func_decl, self.u8_type(), storage);
                }
                "u16_overflowing_add" | "u16_overflowing_sub" => {
                    self.register_libfunc_uint_overflowing_op(func_decl, self.u16_type(), storage);
                }
                "u32_overflowing_add" | "u32_overflowing_sub" => {
                    self.register_libfunc_uint_overflowing_op(func_decl, self.u32_type(), storage);
                }
                "u64_overflowing_add" | "u64_overflowing_sub" => {
                    self.register_libfunc_uint_overflowing_op(func_decl, self.u64_type(), storage);
                }
                "u128_overflowing_add" | "u128_overflowing_sub" => {
                    self.register_libfunc_uint_overflowing_op(func_decl, self.u128_type(), storage);
                }
                "bitwise" => {
                    self.create_libfunc_bitwise(func_decl, parent_block, storage)?;
                }
                "upcast" => {
                    self.create_libfunc_upcast(func_decl, parent_block, storage)?;
                }
                "bool_or_impl" => {
                    self.create_libfunc_bool_binop_impl(
                        func_decl,
                        parent_block,
                        storage,
                        BoolBinaryOp::Or,
                    )?;
                }
                "bool_and_impl" => {
                    self.create_libfunc_bool_binop_impl(
                        func_decl,
                        parent_block,
                        storage,
                        BoolBinaryOp::And,
                    )?;
                }
                "bool_xor_impl" => {
                    self.create_libfunc_bool_binop_impl(
                        func_decl,
                        parent_block,
                        storage,
                        BoolBinaryOp::Xor,
                    )?;
                }
                "bool_not_impl" => {
                    self.create_libfunc_bool_not_impl(func_decl, parent_block, storage)?;
                }
                "bool_to_felt252" => {
                    self.create_libfunc_bool_to_felt252(func_decl, parent_block, storage)?;
                }
                "array_new" => {
                    self.create_libfunc_array_new(func_decl, parent_block, storage)?;
                }
                "array_append" => {
                    self.create_libfunc_array_append(func_decl, parent_block, storage)?;
                }
                "array_len" => {
                    self.create_libfunc_array_len(func_decl, parent_block, storage)?;
                }
                "array_get" => {
                    self.register_libfunc_array_get(func_decl, storage);
                }
                "array_pop_front" => {
                    self.register_libfunc_array_pop_front(func_decl, storage);
                }
                "print" => {
                    self.create_libfunc_print(func_decl, parent_block, storage)?;
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
        storage.libfuncs.insert(id, SierraLibFunc::create_function_all_args(vec![], vec![]));
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

        let mut struct_type_op = self.op_llvm_struct_from_types(&block, &args);

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

        let func =
            self.op_func(&id, &function_type, vec![region], FnAttributes::libfunc(false, true))?;

        storage.libfuncs.insert(
            id,
            SierraLibFunc::create_function_all_args(
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
        let fn_op =
            self.op_func(&fn_id, &fn_ty, vec![region], FnAttributes::libfunc(false, true))?;

        let return_types = field_types.to_vec();
        let struct_type = struct_type.clone();
        storage.libfuncs.insert(
            fn_id,
            SierraLibFunc::create_function_all_args(vec![struct_type], return_types),
        );

        parent_block.append_operation(fn_op);
        Ok(())
    }

    pub fn register_identity_function(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        let results = match &func_decl.long_id.generic_args[0] {
            GenericArg::Type(type_id) => {
                if is_omitted_builtin_type(type_id.debug_name.as_ref().unwrap().as_str()) {
                    vec![]
                } else {
                    vec![PositionalArg {
                        loc: 0,
                        ty: storage
                            .types
                            .get(&type_id.id.to_string())
                            .expect("type to exist")
                            .clone(),
                    }]
                }
            }
            _ => unreachable!("Generic argument to {} should be a Type", id),
        };
        storage.libfuncs.insert(id, SierraLibFunc::InlineDataflow(results));

        Ok(())
    }

    pub fn register_store_local(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        // Since we don't model locals the same as sierra, store_local can just return its payload
        let results = match &func_decl.long_id.generic_args[0] {
            GenericArg::Type(type_id) => {
                if is_omitted_builtin_type(type_id.debug_name.as_ref().unwrap().as_str()) {
                    vec![]
                } else {
                    // Skip over the first argument, which is the uninitialized local variable
                    vec![PositionalArg {
                        loc: 1,
                        ty: storage
                            .types
                            .get(&type_id.id.to_string())
                            .expect("type to exist")
                            .clone(),
                    }]
                }
            }
            _ => unreachable!("Argument to store_local should be a Type"),
        };
        storage.libfuncs.insert(id, SierraLibFunc::InlineDataflow(results));

        Ok(())
    }

    pub fn register_libfunc_dup(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let results = match &func_decl.long_id.generic_args[0] {
            GenericArg::UserType(_) => todo!(),
            GenericArg::Type(type_id) => {
                if is_omitted_builtin_type(type_id.debug_name.as_ref().unwrap().as_str()) {
                    vec![]
                } else {
                    let arg_type =
                        storage.types.get(&type_id.id.to_string()).expect("type to exist").clone();
                    vec![
                        PositionalArg { loc: 0, ty: arg_type.clone() },
                        PositionalArg { loc: 0, ty: arg_type },
                    ]
                }
            }
            GenericArg::Value(_) => todo!(),
            GenericArg::UserFunc(_) => todo!(),
            GenericArg::Libfunc(_) => todo!(),
        };

        storage.libfuncs.insert(id, SierraLibFunc::InlineDataflow(results));

        Ok(())
    }

    pub fn register_libfunc_snapshot_take(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
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

        storage.libfuncs.insert(
            id,
            SierraLibFunc::InlineDataflow(vec![
                PositionalArg { loc: 0, ty: arg_type.clone() },
                PositionalArg { loc: 0, ty: arg_type },
            ]),
        );

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
        let gte_prime_op = self.op_cmp(&entry_block, CmpOp::UnsignedGreaterThanEqual, res, prime);
        let gte_prime = gte_prime_op.result(0)?.into();

        // if gt_prime
        self.op_cond_br(&entry_block, gte_prime, &gte_prime_block, &in_range_block, &[], &[]);

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
            FnAttributes::libfunc(false, true),
        )?;

        parent_block.append_operation(func);
        storage.libfuncs.insert(
            id,
            SierraLibFunc::create_function_all_args(
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
        self.op_cond_br(&entry_block, lt_zero, &lt_zero_block, &in_range_block, &[], &[]);

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
            FnAttributes::libfunc(false, true),
        )?;

        parent_block.append_operation(func);
        storage.libfuncs.insert(
            id,
            SierraLibFunc::create_function_all_args(
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
            FnAttributes::libfunc(false, true),
        )?;
        parent_block.append_operation(func);

        storage.libfuncs.insert(
            id,
            SierraLibFunc::create_function_all_args(
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
            FnAttributes::libfunc(false, true),
        )?;
        parent_block.append_operation(func);

        storage.libfuncs.insert(
            id,
            SierraLibFunc::create_function_all_args(
                vec![sierra_felt_type.clone(), sierra_felt_type.clone()],
                vec![sierra_felt_type],
            ),
        );
        Ok(())
    }

    pub fn register_libfunc_int_is_zero(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        op_type: Type<'ctx>,
        storage: &mut Storage<'ctx>,
    ) {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        storage.libfuncs.insert(
            id,
            SierraLibFunc::Branching {
                args: vec![PositionalArg { loc: 0, ty: SierraType::Simple(op_type) }],
                return_types: vec![
                    vec![],
                    vec![PositionalArg { loc: 0, ty: SierraType::Simple(op_type) }],
                ],
            },
        );
    }

    pub fn register_libfunc_int_eq(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        op_type: Type<'ctx>,
        storage: &mut Storage<'ctx>,
    ) {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        storage.libfuncs.insert(
            id,
            SierraLibFunc::Branching {
                args: vec![
                    PositionalArg { loc: 0, ty: SierraType::Simple(op_type) },
                    PositionalArg { loc: 1, ty: SierraType::Simple(op_type) },
                ],
                return_types: vec![vec![], vec![]],
            },
        );
    }

    pub fn register_libfunc_int_le(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        op_type: Type<'ctx>,
        storage: &mut Storage<'ctx>,
    ) {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        storage.libfuncs.insert(
            id,
            SierraLibFunc::Branching {
                args: vec![
                    PositionalArg { loc: 1, ty: SierraType::Simple(op_type) },
                    PositionalArg { loc: 2, ty: SierraType::Simple(op_type) },
                ],
                return_types: vec![vec![], vec![]],
            },
        );
    }

    pub fn register_libfunc_int_lt(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        op_type: Type<'ctx>,
        storage: &mut Storage<'ctx>,
    ) {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        storage.libfuncs.insert(
            id,
            SierraLibFunc::Branching {
                args: vec![
                    PositionalArg { loc: 1, ty: SierraType::Simple(op_type) },
                    PositionalArg { loc: 2, ty: SierraType::Simple(op_type) },
                ],
                return_types: vec![vec![], vec![]],
            },
        );
    }

    pub fn register_libfunc_uint_overflowing_op(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        op_type: Type<'ctx>,
        storage: &mut Storage<'ctx>,
    ) {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        storage.libfuncs.insert(
            id,
            SierraLibFunc::Branching {
                args: vec![
                    PositionalArg { loc: 1, ty: SierraType::Simple(op_type) },
                    PositionalArg { loc: 2, ty: SierraType::Simple(op_type) },
                ],
                return_types: vec![
                    vec![PositionalArg { loc: 1, ty: SierraType::Simple(op_type) }],
                    vec![PositionalArg { loc: 1, ty: SierraType::Simple(op_type) }],
                ],
            },
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
                SierraLibFunc::Branching {
                    args: vec![PositionalArg { loc: 0, ty: arg_type }],
                    return_types: variants_types
                        .into_iter()
                        .map(|x| vec![PositionalArg { loc: 0, ty: x }])
                        .collect_vec(),
                },
            );
        } else {
            panic!("enum_match arg_type should be a enum")
        }
    }

    pub fn register_libfunc_array_get(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        let arg = if let GenericArg::Type(x) = &func_decl.long_id.generic_args[0] {
            x
        } else {
            unreachable!("array_get argument should be a type")
        };

        let arg_type = storage.types.get(&arg.id.to_string()).cloned().expect("type should exist");

        let sierra_type = SierraType::get_array_type(self, arg_type.clone());

        // 2 branches:
        // - falthrough with return args: 0 = rangecheck, 1 = the value at index
        // - branch jump: if out of bounds jump, return arg 0 = range check

        storage.libfuncs.insert(
            id,
            SierraLibFunc::Branching {
                args: vec![
                    PositionalArg { loc: 1, ty: sierra_type.clone() }, // array
                    PositionalArg { loc: 2, ty: SierraType::Simple(self.u32_type()) }, // index
                ],
                return_types: vec![
                    vec![PositionalArg { loc: 1, ty: arg_type }], // fallthrough
                    vec![],                                       // panic branch
                ],
            },
        );
    }

    pub fn register_libfunc_array_pop_front(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        let arg = if let GenericArg::Type(x) = &func_decl.long_id.generic_args[0] {
            x
        } else {
            unreachable!("array_pop_front argument should be a type")
        };

        let arg_type = storage.types.get(&arg.id.to_string()).cloned().expect("type should exist");

        let sierra_type = SierraType::get_array_type(self, arg_type.clone());

        storage.libfuncs.insert(
            id,
            SierraLibFunc::Branching {
                args: vec![
                    PositionalArg { loc: 0, ty: sierra_type.clone() }, // array
                ],
                return_types: vec![
                    // fallthrough (pop returned something): array, popped value
                    vec![
                        PositionalArg { loc: 0, ty: sierra_type.clone() },
                        PositionalArg { loc: 1, ty: arg_type },
                    ],
                    // jump (pop returned none): array
                    vec![PositionalArg { loc: 0, ty: sierra_type }],
                ],
            },
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

        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let func = self.op_func(
            &id,
            &create_fn_signature(&[src_type], &[self.felt_type()]),
            vec![region],
            FnAttributes::libfunc(false, true),
        )?;

        storage.libfuncs.insert(
            id,
            SierraLibFunc::create_function_all_args(
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
            FnAttributes::libfunc(false, true),
        )?;

        storage.libfuncs.insert(
            id,
            SierraLibFunc::create_function_all_args(
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
            (self.u128_type(), Location::unknown(&self.context)),
            (self.u128_type(), Location::unknown(&self.context)),
        ]));

        let op_zext_lhs = self.op_zext(&block, block.argument(0)?.into(), self.u256_type());
        let op_zext_rhs = self.op_zext(&block, block.argument(1)?.into(), self.u256_type());

        let op_mul =
            self.op_mul(&block, op_zext_lhs.result(0)?.into(), op_zext_rhs.result(0)?.into());

        let op_mul_lo = self.op_trunc(&block, op_mul.result(0)?.into(), self.u128_type());
        let op_mul_hi = {
            let op_const = self.op_const(&block, "128", self.u256_type());
            let op_shru =
                self.op_shru(&block, op_mul.result(0)?.into(), op_const.result(0)?.into());
            self.op_trunc(&block, op_shru.result(0)?.into(), self.u128_type())
        };

        self.op_return(&block, &[op_mul_hi.result(0)?.into(), op_mul_lo.result(0)?.into()]);

        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let func = self.op_func(
            &id,
            &create_fn_signature(
                &[self.u128_type(), self.u128_type()],
                &[self.u128_type(), self.u128_type()],
            ),
            vec![region],
            FnAttributes::libfunc(false, true),
        )?;

        storage.libfuncs.insert(
            id,
            SierraLibFunc::Function {
                // Skip the range check argument and return
                args: vec![
                    PositionalArg { loc: 1, ty: SierraType::Simple(self.u128_type()) },
                    PositionalArg { loc: 2, ty: SierraType::Simple(self.u128_type()) },
                ],
                return_types: vec![
                    PositionalArg { loc: 1, ty: SierraType::Simple(self.u128_type()) },
                    PositionalArg { loc: 2, ty: SierraType::Simple(self.u128_type()) },
                ],
            },
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
            (src_type, Location::unknown(&self.context)),
            (src_type, Location::unknown(&self.context)),
        ]));

        let lhs = block.argument(0)?.into();
        let rhs = block.argument(1)?.into();

        let op_div = self.op_div(&block, lhs, rhs);
        let op_rem = self.op_rem(&block, lhs, rhs);

        self.op_return(&block, &[op_div.result(0)?.into(), op_rem.result(0)?.into()]);

        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let func = self.op_func(
            &id,
            &create_fn_signature(&[src_type, src_type], &[src_type, src_type]),
            vec![region],
            FnAttributes::libfunc(false, true),
        )?;

        storage.libfuncs.insert(
            id,
            SierraLibFunc::Function {
                // Skip range check
                args: vec![
                    PositionalArg { loc: 1, ty: SierraType::Simple(src_type) },
                    PositionalArg { loc: 2, ty: SierraType::Simple(src_type) },
                ],
                return_types: vec![
                    PositionalArg { loc: 1, ty: SierraType::Simple(src_type) },
                    PositionalArg { loc: 2, ty: SierraType::Simple(src_type) },
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
        let fn_op =
            self.op_func(&fn_id, &fn_ty, vec![region], FnAttributes::libfunc(false, true))?;

        storage.libfuncs.insert(
            fn_id,
            SierraLibFunc::Function {
                // Skip bitwise argument and return
                args: vec![
                    PositionalArg { loc: 1, ty: SierraType::Simple(data_in[0]) },
                    PositionalArg { loc: 2, ty: SierraType::Simple(data_in[1]) },
                ],
                return_types: data_out
                    .iter()
                    .enumerate()
                    .map(|(idx, ty)| PositionalArg { loc: idx + 1, ty: SierraType::Simple(*ty) })
                    .collect_vec(),
            },
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
                    FnAttributes::libfunc(false, true),
                )?;

                storage.libfuncs.insert(
                    id,
                    SierraLibFunc::create_function_all_args(
                        vec![src_sierra_type],
                        vec![dst_sierra_type],
                    ),
                );

                parent_block.append_operation(func);
            }
            Ordering::Equal => {
                // Similar to store_local and rename, create a libfuncdef that tells statement processing to just forward its argument
                self.register_identity_function(func_decl, storage)?;
            }
            Ordering::Greater => todo!("invalid generics for libfunc `upcast`"),
        }

        Ok(())
    }

    /// 2 boolean enums -> 1 bool enum
    pub fn create_libfunc_bool_binop_impl(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
        bool_op: BoolBinaryOp,
    ) -> Result<()> {
        let data_in = &[self.boolean_enum_type(), self.boolean_enum_type()];
        let data_out = &[self.boolean_enum_type()];

        let bool_variant = SierraType::Struct {
            ty: self.struct_type(&[Type::none(&self.context)]),
            field_types: vec![],
        };

        let bool_sierra_type = SierraType::Enum {
            ty: self.boolean_enum_type(),
            tag_type: self.u16_type(),
            storage_bytes_len: 0,
            storage_type: Type::parse(&self.context, "!llvm.array<0 x i8>").unwrap(),
            variants_types: vec![bool_variant.clone(), bool_variant],
        };

        let region = Region::new();
        region.append_block({
            let block = self.new_block(data_in);

            let lhs = block.argument(0)?;
            let rhs = block.argument(1)?;

            let lhs_tag_value_op =
                self.op_llvm_extractvalue(&block, 0, lhs.into(), self.u16_type())?;
            let lhs_tag_value: Value = lhs_tag_value_op.result(0)?.into();

            let rhs_tag_value_op =
                self.op_llvm_extractvalue(&block, 0, rhs.into(), self.u16_type())?;
            let rhs_tag_value: Value = rhs_tag_value_op.result(0)?.into();

            let bool_op_ref = match bool_op {
                BoolBinaryOp::And => {
                    self.op_and(&block, lhs_tag_value, rhs_tag_value, self.u16_type())
                }
                BoolBinaryOp::Xor => {
                    self.op_xor(&block, lhs_tag_value, rhs_tag_value, self.u16_type())
                }
                BoolBinaryOp::Or => {
                    self.op_or(&block, lhs_tag_value, rhs_tag_value, self.u16_type())
                }
            };

            let enum_op = self.op_llvm_struct(&block, self.boolean_enum_type());
            let enum_value: Value = enum_op.result(0)?.into();

            let enum_res = self.op_llvm_insertvalue(
                &block,
                0,
                enum_value,
                bool_op_ref.result(0)?.into(),
                self.boolean_enum_type(),
            )?;

            self.op_return(&block, &[enum_res.result(0)?.into()]);

            block
        });

        let fn_id = func_decl.id.debug_name.as_deref().unwrap().to_string();
        let fn_ty = create_fn_signature(data_in, data_out);
        let fn_op =
            self.op_func(&fn_id, &fn_ty, vec![region], FnAttributes::libfunc(false, true))?;

        storage.libfuncs.insert(
            fn_id,
            SierraLibFunc::create_function_all_args(
                vec![bool_sierra_type.clone(), bool_sierra_type.clone()],
                vec![bool_sierra_type.clone()],
            ),
        );

        parent_block.append_operation(fn_op);
        Ok(())
    }

    pub fn create_libfunc_bool_not_impl(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let data_in = &[self.boolean_enum_type()];
        let data_out = &[self.boolean_enum_type()];

        let bool_variant = SierraType::Struct {
            ty: self.struct_type(&[Type::none(&self.context)]),
            field_types: vec![],
        };

        let bool_sierra_type = SierraType::Enum {
            ty: self.boolean_enum_type(),
            tag_type: self.u16_type(),
            storage_bytes_len: 0,
            storage_type: Type::parse(&self.context, "!llvm.array<0 x i8>").unwrap(),
            variants_types: vec![bool_variant.clone()],
        };

        let region = Region::new();
        region.append_block({
            let block = self.new_block(data_in);

            let lhs = block.argument(0)?;

            let lhs_tag_value_op =
                self.op_llvm_extractvalue(&block, 0, lhs.into(), self.u16_type())?;
            let lhs_tag_value: Value = lhs_tag_value_op.result(0)?.into();

            let const_1_op = self.op_const(&block, "1", self.u16_type());

            let bool_op_ref =
                self.op_xor(&block, lhs_tag_value, const_1_op.result(0)?.into(), self.u16_type());

            let enum_op = self.op_llvm_struct(&block, self.boolean_enum_type());
            let enum_value: Value = enum_op.result(0)?.into();

            let enum_res = self.op_llvm_insertvalue(
                &block,
                0,
                enum_value,
                bool_op_ref.result(0)?.into(),
                self.boolean_enum_type(),
            )?;

            self.op_return(&block, &[enum_res.result(0)?.into()]);

            block
        });

        let fn_id = func_decl.id.debug_name.as_deref().unwrap().to_string();
        let fn_ty = create_fn_signature(data_in, data_out);
        let fn_op =
            self.op_func(&fn_id, &fn_ty, vec![region], FnAttributes::libfunc(false, true))?;

        storage.libfuncs.insert(
            fn_id,
            SierraLibFunc::create_function_all_args(
                vec![bool_sierra_type.clone()],
                vec![bool_sierra_type],
            ),
        );

        parent_block.append_operation(fn_op);
        Ok(())
    }

    /// bool to felt
    ///
    /// Sierra:
    /// `extern fn bool_to_felt252(a: bool) -> felt252 implicits() nopanic;`
    pub fn create_libfunc_bool_to_felt252(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let data_in = &[self.boolean_enum_type()];
        let data_out = &[self.felt_type()];

        let bool_variant = SierraType::Struct {
            ty: self.struct_type(&[Type::none(&self.context)]),
            field_types: vec![],
        };

        let bool_sierra_type = SierraType::Enum {
            ty: self.boolean_enum_type(),
            tag_type: self.u16_type(),
            storage_bytes_len: 0,
            storage_type: Type::parse(&self.context, "!llvm.array<0 x i8>").unwrap(),
            variants_types: vec![bool_variant.clone()],
        };

        let region = Region::new();
        region.append_block({
            let block = self.new_block(data_in);

            let bool_enum = block.argument(0)?;

            let tag_value_op =
                self.op_llvm_extractvalue(&block, 0, bool_enum.into(), self.u16_type())?;
            let tag_value: Value = tag_value_op.result(0)?.into();

            let felt_value = self.op_zext(&block, tag_value, self.felt_type());

            self.op_return(&block, &[felt_value.result(0)?.into()]);

            block
        });

        let fn_id = func_decl.id.debug_name.as_deref().unwrap().to_string();
        let fn_ty = create_fn_signature(data_in, data_out);
        let fn_op =
            self.op_func(&fn_id, &fn_ty, vec![region], FnAttributes::libfunc(false, true))?;

        storage.libfuncs.insert(
            fn_id,
            SierraLibFunc::create_function_all_args(
                vec![bool_sierra_type],
                vec![SierraType::Simple(self.felt_type())],
            ),
        );

        parent_block.append_operation(fn_op);
        Ok(())
    }

    // `extern fn array_new<T>() -> Array<T> nopanic;`
    // in sierra: `array_new<felt252>() -> ([0]);`
    pub fn create_libfunc_array_new(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let arg_type = match &func_decl.long_id.generic_args[0] {
            GenericArg::Type(type_id) => {
                storage.types.get(&type_id.id.to_string()).cloned().expect("type to exist")
            }
            _ => unreachable!(),
        };

        let region = Region::new();

        let block = Block::new(&[]);

        let sierra_type = SierraType::get_array_type(self, arg_type.clone());

        let array_value_op = self.op_llvm_struct(&block, sierra_type.get_type());
        let array_value: Value = array_value_op.result(0)?.into();

        let array_len_op = self.op_u32_const(&block, "0");
        let array_len = array_len_op.result(0)?.into();

        let array_capacity_op = self.op_u32_const(&block, "8");
        let array_capacity = array_capacity_op.result(0)?.into();

        let array_element_size_bytes = (arg_type.get_width() + 7) / 8;

        // length
        let insert_op =
            self.op_llvm_insertvalue(&block, 0, array_value, array_len, sierra_type.get_type())?;
        let array_value: Value = insert_op.result(0)?.into();

        // capacity
        let insert_op = self.op_llvm_insertvalue(
            &block,
            1,
            array_value,
            array_capacity,
            sierra_type.get_type(),
        )?;
        let array_value: Value = insert_op.result(0)?.into();

        // 8 here is the capacity
        let const_arr_size_bytes_op =
            self.op_const(&block, &(array_element_size_bytes * 8).to_string(), self.u64_type());
        let const_arr_size_bytes = const_arr_size_bytes_op.result(0)?;

        let null_ptr_op = self.op_llvm_nullptr(&block);
        let null_ptr = null_ptr_op.result(0)?;

        let ptr_op = self.call_realloc(&block, null_ptr.into(), const_arr_size_bytes.into())?;
        let ptr_val = ptr_op.result(0)?;

        let insert_op = self.op_llvm_insertvalue(
            &block,
            2,
            array_value,
            ptr_val.into(),
            sierra_type.get_type(),
        )?;
        let array_value: Value = insert_op.result(0)?.into();

        self.op_return(&block, &[array_value]);

        let function_type = create_fn_signature(&[], &[sierra_type.get_type()]);

        region.append_block(block);

        let func =
            self.op_func(&id, &function_type, vec![region], FnAttributes::libfunc(false, true))?;

        storage
            .libfuncs
            .insert(id, SierraLibFunc::create_function_all_args(vec![], vec![sierra_type]));

        parent_block.append_operation(func);

        Ok(())
    }

    // `extern fn array_append<T>(ref arr: Array<T>, value: T) nopanic;`
    // in sierra `array_append<T>([0], [1]) -> ([2]);`
    pub fn create_libfunc_array_append(
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
        let region = Region::new();

        let sierra_type = SierraType::get_array_type(self, arg_type.clone());

        let block = region.append_block(Block::new(&[
            sierra_type.get_type_location(&self.context),
            arg_type.get_type_location(&self.context),
        ]));

        let array_type = sierra_type.get_type();
        let array_type_with_loc = sierra_type.get_type_location(&self.context);

        let array_value = block.argument(0)?;
        let append_value = block.argument(1)?;

        // check if len < capacity
        let array_len_op =
            self.op_llvm_extractvalue(&block, 0, array_value.into(), self.u32_type())?;
        let array_len = array_len_op.result(0)?;

        let array_capacity_op =
            self.op_llvm_extractvalue(&block, 1, array_value.into(), self.u32_type())?;
        let array_capacity = array_capacity_op.result(0)?;

        let realloc_block = region.append_block(Block::new(&[]));
        let append_value_block = region.append_block(Block::new(&[array_type_with_loc]));

        let is_less =
            self.op_cmp(&block, CmpOp::UnsignedLessThan, array_len.into(), array_capacity.into());

        self.op_cond_br(
            &block,
            is_less.result(0)?.into(),
            &append_value_block,
            &realloc_block,
            &[array_value.into()],
            &[],
        );

        // reallocate with more capacity, for now with a simple algorithm:
        // new_capacity = capacity * 2
        let const_2_op = self.op_u32_const(&realloc_block, "2");
        let const_2 = const_2_op.result(0)?;

        let new_capacity_op = self.op_mul(&realloc_block, array_capacity.into(), const_2.into());
        let new_capacity = new_capacity_op.result(0)?.into();

        let new_capacity_as_u64_op = self.op_zext(&realloc_block, new_capacity, self.u64_type());
        let new_capacity_as_u64 = new_capacity_as_u64_op.result(0)?;

        let data_ptr_op =
            self.op_llvm_extractvalue(&realloc_block, 2, array_value.into(), self.llvm_ptr_type())?;
        let data_ptr: Value = data_ptr_op.result(0)?.into();

        let new_ptr_op = self.call_realloc(&realloc_block, data_ptr, new_capacity_as_u64.into())?;
        let new_ptr = new_ptr_op.result(0)?.into();

        // change the ptr
        let insert_ptr_op =
            self.op_llvm_insertvalue(&realloc_block, 2, array_value.into(), new_ptr, array_type)?;
        let array_value = insert_ptr_op.result(0)?;

        // update the capacity
        let insert_ptr_op = self.op_llvm_insertvalue(
            &realloc_block,
            1,
            array_value.into(),
            new_capacity,
            array_type,
        )?;
        let array_value = insert_ptr_op.result(0)?;

        self.op_br(&realloc_block, &append_value_block, &[array_value.into()]);

        // append value and len + 1
        let array_value = append_value_block.argument(0)?;

        // get the data pointer
        let data_ptr_op = self.op_llvm_extractvalue(
            &append_value_block,
            2,
            array_value.into(),
            self.llvm_ptr_type(),
        )?;
        let data_ptr: Value = data_ptr_op.result(0)?.into();

        // get the pointer to the data index
        let value_ptr_op = self.op_llvm_gep_dynamic(
            &append_value_block,
            &[array_len.into()],
            data_ptr,
            arg_type.get_type(),
        )?;
        let value_ptr = value_ptr_op.result(0)?.into();
        // update the value
        self.op_llvm_store(&append_value_block, append_value.into(), value_ptr)?;

        // increment the length
        let const_1 = self.op_const(&append_value_block, "1", array_len.r#type());
        let len_plus_1 =
            self.op_add(&append_value_block, array_len.into(), const_1.result(0)?.into());

        let insert_op = self.op_llvm_insertvalue(
            &append_value_block,
            0,
            array_value.into(),
            len_plus_1.result(0)?.into(),
            array_type,
        )?;
        let array_value = insert_op.result(0)?;

        self.op_return(&append_value_block, &[array_value.into()]);

        let function_type = create_fn_signature(
            &[sierra_type.get_type(), arg_type.get_type()],
            &[sierra_type.get_type()],
        );

        let func =
            self.op_func(&id, &function_type, vec![region], FnAttributes::libfunc(false, true))?;

        storage.libfuncs.insert(
            id,
            SierraLibFunc::create_function_all_args(
                vec![sierra_type.clone(), arg_type],
                vec![sierra_type],
            ),
        );

        parent_block.append_operation(func);

        Ok(())
    }

    pub fn create_libfunc_array_len(
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
        let region = Region::new();

        let sierra_type = SierraType::get_array_type(self, arg_type.clone());

        let block =
            region.append_block(Block::new(&[sierra_type.get_type_location(&self.context)]));

        let array_type = sierra_type.get_type();
        let array_value = block.argument(0)?;

        let array_len_op =
            self.op_llvm_extractvalue(&block, 0, array_value.into(), self.u32_type())?;
        let array_len = array_len_op.result(0)?.into();

        self.op_return(&block, &[array_len]);

        let function_type = create_fn_signature(&[array_type], &[self.u32_type()]);

        let func =
            self.op_func(&id, &function_type, vec![region], FnAttributes::libfunc(false, true))?;

        storage.libfuncs.insert(
            id,
            SierraLibFunc::create_function_all_args(
                vec![sierra_type],
                vec![SierraType::Simple(self.u32_type())],
            ),
        );

        parent_block.append_operation(func);

        Ok(())
    }

    pub fn create_libfunc_print(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        mlir_asm! { parent_block, opt(
            "--convert-scf-to-cf",
            "--convert-linalg-to-loops",
            "--lower-affine",
            "--convert-scf-to-cf",
            "--canonicalize",
            "--cse",
            "--convert-linalg-to-llvm",
            "--convert-vector-to-llvm=reassociate-fp-reductions",
            "--convert-math-to-llvm",
            "--expand-strided-metadata",
            "--lower-affine",
            "--convert-memref-to-llvm",
            "--convert-func-to-llvm",
            "--convert-index-to-llvm",
            "--reconcile-unrealized-casts",
        ) =>
            func.func @print(%0 : !llvm.struct<(i32, i32, !llvm.ptr)>) -> () {
                // Allocate buffer.
                %1 = memref.alloca() : memref<126xi8>

                // Copy "[DEBUG] ".
                %2 = memref.get_global @lit0 : memref<8xi8>
                %3 = memref.subview %1[0][8][1] : memref<126xi8> to memref<8xi8>
                memref.copy %2, %3 : memref<8xi8> to memref<8xi8>

                // Copy " (raw: ".
                %4 = memref.get_global @lit1 : memref<7xi8>
                %5 = memref.subview %1[39][7][1] : memref<126xi8> to memref<7xi8, strided<[1], offset: 39>>
                memref.copy %4, %5 : memref<7xi8> to memref<7xi8, strided<[1], offset: 39>>

                // For each element in the array:
                %6 = index.constant 0
                %7 = llvm.extractvalue %0[0] : !llvm.struct<(i32, i32, !llvm.ptr)>
                %8 = index.castu %7 : i32 to index
                %9 = index.constant 1
                scf.for %10 = %6 to %8 step %9 {
                    // Load element to print.
                    %11 = llvm.extractvalue %0[2] : !llvm.struct<(i32, i32, !llvm.ptr)>

                    %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
                    %13 = arith.constant 5 : i64
                    %14 = index.castu %10 : index to i64
                    %15 = arith.shli %14, %13 : i64
                    %16 = arith.addi %12, %15 : i64
                    %17 = llvm.inttoptr %16 : i64 to !llvm.ptr
                    %18 = llvm.load %17 : !llvm.ptr -> i256

                    // Copy string value replacing zeros with spaces.
                    %19 = index.constant 32
                    %20 = memref.subview %1[8][32][1] : memref<126xi8> to memref<32xi8, strided<[1], offset: 8>>
                    scf.for %21 = %6 to %19 step %9 {
                        // Compute byte to write.
                        %22 = index.constant 3
                        %23 = index.shl %21, %22
                        %24 = index.castu %23 : index to i256
                        %25 = arith.shrui %18, %24 : i256
                        %26 = arith.trunci %25 : i256 to i8

                        // Map null byte (0) to ' '.
                        %27 = arith.constant 0 : i8
                        %28 = arith.cmpi eq, %26, %27 : i8
                        %29 = arith.constant 32 : i8
                        %30 = arith.select %28, %29, %26 : i8

                        // Write byte into the buffer.
                        %31 = index.constant 30
                        %32 = index.sub %31, %21
                        memref.store %30, %20[%32] : memref<32xi8, strided<[1], offset: 8>>
                    }

                    // Run algorithm to write decimal value.
                    %33 = memref.alloca() : memref<77xi8>
                    %34 = func.call @felt252_bin2dec(%33, %18) : (memref<77xi8>, i256) -> index

                    // Copy the result.
                    %35 = index.constant 76
                    %36 = index.sub %35, %34
                    %37 = memref.subview %33[%34][%36][1] : memref<77xi8> to memref<?xi8, strided<[1], offset: ?>>
                    %38 = memref.subview %1[46][%36][1] : memref<126xi8> to memref<?xi8, strided<[1], offset: 46>>
                    memref.copy %37, %38 : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: 46>>

                    // Copy ")\0".
                    %39 = index.constant 46
                    %40 = index.add %39, %36
                    %41 = memref.subview %1[%40][2][1] : memref<126xi8> to memref<2xi8, strided<[1], offset: ?>>
                    %42 = memref.get_global @lit2 : memref<2xi8>
                    memref.copy %42, %41 : memref<2xi8> to memref<2xi8, strided<[1], offset: ?>>

                    // Call `puts()`.
                    %43 = arith.constant 0 : i8
                    %44 = index.constant 125
                    memref.store %43, %1[%44] : memref<126xi8>
                    %45 = memref.extract_aligned_pointer_as_index %1 : memref<126xi8> -> index
                    %46 = index.castu %45 : index to i64
                    %47 = llvm.inttoptr %46 : i64 to !llvm.ptr
                    func.call @puts(%47) : (!llvm.ptr) -> i32
                }

                func.return
            }

            func.func @felt252_bin2dec(%0 : memref<77xi8>, %1 : i256) -> index {
                // Clear the buffer to zero.
                %2 = memref.extract_aligned_pointer_as_index %0 : memref<77xi8> -> index
                %3 = index.castu %2 : index to i64
                %4 = llvm.inttoptr %3 : i64 to !llvm.ptr<i8>
                %5 = arith.constant 0 : i8
                %6 = arith.constant 77 : i32
                %7 = arith.constant 0 : i1
                "llvm.intr.memset"(%4, %5, %6, %7) : (!llvm.ptr<i8>, i8, i32, i1) -> ()

                // Count number of bits required.
                %8 = math.ctlz %1 : i256
                %9 = arith.trunci %8 : i256 to i8
                %10 = arith.subi %5, %9 : i8

                // Handle special case: zero.
                %11 = arith.constant 0 : i8
                %12 = arith.cmpi eq, %10, %11 : i8
                %13 = scf.if %12 -> index {
                    // Write a zero at the end and return the index.
                    %14 = arith.constant 48 : i8
                    %15 = index.constant 75
                    memref.store %14, %0[%15] : memref<77xi8>
                    scf.yield %15 : index
                } else {
                    // For each (required) bit in MSB to LSB order:
                    %16 = index.constant 0
                    %17 = index.castu %10 : i8 to index
                    %18 = index.constant 1
                    scf.for %19 = %16 to %17 step %18 {
                        %20 = index.sub %17, %18
                        %21 = index.sub %20, %19
                        %22 = index.castu %21 : index to i256
                        %23 = arith.shrui %1, %22 : i256
                        %24 = arith.trunci %23 : i256 to i1

                        // Shift & add.
                        %25 = index.constant 76
                        scf.for %26 = %16 to %25 step %18 iter_args(%27 = %24) -> i1 {
                            // Load byte.
                            %28 = index.constant 75
                            %29 = index.sub %28, %26
                            %30 = memref.load %0[%29] : memref<77xi8>

                            // Add 3 if value >= 5.
                            %31 = arith.constant 5 : i8
                            %32 = arith.cmpi uge, %30, %31 : i8
                            %33 = arith.constant 3 : i8
                            %34 = arith.addi %30, %33 : i8
                            %35 = arith.select %32, %34, %30 : i8

                            // Shift 1 bit to the left.
                            %36 = arith.constant 1 : i8
                            %37 = arith.shli %35, %36 : i8

                            // Insert carry-in bit and truncate to 4 bits.
                            %38 = llvm.zext %27 : i1 to i8
                            %39 = arith.ori %37, %38 : i8
                            %40 = arith.constant 15 : i8
                            %41 = arith.andi %39, %40 : i8

                            // Store byte.
                            memref.store %41, %0[%29] : memref<77xi8>

                            // Extract carry and send it to the next iteration.
                            %42 = arith.shrui %35, %33 : i8
                            %43 = arith.trunci %42 : i8 to i1
                            scf.yield %43 : i1
                        }
                    }

                    // Find first non-zero digit index.
                    %44 = scf.while (%45 = %16) : (index) -> (index) {
                        %46 = memref.load %0[%45] : memref<77xi8>
                        %47 = arith.cmpi eq, %46, %5 : i8
                        scf.condition(%47) %45 : index
                    } do {
                    ^0(%48 : index):
                        %49 = index.add %48, %18
                        scf.yield %49 : index
                    }

                    // Convert BCD to ascii digits.
                    %50 = index.constant 76
                    scf.for %51 = %44 to %50 step %18 {
                        %52 = memref.load %0[%51] : memref<77xi8>
                        %53 = arith.constant 48 : i8
                        %54 = arith.addi %52, %53 : i8
                        memref.store %54, %0[%51] : memref<77xi8>
                    }

                    scf.yield %44 : index
                }

                // Return the first digit offset.
                return %13 : index
            }

            func.func private @puts(!llvm.ptr) -> i32

            memref.global "private" constant @lit0 : memref<8xi8> = dense<[91, 68, 69, 66, 85, 71, 93, 32]>
            memref.global "private" constant @lit1 : memref<7xi8> = dense<[32, 40, 114, 97, 119, 58, 32]>
            memref.global "private" constant @lit2 : memref<2xi8> = dense<[41, 0]>
        };

        let id = func_decl.id.debug_name.as_deref().unwrap();
        storage.libfuncs.insert(
            id.to_string(),
            SierraLibFunc::create_function_all_args(
                vec![SierraType::Array {
                    ty: self.struct_type(&[self.u32_type(), self.u32_type(), self.llvm_ptr_type()]),
                    len_type: self.u32_type(),
                    element_type: Box::new(SierraType::Simple(self.felt_type())),
                }],
                vec![],
            ),
        );

        Ok(())
    }
}
