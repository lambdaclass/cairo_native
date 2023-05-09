use std::cmp::Ordering;

use cairo_lang_sierra::program::{GenericArg, LibfuncDeclaration};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{operation, Block, Location, Type, Value, ValueLike};
use num_bigint::BigInt;
use num_traits::Signed;
use tracing::debug;

use crate::compiler::fn_attributes::FnAttributes;
use crate::compiler::mlir_ops::CmpOp;
use crate::utility::get_type_id;
use crate::{
    compiler::{Compiler, Storage},
    sierra_type::SierraType,
    types::{is_omitted_builtin_type, DEFAULT_PRIME},
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
            let name = func_decl.long_id.generic_id.0.as_str();
            debug!(name, "processing libfunc decl");

            match name {
                // no-ops
                // NOTE jump stops being a nop if return types are stored
                "branch_align"
                | "get_builtin_costs"
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
                    self.create_libfunc_felt_add(func_decl, storage)?;
                }
                "felt252_sub" => {
                    self.create_libfunc_felt_sub(func_decl, storage)?;
                }
                "felt252_mul" => {
                    self.create_libfunc_felt_mul(func_decl, storage)?;
                }
                "felt252_div" => {
                    self.create_libfunc_felt_div(func_decl, storage)?;
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
                    self.create_libfunc_enum_init(func_decl, storage)?;
                }
                "enum_match" => {
                    // Note no actual function is created here, however types are registered
                    self.register_libfunc_enum_match(func_decl, storage);
                }
                "struct_construct" => {
                    self.create_libfunc_struct_construct(func_decl, storage)?;
                }
                "struct_deconstruct" => {
                    self.create_libfunc_struct_deconstruct(func_decl, storage)?;
                }
                "null" => {
                    self.create_libfunc_null(func_decl, storage)?;
                }
                "nullable_from_box" => {
                    self.create_libfunc_nullable_from_box(func_decl, storage)?;
                }
                "match_nullable" => {
                    self.register_match_nullable(func_decl, storage);
                }
                "withdraw_gas" | "withdraw_gas_all" => {
                    self.register_withdraw_gas(func_decl, storage);
                }
                "store_temp" | "rename" | "unbox" | "into_box" => {
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
                    self.create_libfunc_uint_to_felt252(func_decl, self.u8_type(), storage)?;
                }
                "u16_to_felt252" => {
                    self.create_libfunc_uint_to_felt252(func_decl, self.u16_type(), storage)?;
                }
                "u32_to_felt252" => {
                    self.create_libfunc_uint_to_felt252(func_decl, self.u32_type(), storage)?;
                }
                "u64_to_felt252" => {
                    self.create_libfunc_uint_to_felt252(func_decl, self.u64_type(), storage)?;
                }
                "u128_to_felt252" => {
                    self.create_libfunc_uint_to_felt252(func_decl, self.u128_type(), storage)?;
                }
                "u8_wide_mul" => {
                    self.create_libfunc_uint_wide_mul(
                        func_decl,
                        self.u8_type(),
                        self.u16_type(),
                        storage,
                    )?;
                }
                "u16_wide_mul" => {
                    self.create_libfunc_uint_wide_mul(
                        func_decl,
                        self.u16_type(),
                        self.u32_type(),
                        storage,
                    )?;
                }
                "u32_wide_mul" => {
                    self.create_libfunc_uint_wide_mul(
                        func_decl,
                        self.u32_type(),
                        self.u64_type(),
                        storage,
                    )?;
                }
                "u64_wide_mul" => {
                    self.create_libfunc_uint_wide_mul(
                        func_decl,
                        self.u64_type(),
                        self.u128_type(),
                        storage,
                    )?;
                }
                "u128_wide_mul" => {
                    self.create_libfunc_u128_wide_mul(func_decl, storage)?;
                }
                "u8_safe_divmod" => {
                    self.create_libfunc_uint_safe_divmod(func_decl, self.u8_type(), storage)?;
                }
                "u16_safe_divmod" => {
                    self.create_libfunc_uint_safe_divmod(func_decl, self.u16_type(), storage)?;
                }
                "u32_safe_divmod" => {
                    self.create_libfunc_uint_safe_divmod(func_decl, self.u32_type(), storage)?;
                }
                "u64_safe_divmod" => {
                    self.create_libfunc_uint_safe_divmod(func_decl, self.u64_type(), storage)?;
                }
                "u128_safe_divmod" => {
                    self.create_libfunc_uint_safe_divmod(func_decl, self.u128_type(), storage)?;
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
                "u8_try_from_felt252" => {
                    self.register_libfunc_uint_try_from_felt252(func_decl, self.u8_type(), storage);
                }
                "u16_try_from_felt252" => {
                    self.register_libfunc_uint_try_from_felt252(
                        func_decl,
                        self.u16_type(),
                        storage,
                    );
                }
                "u32_try_from_felt252" => {
                    self.register_libfunc_uint_try_from_felt252(
                        func_decl,
                        self.u32_type(),
                        storage,
                    );
                }
                "u64_try_from_felt252" => {
                    self.register_libfunc_uint_try_from_felt252(
                        func_decl,
                        self.u64_type(),
                        storage,
                    );
                }
                "u128_try_from_felt252" => {
                    self.register_libfunc_uint_try_from_felt252(
                        func_decl,
                        self.u128_type(),
                        storage,
                    );
                }
                "bitwise" => {
                    self.create_libfunc_bitwise(func_decl, storage)?;
                }
                "upcast" => {
                    self.create_libfunc_upcast(func_decl, storage)?;
                }
                "downcast" => {
                    self.register_libfunc_downcast(func_decl, storage);
                }
                "bool_or_impl" => {
                    self.create_libfunc_bool_binop_impl(func_decl, storage, BoolBinaryOp::Or)?;
                }
                "bool_and_impl" => {
                    self.create_libfunc_bool_binop_impl(func_decl, storage, BoolBinaryOp::And)?;
                }
                "bool_xor_impl" => {
                    self.create_libfunc_bool_binop_impl(func_decl, storage, BoolBinaryOp::Xor)?;
                }
                "bool_not_impl" => {
                    self.create_libfunc_bool_not_impl(func_decl, storage)?;
                }
                "bool_to_felt252" => {
                    self.create_libfunc_bool_to_felt252(func_decl, storage)?;
                }
                "array_new" => {
                    self.create_libfunc_array_new(func_decl, storage)?;
                }
                "array_append" => {
                    self.create_libfunc_array_append(func_decl, storage)?;
                }
                "array_len" => {
                    self.create_libfunc_array_len(func_decl, storage)?;
                }
                "array_get" => {
                    self.register_libfunc_array_get(func_decl, storage);
                }
                "array_pop_front" => {
                    self.register_libfunc_array_pop_front(func_decl, storage);
                }
                "print" => {
                    self.create_libfunc_print(func_decl, storage)?;
                }
                "pedersen" => {
                    self.create_libfunc_pedersen(func_decl, storage)?;
                }
                "get_available_gas" => {
                    self.create_libfunc_get_available_gas(func_decl, storage)?;
                }
                "hades_permutation" => {
                    self.create_libfunc_hades_permutation(func_decl, storage)?;
                }
                "unwrap_non_zero" => {
                    self.create_libfunc_unwrap_non_zero(func_decl, storage)?;
                }
                _ => todo!(
                    "unhandled libfunc: {:?}",
                    func_decl.id.debug_name.as_ref().unwrap().as_str()
                ),
            }
        }
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
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let type_id = get_type_id(&func_decl.long_id.generic_args[0]);
        let arg_type = storage.types.get(&type_id).cloned().expect("type to exist");

        let args =
            arg_type.get_field_types().expect("arg should be a struct type and have field types");
        let block = self.new_block(&args);

        let struct_type = self.llvm_struct_type(&args, false);
        let mut struct_type_op = self.op_llvm_undef(&block, struct_type);

        for i in 0..block.argument_count() {
            let arg = block.argument(i)?;
            let struct_value = struct_type_op.result(0)?.into();
            struct_type_op =
                self.op_llvm_insertvalue(&block, i, struct_value, arg.into(), arg_type.get_type())?;
        }

        let struct_value: Value = struct_type_op.result(0)?.into();
        self.op_return(&block, &[struct_value]);

        self.create_function(
            &id,
            vec![block],
            &[arg_type.get_type()],
            FnAttributes::libfunc(false, true),
        )?;

        storage.libfuncs.insert(
            id,
            SierraLibFunc::create_function_all_args(
                arg_type.get_field_sierra_types().unwrap().to_vec(),
                vec![arg_type],
            ),
        );

        Ok(())
    }

    /// Extract (destructure) each struct member (in order) into variables.
    pub fn create_libfunc_struct_deconstruct(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let struct_type = storage
            .types
            .get(&get_type_id(&func_decl.long_id.generic_args[0]))
            .expect("struct type not found");
        let (struct_ty, field_types) = match struct_type {
            SierraType::Struct { ty, field_types } => (*ty, field_types.as_slice()),
            _ => todo!("handle non-struct types (error)"),
        };

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

        let fn_id = func_decl.id.debug_name.as_deref().unwrap().to_string();

        self.create_function(
            &fn_id,
            vec![block],
            &field_types.iter().map(SierraType::get_type).collect_vec(),
            FnAttributes::libfunc(false, true),
        )?;

        let return_types = field_types.to_vec();
        let struct_type = struct_type.clone();
        storage.libfuncs.insert(
            fn_id,
            SierraLibFunc::create_function_all_args(vec![struct_type], return_types),
        );
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
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let sierra_felt_type = SierraType::Simple(self.felt_type());

        // Block in which the calculation occurs
        let entry_block = self.new_block(&[self.felt_type(), self.felt_type()]);
        // Block for wrapping values >= PRIME
        let gte_prime_block = self.new_block(&[]);
        // Block for returning values < PRIME
        let in_range_block = self.new_block(&[]);

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

        self.create_function(
            &id,
            vec![entry_block, in_range_block, gte_prime_block],
            &[self.felt_type()],
            FnAttributes::libfunc(false, true),
        )?;

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
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let sierra_felt_type = SierraType::Simple(self.felt_type());

        // Block in which the calculation occurs
        let entry_block = self.new_block(&[self.felt_type(), self.felt_type()]);
        // Block for wrapping values < 0
        let lt_zero_block = self.new_block(&[]);
        // Block for returning values >= 0
        let in_range_block = self.new_block(&[]);

        // res = lhs - rhs
        let lhs = entry_block.argument(0)?.into();
        let rhs = entry_block.argument(1)?.into();
        let res_op = self.op_sub(&entry_block, lhs, rhs);
        let res = res_op.result(0)?.into();

        // lt_zero <=> res_result < 0
        let zero_op = self.op_const(&entry_block, "0", self.felt_type());
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

        self.create_function(
            &id,
            vec![entry_block, in_range_block, lt_zero_block],
            &[self.felt_type()],
            FnAttributes::libfunc(false, true),
        )?;

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
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let sierra_felt_type = SierraType::Simple(self.felt_type());

        let block = self.new_block(&[self.felt_type(), self.felt_type()]);
        let lhs = block.argument(0)?.into();
        let rhs = block.argument(1)?.into();

        let res_op = self.call_felt_mul_impl(&block, lhs, rhs, storage)?;
        let res = res_op.result(0)?.into();

        self.op_return(&block, &[res]);

        self.create_function(
            &id,
            vec![block],
            &[self.felt_type()],
            FnAttributes::libfunc(false, true),
        )?;

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
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        // Second argument is NonZero<felt252>
        let block = self.new_block(&[self.felt_type(), self.felt_type()]);

        let lhs = block.argument(0)?.into();
        let rhs = block.argument(1)?.into();

        // In order to calculate lhs / rhs (mod PRIME), first calculate 1/rhs (mod PRIME)
        let rhs_inverse_op = self.call_egcd_felt_inverse(&block, rhs, storage)?;
        let rhs_inverse = rhs_inverse_op.result(0)?.into();

        // Next calculate lhs * (1/rhs) (mod PRIME)
        let res_op = self.call_felt_mul_impl(&block, lhs, rhs_inverse, storage)?;
        let res = res_op.result(0)?.into();

        self.op_return(&block, &[res]);

        let sierra_felt_type = SierraType::Simple(self.felt_type());
        storage.libfuncs.insert(
            id.clone(),
            SierraLibFunc::create_function_all_args(
                vec![sierra_felt_type.clone(), sierra_felt_type.clone()],
                vec![sierra_felt_type],
            ),
        );

        self.create_function(
            &id,
            vec![block],
            &[self.felt_type()],
            FnAttributes::libfunc(false, true),
        )
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

    pub fn register_libfunc_uint_try_from_felt252(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        op_type: Type<'ctx>,
        storage: &mut Storage<'ctx>,
    ) {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        storage.libfuncs.insert(
            id,
            SierraLibFunc::Branching {
                args: vec![PositionalArg { loc: 1, ty: SierraType::Simple(self.felt_type()) }],
                return_types: vec![
                    vec![PositionalArg { loc: 1, ty: SierraType::Simple(op_type) }],
                    vec![],
                ],
            },
        );
    }

    pub fn register_libfunc_downcast(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) {
        let libfunc_id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        let from_type = &func_decl.long_id.generic_args[0];
        let to_type = &func_decl.long_id.generic_args[1];

        let from_type = match from_type {
            GenericArg::Type(id) => {
                storage.types.get(&id.id.to_string()).cloned().expect("type should exist")
            }
            _ => unreachable!(),
        };
        let to_type = match to_type {
            GenericArg::Type(id) => {
                storage.types.get(&id.id.to_string()).cloned().expect("type should exist")
            }
            _ => unreachable!(),
        };

        storage.libfuncs.insert(
            libfunc_id,
            SierraLibFunc::Branching {
                args: vec![PositionalArg { loc: 1, ty: from_type }],
                return_types: vec![vec![PositionalArg { loc: 1, ty: to_type }], vec![]],
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

        let sierra_type = SierraType::create_array_type(self, arg_type.clone());

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

        let sierra_type = SierraType::create_array_type(self, arg_type.clone());

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
        src_type: Type<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let block = self.new_block(&[src_type]);

        let op_zext = self.op_zext(&block, block.argument(0)?.into(), self.felt_type());
        self.op_return(&block, &[op_zext.result(0)?.into()]);

        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        storage.libfuncs.insert(
            id.clone(),
            SierraLibFunc::create_function_all_args(
                vec![SierraType::Simple(src_type)],
                vec![SierraType::Simple(self.felt_type())],
            ),
        );

        self.create_function(
            &id,
            vec![block],
            &[self.felt_type()],
            FnAttributes::libfunc(false, true),
        )
    }

    pub fn create_libfunc_uint_wide_mul(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        src_type: Type<'ctx>,
        dst_type: Type<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let block = self.new_block(&[src_type, src_type]);

        let op_zext_lhs = self.op_zext(&block, block.argument(0)?.into(), dst_type);
        let op_zext_rhs = self.op_zext(&block, block.argument(1)?.into(), dst_type);

        let op_mul =
            self.op_mul(&block, op_zext_lhs.result(0)?.into(), op_zext_rhs.result(0)?.into());
        self.op_return(&block, &[op_mul.result(0)?.into()]);

        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        storage.libfuncs.insert(
            id.clone(),
            SierraLibFunc::create_function_all_args(
                vec![SierraType::Simple(src_type), SierraType::Simple(src_type)],
                vec![SierraType::Simple(dst_type)],
            ),
        );

        self.create_function(&id, vec![block], &[dst_type], FnAttributes::libfunc(false, true))
    }

    pub fn create_libfunc_u128_wide_mul(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let block = self.new_block(&[self.u128_type(), self.u128_type()]);

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

        storage.libfuncs.insert(
            id.clone(),
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

        self.create_function(
            &id,
            vec![block],
            &[self.u128_type(), self.u128_type()],
            FnAttributes::libfunc(false, true),
        )
    }

    pub fn create_libfunc_uint_safe_divmod(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        src_type: Type<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let block = self.new_block(&[src_type, src_type]);

        let lhs = block.argument(0)?.into();
        let rhs = block.argument(1)?.into();

        let op_div = self.op_div(&block, lhs, rhs);
        let op_rem = self.op_rem(&block, lhs, rhs);

        self.op_return(&block, &[op_div.result(0)?.into(), op_rem.result(0)?.into()]);

        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        storage.libfuncs.insert(
            id.clone(),
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

        self.create_function(
            &id,
            vec![block],
            &[src_type, src_type],
            FnAttributes::libfunc(false, true),
        )
    }

    pub fn create_libfunc_bitwise(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let data_in = &[self.u128_type(), self.u128_type()];
        let data_out = &[self.u128_type(), self.u128_type(), self.u128_type()];

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

        let fn_id = func_decl.id.debug_name.as_deref().unwrap().to_string();

        storage.libfuncs.insert(
            fn_id.clone(),
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

        self.create_function(&fn_id, vec![block], data_out, FnAttributes::libfunc(false, true))
    }

    pub fn create_libfunc_upcast(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        let src_type = storage
            .types
            .get(&get_type_id(&func_decl.long_id.generic_args[0]))
            .expect("type to exist")
            .clone();
        let dst_type = storage
            .types
            .get(&get_type_id(&func_decl.long_id.generic_args[1]))
            .expect("type to exist")
            .clone();

        match src_type.get_width().cmp(&dst_type.get_width()) {
            Ordering::Less => {
                let block = self.new_block(&[src_type.get_type()]);

                let op_ref = self.op_zext(&block, block.argument(0)?.into(), dst_type.get_type());

                self.op_return(&block, &[op_ref.result(0)?.into()]);

                storage.libfuncs.insert(
                    id.clone(),
                    SierraLibFunc::create_function_all_args(
                        vec![src_type.clone()],
                        vec![dst_type.clone()],
                    ),
                );

                self.create_function(
                    &id,
                    vec![block],
                    &[dst_type.get_type()],
                    FnAttributes::libfunc(false, true),
                )
            }
            Ordering::Equal => {
                // Similar to store_local and rename, create a libfuncdef that tells statement processing to just forward its argument
                self.register_identity_function(func_decl, storage)
            }
            Ordering::Greater => todo!("invalid generics for libfunc `upcast`"),
        }
    }

    /// 2 boolean enums -> 1 bool enum
    pub fn create_libfunc_bool_binop_impl(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
        bool_op: BoolBinaryOp,
    ) -> Result<()> {
        let data_in = &[self.boolean_enum_type(), self.boolean_enum_type()];
        let data_out = &[self.boolean_enum_type()];

        let bool_variant = SierraType::Struct {
            ty: self.llvm_struct_type(&[Type::none(&self.context)], false),
            field_types: vec![],
        };

        let bool_sierra_type = SierraType::Enum {
            ty: self.boolean_enum_type(),
            tag_type: self.u16_type(),
            storage_bytes_len: 0,
            storage_type: self.llvm_array_type(self.u8_type(), 0),
            variants_types: vec![bool_variant.clone(), bool_variant],
        };

        let block = self.new_block(data_in);

        let lhs = block.argument(0)?;
        let rhs = block.argument(1)?;

        let lhs_tag_value_op = self.op_llvm_extractvalue(&block, 0, lhs.into(), self.u16_type())?;
        let lhs_tag_value: Value = lhs_tag_value_op.result(0)?.into();

        let rhs_tag_value_op = self.op_llvm_extractvalue(&block, 0, rhs.into(), self.u16_type())?;
        let rhs_tag_value: Value = rhs_tag_value_op.result(0)?.into();

        let bool_op_ref = match bool_op {
            BoolBinaryOp::And => self.op_and(&block, lhs_tag_value, rhs_tag_value, self.u16_type()),
            BoolBinaryOp::Xor => self.op_xor(&block, lhs_tag_value, rhs_tag_value, self.u16_type()),
            BoolBinaryOp::Or => self.op_or(&block, lhs_tag_value, rhs_tag_value, self.u16_type()),
        };

        let enum_op = self.op_llvm_undef(&block, self.boolean_enum_type());
        let enum_value: Value = enum_op.result(0)?.into();

        let enum_res = self.op_llvm_insertvalue(
            &block,
            0,
            enum_value,
            bool_op_ref.result(0)?.into(),
            self.boolean_enum_type(),
        )?;

        self.op_return(&block, &[enum_res.result(0)?.into()]);

        let fn_id = func_decl.id.debug_name.as_deref().unwrap().to_string();

        storage.libfuncs.insert(
            fn_id.clone(),
            SierraLibFunc::create_function_all_args(
                vec![bool_sierra_type.clone(), bool_sierra_type.clone()],
                vec![bool_sierra_type.clone()],
            ),
        );

        self.create_function(&fn_id, vec![block], data_out, FnAttributes::libfunc(false, true))
    }

    pub fn create_libfunc_bool_not_impl(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let data_in = &[self.boolean_enum_type()];
        let data_out = &[self.boolean_enum_type()];

        let bool_variant = SierraType::Struct {
            ty: self.llvm_struct_type(&[Type::none(&self.context)], false),
            field_types: vec![],
        };

        let bool_sierra_type = SierraType::Enum {
            ty: self.boolean_enum_type(),
            tag_type: self.u16_type(),
            storage_bytes_len: 0,
            storage_type: self.llvm_array_type(self.u8_type(), 0),
            variants_types: vec![bool_variant.clone()],
        };

        let block = self.new_block(data_in);

        let lhs = block.argument(0)?;

        let lhs_tag_value_op = self.op_llvm_extractvalue(&block, 0, lhs.into(), self.u16_type())?;
        let lhs_tag_value: Value = lhs_tag_value_op.result(0)?.into();

        let const_1_op = self.op_const(&block, "1", self.u16_type());

        let bool_op_ref =
            self.op_xor(&block, lhs_tag_value, const_1_op.result(0)?.into(), self.u16_type());

        let enum_op = self.op_llvm_undef(&block, self.boolean_enum_type());
        let enum_value: Value = enum_op.result(0)?.into();

        let enum_res = self.op_llvm_insertvalue(
            &block,
            0,
            enum_value,
            bool_op_ref.result(0)?.into(),
            self.boolean_enum_type(),
        )?;

        self.op_return(&block, &[enum_res.result(0)?.into()]);

        let fn_id = func_decl.id.debug_name.as_deref().unwrap().to_string();

        storage.libfuncs.insert(
            fn_id.clone(),
            SierraLibFunc::create_function_all_args(
                vec![bool_sierra_type.clone()],
                vec![bool_sierra_type],
            ),
        );

        self.create_function(&fn_id, vec![block], data_out, FnAttributes::libfunc(false, true))
    }

    /// bool to felt
    ///
    /// Sierra:
    /// `extern fn bool_to_felt252(a: bool) -> felt252 implicits() nopanic;`
    pub fn create_libfunc_bool_to_felt252(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let data_in = &[self.boolean_enum_type()];
        let data_out = &[self.felt_type()];

        let bool_variant = SierraType::Struct {
            ty: self.llvm_struct_type(&[Type::none(&self.context)], false),
            field_types: vec![],
        };

        let bool_sierra_type = SierraType::Enum {
            ty: self.boolean_enum_type(),
            tag_type: self.u16_type(),
            storage_bytes_len: 0,
            storage_type: self.llvm_array_type(self.u8_type(), 0),
            variants_types: vec![bool_variant.clone()],
        };

        let block = self.new_block(data_in);

        let bool_enum = block.argument(0)?;

        let tag_value_op =
            self.op_llvm_extractvalue(&block, 0, bool_enum.into(), self.u16_type())?;
        let tag_value: Value = tag_value_op.result(0)?.into();

        let felt_value = self.op_zext(&block, tag_value, self.felt_type());

        self.op_return(&block, &[felt_value.result(0)?.into()]);

        let fn_id = func_decl.id.debug_name.as_deref().unwrap().to_string();

        storage.libfuncs.insert(
            fn_id.clone(),
            SierraLibFunc::create_function_all_args(
                vec![bool_sierra_type],
                vec![SierraType::Simple(self.felt_type())],
            ),
        );

        self.create_function(&fn_id, vec![block], data_out, FnAttributes::libfunc(false, true))
    }

    // `extern fn array_new<T>() -> Array<T> nopanic;`
    // in sierra: `array_new<felt252>() -> ([0]);`
    pub fn create_libfunc_array_new(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let arg_type = storage
            .types
            .get(&get_type_id(&func_decl.long_id.generic_args[0]))
            .cloned()
            .expect("type to exist");

        let block = self.new_block(&[]);

        let sierra_type = SierraType::create_array_type(self, arg_type.clone());

        // Initially create an undefined struct to insert the values into
        let array_value_op = self.op_llvm_undef(&block, sierra_type.get_type());
        let array_value: Value = array_value_op.result(0)?.into();

        // Length is 0 on creation
        let array_len_op = self.op_u32_const(&block, "0");
        let array_len = array_len_op.result(0)?.into();

        // Capacity is 8 on creation to avoid lots of early reallocations
        let array_capacity_op = self.op_u32_const(&block, "8");
        let array_capacity = array_capacity_op.result(0)?.into();

        // The size in bytes of one array element
        let array_element_size_bytes = (arg_type.get_width() + 7) / 8;

        // length
        let set_len_op =
            self.call_array_set_len_impl(&block, array_value, array_len, &sierra_type, storage)?;
        let array_value: Value = set_len_op.result(0)?.into();

        // capacity
        let set_capacity_op = self.call_array_set_capacity_impl(
            &block,
            array_value,
            array_capacity,
            &sierra_type,
            storage,
        )?;
        let array_value: Value = set_capacity_op.result(0)?.into();

        // 8 here is the capacity
        let const_arr_size_bytes_op =
            self.op_const(&block, &(array_element_size_bytes * 8).to_string(), self.u64_type());
        let const_arr_size_bytes = const_arr_size_bytes_op.result(0)?;

        let null_ptr_op = self.op_llvm_nullptr(&block);
        let null_ptr = null_ptr_op.result(0)?;

        let ptr_op =
            self.call_realloc(&block, null_ptr.into(), const_arr_size_bytes.into(), storage)?;
        let ptr_val = ptr_op.result(0)?.into();

        let set_data_ptr_op =
            self.call_array_set_data_ptr(&block, array_value, ptr_val, &sierra_type, storage)?;
        let array_value: Value = set_data_ptr_op.result(0)?.into();

        self.op_return(&block, &[array_value]);

        storage.libfuncs.insert(
            id.clone(),
            SierraLibFunc::create_function_all_args(vec![], vec![sierra_type.clone()]),
        );

        self.create_function(
            &id,
            vec![block],
            &[sierra_type.get_type()],
            FnAttributes::libfunc(false, true),
        )
    }

    // `extern fn array_append<T>(ref arr: Array<T>, value: T) nopanic;`
    // in sierra `array_append<T>([0], [1]) -> ([2]);`
    pub fn create_libfunc_array_append(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let arg_type = storage
            .types
            .get(&get_type_id(&func_decl.long_id.generic_args[0]))
            .cloned()
            .expect("type to exist");

        let array_type = SierraType::create_array_type(self, arg_type.clone());

        // ---entry block---
        let entry_block = self.new_block(&[array_type.get_type(), arg_type.get_type()]);
        let array_value = entry_block.argument(0)?.into();
        let append_value = entry_block.argument(1)?.into();

        // check if len < capacity
        let array_len_op =
            self.call_array_len_impl(&entry_block, array_value, &array_type, storage)?;
        let array_len = array_len_op.result(0)?.into();

        let array_capacity_op =
            self.call_array_capacity_impl(&entry_block, array_value, &array_type, storage)?;
        let array_capacity = array_capacity_op.result(0)?.into();

        let realloc_block = self.new_block(&[]);
        let append_value_block = self.new_block(&[array_type.get_type()]);

        let is_less_op =
            self.op_cmp(&entry_block, CmpOp::UnsignedLessThan, array_len, array_capacity);
        let is_less = is_less_op.result(0)?.into();

        self.op_cond_br(
            &entry_block,
            is_less,
            &append_value_block,
            &realloc_block,
            &[array_value],
            &[],
        );

        // ---reallocation block---
        // reallocate with more capacity, for now with a simple algorithm:
        // new_capacity = capacity * 2
        let const_2_op = self.op_u32_const(&realloc_block, "2");
        let const_2 = const_2_op.result(0)?;

        let new_capacity_op = self.op_mul(&realloc_block, array_capacity, const_2.into());
        let new_capacity = new_capacity_op.result(0)?.into();

        let new_capacity_as_u64_op = self.op_zext(&realloc_block, new_capacity, self.u64_type());
        let new_capacity_as_u64 = new_capacity_as_u64_op.result(0)?.into();
        let element_byte_width = (arg_type.get_width() + 7) / 8;
        let element_size_op = self.op_u64_const(&realloc_block, &element_byte_width.to_string());
        let element_size = element_size_op.result(0)?.into();
        let new_capacity_in_bytes_op =
            self.op_mul(&realloc_block, new_capacity_as_u64, element_size);
        let new_capacity_in_bytes = new_capacity_in_bytes_op.result(0)?.into();

        let data_ptr_op =
            self.call_array_get_data_ptr(&realloc_block, array_value, &array_type, storage)?;
        let data_ptr: Value = data_ptr_op.result(0)?.into();

        let new_ptr_op =
            self.call_realloc(&realloc_block, data_ptr, new_capacity_in_bytes, storage)?;
        let new_ptr = new_ptr_op.result(0)?.into();

        // change the ptr
        let insert_ptr_op = self.call_array_set_data_ptr(
            &realloc_block,
            array_value,
            new_ptr,
            &array_type,
            storage,
        )?;
        let array_value = insert_ptr_op.result(0)?.into();

        let update_capacity_op = self.call_array_set_capacity_impl(
            &realloc_block,
            array_value,
            new_capacity,
            &array_type,
            storage,
        )?;
        let array_value = update_capacity_op.result(0)?;

        self.op_br(&realloc_block, &append_value_block, &[array_value.into()]);

        // ---append value block---
        // append value and len + 1
        let array_value = append_value_block.argument(0)?.into();

        // update the value
        self.call_array_set_unchecked(
            &append_value_block,
            array_value,
            array_len,
            append_value,
            &array_type,
            storage,
        )?;

        // increment the length
        let const_1 = self.op_const(&append_value_block, "1", array_len.r#type());
        let len_plus_1_op = self.op_add(&append_value_block, array_len, const_1.result(0)?.into());
        let len_plus_1 = len_plus_1_op.result(0)?.into();

        let set_len_op = self.call_array_set_len_impl(
            &append_value_block,
            array_value,
            len_plus_1,
            &array_type,
            storage,
        )?;
        let array_value = set_len_op.result(0)?.into();

        self.op_return(&append_value_block, &[array_value]);

        storage.libfuncs.insert(
            id.clone(),
            SierraLibFunc::create_function_all_args(
                vec![array_type.clone(), arg_type.clone()],
                vec![array_type.clone()],
            ),
        );

        self.create_function(
            &id,
            vec![entry_block, realloc_block, append_value_block],
            &[array_type.get_type()],
            FnAttributes::libfunc(false, true),
        )
    }

    pub fn create_libfunc_array_len(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        let arg_type = storage
            .types
            .get(&get_type_id(&func_decl.long_id.generic_args[0]))
            .cloned()
            .expect("type to exist");

        let sierra_type = SierraType::create_array_type(self, arg_type.clone());

        let block = self.new_block(&[sierra_type.get_type()]);

        let arg = block.argument(0)?.into();
        let impl_call = self.call_array_len_impl(&block, arg, &sierra_type, storage)?;
        let impl_result = impl_call.result(0)?.into();

        self.op_return(&block, &[impl_result]);

        storage.libfuncs.insert(
            id.clone(),
            SierraLibFunc::create_function_all_args(
                vec![sierra_type.clone()],
                vec![SierraType::Simple(self.u32_type())],
            ),
        );

        // TODO replace self.u32_type() once a helper to get array properties from a SierraType is implemented
        self.create_function(
            &id,
            vec![block],
            &[self.u32_type()],
            FnAttributes::libfunc(false, true),
        )
    }

    pub fn create_libfunc_print(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let block = self.module.body();
        mlir_asm! { block, opt(
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
            func.func @print(%0 : !llvm.struct<packed (i32, i32, !llvm.ptr)>) -> () {
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
                %7 = llvm.extractvalue %0[0] : !llvm.struct<packed (i32, i32, !llvm.ptr)>
                %8 = index.castu %7 : i32 to index
                %9 = index.constant 1
                scf.for %10 = %6 to %8 step %9 {
                    // Load element to print.
                    %11 = llvm.extractvalue %0[2] : !llvm.struct<packed (i32, i32, !llvm.ptr)>

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
                    ty: self.llvm_struct_type(
                        &[self.u32_type(), self.u32_type(), self.llvm_ptr_type()],
                        false,
                    ),
                    len_type: self.u32_type(),
                    element_type: Box::new(SierraType::Simple(self.felt_type())),
                }],
                vec![],
            ),
        );

        Ok(())
    }

    pub fn create_libfunc_pedersen(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let parent_block = self.module.body();
        mlir_asm! { parent_block =>
            func.func @pedersen(%0 : i256, %1 : i256) -> i256 {
                // Allocate temporary buffers.
                %2 = memref.alloca() : memref<32xi8>
                %3 = memref.alloca() : memref<32xi8>
                %4 = memref.alloca() : memref<32xi8>

                // Swap endianness (LE -> BE).
                // TODO: Find a way to check the target's endianness.
                %5 = llvm.call_intrinsic "llvm.bswap.i256"(%0) : (i256) -> i256
                %6 = llvm.call_intrinsic "llvm.bswap.i256"(%1) : (i256) -> i256

                // Store lhs and rhs into the temporary buffers.
                %7 = index.constant 0
                %8 = memref.view %3[%7][] : memref<32xi8> to memref<i256>
                %9 = memref.view %4[%7][] : memref<32xi8> to memref<i256>
                memref.store %5, %8[] : memref<i256>
                memref.store %6, %9[] : memref<i256>

                // Call the auxiliary library's pedersen function.
                %10 = memref.extract_aligned_pointer_as_index %2 : memref<32xi8> -> index
                %11 = memref.extract_aligned_pointer_as_index %3 : memref<32xi8> -> index
                %12 = memref.extract_aligned_pointer_as_index %4 : memref<32xi8> -> index
                %13 = index.castu %10 : index to i64
                %14 = index.castu %11 : index to i64
                %15 = index.castu %12 : index to i64
                %16 = llvm.inttoptr %13 : i64 to !llvm.ptr
                %17 = llvm.inttoptr %14 : i64 to !llvm.ptr
                %18 = llvm.inttoptr %15 : i64 to !llvm.ptr
                func.call @sierra2mlir_util_pedersen(%16, %17, %18) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()

                // Load dst from the temporary buffer.
                %19 = memref.view %2[%7][] : memref<32xi8> to memref<i256>
                %20 = memref.load %19[] : memref<i256>

                // Swap endianness (BE -> LE).
                // TODO: Find a way to check the target's endianness.
                %21 = llvm.call_intrinsic "llvm.bswap.i256"(%20) : (i256) -> i256

                return %21 : i256
            }

            func.func private @sierra2mlir_util_pedersen(!llvm.ptr, !llvm.ptr, !llvm.ptr)
        }

        let id = func_decl.id.debug_name.as_deref().unwrap();
        storage.libfuncs.insert(
            id.to_string(),
            SierraLibFunc::Function {
                args: vec![
                    PositionalArg { loc: 1, ty: SierraType::Simple(self.felt_type()) },
                    PositionalArg { loc: 2, ty: SierraType::Simple(self.felt_type()) },
                ],
                return_types: vec![PositionalArg {
                    loc: 1,
                    ty: SierraType::Simple(self.felt_type()),
                }],
            },
        );

        Ok(())
    }

    pub fn create_libfunc_hades_permutation(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let parent_block = self.module.body();
        mlir_asm! { parent_block =>
            func.func @hades_permutation(%0 : i256, %1 : i256, %2 : i256) -> (i256, i256, i256) {
                // Allocate temporary buffers.
                %3 = memref.alloca() : memref<32xi8>
                %4 = memref.alloca() : memref<32xi8>
                %5 = memref.alloca() : memref<32xi8>

                // Swap endianness (LE -> BE).
                // TODO: Find a way to check the target's endianness.
                %6 = llvm.call_intrinsic "llvm.bswap.i256"(%0) : (i256) -> i256
                %7 = llvm.call_intrinsic "llvm.bswap.i256"(%1) : (i256) -> i256
                %8 = llvm.call_intrinsic "llvm.bswap.i256"(%2) : (i256) -> i256

                // Store the operands into the temporary buffers.
                %9 = index.constant 0
                %10 = memref.view %3[%9][] : memref<32xi8> to memref<i256>
                %11 = memref.view %4[%9][] : memref<32xi8> to memref<i256>
                %12 = memref.view %5[%9][] : memref<32xi8> to memref<i256>
                memref.store %6, %10[] : memref<i256>
                memref.store %7, %11[] : memref<i256>
                memref.store %8, %12[] : memref<i256>

                // Call the auxiliary library's pedersen function.
                %13 = memref.extract_aligned_pointer_as_index %3 : memref<32xi8> -> index
                %14 = memref.extract_aligned_pointer_as_index %4 : memref<32xi8> -> index
                %15 = memref.extract_aligned_pointer_as_index %5 : memref<32xi8> -> index
                %16 = index.castu %13 : index to i64
                %17 = index.castu %14 : index to i64
                %18 = index.castu %15 : index to i64
                %19 = llvm.inttoptr %16 : i64 to !llvm.ptr
                %20 = llvm.inttoptr %17 : i64 to !llvm.ptr
                %21 = llvm.inttoptr %18 : i64 to !llvm.ptr
                func.call @sierra2mlir_util_hades_permutation(%19, %20, %21) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()

                // Load the results from the temporary buffer.
                %22 = memref.load %10[] : memref<i256>
                %23 = memref.load %11[] : memref<i256>
                %24 = memref.load %12[] : memref<i256>

                // Swap endianness (BE -> LE).
                // TODO: Find a way to check the target's endianness.
                %25 = llvm.call_intrinsic "llvm.bswap.i256"(%22) : (i256) -> i256
                %26 = llvm.call_intrinsic "llvm.bswap.i256"(%23) : (i256) -> i256
                %27 = llvm.call_intrinsic "llvm.bswap.i256"(%24) : (i256) -> i256

                return %25, %26, %27 : i256, i256, i256
            }

            func.func private @sierra2mlir_util_hades_permutation(!llvm.ptr, !llvm.ptr, !llvm.ptr)
        }

        let id = func_decl.id.debug_name.as_deref().unwrap();
        storage.libfuncs.insert(
            id.to_string(),
            SierraLibFunc::Function {
                args: vec![
                    PositionalArg { loc: 1, ty: SierraType::Simple(self.felt_type()) },
                    PositionalArg { loc: 2, ty: SierraType::Simple(self.felt_type()) },
                    PositionalArg { loc: 3, ty: SierraType::Simple(self.felt_type()) },
                ],
                return_types: vec![
                    PositionalArg { loc: 1, ty: SierraType::Simple(self.felt_type()) },
                    PositionalArg { loc: 2, ty: SierraType::Simple(self.felt_type()) },
                    PositionalArg { loc: 3, ty: SierraType::Simple(self.felt_type()) },
                ],
            },
        );

        Ok(())
    }

    pub fn create_libfunc_null(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let arg_type = storage
            .types
            .get(&get_type_id(&func_decl.long_id.generic_args[0]))
            .cloned()
            .expect("type to exist");

        let block = self.new_block(&[]);

        let nullable_sierra_type = SierraType::create_nullable_type(self, arg_type.clone());
        let struct_type_op = self.op_llvm_undef(&block, nullable_sierra_type.get_type());

        let const_0 = self.op_const(&block, "0", self.bool_type());
        let struct_value = struct_type_op.result(0)?.into();
        let struct_type_op = self.op_llvm_insertvalue(
            &block,
            1,
            struct_value,
            const_0.result(0)?.into(),
            nullable_sierra_type.get_type(),
        )?;

        let struct_value = struct_type_op.result(0)?.into();
        self.op_return(&block, &[struct_value]);

        storage.libfuncs.insert(
            id.clone(),
            SierraLibFunc::create_function_all_args(vec![], vec![nullable_sierra_type.clone()]),
        );

        self.create_function(
            &id,
            vec![block],
            &[nullable_sierra_type.get_type()],
            FnAttributes::libfunc(false, true),
        )
    }

    pub fn create_libfunc_nullable_from_box(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();
        let arg_type = storage
            .types
            .get(&get_type_id(&func_decl.long_id.generic_args[0]))
            .cloned()
            .expect("type to exist");

        let block = self.new_block(&[arg_type.get_type()]);

        let nullable_sierra_type = SierraType::create_nullable_type(self, arg_type.clone());
        let struct_type_op = self.op_llvm_undef(&block, nullable_sierra_type.get_type());

        let const_1 = self.op_const(&block, "1", self.bool_type());
        let struct_value = struct_type_op.result(0)?.into();
        let struct_type_op = self.op_llvm_insertvalue(
            &block,
            1,
            struct_value,
            const_1.result(0)?.into(),
            nullable_sierra_type.get_type(),
        )?;
        let struct_value = struct_type_op.result(0)?.into();

        let struct_type_op = self.op_llvm_insertvalue(
            &block,
            0,
            struct_value,
            block.argument(0)?.into(),
            nullable_sierra_type.get_type(),
        )?;
        let struct_value = struct_type_op.result(0)?.into();

        self.op_return(&block, &[struct_value]);

        storage.libfuncs.insert(
            id.clone(),
            SierraLibFunc::create_function_all_args(
                vec![arg_type],
                vec![nullable_sierra_type.clone()],
            ),
        );

        self.create_function(
            &id,
            vec![block],
            &[nullable_sierra_type.get_type()],
            FnAttributes::libfunc(false, true),
        )
    }

    pub fn register_match_nullable(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        let arg = if let GenericArg::Type(x) = &func_decl.long_id.generic_args[0] {
            x
        } else {
            unreachable!("match_nullable argument should be a type")
        };

        let arg_type = storage.types.get(&arg.id.to_string()).cloned().expect("type should exist");
        let nullable_type = SierraType::create_nullable_type(self, arg_type.clone());

        storage.libfuncs.insert(
            id,
            SierraLibFunc::Branching {
                args: vec![
                    PositionalArg { loc: 0, ty: nullable_type }, // array
                ],
                return_types: vec![
                    // fallthrough: null, nothing
                    vec![],
                    // jump: value
                    vec![PositionalArg { loc: 0, ty: arg_type }],
                ],
            },
        );
    }

    pub fn register_withdraw_gas(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        storage.libfuncs.insert(
            id,
            SierraLibFunc::Branching {
                args: vec![],
                return_types: vec![
                    // fallthrough: success
                    vec![],
                    // jump: failure
                    vec![],
                ],
            },
        );
    }

    pub fn create_libfunc_get_available_gas(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        let block = Block::new(&[]);

        let (_, gas_value_op) = self.call_get_gas_counter(&block)?;

        self.op_return(&block, &[gas_value_op.result(0)?.into()]);

        storage.libfuncs.insert(
            id.clone(),
            SierraLibFunc::Function {
                args: vec![],
                return_types: vec![PositionalArg {
                    loc: 1,
                    ty: SierraType::Simple(self.u128_type()),
                }],
            },
        );

        self.create_function(
            &id,
            vec![block],
            &[self.u128_type()],
            FnAttributes::libfunc(false, true),
        )?;

        Ok(())
    }

    pub fn create_libfunc_unwrap_non_zero(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let nz_ty = &storage.types[&match &func_decl.long_id.generic_args[0] {
            GenericArg::Type(x) => x.id.to_string(),
            _ => unreachable!(),
        }];

        let ty = match nz_ty {
            SierraType::Simple(x) => x,
            _ => unreachable!(),
        };

        self.create_function(
            func_decl.id.debug_name.as_deref().unwrap(),
            vec![{
                let block = Block::new(&[(nz_ty.get_type(), Location::unknown(&self.context))]);

                block.append_operation(
                    operation::Builder::new("func.return", Location::unknown(&self.context))
                        .add_operands(&[block.argument(0)?.into()])
                        .build(),
                );

                block
            }],
            &[*ty],
            FnAttributes::libfunc(false, true),
        )?;

        let id = func_decl.id.debug_name.as_deref().unwrap();
        storage.libfuncs.insert(
            id.to_string(),
            SierraLibFunc::Function {
                args: vec![PositionalArg { loc: 0, ty: nz_ty.clone() }],
                return_types: vec![PositionalArg { loc: 0, ty: SierraType::Simple(*ty) }],
            },
        );

        Ok(())
    }
}
