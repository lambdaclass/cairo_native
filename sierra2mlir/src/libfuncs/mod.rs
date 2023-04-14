use std::cmp::Ordering;

use cairo_lang_sierra::program::{GenericArg, LibfuncDeclaration};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{
    operation, Block, BlockRef, Location, NamedAttribute, Region, Type, TypeLike, Value, ValueLike,
};
use num_bigint::BigInt;
use num_traits::Signed;
use tracing::debug;

use crate::{
    compiler::{CmpOp, Compiler, FnAttributes, SierraType, Storage},
    types::{is_omitted_builtin_type, DEFAULT_PRIME},
    utility::create_fn_signature,
};

use self::lib_func_def::{PositionalArg, SierraLibFunc};

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
            GenericArg::UserType(_) => todo!(),
            GenericArg::Type(type_id) => {
                storage.types.get(&type_id.id.to_string()).cloned().expect("type to exist")
            }
            GenericArg::Value(_) => todo!(),
            GenericArg::UserFunc(_) => todo!(),
            GenericArg::Libfunc(_) => todo!(),
        };

        let region = Region::new();

        let block = Block::new(&[]);

        let sierra_type = SierraType::Array {
            ty: self.struct_type(&[self.u32_type(), self.u32_type(), self.llvm_ptr_type()]),
            len_type: self.u32_type(),
            element_type: Box::new(arg_type.clone()),
        };

        let array_value_op = self.op_llvm_struct(&block, sierra_type.get_type());
        let array_value: Value = array_value_op.result(0)?.into();

        let array_len_op = self.op_u32_const(&block, "0");
        let array_len = array_len_op.result(0)?.into();

        let array_capacity_op = self.op_u32_const(&block, "8");
        let array_capacity = array_capacity_op.result(0)?.into();

        let array_element_size_bytes = arg_type.get_width() / 8;

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

        let sierra_type = SierraType::Array {
            ty: self.struct_type(&[self.u32_type(), self.u32_type(), self.llvm_ptr_type()]),
            len_type: self.u32_type(),
            element_type: Box::new(arg_type.clone()),
        };

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
            self.op_cmp(&block, CmpOp::UnsignedLess, array_len.into(), array_capacity.into());

        self.op_cond_br(
            &block,
            is_less.result(0)?.into(),
            &append_value_block,
            &realloc_block,
            &[array_value.into()],
            &[],
        )?;

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

        let sierra_type = SierraType::Array {
            ty: self.struct_type(&[self.u32_type(), self.u32_type(), self.llvm_ptr_type()]),
            len_type: self.u32_type(),
            element_type: Box::new(arg_type.clone()),
        };

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
        let id = func_decl.id.debug_name.as_deref().unwrap();

        let src_type = SierraType::Array {
            ty: self.struct_type(&[self.u32_type(), self.u32_type(), self.llvm_ptr_type()]),
            len_type: self.u32_type(),
            element_type: Box::new(SierraType::Simple(self.felt_type())),
        };

        // Blocks:
        //   - 0: Entry point.
        //   - 1: Source bits loop body.
        //   - 2: Add & shift loop body.
        //   - 3: Non-zero digit search loop (pre-condition and condition).
        //   - 4: Non-zero digit search loop (post-condition).
        //   - 5: Make ASCII string loop.
        let region = Region::new();
        let blocks = (
            region.append_block(Block::new(&[(
                src_type.get_type(),
                Location::unknown(&self.context),
            )])),
            region.append_block(Block::new(&[(
                Type::parse(&self.context, "index").unwrap(),
                Location::unknown(&self.context),
            )])),
            region.append_block(Block::new(&[
                (Type::parse(&self.context, "index").unwrap(), Location::unknown(&self.context)),
                (Type::parse(&self.context, "i1").unwrap(), Location::unknown(&self.context)),
            ])),
            region.append_block(Block::new(&[(
                Type::parse(&self.context, "index").unwrap(),
                Location::unknown(&self.context),
            )])),
            region.append_block(Block::new(&[(
                Type::parse(&self.context, "index").unwrap(),
                Location::unknown(&self.context),
            )])),
            region.append_block(Block::new(&[(
                src_type.get_type(),
                Location::unknown(&self.context),
            )])),
        );

        //
        // Block #0: Entry point (1/?).
        //

        // Allocate buffer.
        let buf = blocks.0.append_operation(
            operation::Builder::new("memref.alloca", Location::unknown(&self.context))
                .add_results(&[Type::parse(&self.context, "memref<78xi8>").unwrap()])
                .build(),
        );
        let buf: Value = buf.result(0)?.into();

        // Zero the buffer.
        let ptr_as_idx = {
            // Extract LLVM pointer from memref.
            let ptr_as_idx_op = blocks.0.append_operation(
                operation::Builder::new(
                    "memref.extract_aligned_pointer_as_index",
                    Location::unknown(&self.context),
                )
                .add_operands(&[buf])
                .add_results(&[Type::parse(&self.context, "index").unwrap()])
                .build(),
            );
            let ptr_as_idx: Value = ptr_as_idx_op.result(0)?.into();

            let ptr_as_i64 = blocks.0.append_operation(
                operation::Builder::new("arith.index_castui", Location::unknown(&self.context))
                    .add_operands(&[ptr_as_idx])
                    .add_results(&[Type::parse(&self.context, "i64").unwrap()])
                    .build(),
            );
            let ptr_as_i64: Value = ptr_as_i64.result(0)?.into();

            let ptr = blocks.0.append_operation(
                operation::Builder::new("llvm.inttoptr", Location::unknown(&self.context))
                    .add_operands(&[ptr_as_i64])
                    .add_results(&[Type::parse(&self.context, "!llvm.ptr").unwrap()])
                    .build(),
            );
            let ptr: Value = ptr.result(0)?.into();

            // Zero the buffer.
            let k0 = blocks.0.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_attributes(&[
                        NamedAttribute::new_parsed(&self.context, "value", "0 : i8").unwrap()
                    ])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );
            let k0: Value = k0.result(0)?.into();

            let k1 = blocks.0.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_attributes(&[NamedAttribute::new_parsed(
                        &self.context,
                        "value",
                        "78 : i32",
                    )
                    .unwrap()])
                    .add_results(&[Type::parse(&self.context, "i32").unwrap()])
                    .build(),
            );
            let k1: Value = k1.result(0)?.into();

            let k2 = blocks.0.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_attributes(&[
                        NamedAttribute::new_parsed(&self.context, "value", "0 : i1").unwrap()
                    ])
                    .add_results(&[Type::parse(&self.context, "i1").unwrap()])
                    .build(),
            );
            let k2: Value = k2.result(0)?.into();

            blocks.0.append_operation(
                operation::Builder::new("llvm.intr.memset", Location::unknown(&self.context))
                    .add_operands(&[ptr, k0, k1, k2])
                    .build(),
            );

            ptr_as_idx_op
        };
        let ptr_as_idx: Value = ptr_as_idx.result(0)?.into();

        // Find MSB index starting from LSB (count minimum required bits).
        let first_used_bit_index = {
            let leading_zeros = blocks.0.append_operation(
                operation::Builder::new("math.ctlz", Location::unknown(&self.context))
                    .add_operands(&[blocks.0.argument(0)?.into()])
                    .add_results(&[self.felt_type()])
                    .build(),
            );
            let leading_zeros: Value = leading_zeros.result(0)?.into();

            blocks.0.append_operation(
                operation::Builder::new("arith.index_castui", Location::unknown(&self.context))
                    .add_operands(&[leading_zeros])
                    .add_results(&[Type::parse(&self.context, "index").unwrap()])
                    .build(),
            )
        };
        let first_used_bit_index: Value = first_used_bit_index.result(0)?.into();

        // Iterate over used bits (from the first used bit index to the least significant bit).
        {
            let k0 = blocks.0.append_operation(
                operation::Builder::new("index.constant", Location::unknown(&self.context))
                    .add_attributes(&[NamedAttribute::new_parsed(
                        &self.context,
                        "value",
                        "252 : index",
                    )
                    .unwrap()])
                    .add_results(&[Type::parse(&self.context, "index").unwrap()])
                    .build(),
            );
            let k0: Value = k0.result(0)?.into();

            let k1 = blocks.0.append_operation(
                operation::Builder::new("index.constant", Location::unknown(&self.context))
                    .add_attributes(&[NamedAttribute::new_parsed(
                        &self.context,
                        "value",
                        "1 : index",
                    )
                    .unwrap()])
                    .add_results(&[Type::parse(&self.context, "index").unwrap()])
                    .build(),
            );
            let k1: Value = k1.result(0)?.into();

            blocks.0.append_operation(
                operation::Builder::new("scf.for", Location::unknown(&self.context))
                    .add_operands(&[first_used_bit_index, k0, k1])
                    .add_successors(&[&blocks.1])
                    .build(),
            );
        }

        //
        // Block #1: Source bits loop body.
        //

        // Extract current bit from source.
        let current_bit_value = {
            let k0 = blocks.1.append_operation(
                operation::Builder::new("index.constant", Location::unknown(&self.context))
                    .add_attributes(&[NamedAttribute::new_parsed(
                        &self.context,
                        "value",
                        "251 : index",
                    )
                    .unwrap()])
                    .add_results(&[Type::parse(&self.context, "index").unwrap()])
                    .build(),
            );
            let k0: Value = k0.result(0)?.into();

            let shift_amount = blocks.1.append_operation(
                operation::Builder::new("index.sub", Location::unknown(&self.context))
                    .add_operands(&[k0, blocks.1.argument(0)?.into()])
                    .add_results(&[Type::parse(&self.context, "index").unwrap()])
                    .build(),
            );
            let shift_amount: Value = shift_amount.result(0)?.into();

            let shift_amount_felt = blocks.1.append_operation(
                operation::Builder::new("arith.index_castui", Location::unknown(&self.context))
                    .add_operands(&[shift_amount])
                    .add_results(&[self.felt_type()])
                    .build(),
            );
            let shift_amount_felt: Value = shift_amount_felt.result(0)?.into();

            let shifted_value = blocks.1.append_operation(
                operation::Builder::new("arith.shrui", Location::unknown(&self.context))
                    .add_operands(&[blocks.0.argument(0)?.into(), shift_amount_felt])
                    .add_results(&[self.felt_type()])
                    .build(),
            );
            let shifted_value: Value = shifted_value.result(0)?.into();

            blocks.1.append_operation(
                operation::Builder::new("arith.trunci", Location::unknown(&self.context))
                    .add_operands(&[shifted_value])
                    .add_results(&[Type::parse(&self.context, "i1").unwrap()])
                    .build(),
            )
        };
        let current_bit_value: Value = current_bit_value.result(0)?.into();

        // Iterate over each byte for the add & shift steps.
        {
            let k0 = blocks.1.append_operation(
                operation::Builder::new("index.constant", Location::unknown(&self.context))
                    .add_attributes(&[NamedAttribute::new_parsed(
                        &self.context,
                        "value",
                        "0 : index",
                    )
                    .unwrap()])
                    .add_results(&[Type::parse(&self.context, "index").unwrap()])
                    .build(),
            );
            let k0: Value = k0.result(0)?.into();

            let k1 = blocks.1.append_operation(
                operation::Builder::new("index.constant", Location::unknown(&self.context))
                    .add_attributes(&[NamedAttribute::new_parsed(
                        &self.context,
                        "value",
                        "76 : index",
                    )
                    .unwrap()])
                    .add_results(&[Type::parse(&self.context, "index").unwrap()])
                    .build(),
            );
            let k1: Value = k1.result(0)?.into();

            let k2 = blocks.1.append_operation(
                operation::Builder::new("index.constant", Location::unknown(&self.context))
                    .add_attributes(&[NamedAttribute::new_parsed(
                        &self.context,
                        "value",
                        "1 : index",
                    )
                    .unwrap()])
                    .add_results(&[Type::parse(&self.context, "index").unwrap()])
                    .build(),
            );
            let k2: Value = k2.result(0)?.into();

            blocks.1.append_operation(
                operation::Builder::new("scf.for", Location::unknown(&self.context))
                    .add_operands(&[k0, k1, k2, current_bit_value])
                    .add_successors(&[&blocks.2])
                    .build(),
            );
        }

        // Yield.
        blocks.1.append_operation(
            operation::Builder::new("scf.yield", Location::unknown(&self.context)).build(),
        );

        //
        // Block #2: Add & shift loop body.
        //

        // Load value.
        let (value_index, value) = {
            let k0 = blocks.2.append_operation(
                operation::Builder::new("index.constant", Location::unknown(&self.context))
                    .add_attributes(&[NamedAttribute::new_parsed(
                        &self.context,
                        "value",
                        "75 : index",
                    )
                    .unwrap()])
                    .add_results(&[Type::parse(&self.context, "index").unwrap()])
                    .build(),
            );
            let k0: Value = k0.result(0)?.into();

            let value_index_op = blocks.2.append_operation(
                operation::Builder::new("index.sub", Location::unknown(&self.context))
                    .add_operands(&[k0, blocks.2.argument(0)?.into()])
                    .add_results(&[Type::parse(&self.context, "index").unwrap()])
                    .build(),
            );
            let value_index: Value = value_index_op.result(0)?.into();

            let value = blocks.2.append_operation(
                operation::Builder::new("memref.load", Location::unknown(&self.context))
                    .add_operands(&[buf, value_index])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );

            (value_index_op, value)
        };
        let value_index: Value = value_index.result(0)?.into();
        let value: Value = value.result(0)?.into();

        // Add 3 if value >= 5.
        let value = {
            let k0 = blocks.2.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_attributes(&[
                        NamedAttribute::new_parsed(&self.context, "value", "5 : i8").unwrap()
                    ])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );
            let k0: Value = k0.result(0)?.into();

            let is_ge_5 = blocks.2.append_operation(
                operation::Builder::new("arith.cmpi", Location::unknown(&self.context))
                    .add_attributes(&[NamedAttribute::new_parsed(
                        &self.context,
                        "predicate",
                        /* uge */ "9",
                    )
                    .unwrap()])
                    .add_operands(&[value, k0])
                    .add_results(&[Type::parse(&self.context, "i1").unwrap()])
                    .build(),
            );
            let is_ge_5: Value = is_ge_5.result(0)?.into();

            let k1 = blocks.2.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_attributes(&[
                        NamedAttribute::new_parsed(&self.context, "value", "3 : i8").unwrap()
                    ])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );
            let k1: Value = k1.result(0)?.into();

            let value_plus_3 = blocks.2.append_operation(
                operation::Builder::new("arith.addi", Location::unknown(&self.context))
                    .add_operands(&[value, k1])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );
            let value_plus_3: Value = value_plus_3.result(0)?.into();

            blocks.2.append_operation(
                operation::Builder::new("arith.select", Location::unknown(&self.context))
                    .add_operands(&[is_ge_5, value_plus_3, value])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            )
        };
        let value: Value = value.result(0)?.into();

        // Shift left by 1, then bitwise-or the carry in.
        let (shifted_value_with_carry_in, new_value) = {
            let k0 = blocks.2.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_attributes(&[
                        NamedAttribute::new_parsed(&self.context, "value", "1 : i8").unwrap()
                    ])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );
            let k0: Value = k0.result(0)?.into();

            let shifted_value = blocks.2.append_operation(
                operation::Builder::new("arith.shli", Location::unknown(&self.context))
                    .add_operands(&[value, k0])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );
            let shifted_value: Value = shifted_value.result(0)?.into();

            let carry_in = blocks.2.append_operation(
                operation::Builder::new("llvm.zext", Location::unknown(&self.context))
                    .add_operands(&[blocks.2.argument(1)?.into()])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );
            let carry_in: Value = carry_in.result(0)?.into();

            let shifted_value_with_carry_in_op = blocks.2.append_operation(
                operation::Builder::new("arith.ori", Location::unknown(&self.context))
                    .add_operands(&[shifted_value, carry_in])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );
            let shifted_value_with_carry_in: Value =
                shifted_value_with_carry_in_op.result(0)?.into();

            let k1 = blocks.2.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_attributes(&[NamedAttribute::new_parsed(
                        &self.context,
                        "value",
                        "15 : i8",
                    )
                    .unwrap()])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );
            let k1: Value = k1.result(0)?.into();

            let new_value = blocks.2.append_operation(
                operation::Builder::new("arith.andi", Location::unknown(&self.context))
                    .add_operands(&[shifted_value_with_carry_in, k1])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );

            (shifted_value_with_carry_in_op, new_value)
        };
        let shifted_value_with_carry_in: Value = shifted_value_with_carry_in.result(0)?.into();
        let new_value: Value = new_value.result(0)?.into();

        // Store new value.
        blocks.2.append_operation(
            operation::Builder::new("memref.store", Location::unknown(&self.context))
                .add_operands(&[new_value, buf, value_index])
                .build(),
        );

        // Compute next carry and yield.
        {
            let k0 = blocks.2.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_attributes(&[
                        NamedAttribute::new_parsed(&self.context, "value", "4 : i8").unwrap()
                    ])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );
            let k0: Value = k0.result(0)?.into();

            let carry_out = blocks.2.append_operation(
                operation::Builder::new("arith.shrui", Location::unknown(&self.context))
                    .add_operands(&[shifted_value_with_carry_in, k0])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );
            let carry_out: Value = carry_out.result(0)?.into();

            let carry_out = blocks.2.append_operation(
                operation::Builder::new("arith.trunci", Location::unknown(&self.context))
                    .add_operands(&[carry_out])
                    .add_results(&[Type::parse(&self.context, "i1").unwrap()])
                    .build(),
            );
            let carry_out: Value = carry_out.result(0)?.into();

            blocks.2.append_operation(
                operation::Builder::new("scf.yield", Location::unknown(&self.context))
                    .add_operands(&[carry_out])
                    .build(),
            );
        }

        //
        // Block #0: Entry point (2/?).
        //

        // Find first non-zero digit.
        let first_nonzero_digit_index = {
            let k0 = blocks.0.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_attributes(&[NamedAttribute::new_parsed(
                        &self.context,
                        "value",
                        "0 : index",
                    )
                    .unwrap()])
                    .add_results(&[Type::parse(&self.context, "index").unwrap()])
                    .build(),
            );
            let k0: Value = k0.result(0)?.into();

            blocks.0.append_operation(
                operation::Builder::new("scf.while", Location::unknown(&self.context))
                    .add_operands(&[k0])
                    .add_results(&[Type::parse(&self.context, "index").unwrap()])
                    .add_successors(&[&blocks.3, &blocks.4])
                    .build(),
            )
        };
        let first_nonzero_digit_index: Value = first_nonzero_digit_index.result(0)?.into();

        //
        // Block #3: Non-zero digit search loop (pre-condition and condition).
        //
        {
            // Load value.
            let value = blocks.3.append_operation(
                operation::Builder::new("memref.load", Location::unknown(&self.context))
                    .add_operands(&[buf, blocks.3.argument(0)?.into()])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );
            let value: Value = value.result(0)?.into();

            // Compare with zero.
            let k0 = blocks.3.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_attributes(&[
                        NamedAttribute::new_parsed(&self.context, "value", "0 : i8").unwrap()
                    ])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );
            let k0: Value = k0.result(0)?.into();

            let value_is_zero = blocks.3.append_operation(
                operation::Builder::new("arith.cmpi", Location::unknown(&self.context))
                    .add_attributes(&[NamedAttribute::new_parsed(
                        &self.context,
                        "prediate",
                        /* eq */ "0",
                    )
                    .unwrap()])
                    .add_operands(&[value, k0])
                    .add_results(&[Type::parse(&self.context, "i1").unwrap()])
                    .build(),
            );
            let value_is_zero: Value = value_is_zero.result(0)?.into();

            // If equal, continue.
            blocks.3.append_operation(
                operation::Builder::new("scf.condition", Location::unknown(&self.context))
                    .add_operands(&[value_is_zero, blocks.3.argument(0)?.into()])
                    .build(),
            );
        }

        //
        // Block #4: Non-zero digit search loop (post-condition).
        //

        {
            // Increment index.
            let k1 = blocks.4.append_operation(
                operation::Builder::new("index.constant", Location::unknown(&self.context))
                    .add_attributes(&[NamedAttribute::new_parsed(
                        &self.context,
                        "value",
                        "1 : index",
                    )
                    .unwrap()])
                    .add_results(&[Type::parse(&self.context, "index").unwrap()])
                    .build(),
            );
            let k1: Value = k1.result(0)?.into();

            let new_index = blocks.4.append_operation(
                operation::Builder::new("index.add", Location::unknown(&self.context))
                    .add_operands(&[blocks.4.argument(0)?.into(), k1])
                    .add_results(&[Type::parse(&self.context, "index").unwrap()])
                    .build(),
            );
            let new_index: Value = new_index.result(0)?.into();

            blocks.4.append_operation(
                operation::Builder::new("scf.yield", Location::unknown(&self.context))
                    .add_operands(&[new_index])
                    .build(),
            );
        }

        //
        // Block #0: Entry point (3/?).
        //

        // Convert BCD to ascii decimal numbers.
        {
            let k0 = blocks.0.append_operation(
                operation::Builder::new("index.constant", Location::unknown(&self.context))
                    .add_attributes(&[NamedAttribute::new_parsed(
                        &self.context,
                        "value",
                        "76 : index",
                    )
                    .unwrap()])
                    .add_results(&[Type::parse(&self.context, "index").unwrap()])
                    .build(),
            );
            let k0: Value = k0.result(0)?.into();

            let k1 = blocks.0.append_operation(
                operation::Builder::new("index.constant", Location::unknown(&self.context))
                    .add_attributes(&[NamedAttribute::new_parsed(
                        &self.context,
                        "value",
                        "1 : index",
                    )
                    .unwrap()])
                    .add_results(&[Type::parse(&self.context, "index").unwrap()])
                    .build(),
            );
            let k1: Value = k1.result(0)?.into();

            blocks.0.append_operation(
                operation::Builder::new("scf.for", Location::unknown(&self.context))
                    .add_operands(&[first_nonzero_digit_index, k0, k1])
                    .add_successors(&[&blocks.5])
                    .build(),
            );
        }

        //
        // Block #5: Make ASCII string loop.
        //

        {
            // Load value.
            let value = blocks.5.append_operation(
                operation::Builder::new("memref.load", Location::unknown(&self.context))
                    .add_operands(&[buf, blocks.5.argument(0)?.into()])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );
            let value: Value = value.result(0)?.into();

            // Add 48 (or ASCII '0').
            let k0 = blocks.5.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_attributes(&[NamedAttribute::new_parsed(
                        &self.context,
                        "value",
                        "48 : i8",
                    )
                    .unwrap()])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );
            let k0: Value = k0.result(0)?.into();

            let ascii_value = blocks.5.append_operation(
                operation::Builder::new("arith.addi", Location::unknown(&self.context))
                    .add_operands(&[value, k0])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );
            let ascii_value: Value = ascii_value.result(0)?.into();

            // Store new value.
            blocks.5.append_operation(
                operation::Builder::new("memref.store", Location::unknown(&self.context))
                    .add_operands(&[ascii_value, buf, blocks.5.argument(0)?.into()])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );
        }

        // Yield.
        blocks.5.append_operation(
            operation::Builder::new("scf.yield", Location::unknown(&self.context)).build(),
        );

        //
        // Block #0: Entry point (4/?).
        //

        // Set '\n'.
        {
            let k0 = blocks.0.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_attributes(&[NamedAttribute::new_parsed(
                        &self.context,
                        "value",
                        "10 : i8",
                    )
                    .unwrap()])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );
            let k0: Value = k0.result(0)?.into();

            let k1 = blocks.0.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_attributes(&[NamedAttribute::new_parsed(
                        &self.context,
                        "value",
                        "76 : index",
                    )
                    .unwrap()])
                    .add_results(&[Type::parse(&self.context, "index").unwrap()])
                    .build(),
            );
            let k1: Value = k1.result(0)?.into();

            blocks.0.append_operation(
                operation::Builder::new("memref.store", Location::unknown(&self.context))
                    .add_operands(&[k0, buf, k1])
                    .add_results(&[Type::parse(&self.context, "i8").unwrap()])
                    .build(),
            );
        }

        // Call `puts()`.
        {
            // TODO: index.add ptr_as_idx, first_nonzero_digit_index
            // TODO: arith.index_castui
            // TODO: llvm.inttoptr
            // TODO: func.call

            let offset_ptr = blocks.0.append_operation(
                operation::Builder::new("index.add", Location::unknown(&self.context))
                    .add_operands(&[ptr_as_idx, first_nonzero_digit_index])
                    .add_results(&[Type::parse(&self.context, "index").unwrap()])
                    .build(),
            );
            let offset_ptr: Value = offset_ptr.result(0)?.into();

            let offset_ptr_i64 = blocks.0.append_operation(
                operation::Builder::new("arith.index_castui", Location::unknown(&self.context))
                    .add_operands(&[offset_ptr])
                    .add_results(&[Type::parse(&self.context, "i64").unwrap()])
                    .build(),
            );
            let offset_ptr_i64: Value = offset_ptr_i64.result(0)?.into();

            let offset_ptr_llvm = blocks.0.append_operation(
                operation::Builder::new("llvm.inttoptr", Location::unknown(&self.context))
                    .add_operands(&[offset_ptr_i64])
                    .add_results(&[Type::parse(&self.context, "!llvm.ptr").unwrap()])
                    .build(),
            );
            let offset_ptr_llvm: Value = offset_ptr_llvm.result(0)?.into();

            blocks.0.append_operation(
                operation::Builder::new("func.call", Location::unknown(&self.context))
                    .add_operands(&[offset_ptr_llvm])
                    .add_results(&[Type::parse(&self.context, "i32").unwrap()])
                    .build(),
            );
        }

        // Return.
        blocks.0.append_operation(
            operation::Builder::new("func.return", Location::unknown(&self.context)).build(),
        );

        //
        // Register the function.
        //
        let fn_type = create_fn_signature(&[src_type.get_type()], &[]);
        let op_func =
            self.op_func("print", &fn_type, vec![region], FnAttributes::libfunc(false, false))?;

        parent_block.append_operation(op_func);
        storage.libfuncs.insert(
            id.to_string(),
            SierraLibFunc::create_function_all_args(vec![src_type], vec![]),
        );

        Ok(())
    }
}
