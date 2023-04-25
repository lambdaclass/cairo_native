use color_eyre::Result;
use itertools::Itertools;
use melior_next::{
    dialect::{arith, cf},
    ir::{
        operation::{self},
        Attribute, Block, Identifier, Location, NamedAttribute, Operation, OperationRef, Region,
        Type, TypeLike, Value, ValueLike,
    },
};
use std::cmp::Ordering;

use crate::types::DEFAULT_PRIME;

use super::{fn_attributes::FnAttributes, Compiler};

impl<'ctx> Compiler<'ctx> {
    pub fn named_attribute(&self, name: &str, attribute: &str) -> Result<NamedAttribute> {
        Ok(NamedAttribute::new_parsed(&self.context, name, attribute)?)
    }

    pub fn prime_constant<'a>(&self, block: &'a Block) -> OperationRef<'a> {
        self.op_const(block, DEFAULT_PRIME, self.felt_type())
    }

    /// Only the MLIR op, doesn't do modulo.
    pub fn op_add<'a>(&self, block: &'a Block, lhs: Value, rhs: Value) -> OperationRef<'a> {
        block.append_operation(arith::addi(
            lhs,
            rhs,
            lhs.r#type(),
            Location::unknown(&self.context),
        ))
    }

    /// Only the MLIR op, doesn't do modulo.
    pub fn op_sub<'a>(&self, block: &'a Block, lhs: Value, rhs: Value) -> OperationRef<'a> {
        block.append_operation(arith::subi(
            lhs,
            rhs,
            lhs.r#type(),
            Location::unknown(&self.context),
        ))
    }

    /// Only the MLIR op.
    pub fn op_mul<'a>(&self, block: &'a Block, lhs: Value, rhs: Value) -> OperationRef<'a> {
        block.append_operation(arith::muli(
            lhs,
            rhs,
            lhs.r#type(),
            Location::unknown(&self.context),
        ))
    }

    /// Only the MLIR op.
    pub fn op_div<'a>(&self, block: &'a Block, lhs: Value, rhs: Value) -> OperationRef<'a> {
        block.append_operation(arith::divui(
            lhs,
            rhs,
            lhs.r#type(),
            Location::unknown(&self.context),
        ))
    }

    /// Only the MLIR op.
    pub fn op_rem<'a>(&self, block: &'a Block, lhs: Value, rhs: Value) -> OperationRef<'a> {
        block.append_operation(arith::remui(
            lhs,
            rhs,
            lhs.r#type(),
            Location::unknown(&self.context),
        ))
    }

    /// Shift right signed.
    pub fn op_shrs<'a>(&self, block: &'a Block, value: Value, shift_by: Value) -> OperationRef<'a> {
        block.append_operation(arith::shrsi(
            value,
            shift_by,
            value.r#type(),
            Location::unknown(&self.context),
        ))
    }

    /// Shift right unsigned.
    pub fn op_shru<'a>(&self, block: &'a Block, value: Value, shift_by: Value) -> OperationRef<'a> {
        block.append_operation(arith::shrui(
            value,
            shift_by,
            value.r#type(),
            Location::unknown(&self.context),
        ))
    }

    /// Shift left.
    pub fn op_shl<'a>(&self, block: &'a Block, value: Value, shift_by: Value) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("arith.shli", Location::unknown(&self.context))
                .add_operands(&[value, shift_by])
                .add_results(&[value.r#type()])
                .build(),
        )
    }

    /// Only the MLIR op.
    ///
    /// > Source: https://mlir.llvm.org/docs/Dialects/ArithOps/#arithcmpi-mlirarithcmpiop
    pub fn op_cmp<'a>(
        &self,
        block: &'a Block,
        cmp_op: CmpOp,
        lhs: Value,
        rhs: Value,
    ) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("arith.cmpi", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "predicate",
                    cmp_op.to_mlir_val(),
                )
                .unwrap()])
                .add_operands(&[lhs, rhs])
                .add_results(&[Type::integer(&self.context, 1)])
                .build(),
        )
    }

    pub fn op_trunc<'a>(&self, block: &'a Block, value: Value, to: Type) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("arith.trunci", Location::unknown(&self.context))
                .add_operands(&[value])
                .add_results(&[to])
                .build(),
        )
    }

    pub fn op_sext<'a>(&self, block: &'a Block, value: Value, to: Type) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("arith.extsi", Location::unknown(&self.context))
                .add_operands(&[value])
                .add_results(&[to])
                .build(),
        )
    }

    pub fn op_zext<'a>(&self, block: &'a Block, value: Value, to: Type) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("arith.extui", Location::unknown(&self.context))
                .add_operands(&[value])
                .add_results(&[to])
                .build(),
        )
    }

    pub fn op_and<'a>(
        &self,
        block: &'a Block,
        lhs: Value,
        rhs: Value,
        to: Type,
    ) -> OperationRef<'a> {
        block.append_operation(arith::andi(lhs, rhs, to, Location::unknown(&self.context)))
    }

    pub fn op_or<'a>(
        &self,
        block: &'a Block,
        lhs: Value,
        rhs: Value,
        to: Type,
    ) -> OperationRef<'a> {
        block.append_operation(arith::ori(lhs, rhs, to, Location::unknown(&self.context)))
    }

    pub fn op_xor<'a>(
        &self,
        block: &'a Block,
        lhs: Value,
        rhs: Value,
        to: Type,
    ) -> OperationRef<'a> {
        block.append_operation(arith::xori(lhs, rhs, to, Location::unknown(&self.context)))
    }

    /// New constant
    pub fn op_const<'a>(&self, block: &'a Block, val: &str, ty: Type<'ctx>) -> OperationRef<'a> {
        block.append_operation(arith::r#const(
            &self.context,
            val,
            ty,
            Location::unknown(&self.context),
        ))
    }

    /// New felt constant
    pub fn op_felt_const<'a>(&'a self, block: &'a Block, val: &str) -> OperationRef<'a> {
        self.op_const(block, val, self.felt_type())
    }

    /// New u8 constant
    pub fn op_u8_const<'a>(&'a self, block: &'a Block, val: &str) -> OperationRef<'a> {
        self.op_const(block, val, self.u8_type())
    }

    /// New u16 constant
    pub fn op_u16_const<'a>(&'a self, block: &'a Block, val: &str) -> OperationRef<'a> {
        self.op_const(block, val, self.u16_type())
    }

    /// New u32 constant
    pub fn op_u32_const<'a>(&'a self, block: &'a Block, val: &str) -> OperationRef<'a> {
        self.op_const(block, val, self.u32_type())
    }

    /// New u64 constant
    pub fn op_u64_const<'a>(&'a self, block: &'a Block, val: &str) -> OperationRef<'a> {
        self.op_const(block, val, self.u64_type())
    }

    /// New u128 constant
    pub fn op_u128_const<'a>(&'a self, block: &'a Block, val: &str) -> OperationRef<'a> {
        self.op_const(block, val, self.u128_type())
    }

    /// Does modulo prime.
    /// val can be wider that felt, the result will be in felt range, but the same type width
    pub fn op_felt_modulo<'a>(&self, block: &'a Block, val: Value) -> Result<OperationRef<'a>> {
        let prime = self.prime_constant(block);
        let prime_val: Value = prime.result(0)?.into();

        Ok(match val.r#type().get_width().unwrap().cmp(&prime_val.r#type().get_width().unwrap()) {
            // If num_bits(value) < 252, then no modulo is needed (already in range).
            Ordering::Less => {
                // TODO: Remove this modulo when  (it is not necessary).
                self.op_rem(block, val, prime_val)
            }
            // If the value and the modulo have the same width, just apply the modulo.
            Ordering::Equal => self.op_rem(block, val, prime_val),
            // If the value is wider than the prime, zero-extend the prime before the modulo.
            Ordering::Greater => {
                let zext_op = self.op_zext(block, prime_val, val.r#type());
                self.op_rem(block, val, zext_op.result(0)?.into())
            }
        })
    }

    /// Perform modular exponentiation using the binary RTL method.
    ///
    /// > Source: https://en.wikipedia.org/wiki/Modular_exponentiation
    pub fn op_felt_pow<'a>(
        &self,
        region: &'a Region,
        block: &'a Block,
        base: Value,
        exponent: Value,
    ) -> Result<OperationRef<'a>> {
        // <in block> <- setup
        // <new block> <- loop before cond
        // <new block> <- loop after cond
        // <new block> <- final (returned)

        // result = 1
        // while exponent != 0 {
        //     if lsb(exponent) {}
        //         result = (result * base) % prime
        //     }
        //     exponent = exponent >> 1
        //     base = (base * base) % prime
        // }

        let init_block = block;
        let loop_before_condition_block = region.append_block(Block::new(&[
            (self.felt_type(), Location::unknown(&self.context)),
            (self.felt_type(), Location::unknown(&self.context)),
            (self.felt_type(), Location::unknown(&self.context)),
        ]));
        let loop_after_condition_block = region.append_block(Block::new(&[
            (self.felt_type(), Location::unknown(&self.context)),
            (self.felt_type(), Location::unknown(&self.context)),
            (self.felt_type(), Location::unknown(&self.context)),
        ]));
        let loop_step_block = region.append_block(Block::new(&[
            (self.felt_type(), Location::unknown(&self.context)),
            (self.felt_type(), Location::unknown(&self.context)),
            (self.felt_type(), Location::unknown(&self.context)),
        ]));
        let loop_after_step_block = region.append_block(Block::new(&[
            (self.felt_type(), Location::unknown(&self.context)),
            (self.felt_type(), Location::unknown(&self.context)),
            (self.felt_type(), Location::unknown(&self.context)),
        ]));

        let const_one = self.op_felt_const(init_block, "1");
        let const_zero = self.op_felt_const(init_block, "0");

        // Block loop_before_condition:
        //   Args: [result, base, exponent].
        //   Rets: [result, base, exponent] (forwarded).
        {
            // Compare exponent with zero.
            let loop_cmp_op = self.op_cmp(
                &loop_before_condition_block,
                CmpOp::Equal,
                loop_before_condition_block.argument(2)?.into(),
                const_zero.result(0)?.into(),
            );

            // Run `scf.condition` operation.
            loop_before_condition_block.append_operation(
                operation::Builder::new("scf.condition", Location::unknown(&self.context))
                    .add_operands(&[
                        loop_cmp_op.result(0)?.into(),
                        loop_before_condition_block.argument(0)?.into(),
                        loop_before_condition_block.argument(1)?.into(),
                        loop_before_condition_block.argument(2)?.into(),
                    ])
                    .build(),
            );
        }

        // Block loop_after_condition:
        //   Args: [result, base, exponent].
        //   Rets: [result, base, exponent] (forward).
        {
            // Compare LSB of exponent.
            let masked_exponent = self.op_and(
                &loop_after_condition_block,
                loop_after_condition_block.argument(2)?.into(),
                const_one.result(0)?.into(),
                self.felt_type(),
            );
            let lsb_bit = self.op_cmp(
                &loop_after_condition_block,
                CmpOp::Equal,
                masked_exponent.result(0)?.into(),
                const_zero.result(0)?.into(),
            );

            // Jump to either loop_step or loop_after_step.
            self.op_cond_br(
                &loop_after_condition_block,
                lsb_bit.result(0)?.into(),
                &loop_after_step_block,
                &loop_step_block,
                &[
                    loop_after_condition_block.argument(0)?.into(),
                    loop_after_condition_block.argument(1)?.into(),
                    loop_after_condition_block.argument(2)?.into(),
                ],
                &[
                    loop_after_condition_block.argument(0)?.into(),
                    loop_after_condition_block.argument(1)?.into(),
                    loop_after_condition_block.argument(2)?.into(),
                ],
            );
        }

        // Block loop_step:
        //   Args: [result, base, exponent].
        //   Rets: [result, base, exponent] (forward after update).
        {
            // result = (result * base) % modulus.
            let lhs = self.op_zext(
                &loop_step_block,
                loop_step_block.argument(0)?.into(),
                self.double_felt_type(),
            );
            let rhs = self.op_zext(
                &loop_step_block,
                loop_step_block.argument(1)?.into(),
                self.double_felt_type(),
            );

            let op_mul =
                self.op_mul(&loop_step_block, lhs.result(0)?.into(), rhs.result(0)?.into());
            let op_mod = self.op_felt_modulo(&loop_step_block, op_mul.result(0)?.into())?;
            let op_trunc =
                self.op_trunc(&loop_step_block, op_mod.result(0)?.into(), self.felt_type());

            // Jump to loop_after_step.
            self.op_br(
                &loop_step_block,
                &loop_after_step_block,
                &[
                    op_trunc.result(0)?.into(),
                    loop_step_block.argument(1)?.into(),
                    loop_step_block.argument(2)?.into(),
                ],
            );
        }

        // Block loop_after_step:
        //   Args: [result, base, exponent].
        //   Rets: [result, base, exponent] (next step after update).
        {
            // exponent = exponent >> 1.
            let shifted_exponent = self.op_shru(
                &loop_after_step_block,
                loop_after_step_block.argument(2)?.into(),
                const_one.result(0)?.into(),
            );

            // base = (base * base) % modulus.
            let lhs = self.op_zext(
                &loop_after_step_block,
                loop_after_step_block.argument(1)?.into(),
                self.double_felt_type(),
            );
            let rhs = self.op_zext(
                &loop_after_step_block,
                loop_after_step_block.argument(1)?.into(),
                self.double_felt_type(),
            );

            let op_mul =
                self.op_mul(&loop_after_step_block, lhs.result(0)?.into(), rhs.result(0)?.into());
            let op_mod = self.op_felt_modulo(&loop_after_step_block, op_mul.result(0)?.into())?;
            let op_trunc =
                self.op_trunc(&loop_after_step_block, op_mod.result(0)?.into(), self.felt_type());

            // scf.yield
            loop_after_step_block.append_operation(
                operation::Builder::new("scf.yield", Location::unknown(&self.context))
                    .add_operands(&[
                        loop_after_step_block.argument(0)?.into(),
                        op_trunc.result(0)?.into(),
                        shifted_exponent.result(0)?.into(),
                    ])
                    .build(),
            );
        }

        Ok(init_block.append_operation(
            operation::Builder::new("scf.while", Location::unknown(&self.context))
                .add_operands(&[const_one.result(0)?.into(), base, exponent])
                .add_successors(&[&loop_before_condition_block, &loop_after_condition_block])
                .add_results(&[self.felt_type(), self.felt_type(), self.felt_type()])
                .build(),
        ))
    }

    /// Compute the multiplicative inverse of a felt.
    ///
    /// > Source: https://en.wikipedia.org/wiki/Modular_multiplicative_inverse
    /// TODO replace with extended euclidean algorithm
    pub fn op_felt_inverse<'a>(
        &self,
        region: &'a Region,
        block: &'a Block,
        value: Value,
    ) -> Result<OperationRef<'a>> {
        let const_p = self.prime_constant(block);
        let const_two = self.op_felt_const(block, "2");
        let p_minus_2 = self.op_sub(block, const_p.result(0)?.into(), const_two.result(0)?.into());

        self.op_felt_pow(region, block, value, p_minus_2.result(0)?.into())
    }

    /// Perform a felt divison (not euclidean, but modular).
    ///
    /// In other words, find x in `a / b = x` such that `x * b = a` in modulo prime.
    pub fn op_felt_div<'a>(
        &self,
        region: &'a Region,
        block: &'a Block,
        dividend: Value,
        divisor: Value,
    ) -> Result<OperationRef<'a>> {
        // Find the multiplicative inverse of the divisor.
        let divisor_inverse = self.op_felt_inverse(region, block, divisor)?;

        // Multiply by the dividend to find the quotient.
        let lhs = self.op_zext(block, dividend, self.double_felt_type());
        let rhs = self.op_zext(block, divisor_inverse.result(0)?.into(), self.double_felt_type());

        let op_mul = self.op_mul(block, lhs.result(0)?.into(), rhs.result(0)?.into());
        let op_mod = self.op_felt_modulo(block, op_mul.result(0)?.into())?;
        let op_trunc = self.op_trunc(block, op_mod.result(0)?.into(), self.felt_type());

        Ok(op_trunc)
    }

    /// Example function_type: "(i64, i64) -> i64"
    pub fn op_func<'a>(
        &'a self,
        name: &str,
        function_type: &str,
        regions: Vec<Region>,
        fn_attrs: FnAttributes,
    ) -> Result<Operation<'a>> {
        let mut attrs = Vec::with_capacity(3);

        attrs.push(NamedAttribute::new_parsed(&self.context, "function_type", function_type)?);
        attrs.push(NamedAttribute::new_parsed(&self.context, "sym_name", &format!("\"{name}\""))?);

        if !fn_attrs.public {
            attrs.push(NamedAttribute::new_parsed(
                &self.context,
                "llvm.linkage",
                "#llvm.linkage<internal>",
            )?);
        }

        if fn_attrs.emit_c_interface {
            attrs.push(NamedAttribute::new_parsed(&self.context, "llvm.emit_c_interface", "unit")?);
        }

        if fn_attrs.local {
            attrs.push(NamedAttribute::new_parsed(&self.context, "llvm.dso_local", "unit")?);
        }

        let mut llvm_attrs = Vec::with_capacity(8);

        if fn_attrs.norecurse {
            llvm_attrs.push(Attribute::parse(&self.context, "\"norecurse\"").unwrap());
        }

        if fn_attrs.inline {
            llvm_attrs.push(Attribute::parse(&self.context, "\"alwaysinline\"").unwrap());
        }

        if fn_attrs.nounwind {
            llvm_attrs.push(Attribute::parse(&self.context, "\"nounwind\"").unwrap());
        }

        if !llvm_attrs.is_empty() {
            attrs.push(NamedAttribute::new(
                Identifier::new(&self.context, "passthrough"),
                Attribute::array(&self.context, &llvm_attrs).unwrap(),
            )?);
        }

        Ok(operation::Builder::new("func.func", Location::unknown(&self.context))
            .add_attributes(&attrs)
            .add_regions(regions)
            .build())
    }

    pub fn op_return<'a>(&self, block: &'a Block, result: &[Value]) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("func.return", Location::unknown(&self.context))
                .add_operands(result)
                .build(),
        )
    }

    pub fn op_unreachable<'a>(&self, block: &'a Block) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("llvm.unreachable", Location::unknown(&self.context)).build(),
        )
    }

    pub fn op_llvm_undef<'a>(&self, block: &'a Block, ty: Type) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("llvm.mlir.undef", Location::unknown(&self.context))
                .add_results(&[ty])
                .build(),
        )
    }

    /// bitcasts a type into another
    ///
    /// https://mlir.llvm.org/docs/Dialects/LLVM/#llvmbitcast-mlirllvmbitcastop
    pub fn op_llvm_bitcast<'a>(
        &self,
        block: &'a Block,
        value: Value,
        to: Type,
    ) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("llvm.bitcast", Location::unknown(&self.context))
                .add_operands(&[value])
                .add_results(&[to])
                .build(),
        )
    }

    pub fn op_llvm_alloca<'a>(
        &self,
        block: &'a Block,
        element_type: Type,
        array_size: usize,
        // align: usize,
    ) -> Result<OperationRef<'a>> {
        let size = self.op_const(block, &array_size.to_string(), Type::integer(&self.context, 64));
        let size_res = size.result(0)?.into();
        Ok(block.append_operation(
            operation::Builder::new("llvm.alloca", Location::unknown(&self.context))
                .add_attributes(&NamedAttribute::new_parsed_vec(
                    &self.context,
                    &[
                        //("alignment", &align.to_string()),
                        ("elem_type", &element_type.to_string()),
                    ],
                )?)
                .add_operands(&[size_res])
                .add_results(&[self.llvm_ptr_type()])
                .build(),
        ))
    }

    pub fn op_llvm_const<'a>(
        &self,
        block: &'a Block,
        val: &str,
        ty: Type<'ctx>,
    ) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("llvm.mlir.constant", Location::unknown(&self.context))
                .add_results(&[ty])
                .add_attributes(&[NamedAttribute::new_parsed(&self.context, "value", val).unwrap()])
                .build(),
        )
    }

    pub fn op_llvm_nullptr<'a>(&self, block: &'a Block) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("llvm.mlir.null", Location::unknown(&self.context))
                .add_results(&[self.llvm_ptr_type()])
                .build(),
        )
    }

    pub fn op_llvm_store<'a>(
        &self,
        block: &'a Block,
        value: Value,
        addr: Value,
        // align: usize,
    ) -> Result<OperationRef<'a>> {
        Ok(block.append_operation(
            operation::Builder::new("llvm.store", Location::unknown(&self.context))
                .add_operands(&[value, addr])
                .build(),
        ))
    }

    pub fn op_llvm_load<'a>(
        &self,
        block: &'a Block,
        addr: Value,
        value_type: Type,
        // align: usize,
    ) -> Result<OperationRef<'a>> {
        Ok(block.append_operation(
            operation::Builder::new("llvm.load", Location::unknown(&self.context))
                .add_operands(&[addr])
                .add_results(&[value_type])
                .build(),
        ))
    }

    // conditional branch
    pub fn op_cond_br<'a>(
        &self,
        block: &'a Block,
        cond: Value,
        true_block: &Block,
        false_block: &Block,
        true_block_args: &[Value],
        false_block_args: &[Value],
    ) -> OperationRef<'a> {
        block.append_operation(cf::cond_br(
            &self.context,
            cond,
            true_block,
            false_block,
            true_block_args,
            false_block_args,
            Location::unknown(&self.context),
        ))
    }

    // unconditional branch
    pub fn op_br<'a>(
        &self,
        block: &'a Block,
        target_block: &Block,
        block_args: &[Value],
    ) -> OperationRef<'a> {
        block.append_operation(cf::br(target_block, block_args, Location::unknown(&self.context)))
    }

    /// inserts a value into the specified struct.
    ///
    /// The struct_llvm_type is made from `struct_type_string`
    ///
    /// The result is the struct with the value inserted.
    pub fn op_llvm_insertvalue<'a>(
        &self,
        block: &'a Block,
        index: usize,
        struct_value: Value,
        value: Value,
        struct_llvm_type: Type,
    ) -> Result<OperationRef<'a>> {
        Ok(block.append_operation(
            operation::Builder::new("llvm.insertvalue", Location::unknown(&self.context))
                .add_attributes(&[
                    self.named_attribute("position", &format!("array<i64: {}>", index))?
                ])
                .add_operands(&[struct_value, value])
                .add_results(&[struct_llvm_type])
                .build(),
        ))
    }

    /// extracts a value from the specified struct.
    ///
    /// The result is the value with tthe given type.
    pub fn op_llvm_extractvalue<'a>(
        &self,
        block: &'a Block,
        index: usize,
        struct_value: Value,
        value_type: Type,
    ) -> Result<OperationRef<'a>> {
        Ok(block.append_operation(
            operation::Builder::new("llvm.extractvalue", Location::unknown(&self.context))
                .add_attributes(&[
                    self.named_attribute("position", &format!("array<i64: {}>", index))?
                ])
                .add_operands(&[struct_value])
                .add_results(&[value_type])
                .build(),
        ))
    }

    /// llvm getelementptr with a constant offset
    pub fn op_llvm_gep<'a>(
        &self,
        block: &'a Block,
        indexes: &[usize],
        struct_ptr: Value,
        struct_type: Type,
    ) -> Result<OperationRef<'a>> {
        Ok(block.append_operation(
            operation::Builder::new("llvm.getelementptr", Location::unknown(&self.context))
                .add_attributes(&[
                    // 0 is the base offset, check out gep docs for more info
                    self.named_attribute("rawConstantIndices", &format!("array<i32: {}>", indexes.iter().join(", ")))?,
                    self.named_attribute("elem_type", &struct_type.to_string())?,
                    self.named_attribute("inbounds", "unit")?,
                ])
                .add_operands(&[struct_ptr]) // base addr
                .add_results(&[self.llvm_ptr_type()]) // always returns a opaque pointer type
                .build(),
        ))
    }

    /// gep with a offset from a value
    ///
    /// https://llvm.org/docs/LangRef.html#getelementptr-instruction
    pub fn op_llvm_gep_dynamic<'a>(
        &self,
        block: &'a Block,
        indexes: &[Value],
        base_ptr: Value,
        ptr_type: Type,
    ) -> Result<OperationRef<'a>> {
        let mut operands = vec![base_ptr];
        operands.extend(indexes);

        Ok(block.append_operation(
            operation::Builder::new("llvm.getelementptr", Location::unknown(&self.context))
                .add_attributes(&[
                    self.named_attribute("rawConstantIndices", &format!("array<i32: {}>", indexes.iter().map(|_| i32::MIN).join(", ")))?,
                    self.named_attribute("elem_type", &ptr_type.to_string())?,
                ])
                .add_operands(&operands) // base addr
                .add_results(&[self.llvm_ptr_type()]) // always returns a opaque pointer type
                .build(),
        ))
    }

    pub fn op_func_call<'a>(
        &self,
        block: &'a Block,
        name: &str,
        args: &[Value],
        results: &[Type],
    ) -> Result<OperationRef<'a>> {
        Ok(block.append_operation(
            operation::Builder::new("func.call", Location::unknown(&self.context))
                .add_attributes(&[self.named_attribute("callee", &format!("@\"{name}\""))?])
                .add_operands(args)
                .add_results(results)
                .build(),
        ))
    }

    pub fn op_llvm_call<'a>(
        &self,
        block: &'a Block,
        name: &str,
        args: &[Value],
        results: &[Type],
    ) -> Result<OperationRef<'a>> {
        Ok(block.append_operation(
            operation::Builder::new("llvm.call", Location::unknown(&self.context))
                .add_attributes(&[self.named_attribute("callee", &format!("@\"{name}\""))?])
                .add_operands(args)
                .add_results(results)
                .build(),
        ))
    }

    /// Creates a new block
    pub fn new_block(&self, args: &[Type<'ctx>]) -> Block {
        let location = Location::unknown(&self.context);
        let args: Vec<_> = args.iter().map(|x| (*x, location)).collect();
        Block::new(&args)
    }
}

// Source: https://mlir.llvm.org/docs/Dialects/ArithOps/#arithcmpi-mlirarithcmpiop
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum CmpOp {
    #[default]
    Equal,
    NotEqual,
    SignedLessThan,
    SignedLessThanEqual,
    SignedGreaterThan,
    SignedGreaterThanEqual,
    UnsignedLessThan,
    UnsignedLessThanEqual,
    UnsignedGreaterThan,
    UnsignedGreaterThanEqual,
}

impl CmpOp {
    pub const fn to_mlir_val(&self) -> &'static str {
        match self {
            Self::Equal => "0",
            Self::NotEqual => "1",
            Self::SignedLessThan => "2",
            Self::SignedLessThanEqual => "3",
            Self::SignedGreaterThan => "4",
            Self::SignedGreaterThanEqual => "5",
            Self::UnsignedLessThan => "6",
            Self::UnsignedLessThanEqual => "7",
            Self::UnsignedGreaterThan => "8",
            Self::UnsignedGreaterThanEqual => "9",
        }
    }
}
