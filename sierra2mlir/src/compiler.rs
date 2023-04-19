use cairo_lang_sierra::program::Program;
use color_eyre::Result;
use itertools::Itertools;
use melior_next::{
    dialect::{self, arith, cf, llvm},
    ir::{
        operation::{self},
        Attribute, Block, Identifier, Location, Module, NamedAttribute, Operation, OperationRef,
        Region, Type, TypeLike, Value, ValueLike,
    },
    utility::{register_all_dialects, register_all_llvm_translations},
    Context,
};
use std::{cmp::Ordering, collections::HashMap};

use crate::{
    libfuncs::lib_func_def::SierraLibFunc, types::DEFAULT_PRIME,
    userfuncs::user_func_def::UserFuncDef,
};

pub struct Compiler<'ctx> {
    pub program: &'ctx Program,
    pub context: Context,
    pub module: Module<'ctx>,
    pub main_print: bool,
    pub print_fd: i32,
}

// We represent a struct as a contiguous list of types, like sierra does, for now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SierraType<'ctx> {
    Simple(Type<'ctx>),
    Struct {
        ty: Type<'ctx>,
        field_types: Vec<Self>,
    },
    Enum {
        ty: Type<'ctx>,
        tag_type: Type<'ctx>,
        storage_bytes_len: u32,
        storage_type: Type<'ctx>, // the array
        variants_types: Vec<Self>,
    },
    Array {
        /// (u32, u32, ptr)
        ///
        /// (length, capacity, data)
        ty: Type<'ctx>,
        len_type: Type<'ctx>, // type of length and capacity: u32
        element_type: Box<Self>,
    },
}

impl<'ctx> SierraType<'ctx> {
    pub fn get_width(&self) -> u32 {
        match self {
            SierraType::Simple(ty) => ty.get_width().unwrap_or(0),
            SierraType::Struct { ty: _, field_types } => {
                let mut width = 0;
                for ty in field_types {
                    width += ty.get_width();
                }
                width
            }
            SierraType::Enum {
                ty: _,
                tag_type,
                storage_bytes_len: storage_type_len,
                storage_type: _,
                variants_types: _,
            } => tag_type.get_width().unwrap() + (storage_type_len * 8),
            SierraType::Array { ty: _, len_type, element_type: _ } => {
                // 64 is the pointer size, assuming here
                // TODO: find a better way to find the pointer size? it would require getting the context here
                len_type.get_width().unwrap() * 2 + 64
            }
        }
    }

    pub fn get_felt_representation_width(&self) -> usize {
        match self {
            SierraType::Simple(_) => 1,
            SierraType::Struct { ty: _, field_types } => {
                field_types.iter().map(Self::get_felt_representation_width).sum()
            }
            SierraType::Enum {
                ty: _,
                tag_type: _,
                storage_bytes_len: _,
                storage_type: _,
                variants_types,
            } => {
                1 + variants_types
                    .iter()
                    .map(Self::get_felt_representation_width)
                    .max()
                    .unwrap_or(0)
            }
            SierraType::Array { .. } => 2,
        }
    }

    /// Returns the MLIR type of this sierra type
    pub const fn get_type(&self) -> Type {
        match self {
            Self::Simple(ty) => *ty,
            Self::Struct { ty, field_types: _ } => *ty,
            Self::Enum {
                ty,
                tag_type: _,
                storage_bytes_len: _,
                storage_type: _,
                variants_types: _,
            } => *ty,
            Self::Array { ty, .. } => *ty,
        }
    }

    pub fn get_type_location(&self, context: &'ctx Context) -> (Type<'ctx>, Location<'ctx>) {
        (
            match self {
                Self::Simple(ty) => *ty,
                Self::Struct { ty, field_types: _ } => *ty,
                Self::Enum {
                    ty,
                    tag_type: _,
                    storage_bytes_len: _,
                    storage_type: _,
                    variants_types: _,
                } => *ty,
                Self::Array { ty, .. } => *ty,
            },
            Location::unknown(context),
        )
    }

    /// Returns a vec of field types if this is a struct type.
    pub fn get_field_types(&self) -> Option<Vec<Type>> {
        match self {
            SierraType::Simple(_) => None,
            SierraType::Struct { ty: _, field_types } => {
                Some(field_types.iter().map(|x| x.get_type()).collect_vec())
            }
            SierraType::Enum {
                ty: _,
                tag_type: _,
                storage_bytes_len: _,
                storage_type: _,
                variants_types,
            } => Some(variants_types.iter().map(|x| x.get_type()).collect()),
            SierraType::Array { .. } => None,
        }
    }

    pub fn get_field_sierra_types(&self) -> Option<&[Self]> {
        match self {
            SierraType::Simple(_) => None,
            SierraType::Struct { ty: _, field_types } => Some(field_types),
            SierraType::Enum {
                ty: _,
                tag_type: _,
                storage_bytes_len: _,
                storage_type: _,
                variants_types,
            } => Some(variants_types),
            SierraType::Array { .. } => None,
        }
    }

    /// gets the sierra array type for the given element
    pub fn get_array_type<'c>(
        compiler: &'c Compiler<'c>,
        element: SierraType<'c>,
    ) -> SierraType<'c> {
        SierraType::Array {
            ty: compiler.struct_type(&[
                compiler.u32_type(),
                compiler.u32_type(),
                compiler.llvm_ptr_type(),
            ]),
            len_type: compiler.u32_type(),
            element_type: Box::new(element),
        }
    }
}

/// Types, functions, etc storage.
/// This aproach works better with lifetimes.
#[derive(Debug, Default, Clone)]
pub struct Storage<'ctx> {
    pub(crate) types: HashMap<String, SierraType<'ctx>>,
    pub(crate) libfuncs: HashMap<String, SierraLibFunc<'ctx>>,
    pub(crate) userfuncs: HashMap<String, UserFuncDef<'ctx>>,
}

/// Function attributes to fine tune the generated definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FnAttributes {
    /// the function linkage
    pub public: bool,
    pub emit_c_interface: bool,
    /// true = dso_local https://llvm.org/docs/LangRef.html#runtime-preemption-specifiers
    pub local: bool,
    pub inline: bool,
    /// This function attribute indicates that the function does not call itself either directly or indirectly down any possible call path. This produces undefined behavior at runtime if the function ever does recurse.
    pub norecurse: bool,
    /// This function attribute indicates that the function never raises an exception. If the function does raise an exception, its runtime behavior is undefined.
    pub nounwind: bool, // never raises exception
}

impl Default for FnAttributes {
    fn default() -> Self {
        Self {
            public: true,
            emit_c_interface: false,
            local: true,
            inline: false,
            norecurse: false,
            nounwind: false,
        }
    }
}

impl FnAttributes {
    pub const fn libfunc(panics: bool, inline: bool) -> Self {
        Self {
            public: false,
            emit_c_interface: false,
            local: true,
            inline,
            norecurse: true,
            nounwind: !panics,
        }
    }
}

impl<'ctx> Compiler<'ctx> {
    pub fn new(
        program: &'ctx Program,
        main_print: bool,
        print_fd: i32,
    ) -> color_eyre::Result<Self> {
        let registry = dialect::Registry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        register_all_llvm_translations(&context);

        let location = Location::unknown(&context);

        let region = Region::new();
        let block = Block::new(&[]);
        region.append_block(block);
        let module_op = operation::Builder::new("builtin.module", location)
            /*
            .add_attributes(&[NamedAttribute::new_parsed(
                &context,
                "gpu.container_module",
                "unit",
            )
            .unwrap()])
            */
            .add_regions(vec![region])
            .build();

        let module = Module::from_operation(module_op).unwrap();

        Ok(Self { program, context, module, main_print, print_fd })
    }
}

impl<'ctx> Compiler<'ctx> {
    pub fn named_attribute(&self, name: &str, attribute: &str) -> Result<NamedAttribute> {
        Ok(NamedAttribute::new_parsed(&self.context, name, attribute)?)
    }

    pub fn llvm_ptr_type(&self) -> Type {
        llvm::r#type::opaque_pointer(&self.context)
    }

    pub fn felt_type(&self) -> Type {
        Type::integer(&self.context, 256)
    }

    pub fn double_felt_type(&self) -> Type {
        Type::integer(&self.context, 512)
    }

    pub fn i32_type(&self) -> Type {
        Type::integer(&self.context, 32)
    }

    pub fn bool_type(&self) -> Type {
        Type::integer(&self.context, 1)
    }

    pub fn u8_type(&self) -> Type {
        Type::integer(&self.context, 8)
    }

    pub fn u16_type(&self) -> Type {
        Type::integer(&self.context, 16)
    }

    pub fn u32_type(&self) -> Type {
        Type::integer(&self.context, 32)
    }

    pub fn u64_type(&self) -> Type {
        Type::integer(&self.context, 64)
    }

    pub fn u128_type(&self) -> Type {
        Type::integer(&self.context, 128)
    }

    pub fn u256_type(&self) -> Type {
        Type::integer(&self.context, 256)
    }

    /// Type `Bitwise`. Points to the bitwise builtin pointer. Since we're not respecting the
    /// classic segments this type makes no sense, therefore it's implemented as `()`.
    pub fn bitwise_type(&self) -> Type {
        Type::none(&self.context)
    }

    /// Type `Bitwise`. Points to the range check builtin pointer. Since we're not respecting the
    /// classic segments this type makes no sense, therefore it's implemented as `()`.
    pub fn range_check_type(&self) -> Type {
        Type::none(&self.context)
    }

    /// The enum struct type. Needed due to some libfuncs using it.
    ///
    /// The tag value is the boolean value: 0, 1
    ///
    /// Sierra: type core::bool = Enum<ut@core::bool, Unit, Unit>;
    pub fn boolean_enum_type(&self) -> Type {
        self.llvm_struct_type(&[self.u16_type(), self.llvm_array_type(self.u8_type(), 0)])
    }

    pub fn llvm_struct_type<'c>(&'c self, fields: &[Type<'c>]) -> Type {
        llvm::r#type::r#struct(&self.context, fields, false)
    }

    pub fn llvm_array_type<'c>(&'c self, element_type: Type<'c>, len: u32) -> Type {
        llvm::r#type::array(element_type, len)
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

    /// creates a llvm struct
    pub fn op_llvm_struct_from_types<'a>(
        &self,
        block: &'a Block,
        types: &[Type],
    ) -> OperationRef<'a> {
        self.op_llvm_struct(
            block,
            Type::parse(&self.context, &self.struct_type_string(types)).unwrap(),
        )
    }

    pub fn op_llvm_struct<'a>(&self, block: &'a Block, ty: Type) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("llvm.mlir.undef", Location::unknown(&self.context))
                .add_results(&[ty])
                .build(),
        )
    }

    /// creates a llvm struct allocating on the stack
    ///
    /// use getelementptr instead of extractvalue
    pub fn op_llvm_struct_alloca<'a>(
        &self,
        block: &'a Block,
        types: &[Type],
    ) -> Result<OperationRef<'a>> {
        self.op_llvm_alloca(
            block,
            Type::parse(&self.context, &self.struct_type_string(types)).unwrap(),
            1,
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

    pub fn struct_type_string(&self, types: &[Type]) -> String {
        let types = types.iter().map(|x| x.to_string()).join(", ");
        format!("!llvm.struct<({})>", types)
    }

    pub fn struct_type(&self, types: &[Type]) -> Type {
        Type::parse(&self.context, &self.struct_type_string(types)).unwrap()
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

    pub fn compile(&'ctx self) -> color_eyre::Result<OperationRef<'ctx>> {
        self.create_libc_funcs()?;
        if self.print_fd > 0 {
            self.create_printf()?;
        }
        let mut storage = Storage::default();
        self.process_types(&mut storage)?;
        self.process_libfuncs(&mut storage)?;
        self.process_functions(&mut storage)?;
        self.process_statements(&mut storage)?;
        Ok(self.module.as_operation())
    }

    pub fn compile_hardcoded_gpu(&'ctx self) -> color_eyre::Result<OperationRef<'ctx>> {
        /*
        fn main(a: i32, b: i32) -> i32 {
            a * b
        }
        */

        let i32_type = Type::integer(&self.context, 32);
        let i256_type = Type::integer(&self.context, 256);
        let index_type = Type::index(&self.context);
        let location = Location::unknown(&self.context);

        let gpu_module = {
            let module_region = Region::new();
            let module_block = Block::new(&[]);

            let region = Region::new();
            let block = Block::new(&[(i256_type, location), (i256_type, location)]);

            let arg1 = block.argument(0)?;
            let arg2 = block.argument(1)?;

            let res = self.op_add(&block, arg1.into(), arg2.into());
            let res_result = res.result(0)?;

            let trunc_op = block.append_operation(
                operation::Builder::new("arith.trunci", Location::unknown(&self.context))
                    .add_operands(&[res_result.into()])
                    .add_results(&[i32_type])
                    .build(),
            );

            let trunc_op_res = trunc_op.result(0)?;

            block.append_operation(
                operation::Builder::new("gpu.printf", Location::unknown(&self.context))
                    .add_attributes(&[self.named_attribute("format", r#""suma: %d ""#)?])
                    .add_operands(&[trunc_op_res.into()])
                    .build(),
            );

            // kernels always return void
            block.append_operation(
                operation::Builder::new("gpu.return", Location::unknown(&self.context)).build(),
            );

            region.append_block(block);

            let func = operation::Builder::new("gpu.func", Location::unknown(&self.context))
                .add_attributes(&NamedAttribute::new_parsed_vec(
                    &self.context,
                    &[
                        ("function_type", "(i256, i256) -> ()"),
                        ("sym_name", "\"kernel1\""),
                        ("gpu.kernel", "unit"),
                    ],
                )?)
                .add_regions(vec![region])
                .build();

            module_block.append_operation(func);

            module_block.append_operation(
                operation::Builder::new("gpu.module_end", Location::unknown(&self.context)).build(),
            );

            module_region.append_block(module_block);

            let gpu_module =
                operation::Builder::new("gpu.module", Location::unknown(&self.context))
                    .add_attributes(&[self.named_attribute("sym_name", "\"kernels\"")?])
                    .add_regions(vec![module_region])
                    .build();

            gpu_module
        };

        self.module.body().append_operation(gpu_module);

        let main_function = {
            let region = Region::new();
            let block = Block::new(&[]);

            let index_op = block.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_results(&[index_type])
                    .add_attributes(&[
                        self.named_attribute("value", &format!("1 : {}", index_type))?
                    ])
                    .build(),
            );
            let index_value = index_op.result(0)?.into();

            let dynamic_shared_memory_size_op = block.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_results(&[i32_type])
                    .add_attributes(&[self.named_attribute("value", &format!("0 : {}", i32_type))?])
                    .build(),
            );
            let dynamic_shared_memory_size = dynamic_shared_memory_size_op.result(0)?.into();

            let arg1_op = block.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_results(&[i256_type])
                    .add_attributes(
                        &[self.named_attribute("value", &format!("4 : {}", i256_type))?],
                    )
                    .build(),
            );
            let arg1 = arg1_op.result(0)?;
            let arg2_op = block.append_operation(
                operation::Builder::new("arith.constant", Location::unknown(&self.context))
                    .add_results(&[i256_type])
                    .add_attributes(
                        &[self.named_attribute("value", &format!("2 : {}", i256_type))?],
                    )
                    .build(),
            );
            let arg2 = arg2_op.result(0)?;

            let gpu_launch =
                operation::Builder::new("gpu.launch_func", Location::unknown(&self.context))
                    .add_attributes(&NamedAttribute::new_parsed_vec(
                        &self.context,
                        &[
                            ("kernel", "@kernels::@kernel1"),
                            ("operand_segment_sizes", "array<i32: 0, 1, 1, 1, 1, 1, 1, 1, 2>"),
                        ],
                    )?)
                    .add_operands(&[
                        index_value,
                        index_value,
                        index_value,
                        index_value,
                        index_value,
                        index_value,
                        dynamic_shared_memory_size,
                        arg1.into(),
                        arg2.into(),
                    ])
                    .build();

            block.append_operation(gpu_launch);

            let main_ret = self.op_const(&block, "0", self.i32_type());
            self.op_return(&block, &[main_ret.result(0)?.into()]);
            region.append_block(block);

            self.op_func(
                "main",
                "() -> i32",
                vec![region],
                FnAttributes { public: true, emit_c_interface: true, ..Default::default() },
            )?
        };

        self.module.body().append_operation(main_function);

        let op = self.module.as_operation();

        if op.verify() {
            Ok(op)
        } else {
            Err(color_eyre::eyre::eyre!("error verifiying"))
        }
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
