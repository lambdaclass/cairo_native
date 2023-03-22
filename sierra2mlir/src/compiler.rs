use cairo_lang_sierra::{program::Program, ProgramParser};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::{
    dialect,
    ir::{
        operation::{self},
        Block, Location, Module, NamedAttribute, Operation, OperationRef, Region, Type, Value,
        ValueLike,
    },
    utility::{register_all_dialects, register_all_llvm_translations},
    Context,
};
use regex::Regex;
use std::{borrow::Cow, cell::RefCell, collections::HashMap, rc::Rc};

use crate::types::DEFAULT_PRIME;

pub struct Compiler<'ctx> {
    pub code: String,
    pub program: Program,
    pub context: Context,
    pub module: Module<'ctx>,
    pub main_print: bool,
}

// We represent a struct as a contiguous list of types, like sierra does, for now.
#[derive(Debug, Clone)]
pub enum SierraType<'ctx> {
    Simple(Type<'ctx>),
    Struct {
        ty: Type<'ctx>,
        field_types: Vec<Type<'ctx>>,
    },
}

#[derive(Debug, Clone)]
pub struct FunctionDef<'ctx> {
    #[allow(unused)]
    pub(crate) args: Vec<Type<'ctx>>,
    pub(crate) return_types: Vec<Type<'ctx>>,
}

/// Types, functions, etc storage.
/// This aproach works better with lifetimes.
#[derive(Debug, Default, Clone)]
pub struct Storage<'ctx> {
    pub(crate) types: HashMap<String, SierraType<'ctx>>,
    pub(crate) u8_consts: HashMap<String, String>,
    pub(crate) u16_consts: HashMap<String, String>,
    pub(crate) u32_consts: HashMap<String, String>,
    pub(crate) u64_consts: HashMap<String, String>,
    pub(crate) u128_consts: HashMap<String, String>,
    pub(crate) felt_consts: HashMap<String, String>,
    pub(crate) functions: HashMap<String, FunctionDef<'ctx>>,
}

impl<'ctx> Compiler<'ctx> {
    pub fn new(code: &str, main_print: bool) -> color_eyre::Result<Self> {
        let code = code.to_string();
        let program: Program = ProgramParser::new().parse(&code).unwrap();

        let registry = dialect::Registry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        register_all_llvm_translations(&context);
        context.get_or_load_dialect("func");
        context.get_or_load_dialect("arith");
        context.get_or_load_dialect("math");
        context.get_or_load_dialect("cf");
        context.get_or_load_dialect("scf");

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

        Ok(Self {
            code,
            program,
            context,
            module,
            main_print,
        })
    }
}

impl<'ctx> Compiler<'ctx> {
    pub fn named_attribute(&self, name: &str, attribute: &str) -> Result<NamedAttribute> {
        Ok(NamedAttribute::new_parsed(&self.context, name, attribute)?)
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

    pub fn prime_constant<'a>(&self, block: &'a Block) -> OperationRef<'a> {
        // The prime number is a double felt as it's always used for modulo.
        self.op_const(block, DEFAULT_PRIME, self.double_felt_type())
    }

    /// Only the MLIR op, doesn't do modulo.
    pub fn op_add<'a>(&self, block: &'a Block, lhs: Value, rhs: Value) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("arith.addi", Location::unknown(&self.context))
                .add_operands(&[lhs, rhs])
                .add_results(&[lhs.r#type()])
                .build(),
        )
    }

    /// Only the MLIR op, doesn't do modulo.
    pub fn op_sub<'a>(&self, block: &'a Block, lhs: Value, rhs: Value) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("arith.subi", Location::unknown(&self.context))
                .add_operands(&[lhs, rhs])
                .add_results(&[lhs.r#type()])
                .build(),
        )
    }

    /// Only the MLIR op.
    pub fn op_mul<'a>(&self, block: &'a Block, lhs: Value, rhs: Value) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("arith.muli", Location::unknown(&self.context))
                .add_operands(&[lhs, rhs])
                .add_results(&[lhs.r#type()])
                .build(),
        )
    }

    /// Only the MLIR op.
    pub fn op_rem<'a>(&self, block: &'a Block, lhs: Value, rhs: Value) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("arith.remsi", Location::unknown(&self.context))
                .add_operands(&[lhs, rhs])
                .add_results(&[lhs.r#type()])
                .build(),
        )
    }

    /// Shift right signed.
    pub fn op_shrs<'a>(&self, block: &'a Block, value: Value, shift_by: Value) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("arith.shrsi", Location::unknown(&self.context))
                .add_operands(&[value, shift_by])
                .add_results(&[value.r#type()])
                .build(),
        )
    }

    /// Only the MLIR op.
    ///
    /// todo adapt to all predicates, probs with a enum
    ///
    /// https://mlir.llvm.org/docs/Dialects/ArithOps/#arithcmpi-mlirarithcmpiop
    pub fn op_eq<'a>(&self, block: &'a Block, lhs: Value, rhs: Value) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("arith.cmpi", Location::unknown(&self.context))
                .add_attributes(&[
                    NamedAttribute::new_parsed(&self.context, "predicate", "0").unwrap()
                ]) // 0 -> eq
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

    /// New constant
    pub fn op_const<'a>(&self, block: &'a Block, val: &str, ty: Type<'ctx>) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("arith.constant", Location::unknown(&self.context))
                .add_results(&[ty])
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "value",
                    &format!("{val} : {}", ty),
                )
                .unwrap()])
                .build(),
        )
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

    /// Does modulo prime and truncates back to felt type.
    ///
    /// Arguments should be of double felt type.
    pub fn op_felt_modulo<'a>(&self, block: &'a Block, val: Value) -> Result<OperationRef<'a>> {
        let prime = self.prime_constant(block);
        let prime_val = prime.result(0)?.into();
        let op = self.op_rem(block, val, prime_val);
        let rem_val = op.result(0)?.into();
        let trunc = self.op_trunc(block, rem_val, self.felt_type());
        Ok(trunc)
    }

    /// Example function_type: "(i64, i64) -> i64"
    pub fn op_func<'a>(
        &'a self,
        name: &str,
        function_type: &str,
        regions: Vec<Region>,
        emit_c_interface: bool,
        public: bool,
    ) -> Result<Operation<'a>> {
        let mut attrs = Vec::with_capacity(3);

        attrs.push(NamedAttribute::new_parsed(
            &self.context,
            "function_type",
            function_type,
        )?);
        attrs.push(NamedAttribute::new_parsed(
            &self.context,
            "sym_name",
            &format!("\"{name}\""),
        )?);

        if !public {
            attrs.push(NamedAttribute::new_parsed(
                &self.context,
                "llvm.linkage",
                "#llvm.linkage<internal>", // found digging llvm code..
            )?);
        }

        if emit_c_interface {
            attrs.push(NamedAttribute::new_parsed(
                &self.context,
                "llvm.emit_c_interface",
                "unit",
            )?);
        }

        Ok(
            operation::Builder::new("func.func", Location::unknown(&self.context))
                .add_attributes(&attrs)
                .add_regions(regions)
                .build(),
        )
    }

    pub fn op_return<'a>(&self, block: &'a Block, result: &[Value]) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("func.return", Location::unknown(&self.context))
                .add_operands(result)
                .build(),
        )
    }

    /// creates a llvm struct
    pub fn op_llvm_struct<'a>(&self, block: &'a Block, types: &[Type]) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("llvm.mlir.undef", Location::unknown(&self.context))
                .add_results(
                    &[Type::parse(&self.context, &self.struct_type_string(types)).unwrap()],
                )
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
        let size = self.op_const(
            block,
            &array_size.to_string(),
            Type::integer(&self.context, 64),
        );
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
                .add_results(&[Type::parse(&self.context, "!llvm.ptr").unwrap()])
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

    // conditional branch
    pub fn op_cond_br<'a>(
        &self,
        block: &'a Block,
        cond: Value,
        true_block: &Block,
        false_block: &Block,
    ) -> Result<OperationRef<'a>> {
        Ok(block.append_operation(
            operation::Builder::new("cf.cond_br", Location::unknown(&self.context))
                .add_attributes(&[NamedAttribute::new_parsed(
                    &self.context,
                    "operand_segment_sizes",
                    "array<i32: 1, 0, 0>",
                )?])
                .add_operands(&[cond])
                .add_successors(&[true_block, false_block])
                .build(),
        ))
    }

    // unconditional branch
    pub fn op_br<'a>(&self, block: &'a Block, target_block: &Block) -> OperationRef<'a> {
        block.append_operation(
            operation::Builder::new("cf.br", Location::unknown(&self.context))
                .add_successors(&[target_block])
                .build(),
        )
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
        struct_llvm_type: &str,
    ) -> Result<OperationRef<'a>> {
        Ok(block.append_operation(
            operation::Builder::new("llvm.insertvalue", Location::unknown(&self.context))
                .add_attributes(&[
                    self.named_attribute("position", &format!("array<i64: {}>", index))?
                ])
                .add_operands(&[struct_value, value])
                .add_results(&[Type::parse(&self.context, struct_llvm_type).unwrap()])
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

    pub fn struct_type_string(&self, types: &[Type]) -> String {
        let types = types.iter().map(|x| x.to_string()).join(", ");
        format!("!llvm.struct<({})>", types)
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

    /// Normalizes a function name.
    ///
    /// a::a::name -> a_a_name()
    pub fn normalize_func_name(name: &str) -> Cow<str> {
        let re = Regex::new(r"[:-]+").unwrap();
        re.replace_all(name, "_")
    }

    pub fn compile(&'ctx self) -> color_eyre::Result<OperationRef<'ctx>> {
        let storage = Rc::new(RefCell::new(Storage::default()));
        self.process_types(storage.clone())?;
        self.process_libfuncs(storage.clone())?;
        self.process_functions(storage)?;
        Ok(self.module.as_operation())
    }

    pub fn compile_hardcoded_fib(&'ctx self) -> color_eyre::Result<OperationRef<'ctx>> {
        // hardcoded fib

        /*
        fn fib(a: felt, b: felt, n: felt) -> felt {
            match n {
                0 => a,
                _ => fib(b, a + b, n - 1),
            }
        }
        fn fib_mid(n: felt) {
            match n {
                0 => (),
                _ => {
                    fib(0, 1, 500);
                    fib_mid(n - 1);
                },
            }
        }
        fn main(a: felt) {
            fib_mid(100);
        }
         */

        let felt_type = self.felt_type();
        let location = Location::unknown(&self.context);
        let prime = DEFAULT_PRIME;

        let fib_function = {
            let fib_region = Region::new();
            // arguments: a: felt, n: felt
            let fib_block = self.new_block(&[felt_type, felt_type, felt_type]);
            let arg_a = fib_block.argument(0)?;
            let arg_b = fib_block.argument(1)?;
            let arg_n = fib_block.argument(2)?;

            // prepare the if comparision: n == 0
            let zero = self.op_felt_const(&fib_block, "0");
            let zero_res = zero.result(0)?.into();
            let eq = self.op_eq(&fib_block, arg_n.into(), zero_res);

            // if else regions
            let if_region = Region::new();
            let else_region = Region::new();

            // if
            {
                //  0 => (a, 0),
                let if_block = self.new_block(&[]);

                if_block.append_operation(
                    operation::Builder::new("scf.yield", location)
                        .add_operands(&[arg_a.into(), zero_res])
                        .build(),
                );

                if_region.append_block(if_block);
            }

            // else
            {
                /*
                    let (v, count) = fib(b, a + b, n - 1);
                    return (v, count + 1)
                */

                let else_block = self.new_block(&[]);

                let prime = self.op_felt_const(&else_block, prime);
                let prime_res = prime.result(0)?;

                let a_plus_b = self.op_add(&else_block, arg_a.into(), arg_b.into());
                let a_plus_b_res = a_plus_b.result(0).unwrap();

                let a_plus_b_mod = self.op_rem(&else_block, a_plus_b_res.into(), prime_res.into());
                let a_plus_b_mod_res = a_plus_b_mod.result(0)?;

                let one = self.op_felt_const(&fib_block, "1");
                let one_res = one.result(0)?.into();

                let n_minus_1 = self.op_sub(&else_block, arg_n.into(), one_res);
                let n_minus_1_res = n_minus_1.result(0).unwrap();

                let n_minus_1_mod =
                    self.op_rem(&else_block, n_minus_1_res.into(), prime_res.into());
                let n_minus_1_mod_res = n_minus_1_mod.result(0)?;

                let func_call = self.op_func_call(
                    &else_block,
                    "fib",
                    &[
                        arg_b.into(),
                        a_plus_b_mod_res.into(),
                        n_minus_1_mod_res.into(),
                    ],
                    &[felt_type, felt_type],
                )?;

                let value = func_call.result(0)?;
                let count = func_call.result(1)?;

                let count_plus_1 = self.op_add(&else_block, count.into(), one_res);
                let count_plus_1_res = count_plus_1.result(0).unwrap();

                let count_plus_1_mod =
                    self.op_rem(&else_block, count_plus_1_res.into(), prime_res.into());
                let count_plus_1_mod_res = count_plus_1_mod.result(0)?;

                else_block.append_operation(
                    operation::Builder::new("scf.yield", location)
                        .add_operands(&[value.into(), count_plus_1_mod_res.into()])
                        .build(),
                );

                else_region.append_block(else_block);
            }

            let isif = fib_block.append_operation(
                operation::Builder::new("scf.if", location)
                    .add_operands(&[eq.result(0)?.into()])
                    .add_results(&[felt_type, felt_type])
                    .add_regions(vec![if_region, else_region])
                    .build(),
            );

            let value = isif.result(0)?;
            let count = isif.result(1)?;

            self.op_return(&fib_block, &[value.into(), count.into()]);

            fib_region.append_block(fib_block);

            self.op_func(
                "fib",
                "(i256, i256, i256) -> (i256, i256)",
                vec![fib_region],
                false,
                true,
            )?
        };

        self.module.body().append_operation(fib_function);

        let fib_mid_function = {
            /*
            fn fib_mid(n: felt) {
                match n {
                    0 => (),
                    _ => {
                        fib(0, 1, 500);
                        fib_mid(n - 1);
                    },
                }
            }
            */
            let fib_mid_region = Region::new();
            // arguments: a: felt, n: felt
            let fib_block = self.new_block(&[felt_type]);
            let arg_n = fib_block.argument(0)?;

            // prepare the if comparision: n == 0
            let zero = self.op_felt_const(&fib_block, "0");
            let zero_res = zero.result(0)?.into();
            let eq = self.op_eq(&fib_block, arg_n.into(), zero_res);

            // if else regions
            let if_region = Region::new();
            let else_region = Region::new();

            // if
            {
                //  0 => (),
                let if_block = self.new_block(&[]);

                if_block.append_operation(operation::Builder::new("scf.yield", location).build());

                if_region.append_block(if_block);
            }

            // else
            {
                /*
                    fib(0, 1, 500);
                    fib_mid(n - 1);
                */

                let else_block = self.new_block(&[]);

                let one = self.op_felt_const(&fib_block, "1");
                let one_res = one.result(0)?.into();
                let times = self.op_felt_const(&fib_block, "500");
                let times_res = times.result(0)?.into();

                self.op_func_call(
                    &else_block,
                    "fib",
                    &[zero_res, one_res, times_res],
                    &[felt_type, felt_type],
                )?;

                let n_minus_1 = self.op_sub(&else_block, arg_n.into(), one_res);
                let n_minus_1_res = n_minus_1.result(0).unwrap();

                let prime = self.op_felt_const(&else_block, prime);
                let prime_res = prime.result(0)?;

                let n_minus_1_mod =
                    self.op_rem(&else_block, n_minus_1_res.into(), prime_res.into());
                let n_minus_1_mod_res = n_minus_1_mod.result(0)?;

                self.op_func_call(&else_block, "fib_mid", &[n_minus_1_mod_res.into()], &[])?;

                else_block.append_operation(operation::Builder::new("scf.yield", location).build());

                else_region.append_block(else_block);
            }

            fib_block.append_operation(
                operation::Builder::new("scf.if", location)
                    .add_operands(&[eq.result(0)?.into()])
                    .add_results(&[])
                    .add_regions(vec![if_region, else_region])
                    .build(),
            );

            self.op_return(&fib_block, &[]);

            fib_mid_region.append_block(fib_block);

            self.op_func("fib_mid", "(i256) -> ()", vec![fib_mid_region], false, true)?
        };

        self.module.body().append_operation(fib_mid_function);

        let main_function = {
            let region = Region::new();
            let block = Block::new(&[]);

            let n_arg = self.op_felt_const(&block, "100");
            let n_arg_res = n_arg.result(0)?.into();

            self.op_func_call(&block, "fib_mid", &[n_arg_res], &[])?;

            let main_ret = self.op_const(&block, "0", self.i32_type());

            self.op_return(&block, &[main_ret.result(0)?.into()]);

            region.append_block(block);

            self.op_func("main", "() -> i32", vec![region], true, true)?
        };

        self.module.body().append_operation(main_function);
        let op = self.module.as_operation();

        if op.verify() {
            Ok(op)
        } else {
            Err(color_eyre::eyre::eyre!("error verifiying"))
        }
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
                            (
                                "operand_segment_sizes",
                                "array<i32: 0, 1, 1, 1, 1, 1, 1, 1, 2>",
                            ),
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

            self.op_func("main", "() -> i32", vec![region], true, true)?
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
