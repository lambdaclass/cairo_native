use std::{path::Path, fs::File};

use cairo_lang_sierra::{program::Program, ProgramParser};
use melior::{
    dialect,
    ir::{
        block,
        operation::{self, ResultValue},
        Attribute, Block, Identifier, Location, Module, Operation, OperationRef, Region, Type,
        Value,
    },
    utility::register_all_dialects,
    Context,
};

pub struct Compiler<'ctx> {
    pub code: String,
    pub program: Program,
    pub context: Context,
    pub module: Module<'ctx>,
}

impl<'ctx> Compiler<'ctx> {
    pub fn new(code: &str) -> color_eyre::Result<Self> {
        let code = code.to_string();
        let program: Program = ProgramParser::new().parse(&code).unwrap();

        let registry = dialect::Registry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.get_or_load_dialect("func"); // needed?
        context.get_or_load_dialect("arith");
        context.get_or_load_dialect("math");
        context.get_or_load_dialect("cf");
        context.get_or_load_dialect("scf");

        let felt_type = Type::integer(&context, 256);
        let location = Location::unknown(&context);
        let module = Module::new(location);

        Ok(Self {
            code,
            program,
            context,
            module,
        })
    }

    pub fn compile_from_code(code: &str) -> color_eyre::Result<()> {
        let code = code.to_string();
        let program: Program = ProgramParser::new().parse(&code).unwrap();

        let registry = dialect::Registry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.get_or_load_dialect("func"); // needed?
        context.get_or_load_dialect("arith");
        context.get_or_load_dialect("math");
        context.get_or_load_dialect("cf");

        let felt_type = Type::integer(&context, 256);
        let location = Location::unknown(&context);
        let module = Module::new(location);

        let compiler = Self {
            code,
            program,
            context,
            module,
        };

        Ok(())
    }
}

impl<'ctx> Compiler<'ctx> {
    pub fn named_attribute(
        &'ctx self,
        name: &str,
        attribute: &str,
    ) -> (Identifier<'ctx>, Attribute<'ctx>) {
        (
            Identifier::new(&self.context, name),
            Attribute::parse(&self.context, attribute).unwrap(),
        )
    }

    pub fn felt_type(&'ctx self) -> Type<'ctx> {
        Type::integer(&self.context, 256)
    }

    pub fn bool_type(&'ctx self) -> Type<'ctx> {
        Type::integer(&self.context, 1)
    }

    /// Only the MLIR op, doesn't do modulo.
    pub fn op_felt_add(
        &'ctx self,
        block: &'ctx Block<'ctx>,
        lhs: Value<'ctx>,
        rhs: Value<'ctx>,
    ) -> OperationRef<'ctx> {
        block.append_operation(
            operation::Builder::new("arith.addi", Location::unknown(&self.context))
                .add_operands(&[lhs, rhs])
                .add_results(&[self.felt_type()])
                .build(),
        )
    }

    /// Only the MLIR op, doesn't do modulo.
    pub fn op_felt_sub(
        &'ctx self,
        block: &'ctx Block<'ctx>,
        lhs: Value<'ctx>,
        rhs: Value<'ctx>,
    ) -> OperationRef<'ctx> {
        block.append_operation(
            operation::Builder::new("arith.subi", Location::unknown(&self.context))
                .add_operands(&[lhs, rhs])
                .add_results(&[self.felt_type()])
                .build(),
        )
    }

    /// Only the MLIR op.
    pub fn op_felt_mul(
        &'ctx self,
        block: &'ctx Block,
        lhs: Value,
        rhs: Value,
    ) -> OperationRef<'ctx> {
        block.append_operation(
            operation::Builder::new("arith.muli", Location::unknown(&self.context))
                .add_operands(&[lhs, rhs])
                .add_results(&[self.felt_type()])
                .build(),
        )
    }

    /// Only the MLIR op.
    pub fn op_felt_rem(
        &'ctx self,
        block: &'ctx Block,
        lhs: Value,
        rhs: Value,
    ) -> OperationRef<'ctx> {
        block.append_operation(
            operation::Builder::new("arith.remsi", Location::unknown(&self.context))
                .add_operands(&[lhs, rhs])
                .add_results(&[self.felt_type()])
                .build(),
        )
    }

    /// Only the MLIR op.
    ///
    /// todo adapt to all predicates, probs with a enum
    ///
    /// https://mlir.llvm.org/docs/Dialects/ArithOps/#arithcmpi-mlirarithcmpiop
    pub fn op_eq(&'ctx self, block: &'ctx Block, lhs: Value, rhs: Value) -> OperationRef<'ctx> {
        block.append_operation(
            operation::Builder::new("arith.cmpi", Location::unknown(&self.context))
                .add_attributes(&[self.named_attribute("predicate", "0")]) // 0 -> eq
                .add_operands(&[lhs, rhs])
                .add_results(&[Type::integer(&self.context, 1)])
                .build(),
        )
    }

    /// New felt constant
    pub fn op_felt_const(&'ctx self, block: &'ctx Block, val: &str) -> OperationRef<'ctx> {
        block.append_operation(
            operation::Builder::new("arith.constant", Location::unknown(&self.context))
                .add_results(&[self.felt_type()])
                .add_attributes(&[(
                    Identifier::new(&self.context, "value"),
                    Attribute::parse(&self.context, &format!("{val} : {}", self.felt_type()))
                        .unwrap(),
                )])
                .build(),
        )
    }

    ///
    /// Example function_type: "(i64, i64) -> i64"
    pub fn op_func(
        &'ctx self,
        name: &str,
        function_type: &str,
        regions: Vec<Region>,
    ) -> Operation<'ctx> {
        operation::Builder::new("func.func", Location::unknown(&self.context))
            .add_attributes(&[
                (
                    Identifier::new(&self.context, "function_type"),
                    Attribute::parse(&self.context, function_type).unwrap(),
                ),
                (
                    Identifier::new(&self.context, "sym_name"),
                    Attribute::parse(&self.context, &format!("\"{name}\"")).unwrap(),
                ),
            ])
            .add_regions(regions)
            .build()
    }

    pub fn op_return(&'ctx self, block: &'ctx Block, result: &[Value]) -> OperationRef<'ctx> {
        block.append_operation(
            operation::Builder::new("func.return", Location::unknown(&self.context))
                .add_operands(result)
                .build(),
        )
    }

    pub fn op_func_call(
        &'ctx self,
        block: &'ctx Block,
        name: &str,
        args: &[Value],
        results: &[Type],
    ) -> OperationRef<'ctx> {
        block.append_operation(
            operation::Builder::new("func.call", Location::unknown(&self.context))
                .add_attributes(&[self.named_attribute("callee", name)])
                .add_operands(args)
                .add_results(results)
                .build(),
        )
    }

    /// Creates a new block
    pub fn new_block(&'ctx self, args: &[Type<'ctx>]) -> Block<'ctx> {
        let location = Location::unknown(&self.context);
        let args: Vec<_> = args.iter().map(|x| (*x, location)).collect();
        Block::new(&args)
    }

    pub fn run(&'ctx self) -> color_eyre::Result<OperationRef<'ctx>> {
        // hardcoded fib

        /*
        fn fib(a: felt, b: felt, n: felt) -> (felt, felt) {
            match n {
                0 => (a, 0),
                _ => {
                    let (v, count) = fib(b, a + b, n - 1);
                    (v, count + 1)
                },
            }
        }
         */

        let felt_type = self.felt_type();
        let location = Location::unknown(&self.context);

        let function = {
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

                let yield_op = if_block.append_operation(
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

                let a_plus_b = self.op_felt_add(&else_block, arg_a.into(), arg_b.into());
                let a_plus_b_res = a_plus_b.result(0).unwrap();

                let one = self.op_felt_const(&fib_block, "1");
                let one_res = one.result(0)?.into();

                let n_minus_1 = self.op_felt_sub(&else_block, arg_a.into(), one_res);
                let n_minus_1_res = n_minus_1.result(0).unwrap();

                let func_call = self.op_func_call(
                    &else_block,
                    "@fib",
                    &[arg_b.into(), a_plus_b_res.into(), n_minus_1_res.into()],
                    &[felt_type, felt_type],
                );

                let value = func_call.result(0)?;
                let count = func_call.result(1)?;

                let count_plus_1 = self.op_felt_add(&else_block, count.into(), one_res);
                let count_plus_1_res = count_plus_1.result(0).unwrap();

                let yield_op = else_block.append_operation(
                    operation::Builder::new("scf.yield", location)
                        .add_operands(&[value.into(), count_plus_1_res.into()])
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

            let func = self.op_func(
                "fib",
                "(i256, i256, i256) -> (i256, i256)",
                vec![fib_region],
            );

            func
        };

        self.module.body().append_operation(function);

        let op = self.module.as_operation();

        if op.verify() {
            Ok(op)
        } else {
            Err(color_eyre::eyre::eyre!("error verifiying"))
        }
    }
}
