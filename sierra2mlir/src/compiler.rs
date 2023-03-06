use std::path::Path;

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
        block: &'ctx Block,
        lhs: Value,
        rhs: Value,
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
        block: &'ctx Block,
        lhs: Value,
        rhs: Value,
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
        block: &'ctx Block,
        name: &str,
        function_type: &str,
    ) -> OperationRef<'ctx> {
        let region = Region::new();
        block.append_operation(
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
                .add_regions(vec![region])
                .build(),
        )
    }

    pub fn op_return(&'ctx self, block: &'ctx Block, result: &[Value]) -> OperationRef<'ctx> {
        block.append_operation(
            operation::Builder::new("func.return", Location::unknown(&self.context))
                .add_operands(result)
                .build(),
        )
    }

    /// Creates a new block
    pub fn new_block(&'ctx self, args: &[Type<'ctx>]) -> Block<'ctx> {
        let location = Location::unknown(&self.context);
        let args: Vec<_> = args.iter().map(|x| (*x, location)).collect();
        Block::new(&args)
    }

    pub fn run(&'ctx self) -> color_eyre::Result<()> {
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
            // arguments: a: felt, n: felt
            let block = self.new_block(&[felt_type, felt_type]);
            let arg_a = block.argument(0)?;
            let arg_n = block.argument(1)?;

            // if n == 0

            let zero = self.op_felt_const(&block, "0");
            dbg!(zero.to_string());
            let eq = self.op_eq(&block, arg_n.into(), zero.result(0)?.into());

            let isif = block.append_operation(
                operation::Builder::new("scf.if", location)
                    .add_operands(&[eq.result(0)?.into()])
                    .add_results(&[felt_type, felt_type])
                    .build(),
            );

            dbg!(isif.to_string());
            dbg!(block.to_string());

            /*
            block.append_operation(
                operation::Builder::new("func.return", Location::unknown(&context))
                    .add_operands(&[sum.result(0)?.into()])
                    .build(),
            );

            region.append_block(block);

            operation::Builder::new("func.func", Location::unknown(&context))
                .add_attributes(&[
                    (
                        Identifier::new(&context, "function_type"),
                        Attribute::parse(&context, "(i64, i64) -> i64").unwrap(),
                    ),
                    (
                        Identifier::new(&context, "sym_name"),
                        Attribute::parse(&context, "\"add\"").unwrap(),
                    ),
                ])
                .add_regions(vec![region])
                .build()

            */
            todo!()
        };

        /*
        module.body().append_operation(function);

        let op = module.as_operation();
        dbg!(op.verify());

        let x = op.to_string();
        dbg!(x);
         */

        Ok(())
    }
}
