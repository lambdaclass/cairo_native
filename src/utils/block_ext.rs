use melior::{
    dialect::llvm,
    helpers::BuiltinBlockExt,
    ir::{
        attribute::{DenseI32ArrayAttribute, FlatSymbolRefAttribute},
        operation::OperationBuilder,
        Attribute, Block, Identifier, Location, Type, Value,
    },
    Context,
};

use crate::error::Result;

pub(crate) enum LLVMCalleType<'ctx, 'this> {
    Symbol(&'this str),
    FuncPtr(Value<'ctx, 'this>),
}

pub(crate) trait BlockExt<'ctx, 'this> {
    fn llvm_call(
        &'this self,
        context: &'ctx Context,
        calle_type: LLVMCalleType<'ctx, 'this>,
        args: &[Value<'ctx, 'this>],
        attrs: &[(Identifier<'ctx>, Attribute<'ctx>)],
        ret_types: &[Type<'ctx>],
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, 'this>>;
}

impl<'ctx, 'this> BlockExt<'ctx, 'this> for Block<'ctx> {
    /// Helper function to perform an `llvm.call` operation.
    ///
    /// The function allows to use either the function pointer or it's symbol. It will also calculate
    /// the `operandSegmentSizes` attribute, needed by the operation, from `calle_type` and the
    /// arguments' size. This is to avoid having to calculate this attribute by hand.
    ///
    /// # Safety
    ///
    /// The `attrs` argument should no contain the calle or the operandSegmentSizes attributes as specified
    /// by the function itself. Adding them result in an error for attribute duplication.
    ///
    /// If the call was to be performed with the function pointer, the latter should not be included in `args`
    /// as part of the call operands since that is already handled by the function itself.
    fn llvm_call(
        &'this self,
        context: &'ctx Context,
        calle_type: LLVMCalleType<'ctx, 'this>,
        args: &[Value<'ctx, 'this>],
        attrs: &[(Identifier<'ctx>, Attribute<'ctx>)],
        ret_types: &[Type<'ctx>],
        location: Location<'ctx>,
    ) -> Result<Value<'ctx, 'this>> {
        let op = {
            let op = OperationBuilder::new("llvm.call", location);

            // llvm.call is an operation that takes two groups of variadic operands (calle-operands and "op-bundle-operands").
            //
            // * The calle-operans are the operands we are used to:
            //     1. function-pointer (if it was an indirect call).
            //     2. function-args (if any).
            //
            // * The op-bundle-operands are a way to specify operands without changing the function's firm.
            //
            // Since we have 2 groups of variadic operands, we are expected to tell the amount of operands for each group. We
            // do this by specifying the attribute "operandSegmentSizes". If we were to call the function from a pointer, we would 
            // have 1 + <function-operands> for the "calle-operands" group and 0 from the op-bundle-operands group.
            match calle_type {
                LLVMCalleType::Symbol(sym) => op.add_attributes(&[
                    (
                        Identifier::new(context, "callee"),
                        FlatSymbolRefAttribute::new(context, sym).into(),
                    ),
                    (
                        Identifier::new(context, "operandSegmentSizes"),
                        DenseI32ArrayAttribute::new(context, &[args.len() as i32, 0]).into(),
                    ),
                    (
                        Identifier::new(context, "op_bundle_sizes"),
                        DenseI32ArrayAttribute::new(context, &[]).into(),
                    ),
                ]),
                LLVMCalleType::FuncPtr(ptr) => op
                    .add_attributes(&[
                        (
                            Identifier::new(context, "operandSegmentSizes"),
                            DenseI32ArrayAttribute::new(context, &[args.len() as i32 + 1, 0])
                                .into(),
                        ),
                        (
                            Identifier::new(context, "op_bundle_sizes"),
                            DenseI32ArrayAttribute::new(context, &[]).into(),
                        ),
                    ])
                    .add_operands(&[ptr]),
            }
        };

        Ok(self.append_op_result(
            op.add_attributes(attrs)
                .add_operands(args)
                .add_results(&[llvm::r#type::r#struct(context, ret_types, false)])
                .build()?,
        )?)
    }
}
