use melior::{
    ir::{
        attribute::{DenseI32ArrayAttribute, FlatSymbolRefAttribute},
        operation::{Operation, OperationBuilder},
        Attribute, Identifier, Location, Type, Value,
    },
    Context,
};

use crate::error::Result;

pub enum LLVMCalleType<'c, 'a> {
    Symbol(&'a str),
    FuncPtr(Value<'c, 'a>),
}

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
pub fn llvm_call<'c, 'a>(
    context: &'c Context,
    calle_type: LLVMCalleType<'c, 'a>,
    args: &[Value<'c, 'a>],
    attrs: &[(Identifier<'c>, Attribute<'c>)],
    ret_types: &[Type<'c>],
    location: Location<'c>,
) -> Result<Operation<'c>> {
    let op = {
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
        let op = match calle_type {
            LLVMCalleType::Symbol(sym) => OperationBuilder::new("llvm.call", location)
                .add_attributes(&[
                    (
                        Identifier::new(context, "callee"),
                        FlatSymbolRefAttribute::new(context, sym).into(),
                    ),
                    (
                        Identifier::new(context, "operandSegmentSizes"),
                        DenseI32ArrayAttribute::new(context, &[args.len() as i32, 0]).into(),
                    ),
                ]),
            LLVMCalleType::FuncPtr(ptr) => OperationBuilder::new("llvm.call", location)
                .add_attributes(&[(
                    Identifier::new(context, "operandSegmentSizes"),
                    DenseI32ArrayAttribute::new(context, &[args.len() as i32 + 1, 0]).into(),
                )])
                .add_operands(&[ptr]),
        };

        // We don't use op-bundle-operands in the call, so "op_bundle_sizes" should be empty.
        op.add_attributes(&[(
            Identifier::new(context, "op_bundle_sizes"),
            DenseI32ArrayAttribute::new(context, &[]).into(),
        )])
    };

    Ok(op
        .add_operands(args)
        .add_attributes(attrs)
        .add_results(ret_types)
        .build()?)
}
