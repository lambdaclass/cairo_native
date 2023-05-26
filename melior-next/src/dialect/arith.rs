use crate::{
    ir::{operation, Location, NamedAttribute, Operation, Type, Value},
    Context,
};

macro_rules! impl_arith_binary_op {
    ($name:ident, $op:literal) => {
        pub fn $name<'c>(
            lhs: Value<'c>,
            rhs: Value<'c>,
            result: Type<'c>,
            location: Location<'c>,
        ) -> Operation<'c> {
            operation::Builder::new(concat!("arith.", $op), location)
                .add_operands(&[lhs, rhs])
                .add_results(&[result])
                .build()
        }
    };
}

impl_arith_binary_op!(addi, "addi");
impl_arith_binary_op!(subi, "subi");
impl_arith_binary_op!(muli, "muli");
impl_arith_binary_op!(divui, "divui");
impl_arith_binary_op!(remui, "remui");
impl_arith_binary_op!(shrsi, "shrsi");
impl_arith_binary_op!(shrui, "shrui");
impl_arith_binary_op!(andi, "andi");
impl_arith_binary_op!(ori, "ori");
impl_arith_binary_op!(xori, "xori");

/// arith.constant
pub fn r#const<'c>(
    context: &'c Context,
    val: &str,
    result: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    operation::Builder::new("arith.constant", location)
        .add_results(&[result])
        .add_attributes(&[NamedAttribute::new_parsed(
            context,
            "value",
            &format!("{val} : {}", result),
        )
        .unwrap()])
        .build()
}
