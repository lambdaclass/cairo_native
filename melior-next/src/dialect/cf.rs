use itertools::Itertools;

use crate::{
    ir::{
        operation::{self, Operation},
        Block, Location, NamedAttribute, Value, ValueLike,
    },
    Context,
};

/// branch operation
///
/// `cf.br (::mlir::cf::BranchOp)`
///
/// # Arguments
/// * `dest` - any successor
/// * `dest_operands` - any type
pub fn br<'c>(dest: &Block<'c>, dest_operands: &[Value], location: Location<'c>) -> Operation<'c> {
    operation::Builder::new("cf.br", location)
        .add_operands(dest_operands)
        .add_successors(&[dest])
        .build()
}

/// conditional branch operation
///
/// `cf.cond_br (::mlir::cf::CondBranchOp)`
///
/// # Arguments
/// * `condition` - 1-bit signless integer
/// * `true_dest` - any successor
/// * `false_dest` - any successor
/// * `true_dest_operands` - any type
/// * `false_dest_operands` - any type
pub fn cond_br<'c>(
    context: &'c Context,
    condition: Value,
    true_dest: &Block<'c>,
    false_dest: &Block<'c>,
    true_dest_operands: &[Value],
    false_dest_operands: &[Value],
    location: Location<'c>,
) -> Operation<'c> {
    let mut operands = vec![condition];
    operands.extend(true_dest_operands);
    operands.extend(false_dest_operands);

    operation::Builder::new("cf.cond_br", location)
        .add_attributes(&[NamedAttribute::new_parsed(
            context,
            "operand_segment_sizes",
            &format!("array<i32: 1, {}, {}>", true_dest_operands.len(), false_dest_operands.len()),
        )
        .unwrap()])
        .add_operands(&operands)
        .add_successors(&[true_dest, false_dest])
        .build()
}

/// cf switch
///
/// The switch terminator operation represents a switch on a signless integer value.
/// If the flag matches one of the specified cases, then the corresponding destination is jumped to.
/// If the flag does not match any of the cases, the default destination is jumped to.
/// The count and types of operands must align with the arguments in the corresponding target blocks.
///
/// `cf.switch (::mlir::cf::SwitchOp)`
///
/// # Arguments
/// * `case_values` - The hard-coded constant values the flag matches against.
/// * `default_destination` - the default case successor, with the operands it requires.
/// * `case_destinations` - case successors, with the operands each successor requires.
pub fn switch<'c>(
    context: &'c Context,
    case_values: &[String],
    flag: Value,
    default_destination: (&Block<'c>, &[Value]),
    case_destinations: &[(&Block<'c>, &[Value])],
    location: Location<'c>,
) -> Operation<'c> {
    let case_segment_sizes = std::iter::once(default_destination.1.len())
        .chain(case_destinations.iter().map(|x| x.1.len()))
        .join(", ");

    let default_op_segments = default_destination.1.len();
    let case_op_segments: usize = case_destinations.iter().map(|x| x.1.len()).sum();

    let (dests, operands): (Vec<_>, Vec<_>) =
        std::iter::once(&default_destination).chain(case_destinations.iter()).cloned().unzip();

    operation::Builder::new("cf.switch", location)
        .add_attributes(
            &NamedAttribute::new_parsed_vec(
                context,
                &[
                    (
                        "case_values",
                        &format!(
                            "dense<[{}]> : tensor<{} x {}>",
                            case_values.iter().join(", "),
                            case_values.len(),
                            flag.r#type()
                        ),
                    ),
                    (
                        // number of operands passed to each case
                        "case_operand_segments",
                        &format!("array<i32: {}>", case_segment_sizes),
                    ),
                    (
                        "operand_segment_sizes",
                        &format!(
                            // flag, defaultops, caseops
                            "array<i32: 1, {}, {}>",
                            default_op_segments, case_op_segments
                        ),
                    ),
                ],
            )
            .unwrap(),
        )
        .add_operands(
            std::iter::once([flag].as_slice())
                .chain(operands.into_iter())
                .flatten()
                .cloned()
                .collect_vec()
                .as_slice(),
        )
        .add_successors(dests.iter().map(|x| *x).collect_vec().as_slice())
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect;

    fn _create_context() -> Context {
        let context = Context::new();

        dialect::Handle::cf().register_dialect(&context);
        context.get_or_load_dialect("cf");

        context
    }

    /* todo: add test when arith constant is done
    #[test]
    fn br() {
        let context = create_context();
        let i32 = Type::integer(&context, 32);
        let i64 = Type::integer(&context, 64);

        assert_eq!(
            super::br(),
        );
    }
     */
}
