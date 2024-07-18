use cairo_lang_compiler::db::RootDatabase;
use cairo_lang_diagnostics::DiagnosticAdded;
use cairo_lang_lowering::{
    db::LoweringGroup, ids::LocationId, FlatLowered, Statement as LoweringStatement, Variable,
};
use cairo_lang_sierra::{
    ids::ConcreteLibfuncId,
    program::{Function, GenStatement, Program, Statement, StatementIdx},
};
use cairo_lang_sierra_generator::{
    db::SierraGenGroup,
    pre_sierra::{LabelId, Statement as PreSierraStatement, StatementWithLocation},
};
use id_arena::Arena;
use itertools::Itertools;
use std::collections::{BTreeMap, HashMap};

pub fn find_all_statements(
    db: &RootDatabase,
    contains_libfunc: impl Fn(&ConcreteLibfuncId) -> bool,
    program: &Program,
) -> Result<HashMap<StatementIdx, LocationId>, DiagnosticAdded> {
    program
        .funcs
        .iter()
        .map(|function| {
            Ok(
                find_statement_locations(db, &contains_libfunc, function, &program.statements)?
                    .into_iter(),
            )
        })
        .flatten_ok()
        .try_collect()
}

fn find_statement_locations(
    db: &RootDatabase,
    contains_libfunc: &impl Fn(&ConcreteLibfuncId) -> bool,
    function: &Function,
    statements: &[Statement],
) -> Result<HashMap<StatementIdx, LocationId>, DiagnosticAdded> {
    let function_id = db.lookup_intern_sierra_function(function.id.clone());
    let function_impl = db.function_with_body_sierra(function_id.body(db)?.unwrap())?;

    let function_long_id = function_id
        .body(db)
        .unwrap()
        .unwrap()
        .function_id(db)
        .unwrap(); // TODO: make it more declarative
    let flat_lowered =
        db.concrete_function_with_body_postpanic_lowered(function_long_id.body(db)?.unwrap())?;

    // Map Sierra to pre-Sierra statements.
    let sierra_to_pre_sierra_mappings =
        map_sierra_to_pre_sierra_statements(function.entry_point, statements, &function_impl.body);

    // Remove Sierra-specific invocations (they have no location since they are compiler-generated).
    let sierra_to_pre_sierra_mappings = sierra_to_pre_sierra_mappings
        .into_iter()
        .filter(|(_, statement)| {
            if let GenStatement::Invocation(invocation) = statement {
                !contains_libfunc(&invocation.libfunc_id)
            } else {
                false
            }
        })
        .collect();

    // Map Sierra to lowering statements by using the pre-Sierra mappings.
    Ok(remap_sierra_statements_to_locations(
        sierra_to_pre_sierra_mappings,
        &flat_lowered,
    ))
}

fn map_sierra_to_pre_sierra_statements<'a>(
    lhs_entry_point: StatementIdx,
    lhs_statements: &[Statement],
    rhs_statements: &'a [StatementWithLocation],
) -> HashMap<StatementIdx, &'a GenStatement<LabelId>> {
    let mut mappings = HashMap::new();

    let mut lhs_iter = lhs_statements.iter().enumerate().skip(lhs_entry_point.0);
    for rhs_statement in rhs_statements {
        let rhs_statement = match &rhs_statement.statement {
            PreSierraStatement::Sierra(x) => x,
            PreSierraStatement::Label(_) => continue,
            PreSierraStatement::PushValues(_) => panic!(),
        };

        if let Some((lhs_idx, _lhs_statement)) = lhs_iter.next() {
            // TODO: Assert lhs_statement == rhs_statement.

            mappings.insert(StatementIdx(lhs_idx), rhs_statement);
        }
    }

    mappings
}

fn remap_sierra_statements_to_locations(
    sierra_to_pre_sierra_mappings: HashMap<StatementIdx, &GenStatement<LabelId>>,
    flat_lowered: &FlatLowered,
) -> HashMap<StatementIdx, LocationId> {
    let sierra_to_pre_sierra_mappings = sierra_to_pre_sierra_mappings
        .into_iter()
        .map(|(k, v)| (k.0, v))
        .collect::<BTreeMap<_, _>>();

    let lowering_iter = flat_lowered
        .blocks
        .iter()
        .flat_map(|(_, block)| &block.statements);

    sierra_to_pre_sierra_mappings
        .into_iter()
        .zip(lowering_iter)
        .map(|((statement_idx, _lhs_statement), rhs_statement)| {
            (
                StatementIdx(statement_idx),
                locate_statement(&flat_lowered.variables, rhs_statement),
            )
        })
        .collect()
}

fn locate_statement(variables: &Arena<Variable>, statement: &LoweringStatement) -> LocationId {
    match statement {
        LoweringStatement::Call(x) => x.location,
        LoweringStatement::StructConstruct(x) => variables[x.output].location,
        LoweringStatement::StructDestructure(x) => x.input.location,
        LoweringStatement::EnumConstruct(x) => variables[x.output].location,
        LoweringStatement::Snapshot(x) => variables[x.snapshot()].location,
        LoweringStatement::Desnap(x) => variables[x.output].location,
        LoweringStatement::Const(x) => variables[x.output].location,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cairo_lang_defs::ids::VariantId;
    use cairo_lang_lowering::ids::FunctionId;
    use cairo_lang_lowering::ids::LocationId;
    use cairo_lang_lowering::objects::{
        StatementCall, StatementConst, StatementDesnap, StatementEnumConstruct, StatementSnapshot,
        StatementStructConstruct, StatementStructDestructure, VarUsage,
    };
    use cairo_lang_semantic::items::constant::ConstValue;
    use cairo_lang_semantic::items::enm::ConcreteVariant;
    use cairo_lang_semantic::items::imp::{ConcreteImplId, ImplId};
    use cairo_lang_semantic::types::ConcreteEnumId;
    use cairo_lang_semantic::TypeId;
    use salsa::InternKey;

    #[test]
    fn test_locate_statement_call() {
        let arena = Arena::<Variable>::new();

        let statement = LoweringStatement::Call(StatementCall {
            function: FunctionId::from_intern_id((22_u32).into()),
            inputs: Default::default(),
            with_coupon: false,
            outputs: Default::default(),
            location: LocationId::from_intern_id((22_u32).into()),
        });

        assert_eq!(
            locate_statement(&arena, &statement),
            LocationId::from_intern_id((22_u32).into())
        );
    }

    // #[test]
    // fn test_locate_statement_struct_construct() {
    //     let mut arena = Arena::<Variable>::new();

    //     let a = arena.alloc(Variable {
    //         droppable: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             45_u32.into(),
    //         ))),
    //         copyable: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             40_u32.into(),
    //         ))),
    //         destruct_impl: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             30_u32.into(),
    //         ))),
    //         panic_destruct_impl: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             20_u32.into(),
    //         ))),
    //         ty: TypeId::from_intern_id(10_u32.into()),
    //         location: LocationId::from_intern_id(15_u32.into()),
    //     });

    //     let statement = LoweringStatement::StructConstruct(StatementStructConstruct {
    //         inputs: Default::default(),
    //         output: a,
    //     });

    //     assert_eq!(
    //         locate_statement(&arena, &statement),
    //         LocationId::from_intern_id((15_u32).into())
    //     );
    // }

    // #[test]
    // fn test_locate_statement_struct_destructure() {
    //     let mut arena = Arena::<Variable>::new();

    //     let a = arena.alloc(Variable {
    //         droppable: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             45_u32.into(),
    //         ))),
    //         copyable: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             40_u32.into(),
    //         ))),
    //         destruct_impl: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             30_u32.into(),
    //         ))),
    //         panic_destruct_impl: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             20_u32.into(),
    //         ))),
    //         ty: TypeId::from_intern_id(10_u32.into()),
    //         location: LocationId::from_intern_id(15_u32.into()),
    //     });

    //     let statement = LoweringStatement::StructDestructure(StatementStructDestructure {
    //         input: VarUsage {
    //             var_id: a,
    //             location: LocationId::from_intern_id(23_u32.into()),
    //         },
    //         outputs: Default::default(),
    //     });

    //     assert_eq!(
    //         locate_statement(&arena, &statement),
    //         LocationId::from_intern_id((23_u32).into())
    //     );
    // }

    // #[test]
    // fn test_locate_statement_enum_construct() {
    //     let mut arena = Arena::<Variable>::new();

    //     let a = arena.alloc(Variable {
    //         droppable: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             45_u32.into(),
    //         ))),
    //         copyable: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             40_u32.into(),
    //         ))),
    //         destruct_impl: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             30_u32.into(),
    //         ))),
    //         panic_destruct_impl: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             20_u32.into(),
    //         ))),
    //         ty: TypeId::from_intern_id(10_u32.into()),
    //         location: LocationId::from_intern_id(15_u32.into()),
    //     });

    //     let statement = LoweringStatement::EnumConstruct(StatementEnumConstruct {
    //         variant: ConcreteVariant {
    //             concrete_enum_id: ConcreteEnumId::from_intern_id(55_u32.into()),
    //             id: VariantId::from_intern_id(432_u32.into()),
    //             ty: TypeId::from_intern_id(54_u32.into()),
    //             idx: 232,
    //         },
    //         input: VarUsage {
    //             var_id: a,
    //             location: LocationId::from_intern_id(23_u32.into()),
    //         },
    //         output: a,
    //     });

    //     assert_eq!(
    //         locate_statement(&arena, &statement),
    //         LocationId::from_intern_id((15_u32).into())
    //     );
    // }

    // #[test]
    // fn test_locate_statement_snapshot() {
    //     let mut arena = Arena::<Variable>::new();

    //     let a = arena.alloc(Variable {
    //         droppable: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             45_u32.into(),
    //         ))),
    //         copyable: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             40_u32.into(),
    //         ))),
    //         destruct_impl: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             30_u32.into(),
    //         ))),
    //         panic_destruct_impl: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             20_u32.into(),
    //         ))),
    //         ty: TypeId::from_intern_id(10_u32.into()),
    //         location: LocationId::from_intern_id(15_u32.into()),
    //     });

    //     let b = arena.alloc(Variable {
    //         droppable: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             4534_u32.into(),
    //         ))),
    //         copyable: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             4320_u32.into(),
    //         ))),
    //         destruct_impl: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             31_u32.into(),
    //         ))),
    //         panic_destruct_impl: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             223_u32.into(),
    //         ))),
    //         ty: TypeId::from_intern_id(111_u32.into()),
    //         location: LocationId::from_intern_id(125_u32.into()),
    //     });

    //     let statement = LoweringStatement::Snapshot(StatementSnapshot::new(
    //         VarUsage {
    //             var_id: a,
    //             location: LocationId::from_intern_id(23_u32.into()),
    //         },
    //         a,
    //         b,
    //     ));

    //     assert_eq!(
    //         locate_statement(&arena, &statement),
    //         LocationId::from_intern_id((125_u32).into())
    //     );
    // }

    // #[test]
    // fn test_locate_statement_desnap() {
    //     let mut arena = Arena::<Variable>::new();

    //     let a = arena.alloc(Variable {
    //         droppable: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             45_u32.into(),
    //         ))),
    //         copyable: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             40_u32.into(),
    //         ))),
    //         destruct_impl: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             30_u32.into(),
    //         ))),
    //         panic_destruct_impl: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             20_u32.into(),
    //         ))),
    //         ty: TypeId::from_intern_id(10_u32.into()),
    //         location: LocationId::from_intern_id(15_u32.into()),
    //     });

    //     let b = arena.alloc(Variable {
    //         droppable: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             4534_u32.into(),
    //         ))),
    //         copyable: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             4320_u32.into(),
    //         ))),
    //         destruct_impl: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             31_u32.into(),
    //         ))),
    //         panic_destruct_impl: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             223_u32.into(),
    //         ))),
    //         ty: TypeId::from_intern_id(111_u32.into()),
    //         location: LocationId::from_intern_id(122_u32.into()),
    //     });

    //     let statement = LoweringStatement::Desnap(StatementDesnap {
    //         input: VarUsage {
    //             var_id: a,
    //             location: LocationId::from_intern_id(23_u32.into()),
    //         },
    //         output: b,
    //     });

    //     assert_eq!(
    //         locate_statement(&arena, &statement),
    //         LocationId::from_intern_id((122_u32).into())
    //     );
    // }

    // #[test]
    // fn test_locate_statement_const() {
    //     let mut arena = Arena::<Variable>::new();

    //     let a = arena.alloc(Variable {
    //         droppable: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             4534_u32.into(),
    //         ))),
    //         copyable: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             4320_u32.into(),
    //         ))),
    //         destruct_impl: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             31_u32.into(),
    //         ))),
    //         panic_destruct_impl: Ok(ImplId::Concrete(ConcreteImplId::from_intern_id(
    //             223_u32.into(),
    //         ))),
    //         ty: TypeId::from_intern_id(111_u32.into()),
    //         location: LocationId::from_intern_id(1223_u32.into()),
    //     });

    //     let statement = LoweringStatement::Const(StatementConst {
    //         value: ConstValue::Missing(cairo_lang_diagnostics::DiagnosticAdded {}),
    //         output: a,
    //     });

    //     assert_eq!(
    //         locate_statement(&arena, &statement),
    //         LocationId::from_intern_id((1223_u32).into())
    //     );
    // }
}
