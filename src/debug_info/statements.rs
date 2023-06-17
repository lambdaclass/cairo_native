use cairo_lang_compiler::db::RootDatabase;
use cairo_lang_defs::diagnostic_utils::StableLocation;
use cairo_lang_diagnostics::DiagnosticAdded;
use cairo_lang_lowering::{
    db::LoweringGroup, FlatLowered, Statement as LoweringStatement, Variable,
};
use cairo_lang_sierra::{
    ids::ConcreteLibfuncId,
    program::{Function, GenStatement, Program, Statement, StatementIdx},
};
use cairo_lang_sierra_generator::{
    db::SierraGenGroup,
    pre_sierra::{LabelId, Statement as PreSierraStatement},
};
use id_arena::Arena;
use itertools::Itertools;
use std::collections::{BTreeMap, HashMap};

pub fn find_all_statements(
    db: &RootDatabase,
    contains_libfunc: impl Fn(&ConcreteLibfuncId) -> bool,
    program: &Program,
) -> Result<HashMap<StatementIdx, StableLocation>, DiagnosticAdded> {
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
) -> Result<HashMap<StatementIdx, StableLocation>, DiagnosticAdded> {
    let function_id = db.lookup_intern_sierra_function(function.id.clone());
    let function_impl = db.function_with_body_sierra(function_id.body(db)?.unwrap())?;

    let function_long_id = function_id.lookup(db);
    let flat_lowered =
        db.concrete_function_with_body_postpanic_lowered(function_long_id.body(db)?.unwrap())?;

    // Map Sierra to pre-Sierra statements.
    let mut sierra_to_pre_sierra_mappings =
        map_sierra_to_pre_sierra_statements(function.entry_point, statements, &function_impl.body);

    // Remove Sierra-specific invocations (they have no location since they are compiler-generated).
    sierra_to_pre_sierra_mappings
        .drain_filter(|_, statement| match statement {
            GenStatement::Invocation(invocation) => contains_libfunc(&invocation.libfunc_id),
            GenStatement::Return(_) => false,
        })
        .for_each(|_| {});

    // Map Sierra to lowering statements by using the pre-Sierra mappings.
    Ok(remap_sierra_statements_to_locations(
        sierra_to_pre_sierra_mappings,
        &flat_lowered,
    ))
}

fn map_sierra_to_pre_sierra_statements<'a>(
    lhs_entry_point: StatementIdx,
    lhs_statements: &[Statement],
    rhs_statements: &'a [PreSierraStatement],
) -> HashMap<StatementIdx, &'a GenStatement<LabelId>> {
    let mut mappings = HashMap::new();

    let mut lhs_iter = lhs_statements.iter().enumerate().skip(lhs_entry_point.0);
    for rhs_statement in rhs_statements {
        let rhs_statement = match rhs_statement {
            PreSierraStatement::Sierra(x) => x,
            PreSierraStatement::Label(_) => continue,
            PreSierraStatement::PushValues(_) => panic!(),
        };

        let (lhs_idx, _lhs_statement) = lhs_iter.next().unwrap();
        // TODO: Assert lhs_statement == rhs_statement.

        mappings.insert(StatementIdx(lhs_idx), rhs_statement);
    }

    mappings
}

fn remap_sierra_statements_to_locations(
    sierra_to_pre_sierra_mappings: HashMap<StatementIdx, &GenStatement<LabelId>>,
    flat_lowered: &FlatLowered,
) -> HashMap<StatementIdx, StableLocation> {
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
        .filter_map(|((statement_idx, _lhs_statement), rhs_statement)| {
            locate_statement(&flat_lowered.variables, rhs_statement)
                .map(|location| (StatementIdx(statement_idx), location))
        })
        .collect()
}

fn locate_statement(
    variables: &Arena<Variable>,
    statement: &LoweringStatement,
) -> Option<StableLocation> {
    match statement {
        LoweringStatement::Literal(x) => Some(variables[x.output].location.unwrap()),
        LoweringStatement::Call(x) => Some(x.location.unwrap()),
        LoweringStatement::StructConstruct(_) => None,
        LoweringStatement::StructDestructure(_) => None,
        LoweringStatement::EnumConstruct(_) => None,
        LoweringStatement::Snapshot(_) => None,
        LoweringStatement::Desnap(_) => None,
    }
}
