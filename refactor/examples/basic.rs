use cairo_lang_sierra::{
    ids::{ConcreteLibfuncId, ConcreteTypeId, FunctionId, GenericLibfuncId, GenericTypeId, VarId},
    program::{
        BranchInfo, BranchTarget, ConcreteLibfuncLongId, ConcreteTypeLongId, Function,
        FunctionSignature, GenericArg, Invocation, LibfuncDeclaration, Statement, StatementIdx,
        TypeDeclaration,
    },
};
use sierra2mlir::{
    compiler::Compiler,
    database::{libfuncs::LibfuncDatabase, types::TypeDatabase},
};

fn main() {
    let mut compiler = Compiler::default();

    let type_database = TypeDatabase::default();
    let libfunc_database = LibfuncDatabase::default();

    compiler.process_type(
        &type_database,
        &TypeDeclaration {
            id: ConcreteTypeId::from_string("felt252"),
            long_id: ConcreteTypeLongId {
                generic_id: GenericTypeId::from_string("felt252"),
                generic_args: vec![],
            },
            declared_type_info: None,
        },
    );

    compiler.declare_func(&Function {
        id: FunctionId::from_string("basic::basic::main"),
        signature: FunctionSignature {
            param_types: vec![],
            ret_types: vec![ConcreteTypeId::from_string("felt252")],
        },
        params: vec![],
        entry_point: StatementIdx(0),
    });

    compiler.process_libfunc(
        &libfunc_database,
        &LibfuncDeclaration {
            id: ConcreteLibfuncId::from_string("felt252_const<42>"),
            long_id: ConcreteLibfuncLongId {
                generic_id: GenericLibfuncId::from_string("felt252_const"),
                generic_args: vec![GenericArg::Value(42.into())],
            },
        },
    );
    compiler.process_libfunc(
        &libfunc_database,
        &LibfuncDeclaration {
            id: ConcreteLibfuncId::from_string("store_temp<felt252>"),
            long_id: ConcreteLibfuncLongId {
                generic_id: GenericLibfuncId::from_string("store_temp"),
                generic_args: vec![GenericArg::Type(ConcreteTypeId::from_string("felt252"))],
            },
        },
    );

    let compiled_program = compiler.build(&[
        Statement::Invocation(Invocation {
            libfunc_id: ConcreteLibfuncId::from_string("felt252_const<42>"),
            args: vec![],
            branches: vec![BranchInfo {
                target: BranchTarget::Fallthrough,
                results: vec![VarId::new(0)],
            }],
        }),
        Statement::Invocation(Invocation {
            libfunc_id: ConcreteLibfuncId::from_string("store_temp<felt252>"),
            args: vec![VarId::new(0)],
            branches: vec![BranchInfo {
                target: BranchTarget::Fallthrough,
                results: vec![VarId::new(1)],
            }],
        }),
        Statement::Return(vec![VarId::new(1)]),
    ]);
}
