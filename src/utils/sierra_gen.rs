#![cfg(test)]

use cairo_lang_sierra::{
    extensions::{
        lib_func::{SierraApChange, SignatureSpecializationContext},
        type_specialization_context::TypeSpecializationContext,
        GenericLibfunc,
    },
    ids::{ConcreteLibfuncId, ConcreteTypeId, FunctionId, GenericTypeId, VarId},
    program::{
        BranchInfo, ConcreteLibfuncLongId, ConcreteTypeLongId, Function, FunctionSignature,
        GenBranchTarget, GenericArg, Invocation, LibfuncDeclaration, Param, Program, Statement,
        StatementIdx, TypeDeclaration,
    },
};
use std::cell::RefCell;

pub fn generate_program<T>(args: &[GenericArg]) -> Program
where
    T: GenericLibfunc,
{
    // Initialize the Sierra generation context (which contains an empty program).
    let context = Context(RefCell::new(Program {
        type_declarations: Vec::new(),
        libfunc_declarations: Vec::new(),
        statements: Vec::new(),
        funcs: Vec::new(),
    }));

    // Extract the libfunc id.
    let libfunc_ids = T::supported_ids();
    let libfunc = T::by_id(&libfunc_ids[0]).unwrap();
    assert_eq!(libfunc_ids.len(), 1);

    // Specialize the target libfunc signature. This will generate the required types within the
    // program.
    let libfunc_signature = libfunc.specialize_signature(&context, args).unwrap();

    // Generate the target libfunc declaration.
    let mut program = context.0.into_inner();
    let libfunc_id = ConcreteLibfuncId::new(program.libfunc_declarations.len() as u64);
    program.libfunc_declarations.push(LibfuncDeclaration {
        id: libfunc_id.clone(),
        long_id: ConcreteLibfuncLongId {
            generic_id: libfunc_ids[0].clone(),
            generic_args: args.to_vec(),
        },
    });

    // Generate the test's entry point.
    let num_builtins;
    let ret_types = {
        // Add all builtins.
        let mut ret_types: Vec<ConcreteTypeId> = libfunc_signature
            .param_signatures
            .iter()
            .take_while(|param_signature| {
                let ty = program
                    .type_declarations
                    .iter()
                    .find(|ty| ty.id == param_signature.ty)
                    .unwrap();
                matches!(
                    ty.long_id.generic_id.0.as_str(),
                    "Bitwise"
                        | "EcOp"
                        | "GasBuiltin"
                        | "BuiltinCosts"
                        | "RangeCheck"
                        | "RangeCheck96"
                        | "Pedersen"
                        | "Poseidon"
                        | "Coupon"
                        | "System"
                        | "SegmentArena"
                        | "AddMod"
                        | "MulMod"
                )
            })
            .map(|param_signature| param_signature.ty.clone())
            .collect();
        num_builtins = ret_types.len();

        // Push the return value.
        ret_types.push({
            let num_branches = libfunc_signature.branch_signatures.len();
            let mut iter = libfunc_signature
                .branch_signatures
                .iter()
                .map(|branch_signature| match branch_signature.vars.len() {
                    1 => branch_signature.vars[0].ty.clone(),
                    _ => todo!(),
                });

            match num_branches {
                0 => todo!(),
                1 => iter.next().unwrap(),
                _ => todo!(),
            }
        });

        ret_types
    };

    program.funcs.push(Function {
        id: FunctionId::new(0),
        signature: FunctionSignature {
            param_types: libfunc_signature
                .param_signatures
                .iter()
                .map(|param_signature| param_signature.ty.clone())
                .collect(),
            ret_types,
        },
        params: libfunc_signature
            .param_signatures
            .iter()
            .enumerate()
            .map(|(id, param_signature)| Param {
                id: VarId::new(id as u64),
                ty: param_signature.ty.clone(),
            })
            .collect(),
        entry_point: StatementIdx(0),
    });

    // Generate the statements.
    let mut libfunc_invocation = Invocation {
        libfunc_id,
        args: libfunc_signature
            .param_signatures
            .iter()
            .enumerate()
            .map(|(idx, _)| VarId::new(idx as u64))
            .collect(),
        branches: Vec::new(),
    };

    for branch_signature in &libfunc_signature.branch_signatures {
        libfunc_invocation.branches.push(BranchInfo {
            target: GenBranchTarget::Statement(StatementIdx(program.statements.len() + 1)),
            results: branch_signature
                .vars
                .iter()
                .enumerate()
                .map(|(idx, _)| VarId::new(idx as u64))
                .collect(),
        });

        // TODO: Handle multiple return values (struct_construct).
        // TODO: Handle multiple branches (enum_init).

        program.statements.push(Statement::Return(
            (0..=num_builtins)
                .map(|idx| VarId::new(idx as u64))
                .collect(),
        ));
    }

    program
        .statements
        .insert(0, Statement::Invocation(libfunc_invocation));

    program
}

struct Context(RefCell<Program>);

impl TypeSpecializationContext for Context {
    fn try_get_type_info(
        &self,
        _id: ConcreteTypeId,
    ) -> Option<cairo_lang_sierra::extensions::types::TypeInfo> {
        todo!()
    }
}

impl SignatureSpecializationContext for Context {
    fn try_get_concrete_type(
        &self,
        id: GenericTypeId,
        generic_args: &[GenericArg],
    ) -> Option<ConcreteTypeId> {
        let mut program = self.0.borrow_mut();

        let long_id = ConcreteTypeLongId {
            generic_id: id,
            generic_args: generic_args.to_vec(),
        };
        match program
            .type_declarations
            .iter()
            .find_map(|ty| (ty.long_id == long_id).then_some(ty.id.clone()))
        {
            Some(x) => Some(x),
            None => {
                let type_id = ConcreteTypeId {
                    id: program.type_declarations.len() as u64,
                    debug_name: None,
                };
                program.type_declarations.push(TypeDeclaration {
                    id: type_id.clone(),
                    long_id,
                    declared_type_info: None,
                });

                Some(type_id)
            }
        }
    }

    fn try_get_function_signature(&self, _function_id: &FunctionId) -> Option<FunctionSignature> {
        todo!()
    }

    fn try_get_function_ap_change(&self, _function_id: &FunctionId) -> Option<SierraApChange> {
        todo!()
    }

    fn as_type_specialization_context(&self) -> &dyn TypeSpecializationContext {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use cairo_lang_sierra::extensions::int::{unsigned::Uint64Traits, IntConstLibfunc};

    #[test]
    fn sierra_generator() {
        let program =
            generate_program::<IntConstLibfunc<Uint64Traits>>(&[GenericArg::Value(0.into())]);
        println!("{program}");
    }
}
