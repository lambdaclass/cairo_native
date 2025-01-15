#![cfg(test)]

use cairo_lang_sierra::{
    extensions::{
        branch_align::BranchAlignLibfunc,
        enm::{EnumInitLibfunc, EnumType},
        lib_func::{SierraApChange, SignatureSpecializationContext},
        structure::{StructConstructLibfunc, StructType},
        type_specialization_context::TypeSpecializationContext,
        types::TypeInfo,
        GenericLibfunc, NamedLibfunc, NamedType,
    },
    ids::{
        ConcreteLibfuncId, ConcreteTypeId, FunctionId, GenericLibfuncId, GenericTypeId, UserTypeId,
        VarId,
    },
    program::{
        BranchInfo, BranchTarget, ConcreteLibfuncLongId, ConcreteTypeLongId, Function,
        FunctionSignature, GenericArg, Invocation, LibfuncDeclaration, Param, Program, Statement,
        StatementIdx, TypeDeclaration,
    },
};
use std::{
    cell::{OnceCell, RefCell},
    iter::once,
    marker::PhantomData,
};

#[derive(Debug)]
pub struct SierraGenerator<T>
where
    T: GenericLibfunc,
{
    program: Program,
    phantom: PhantomData<T>,
}

impl<T> Default for SierraGenerator<T>
where
    T: GenericLibfunc,
{
    fn default() -> Self {
        Self {
            program: Program {
                type_declarations: Vec::new(),
                libfunc_declarations: Vec::new(),
                statements: Vec::new(),
                funcs: Vec::new(),
            },
            phantom: PhantomData,
        }
    }
}

impl<T> SierraGenerator<T>
where
    T: GenericLibfunc,
{
    pub fn build(self, generic_args: impl Into<Vec<GenericArg>>) -> Program {
        match T::supported_ids().as_slice() {
            [generic_id] => self.build_with_generic_id(generic_id.clone(), generic_args.into()),
            _ => panic!("multiple generic ids detected, please use build_with_generic_id directly"),
        }
    }

    pub fn build_with_generic_id(
        self,
        generic_id: GenericLibfuncId,
        generic_args: impl Into<Vec<GenericArg>>,
    ) -> Program {
        let context = SierraGeneratorWrapper(RefCell::new(self));
        let generic_args = generic_args.into();

        let libfunc = T::by_id(&generic_id).unwrap();
        let libfunc_signature = libfunc
            .specialize_signature(&context, &generic_args)
            .unwrap();

        let mut context = RefCell::into_inner(context.0);

        // Push the libfunc declaration.
        let libfunc_id = context
            .push_libfunc_declaration(ConcreteLibfuncLongId {
                generic_id,
                generic_args: generic_args.to_vec(),
            })
            .clone();

        // Generate packed types.
        let num_builtins = libfunc_signature
            .param_signatures
            .iter()
            .take_while(|param_signature| {
                let long_id = &context
                    .program
                    .type_declarations
                    .iter()
                    .find(|type_declaration| type_declaration.id == param_signature.ty)
                    .unwrap()
                    .long_id;

                matches!(
                    long_id.generic_id.0.as_str(),
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
            .count();

        let mut return_types = Vec::with_capacity(libfunc_signature.branch_signatures.len());
        let mut packed_unit_type_id = None;
        for branch_signature in &libfunc_signature.branch_signatures {
            assert!(branch_signature
                .vars
                .iter()
                .zip(libfunc_signature.param_signatures.iter().take(num_builtins))
                .all(|(lhs, rhs)| lhs.ty == rhs.ty));

            return_types.push(match branch_signature.vars.len() - num_builtins {
                0 => match libfunc_signature.branch_signatures.len() {
                    1 => ResultVarType::Empty(None),
                    _ => ResultVarType::Empty(Some(
                        packed_unit_type_id
                            .get_or_insert_with(|| {
                                context
                                    .push_type_declaration(ConcreteTypeLongId {
                                        generic_id: StructType::ID,
                                        generic_args: vec![GenericArg::UserType(
                                            UserTypeId::from_string("Tuple"),
                                        )],
                                    })
                                    .clone()
                            })
                            .clone(),
                    )),
                },
                1 => ResultVarType::Single(branch_signature.vars[num_builtins].ty.clone()),
                _ => ResultVarType::Multi(
                    context
                        .push_type_declaration(ConcreteTypeLongId {
                            generic_id: StructType::ID,
                            generic_args: once(GenericArg::UserType(UserTypeId::from_string(
                                "Tuple",
                            )))
                            .chain(
                                branch_signature
                                    .vars
                                    .iter()
                                    .skip(num_builtins)
                                    .map(|var_info| GenericArg::Type(var_info.ty.clone())),
                            )
                            .collect(),
                        })
                        .clone(),
                ),
            });
        }

        // Generate switch type.
        let return_type = match return_types.len() {
            1 => match return_types[0].clone() {
                ResultVarType::Empty(ty) => ty.unwrap().clone(),
                ResultVarType::Single(ty) => ty.clone(),
                ResultVarType::Multi(ty) => ty.clone(),
            },
            _ => context
                .push_type_declaration(ConcreteTypeLongId {
                    generic_id: EnumType::ID,
                    generic_args: once(GenericArg::UserType(UserTypeId::from_string("Tuple")))
                        .chain(return_types.iter().map(|ty| {
                            GenericArg::Type(match ty {
                                ResultVarType::Empty(ty) => ty.clone().unwrap(),
                                ResultVarType::Single(ty) => ty.clone(),
                                ResultVarType::Multi(ty) => ty.clone(),
                            })
                        }))
                        .collect(),
                })
                .clone(),
        };

        // Generate function declaration.
        context.program.funcs.push(Function {
            id: FunctionId::new(0),
            signature: FunctionSignature {
                param_types: libfunc_signature
                    .param_signatures
                    .iter()
                    .map(|param_signature| param_signature.ty.clone())
                    .collect(),
                ret_types: libfunc_signature.param_signatures[..num_builtins]
                    .iter()
                    .map(|param_signature| param_signature.ty.clone())
                    .chain(once(return_type.clone()))
                    .collect(),
            },
            params: libfunc_signature
                .param_signatures
                .iter()
                .enumerate()
                .map(|(idx, param_signature)| Param {
                    id: VarId::new(idx as u64),
                    ty: param_signature.ty.clone(),
                })
                .collect(),
            entry_point: StatementIdx(0),
        });

        // Generate statements.
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

        let branch_align_libfunc = OnceCell::new();
        let construct_unit_libfunc = packed_unit_type_id.map(|ty| {
            context
                .push_libfunc_declaration(ConcreteLibfuncLongId {
                    generic_id: GenericLibfuncId::from_string(StructConstructLibfunc::STR_ID),
                    generic_args: vec![GenericArg::Type(ty)],
                })
                .clone()
        });

        for (branch_index, branch_signature) in
            libfunc_signature.branch_signatures.iter().enumerate()
        {
            let branch_target = match branch_index {
                0 => BranchTarget::Fallthrough,
                _ => {
                    let statement_idx = StatementIdx(context.program.statements.len() + 1);
                    let branch_align_libfunc_id = branch_align_libfunc
                        .get_or_init(|| {
                            context
                                .push_libfunc_declaration(ConcreteLibfuncLongId {
                                    generic_id: GenericLibfuncId::from_string(
                                        BranchAlignLibfunc::STR_ID,
                                    ),
                                    generic_args: Vec::new(),
                                })
                                .clone()
                        })
                        .clone();

                    context
                        .program
                        .statements
                        .push(Statement::Invocation(Invocation {
                            libfunc_id: branch_align_libfunc_id,
                            args: Vec::new(),
                            branches: vec![BranchInfo {
                                target: BranchTarget::Fallthrough,
                                results: Vec::new(),
                            }],
                        }));

                    BranchTarget::Statement(statement_idx)
                }
            };

            // Maybe pack values.
            match &return_types[branch_index] {
                ResultVarType::Empty(Some(_)) => {
                    context
                        .program
                        .statements
                        .push(Statement::Invocation(Invocation {
                            libfunc_id: construct_unit_libfunc.clone().unwrap(),
                            args: Vec::new(),
                            branches: vec![BranchInfo {
                                target: BranchTarget::Fallthrough,
                                results: vec![VarId::new(num_builtins as u64)],
                            }],
                        }));
                }
                ResultVarType::Multi(type_id) => {
                    let construct_libfunc_id = context
                        .push_libfunc_declaration(ConcreteLibfuncLongId {
                            generic_id: GenericLibfuncId::from_string(
                                StructConstructLibfunc::STR_ID,
                            ),
                            generic_args: vec![GenericArg::Type(type_id.clone())],
                        })
                        .clone();

                    context
                        .program
                        .statements
                        .push(Statement::Invocation(Invocation {
                            libfunc_id: construct_libfunc_id,
                            args: (num_builtins..branch_signature.vars.len())
                                .map(|x| VarId::new(x as u64))
                                .collect(),
                            branches: vec![BranchInfo {
                                target: BranchTarget::Fallthrough,
                                results: vec![VarId::new(num_builtins as u64)],
                            }],
                        }));
                }
                _ => {}
            }

            // Maybe enum values.
            if libfunc_signature.branch_signatures.len() > 1 {
                let enum_libfunc_id = context
                    .push_libfunc_declaration(ConcreteLibfuncLongId {
                        generic_id: GenericLibfuncId::from_string(EnumInitLibfunc::STR_ID),
                        generic_args: vec![
                            GenericArg::Type(return_type.clone()),
                            GenericArg::Value(branch_index.into()),
                        ],
                    })
                    .clone();

                context
                    .program
                    .statements
                    .push(Statement::Invocation(Invocation {
                        libfunc_id: enum_libfunc_id,
                        args: vec![VarId::new(num_builtins as u64)],
                        branches: vec![BranchInfo {
                            target: BranchTarget::Fallthrough,
                            results: vec![VarId::new(num_builtins as u64)],
                        }],
                    }));
            }

            // Return.
            context.program.statements.push(Statement::Return(
                (0..=num_builtins).map(|x| VarId::new(x as u64)).collect(),
            ));

            // Push the branch target.
            libfunc_invocation.branches.push(BranchInfo {
                target: branch_target,
                results: branch_signature
                    .vars
                    .iter()
                    .enumerate()
                    .map(|(idx, _)| VarId::new(idx as u64))
                    .collect(),
            });
        }

        context
            .program
            .statements
            .insert(0, Statement::Invocation(libfunc_invocation));

        context.program
    }

    pub fn push_type_declaration(&mut self, long_id: ConcreteTypeLongId) -> &ConcreteTypeId {
        let id = ConcreteTypeId::new(self.program.type_declarations.len() as u64);
        self.program.type_declarations.push(TypeDeclaration {
            id,
            long_id,
            declared_type_info: None,
        });

        &self.program.type_declarations.last().unwrap().id
    }

    fn push_libfunc_declaration(&mut self, long_id: ConcreteLibfuncLongId) -> &ConcreteLibfuncId {
        let id = ConcreteLibfuncId::new(self.program.libfunc_declarations.len() as u64);
        self.program
            .libfunc_declarations
            .push(LibfuncDeclaration { id, long_id });

        &self.program.libfunc_declarations.last().unwrap().id
    }
}

struct SierraGeneratorWrapper<T>(RefCell<SierraGenerator<T>>)
where
    T: GenericLibfunc;

impl<T> SignatureSpecializationContext for SierraGeneratorWrapper<T>
where
    T: GenericLibfunc,
{
    fn try_get_concrete_type(
        &self,
        id: GenericTypeId,
        generic_args: &[GenericArg],
    ) -> Option<ConcreteTypeId> {
        let mut context = self.0.borrow_mut();

        let long_id = ConcreteTypeLongId {
            generic_id: id,
            generic_args: generic_args.to_vec(),
        };
        assert!(!context
            .program
            .type_declarations
            .iter()
            .any(|type_declaration| type_declaration.long_id == long_id));

        let id = ConcreteTypeId::new(context.program.type_declarations.len() as u64);
        context.program.type_declarations.push(TypeDeclaration {
            id: id.clone(),
            long_id,
            declared_type_info: None,
        });

        Some(id)
    }

    fn try_get_function_signature(&self, _function_id: &FunctionId) -> Option<FunctionSignature> {
        todo!()
    }

    fn try_get_function_ap_change(&self, _function_id: &FunctionId) -> Option<SierraApChange> {
        todo!()
    }

    fn as_type_specialization_context(&self) -> &dyn TypeSpecializationContext {
        self
    }
}

impl<T> TypeSpecializationContext for SierraGeneratorWrapper<T>
where
    T: GenericLibfunc,
{
    fn try_get_type_info(&self, _id: ConcreteTypeId) -> Option<TypeInfo> {
        todo!()
    }
}

#[derive(Clone)]
enum ResultVarType {
    Empty(Option<ConcreteTypeId>),
    Single(ConcreteTypeId),
    Multi(ConcreteTypeId),
}

#[cfg(test)]
mod test {
    use super::*;
    use cairo_lang_sierra::extensions::{
        array::ArrayNewLibfunc,
        int::{
            signed::{Sint8Traits, SintDiffLibfunc},
            unsigned::{Uint64Traits, Uint8Type},
            unsigned128::U128GuaranteeMulLibfunc,
            IntConstLibfunc,
        },
    };

    #[test]
    fn sierra_generator() {
        let program = SierraGenerator::<IntConstLibfunc<Uint64Traits>>::default()
            .build(&[GenericArg::Value(0.into())]);
        println!("{program}");
    }

    #[test]
    fn sierra_generator_multiret() {
        let program = SierraGenerator::<U128GuaranteeMulLibfunc>::default().build(&[]);
        println!("{program}");
    }

    #[test]
    fn sierra_generator_multibranch() {
        let program = SierraGenerator::<SintDiffLibfunc<Sint8Traits>>::default().build(&[]);
        println!("{program}");
    }

    #[test]
    fn sierra_generator_template() {
        let program = {
            let mut generator = SierraGenerator::<ArrayNewLibfunc>::default();

            let u8_type = generator
                .push_type_declaration(ConcreteTypeLongId {
                    generic_id: Uint8Type::ID,
                    generic_args: Vec::new(),
                })
                .clone();

            generator.build(&[GenericArg::Type(u8_type)])
        };
        println!("{program}");
    }
}
