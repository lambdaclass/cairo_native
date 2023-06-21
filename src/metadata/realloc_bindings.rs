use melior::{
    dialect::{func, llvm},
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType},
        Identifier, Location, Module, Region,
    },
    Context,
};
use std::marker::PhantomData;

#[derive(Debug)]
pub struct ReallocBindings {
    phantom: PhantomData<()>,
}

impl ReallocBindings {
    pub fn new(context: &Context, module: &Module) -> Self {
        module.body().append_operation(func::func(
            context,
            StringAttribute::new(context, "realloc"),
            TypeAttribute::new(
                FunctionType::new(
                    context,
                    &[
                        llvm::r#type::opaque_pointer(context),
                        IntegerType::new(context, 64).into(),
                    ],
                    &[llvm::r#type::opaque_pointer(context)],
                )
                .into(),
            ),
            Region::new(),
            &[(
                Identifier::new(context, "sym_visibility"),
                StringAttribute::new(context, "private").into(),
            )],
            Location::unknown(context),
        ));

        Self {
            phantom: PhantomData,
        }
    }
}
