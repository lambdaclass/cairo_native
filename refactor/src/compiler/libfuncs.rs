use super::{OpFactory, Successor, Value};
use cairo_lang_sierra::ids::ConcreteLibfuncId;
use std::collections::BTreeMap;

pub type LibfuncStorage = BTreeMap<SierraLibfuncId, CompiledLibfunc>;
type LibfuncInvoke = dyn Fn(&OpFactory, &[Value], &[Successor]) -> Vec<Vec<Value>>;

/// Compiled sierra libfunc identifier.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SierraLibfuncId(u64);

impl SierraLibfuncId {
    pub(super) const fn new(value: u64) -> Self {
        Self(value)
    }
}

/// Libfunc implementation.
pub struct LibfuncImpl {
    pub(crate) libfunc_setup: Option<Box<dyn FnOnce()>>,
    pub(crate) libfunc_invoke: Box<LibfuncInvoke>,
}

impl LibfuncImpl {
    /// Create a new simple libfunc.
    pub fn new(
        invoke: impl 'static + Fn(&OpFactory, &[Value], &[Successor]) -> Vec<Vec<Value>>,
    ) -> Self {
        Self {
            libfunc_setup: None,
            libfunc_invoke: Box::new(invoke),
        }
    }

    // /// Create a new libfunc with setup.
    // pub fn new_with_setup(setup: impl 'a + FnOnce(), invoke: impl 'a + Fn()) -> Self {
    //     Self {
    //         libfunc_setup: Some(Box::new(setup)),
    //         libfunc_invoke: Box::new(invoke),
    //         phantom: PhantomData,
    //     }
    // }
}

pub struct CompiledLibfunc {
    pub _id: ConcreteLibfuncId,

    pub _libfunc_setup: Option<Box<dyn FnOnce()>>,
    pub libfunc_invoke: Box<LibfuncInvoke>,
}
