use crate::{
    context::NativeContext, executor::AotNativeExecutor, metadata::MetadataStorage, OptLevel,
};
use cairo_lang_sierra::program::Program;
use std::{
    collections::HashMap,
    fmt::{self, Debug},
    hash::Hash,
    rc::Rc,
};

pub struct AotProgramCache<'a, K>
where
    K: PartialEq + Eq + Hash,
{
    context: &'a NativeContext,
    cache: HashMap<K, Rc<AotNativeExecutor<'a>>>,
}

impl<'a, K> AotProgramCache<'a, K>
where
    K: PartialEq + Eq + Hash,
{
    pub fn new(context: &'a NativeContext) -> Self {
        Self {
            context,
            cache: Default::default(),
        }
    }

    pub fn get(&self, key: &K) -> Option<Rc<AotNativeExecutor>> {
        self.cache.get(key).cloned()
    }

    pub fn compile_and_insert(
        &mut self,
        key: K,
        program: &Program,
        metadata: MetadataStorage,
        opt_level: OptLevel,
    ) -> Rc<AotNativeExecutor<'a>> {
        let module = self
            .context
            .compile(program, metadata)
            .expect("should compile");
        let executor = AotNativeExecutor::from_native_module(module, opt_level);

        let executor = Rc::new(executor);
        self.cache.insert(key, executor.clone());

        executor
    }
}

impl<'a, K> Debug for AotProgramCache<'a, K>
where
    K: PartialEq + Eq + Hash,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("AotProgramCache")
    }
}
