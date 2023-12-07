use crate::{context::NativeContext, executor::AotNativeExecutor, utils::SHARED_LIBRARY_EXT};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    program::Program,
    program_registry::ProgramRegistry,
};
use libloading::Library;
use std::{collections::HashMap, hash::Hash, rc::Rc};

pub struct AotProgramCache<'a, K: PartialEq + Eq + Hash> {
    context: &'a NativeContext,
    cache: HashMap<K, Rc<AotNativeExecutor<CoreType, CoreLibfunc>>>,
}

impl<'a, K: PartialEq + Eq + Hash> AotProgramCache<'a, K> {
    pub fn new(context: &'a NativeContext) -> Self {
        Self {
            context,
            cache: Default::default(),
        }
    }

    pub fn get(&self, key: &K) -> Option<Rc<AotNativeExecutor<CoreType, CoreLibfunc>>> {
        self.cache.get(key).cloned()
    }

    pub fn compile_and_insert(
        &mut self,
        key: K,
        program: &Program,
    ) -> Rc<AotNativeExecutor<CoreType, CoreLibfunc>> {
        let module = self.context.compile(program).expect("should compile");

        // Compile module into an object.
        let object_data = crate::ffi::module_to_object(module.module()).unwrap();

        // Compile object into a shared library.
        let shared_library_path = tempfile::Builder::new()
            .prefix("lib")
            .suffix(SHARED_LIBRARY_EXT)
            .tempfile()
            .unwrap()
            .into_temp_path();
        crate::ffi::object_to_shared_lib(&object_data, &shared_library_path).unwrap();

        let shared_library = unsafe { Library::new(shared_library_path).unwrap() };
        let executor =
            AotNativeExecutor::new(shared_library, ProgramRegistry::new(program).unwrap());

        let executor = Rc::new(executor);
        self.cache.insert(key, executor.clone());

        executor
    }
}
