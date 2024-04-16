use crate::{
    context::NativeContext, executor::AotNativeExecutor, metadata::gas::GasMetadata,
    module::NativeModule, utils::SHARED_LIBRARY_EXT, OptLevel,
};
use cairo_lang_sierra::program::Program;
use libloading::Library;
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
    cache: HashMap<K, Rc<AotNativeExecutor>>,
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
        opt_level: OptLevel,
    ) -> Rc<AotNativeExecutor> {
        let NativeModule {
            module,
            registry,
            metadata,
        } = self.context.compile(program, None).expect("should compile");

        // Compile module into an object.
        let object_data = crate::ffi::module_to_object(&module, opt_level).unwrap();

        // Compile object into a shared library.
        let shared_library_path = tempfile::Builder::new()
            .prefix("lib")
            .suffix(SHARED_LIBRARY_EXT)
            .tempfile()
            .unwrap()
            .into_temp_path();
        crate::ffi::object_to_shared_lib(&object_data, &shared_library_path).unwrap();

        let shared_library = unsafe { Library::new(shared_library_path).unwrap() };
        let executor = AotNativeExecutor::new(
            shared_library,
            registry,
            metadata.get::<GasMetadata>().cloned().unwrap(),
        );

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
