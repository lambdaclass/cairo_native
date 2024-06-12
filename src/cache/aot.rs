//use crate::{
use crate::{
//    context::NativeContext, executor::AotNativeExecutor, metadata::gas::GasMetadata,
    context::NativeContext, executor::AotNativeExecutor, metadata::gas::GasMetadata,
//    module::NativeModule, utils::SHARED_LIBRARY_EXT, OptLevel,
    module::NativeModule, utils::SHARED_LIBRARY_EXT, OptLevel,
//};
};
//use cairo_lang_sierra::program::Program;
use cairo_lang_sierra::program::Program;
//use libloading::Library;
use libloading::Library;
//use std::{
use std::{
//    collections::HashMap,
    collections::HashMap,
//    fmt::{self, Debug},
    fmt::{self, Debug},
//    hash::Hash,
    hash::Hash,
//    rc::Rc,
    rc::Rc,
//};
};
//

//pub struct AotProgramCache<'a, K>
pub struct AotProgramCache<'a, K>
//where
where
//    K: PartialEq + Eq + Hash,
    K: PartialEq + Eq + Hash,
//{
{
//    context: &'a NativeContext,
    context: &'a NativeContext,
//    cache: HashMap<K, Rc<AotNativeExecutor>>,
    cache: HashMap<K, Rc<AotNativeExecutor>>,
//}
}
//

//impl<'a, K> AotProgramCache<'a, K>
impl<'a, K> AotProgramCache<'a, K>
//where
where
//    K: PartialEq + Eq + Hash,
    K: PartialEq + Eq + Hash,
//{
{
//    pub fn new(context: &'a NativeContext) -> Self {
    pub fn new(context: &'a NativeContext) -> Self {
//        Self {
        Self {
//            context,
            context,
//            cache: Default::default(),
            cache: Default::default(),
//        }
        }
//    }
    }
//

//    pub fn get(&self, key: &K) -> Option<Rc<AotNativeExecutor>> {
    pub fn get(&self, key: &K) -> Option<Rc<AotNativeExecutor>> {
//        self.cache.get(key).cloned()
        self.cache.get(key).cloned()
//    }
    }
//

//    pub fn compile_and_insert(
    pub fn compile_and_insert(
//        &mut self,
        &mut self,
//        key: K,
        key: K,
//        program: &Program,
        program: &Program,
//        opt_level: OptLevel,
        opt_level: OptLevel,
//    ) -> Rc<AotNativeExecutor> {
    ) -> Rc<AotNativeExecutor> {
//        let NativeModule {
        let NativeModule {
//            module,
            module,
//            registry,
            registry,
//            metadata,
            metadata,
//        } = self.context.compile(program, None).expect("should compile");
        } = self.context.compile(program, None).expect("should compile");
//

//        // Compile module into an object.
        // Compile module into an object.
//        let object_data = crate::ffi::module_to_object(&module, opt_level).unwrap();
        let object_data = crate::ffi::module_to_object(&module, opt_level).unwrap();
//

//        // Compile object into a shared library.
        // Compile object into a shared library.
//        let shared_library_path = tempfile::Builder::new()
        let shared_library_path = tempfile::Builder::new()
//            .prefix("lib")
            .prefix("lib")
//            .suffix(SHARED_LIBRARY_EXT)
            .suffix(SHARED_LIBRARY_EXT)
//            .tempfile()
            .tempfile()
//            .unwrap()
            .unwrap()
//            .into_temp_path();
            .into_temp_path();
//        crate::ffi::object_to_shared_lib(&object_data, &shared_library_path).unwrap();
        crate::ffi::object_to_shared_lib(&object_data, &shared_library_path).unwrap();
//

//        let shared_library = unsafe { Library::new(shared_library_path).unwrap() };
        let shared_library = unsafe { Library::new(shared_library_path).unwrap() };
//        let executor = AotNativeExecutor::new(
        let executor = AotNativeExecutor::new(
//            shared_library,
            shared_library,
//            registry,
            registry,
//            metadata.get::<GasMetadata>().cloned().unwrap(),
            metadata.get::<GasMetadata>().cloned().unwrap(),
//        );
        );
//

//        let executor = Rc::new(executor);
        let executor = Rc::new(executor);
//        self.cache.insert(key, executor.clone());
        self.cache.insert(key, executor.clone());
//

//        executor
        executor
//    }
    }
//}
}
//

//impl<'a, K> Debug for AotProgramCache<'a, K>
impl<'a, K> Debug for AotProgramCache<'a, K>
//where
where
//    K: PartialEq + Eq + Hash,
    K: PartialEq + Eq + Hash,
//{
{
//    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//        f.write_str("AotProgramCache")
        f.write_str("AotProgramCache")
//    }
    }
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod tests {
mod tests {
//    use super::*;
    use super::*;
//    use crate::{utils::test::load_cairo, values::JitValue};
    use crate::{utils::test::load_cairo, values::JitValue};
//    use starknet_types_core::felt::Felt;
    use starknet_types_core::felt::Felt;
//

//    #[test]
    #[test]
//    fn test_aot_compile_and_insert() {
    fn test_aot_compile_and_insert() {
//        let native_context = NativeContext::new();
        let native_context = NativeContext::new();
//        let mut cache = AotProgramCache::new(&native_context);
        let mut cache = AotProgramCache::new(&native_context);
//

//        let (_, program) = load_cairo! {
        let (_, program) = load_cairo! {
//            fn run_test() -> felt252 {
            fn run_test() -> felt252 {
//                42
                42
//            }
            }
//        };
        };
//

//        let function_id = &program.funcs.first().expect("should have a function").id;
        let function_id = &program.funcs.first().expect("should have a function").id;
//        let executor = cache.compile_and_insert((), &program, OptLevel::default());
        let executor = cache.compile_and_insert((), &program, OptLevel::default());
//        let res = executor
        let res = executor
//            .invoke_dynamic(function_id, &[], Some(u128::MAX))
            .invoke_dynamic(function_id, &[], Some(u128::MAX))
//            .expect("should run");
            .expect("should run");
//

//        // After compiling and inserting the program, we should be able to run it.
        // After compiling and inserting the program, we should be able to run it.
//        assert_eq!(res.return_value, JitValue::Felt252(Felt::from(42)));
        assert_eq!(res.return_value, JitValue::Felt252(Felt::from(42)));
//    }
    }
//}
}
