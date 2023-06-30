//! # Various utilities

use cairo_lang_sierra::ids::FunctionId;
use std::{alloc::Layout, borrow::Cow, fmt};

/// Generate a function name.
///
/// If the program includes function identifiers, return those. Otherwise return `f` followed by the
/// identifier number.
pub fn generate_function_name(function_id: &FunctionId) -> Cow<str> {
    function_id
        .debug_name
        .as_deref()
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(format!("f{}", function_id.id)))
}

/// Return the layout for an integer of arbitrary width.
///
/// This assumes the platform's maximum (effective) alignment is 8 bytes, and that every integer
/// with a size in bytes of a power of two has the same alignment as its size.
pub fn get_integer_layout(width: u32) -> Layout {
    if width == 0 {
        Layout::new::<()>()
    } else if width <= 8 {
        Layout::new::<u8>()
    } else if width <= 16 {
        Layout::new::<u16>()
    } else if width <= 32 {
        Layout::new::<u32>()
    } else {
        Layout::array::<u64>(width.next_multiple_of(8) as usize >> 3).unwrap()
    }
}

/// Return a type that calls a closure when formatted using [Debug](std::fmt::Debug).
pub fn debug_with<F>(fmt: F) -> impl fmt::Debug
where
    F: Fn(&mut fmt::Formatter) -> fmt::Result,
{
    struct FmtWrapper<F>(F)
    where
        F: Fn(&mut fmt::Formatter) -> fmt::Result;

    impl<F> fmt::Debug for FmtWrapper<F>
    where
        F: Fn(&mut fmt::Formatter) -> fmt::Result,
    {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            self.0(f)
        }
    }

    FmtWrapper(fmt)
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::metadata::MetadataStorage;
    use cairo_lang_compiler::{
        compile_prepared_db, db::RootDatabase, project::setup_project, CompilerConfig,
    };
    use cairo_lang_filesystem::db::init_dev_corelib;
    use cairo_lang_sierra::{
        extensions::core::{CoreLibfunc, CoreType},
        program_registry::ProgramRegistry,
    };
    use melior::{
        dialect::DialectRegistry,
        ir::{Location, Module},
        pass::{self, PassManager},
        utility::{register_all_dialects, register_all_passes},
        Context, ExecutionEngine,
    };
    use std::{env::var, fs, path::Path, sync::Arc};

    macro_rules! run_cairo {
        ( $entry_point:ident( $( $args:tt ),* ) in mod { $( $program:tt )+ } ) => {
            $crate::utils::test::run_cairo_str(
                stringify!($($program)+),
                stringify!($entry_point),
                json!([$($args),*]),
            )
        };
    }
    pub(crate) use run_cairo;

    pub fn run_cairo_str(
        program_str: &str,
        entry_point: &str,
        args: serde_json::Value,
    ) -> serde_json::Value {
        let (program, entry_point) = {
            let mut program_file = tempfile::Builder::new()
                .prefix("test_")
                .suffix(".cairo")
                .tempfile()
                .unwrap();
            fs::write(&mut program_file, program_str).unwrap();

            let mut db = RootDatabase::default();
            init_dev_corelib(
                &mut db,
                Path::new(&var("CARGO_MANIFEST_DIR").unwrap()).join("corelib/src"),
            );
            let main_crate_ids = setup_project(&mut db, program_file.path()).unwrap();
            let program = Arc::unwrap_or_clone(
                compile_prepared_db(
                    &mut db,
                    main_crate_ids,
                    CompilerConfig {
                        replace_ids: true,
                        ..Default::default()
                    },
                )
                .unwrap(),
            );

            let module_name = program_file.path().with_extension("");
            let module_name = module_name.file_name().unwrap().to_str().unwrap();
            (
                program,
                format!("{module_name}::{module_name}::{entry_point}"),
            )
        };
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program)
            .expect("Could not create the test program registry.");

        let context = Context::new();
        context.append_dialect_registry(&{
            let registry = DialectRegistry::new();
            register_all_dialects(&registry);
            registry
        });
        context.load_all_available_dialects();
        register_all_passes();

        let mut module = Module::new(Location::unknown(&context));

        let mut metadata = MetadataStorage::new();
        crate::compile::<CoreType, CoreLibfunc>(
            &context,
            &module,
            &program,
            &registry,
            &mut metadata,
            None,
        )
        .expect("Could not compile test program to MLIR.");

        assert!(
            module.as_operation().verify(),
            "Test program generated invalid MLIR."
        );

        let pass_manager = PassManager::new(&context);
        pass_manager.enable_verifier(true);
        pass_manager.add_pass(pass::transform::create_canonicalizer());

        pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());

        pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
        pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
        pass_manager.add_pass(pass::conversion::create_func_to_llvm());
        pass_manager.add_pass(pass::conversion::create_index_to_llvm_pass());
        pass_manager.add_pass(pass::conversion::create_mem_ref_to_llvm());
        pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());

        pass_manager
            .run(&mut module)
            .expect("Could not apply passes to the compiled test program.");

        let engine = ExecutionEngine::new(&module, 0, &[], false);
        crate::execute::<CoreType, CoreLibfunc, _, _>(
            &engine,
            &registry,
            &program
                .funcs
                .iter()
                .find(|x| x.id.debug_name.as_deref() == Some(&entry_point))
                .expect("Test program entry point not found.")
                .id,
            args,
            serde_json::value::Serializer,
        )
        .expect("Test program execution failed.")
    }

    /// Ensures that the host's `u8` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u8() {
        assert_eq!(get_integer_layout(8).align(), 1);
    }

    /// Ensures that the host's `u16` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u16() {
        assert_eq!(get_integer_layout(16).align(), 2);
    }

    /// Ensures that the host's `u32` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u32() {
        assert_eq!(get_integer_layout(32).align(), 4);
    }

    /// Ensures that the host's `u64` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u64() {
        assert_eq!(get_integer_layout(64).align(), 8);
    }

    /// Ensures that the host's `u128` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u128() {
        assert_eq!(get_integer_layout(128).align(), 8);
    }

    /// Ensures that the host's `u256` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u256() {
        assert_eq!(get_integer_layout(256).align(), 8);
    }

    /// Ensures that the host's `u512` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u512() {
        assert_eq!(get_integer_layout(512).align(), 8);
    }

    /// Ensures that the host's `felt252` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_felt252() {
        assert_eq!(get_integer_layout(252).align(), 8);
    }
}
