//! # FFI Wrappers
//!
//! This is a "hotfix" for missing Rust interfaces to the C/C++ libraries we use, namely LLVM/MLIR
//! APIs that are missing from melior.

use crate::error::{Error, Result};
use llvm_sys::{
    core::{
        LLVMContextCreate, LLVMContextDispose, LLVMDisposeMemoryBuffer, LLVMDisposeMessage,
        LLVMDisposeModule, LLVMGetBufferSize, LLVMGetBufferStart,
    },
    error::LLVMGetErrorMessage,
    prelude::LLVMMemoryBufferRef,
    target::{
        LLVM_InitializeAllAsmParsers, LLVM_InitializeAllAsmPrinters, LLVM_InitializeAllTargetInfos,
        LLVM_InitializeAllTargetMCs, LLVM_InitializeAllTargets,
    },
    target_machine::{
        LLVMCodeGenFileType, LLVMCodeGenOptLevel, LLVMCodeModel, LLVMCreateTargetMachine,
        LLVMDisposeTargetMachine, LLVMGetDefaultTargetTriple, LLVMGetHostCPUFeatures,
        LLVMGetHostCPUName, LLVMGetTargetFromTriple, LLVMRelocMode,
        LLVMTargetMachineEmitToMemoryBuffer, LLVMTargetRef,
    },
    transforms::pass_builder::{
        LLVMCreatePassBuilderOptions, LLVMDisposePassBuilderOptions, LLVMRunPasses,
    },
};
use melior::ir::{Module, Type, TypeLike};
use mlir_sys::{mlirLLVMStructTypeGetElementType, mlirTranslateModuleToLLVMIR};
use std::{
    borrow::Cow,
    env,
    ffi::{CStr, CString},
    io::Write,
    mem::MaybeUninit,
    path::Path,
    ptr::{addr_of_mut, null_mut},
    sync::OnceLock,
};
use tempfile::NamedTempFile;

/// For any `!llvm.struct<...>` type, return the MLIR type of the field at the requested index.
pub fn get_struct_field_type_at<'c>(r#type: &Type<'c>, index: usize) -> Type<'c> {
    assert!(r#type.is_llvm_struct_type());
    unsafe {
        Type::from_raw(mlirLLVMStructTypeGetElementType(
            r#type.to_raw(),
            index as isize,
        ))
    }
}

/// Optimization levels.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum OptLevel {
    None,
    Less,
    #[default]
    Default,
    Aggressive,
}

impl From<usize> for OptLevel {
    fn from(value: usize) -> Self {
        match value {
            0 => OptLevel::None,
            1 => OptLevel::Less,
            2 => OptLevel::Default,
            _ => OptLevel::Aggressive,
        }
    }
}

impl From<OptLevel> for usize {
    fn from(val: OptLevel) -> Self {
        match val {
            OptLevel::None => 0,
            OptLevel::Less => 1,
            OptLevel::Default => 2,
            OptLevel::Aggressive => 3,
        }
    }
}

impl From<u8> for OptLevel {
    fn from(value: u8) -> Self {
        match value {
            0 => OptLevel::None,
            1 => OptLevel::Less,
            2 => OptLevel::Default,
            _ => OptLevel::Aggressive,
        }
    }
}

/// Converts a MLIR module to a compile object, that can be linked with a linker.
pub fn module_to_object(module: &Module<'_>, opt_level: OptLevel) -> Result<Vec<u8>> {
    static INITIALIZED: OnceLock<()> = OnceLock::new();

    INITIALIZED.get_or_init(|| unsafe {
        LLVM_InitializeAllTargets();
        LLVM_InitializeAllTargetInfos();
        LLVM_InitializeAllTargetMCs();
        LLVM_InitializeAllAsmPrinters();
        LLVM_InitializeAllAsmParsers();
    });

    unsafe {
        let llvm_context = LLVMContextCreate();

        let op = module.as_operation().to_raw();

        let llvm_module = mlirTranslateModuleToLLVMIR(op, llvm_context as *mut _) as *mut _;

        let mut null = null_mut();
        let mut error_buffer = addr_of_mut!(null);

        let target_triple = LLVMGetDefaultTargetTriple();
        let target_cpu = LLVMGetHostCPUName();
        let target_cpu_features = LLVMGetHostCPUFeatures();

        let mut target: MaybeUninit<LLVMTargetRef> = MaybeUninit::uninit();

        if LLVMGetTargetFromTriple(target_triple, target.as_mut_ptr(), error_buffer) != 0 {
            let error = CStr::from_ptr(*error_buffer);
            let err = error.to_string_lossy().to_string();
            LLVMDisposeMessage(*error_buffer);
            Err(Error::LLVMCompileError(err))?;
        } else if !(*error_buffer).is_null() {
            LLVMDisposeMessage(*error_buffer);
            error_buffer = addr_of_mut!(null);
        }

        let target = target.assume_init();

        let machine = LLVMCreateTargetMachine(
            target,
            target_triple.cast(),
            target_cpu.cast(),
            target_cpu_features.cast(),
            match opt_level {
                OptLevel::None => LLVMCodeGenOptLevel::LLVMCodeGenLevelNone,
                OptLevel::Less => LLVMCodeGenOptLevel::LLVMCodeGenLevelLess,
                OptLevel::Default => LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault,
                OptLevel::Aggressive => LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
            },
            LLVMRelocMode::LLVMRelocDynamicNoPic,
            LLVMCodeModel::LLVMCodeModelDefault,
        );

        let opts = LLVMCreatePassBuilderOptions();

        let opt = match opt_level {
            OptLevel::None => 0,
            OptLevel::Less => 1,
            OptLevel::Default => 1, // todo: change once slp-vectorizer pass is fixed on llvm
            OptLevel::Aggressive => 1, // https://github.com/llvm/llvm-project/issues/107198
        };
        let passes = CString::new(format!("default<O{opt}>")).unwrap();
        let error = LLVMRunPasses(llvm_module, passes.as_ptr(), machine, opts);
        if !error.is_null() {
            let msg = LLVMGetErrorMessage(error);
            let msg = CStr::from_ptr(msg);
            Err(Error::LLVMCompileError(msg.to_string_lossy().into_owned()))?;
        }

        LLVMDisposePassBuilderOptions(opts);

        let mut out_buf: MaybeUninit<LLVMMemoryBufferRef> = MaybeUninit::uninit();

        let ok = LLVMTargetMachineEmitToMemoryBuffer(
            machine,
            llvm_module,
            LLVMCodeGenFileType::LLVMObjectFile,
            error_buffer,
            out_buf.as_mut_ptr(),
        );

        if ok != 0 {
            let error = CStr::from_ptr(*error_buffer);
            let err = error.to_string_lossy().to_string();
            LLVMDisposeMessage(*error_buffer);
            Err(Error::LLVMCompileError(err))?;
        } else if !(*error_buffer).is_null() {
            LLVMDisposeMessage(*error_buffer);
        }

        let out_buf = out_buf.assume_init();

        let out_buf_start: *const u8 = LLVMGetBufferStart(out_buf).cast();
        let out_buf_size = LLVMGetBufferSize(out_buf);

        // keep it in rust side
        let data = std::slice::from_raw_parts(out_buf_start, out_buf_size).to_vec();

        LLVMDisposeMemoryBuffer(out_buf);
        LLVMDisposeTargetMachine(machine);
        LLVMDisposeModule(llvm_module);
        LLVMContextDispose(llvm_context);

        Ok(data)
    }
}

/// Links the passed object into a shared library, stored on the given path.
pub fn object_to_shared_lib(object: &[u8], output_filename: &Path) -> Result<()> {
    // linker seems to need a file and doesn't accept stdin
    let mut file = NamedTempFile::new()?;
    file.write_all(object)?;
    let file = file.into_temp_path();

    let file_path = file.display().to_string();
    let output_path = output_filename.display().to_string();
    if let Ok(x) = std::env::var("NATIVE_DEBUG_DUMP") {
        if x == "1" || x == "true" {
            // forget so the temp file is not deleted and the debugger can load it.
            // its still in a temp file directory so eventually the OS will delete it, but just not instantly.
            // todo: maybe remove it when exiting, for example using atexit.
            std::mem::forget(file);
        }
    }

    let runtime_library_path = if let Ok(extra_dir) = std::env::var("CAIRO_NATIVE_RUNTIME_LIBRARY")
    {
        let path = Path::new(&extra_dir);
        if path.is_absolute() {
            extra_dir
        } else {
            let mut absolute_path = env::current_dir()
                .expect("Failed to get the current directory")
                .join(path);
            absolute_path = absolute_path
                .canonicalize()
                .expect("Failed to cannonicalize path");
            String::from(
                absolute_path
                    .to_str()
                    .expect("Absolute path contains non-utf8 characters"),
            )
        }
    } else {
        String::from("libcairo_native_runtime.a")
    };

    let args: Vec<Cow<'static, str>> = {
        #[cfg(target_os = "macos")]
        {
            let mut args: Vec<Cow<'static, str>> = vec![
                "-demangle".into(),
                "-no_deduplicate".into(),
                "-dynamic".into(),
                "-dylib".into(),
                "-L/usr/local/lib".into(),
                "-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib".into(),
            ];

            args.extend([
                Cow::from(file_path),
                "-o".into(),
                Cow::from(output_path),
                "-lSystem".into(),
                Cow::from(runtime_library_path),
            ]);

            args
        }
        #[cfg(target_os = "linux")]
        {
            let mut args: Vec<Cow<'static, str>> = vec![
                "--hash-style=gnu".into(),
                "--eh-frame-hdr".into(),
                "-shared".into(),
                "-L/lib/../lib64".into(),
                "-L/usr/lib/../lib64".into(),
            ];

            args.extend([
                "-o".into(),
                Cow::from(output_path),
                "-lc".into(),
                Cow::from(file_path),
                Cow::from(runtime_library_path),
            ]);

            args
        }
        #[cfg(target_os = "windows")]
        {
            unimplemented!()
        }
    };

    let mut linker = std::process::Command::new("ld");
    let proc = linker.args(args.iter().map(|x| x.as_ref())).output()?;
    if proc.status.success() {
        Ok(())
    } else {
        let msg = String::from_utf8_lossy(&proc.stderr);
        Err(Error::LinkError(msg.to_string()))
    }
}

/// Gets the target triple, which identifies the platform and ABI.
pub fn get_target_triple() -> String {
    let target_triple = unsafe {
        let value = LLVMGetDefaultTargetTriple();
        CStr::from_ptr(value).to_string_lossy().into_owned()
    };
    target_triple
}

/// Gets the data layout reprrsentation as a string, to be given to the MLIR module.
/// LLVM uses this to know the proper alignments for the given sizes, etc.
/// This function gets the data layout of the host target triple.
pub fn get_data_layout_rep() -> Result<String> {
    unsafe {
        let mut null = null_mut();
        let error_buffer = addr_of_mut!(null);

        let target_triple = LLVMGetDefaultTargetTriple();

        let target_cpu = LLVMGetHostCPUName();

        let target_cpu_features = LLVMGetHostCPUFeatures();

        let mut target: MaybeUninit<LLVMTargetRef> = MaybeUninit::uninit();

        if LLVMGetTargetFromTriple(target_triple, target.as_mut_ptr(), error_buffer) != 0 {
            let error = CStr::from_ptr(*error_buffer);
            let err = error.to_string_lossy().to_string();
            tracing::error!("error getting target triple: {}", err);
            LLVMDisposeMessage(*error_buffer);
            Err(Error::LLVMCompileError(err))?;
        }
        if !(*error_buffer).is_null() {
            LLVMDisposeMessage(*error_buffer);
        }

        let target = target.assume_init();

        let machine = LLVMCreateTargetMachine(
            target,
            target_triple.cast(),
            target_cpu.cast(),
            target_cpu_features.cast(),
            LLVMCodeGenOptLevel::LLVMCodeGenLevelNone,
            LLVMRelocMode::LLVMRelocDynamicNoPic,
            LLVMCodeModel::LLVMCodeModelDefault,
        );

        let data_layout = llvm_sys::target_machine::LLVMCreateTargetDataLayout(machine);
        let data_layout_str =
            CStr::from_ptr(llvm_sys::target::LLVMCopyStringRepOfTargetData(data_layout));
        Ok(data_layout_str.to_string_lossy().into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opt_level_default() {
        // Asserts that the default implementation of `OptLevel` returns `OptLevel::Default`.
        assert_eq!(OptLevel::default(), OptLevel::Default);

        // Asserts that converting from usize value 2 returns `OptLevel::Default`.
        assert_eq!(OptLevel::from(2usize), OptLevel::Default);

        // Asserts that converting from u8 value 2 returns `OptLevel::Default`.
        assert_eq!(OptLevel::from(2u8), OptLevel::Default);
    }

    #[test]
    fn test_opt_level_conversion() {
        // Test conversion from usize to OptLevel
        assert_eq!(OptLevel::from(0usize), OptLevel::None);
        assert_eq!(OptLevel::from(1usize), OptLevel::Less);
        assert_eq!(OptLevel::from(2usize), OptLevel::Default);
        assert_eq!(OptLevel::from(3usize), OptLevel::Aggressive);
        assert_eq!(OptLevel::from(30usize), OptLevel::Aggressive);

        // Test conversion from OptLevel to usize
        assert_eq!(usize::from(OptLevel::None), 0usize);
        assert_eq!(usize::from(OptLevel::Less), 1usize);
        assert_eq!(usize::from(OptLevel::Default), 2usize);
        assert_eq!(usize::from(OptLevel::Aggressive), 3usize);

        // Test conversion from u8 to OptLevel
        assert_eq!(OptLevel::from(0u8), OptLevel::None);
        assert_eq!(OptLevel::from(1u8), OptLevel::Less);
        assert_eq!(OptLevel::from(2u8), OptLevel::Default);
        assert_eq!(OptLevel::from(3u8), OptLevel::Aggressive);
        assert_eq!(OptLevel::from(30u8), OptLevel::Aggressive);
    }
}
