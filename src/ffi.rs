////! # FFI Wrappers
//! # FFI Wrappers
////!
//!
////! This is a "hotfix" for missing Rust interfaces to the C/C++ libraries we use, namely LLVM/MLIR
//! This is a "hotfix" for missing Rust interfaces to the C/C++ libraries we use, namely LLVM/MLIR
////! APIs that are missing from melior.
//! APIs that are missing from melior.
//

//use crate::error::Error as CompileError;
use crate::error::Error as CompileError;
//use llvm_sys::{
use llvm_sys::{
//    core::{
    core::{
//        LLVMContextCreate, LLVMContextDispose, LLVMDisposeMemoryBuffer, LLVMDisposeMessage,
        LLVMContextCreate, LLVMContextDispose, LLVMDisposeMemoryBuffer, LLVMDisposeMessage,
//        LLVMDisposeModule, LLVMGetBufferSize, LLVMGetBufferStart,
        LLVMDisposeModule, LLVMGetBufferSize, LLVMGetBufferStart,
//    },
    },
//    prelude::{LLVMContextRef, LLVMMemoryBufferRef, LLVMModuleRef},
    prelude::{LLVMContextRef, LLVMMemoryBufferRef, LLVMModuleRef},
//    target::{
    target::{
//        LLVM_InitializeAllAsmParsers, LLVM_InitializeAllAsmPrinters, LLVM_InitializeAllTargetInfos,
        LLVM_InitializeAllAsmParsers, LLVM_InitializeAllAsmPrinters, LLVM_InitializeAllTargetInfos,
//        LLVM_InitializeAllTargetMCs, LLVM_InitializeAllTargets,
        LLVM_InitializeAllTargetMCs, LLVM_InitializeAllTargets,
//    },
    },
//    target_machine::{
    target_machine::{
//        LLVMCodeGenFileType, LLVMCodeGenOptLevel, LLVMCodeModel, LLVMCreateTargetMachine,
        LLVMCodeGenFileType, LLVMCodeGenOptLevel, LLVMCodeModel, LLVMCreateTargetMachine,
//        LLVMDisposeTargetMachine, LLVMGetDefaultTargetTriple, LLVMGetHostCPUFeatures,
        LLVMDisposeTargetMachine, LLVMGetDefaultTargetTriple, LLVMGetHostCPUFeatures,
//        LLVMGetHostCPUName, LLVMGetTargetFromTriple, LLVMRelocMode,
        LLVMGetHostCPUName, LLVMGetTargetFromTriple, LLVMRelocMode,
//        LLVMTargetMachineEmitToMemoryBuffer, LLVMTargetRef,
        LLVMTargetMachineEmitToMemoryBuffer, LLVMTargetRef,
//    },
    },
//};
};
//use melior::ir::{Module, Type, TypeLike};
use melior::ir::{Module, Type, TypeLike};
//use mlir_sys::MlirOperation;
use mlir_sys::MlirOperation;
//use std::{
use std::{
//    borrow::Cow,
    borrow::Cow,
//    error::Error,
    error::Error,
//    ffi::{c_void, CStr},
    ffi::{c_void, CStr},
//    fmt::Display,
    fmt::Display,
//    io::Write,
    io::Write,
//    mem::MaybeUninit,
    mem::MaybeUninit,
//    path::Path,
    path::Path,
//    ptr::{addr_of_mut, null_mut},
    ptr::{addr_of_mut, null_mut},
//    sync::OnceLock,
    sync::OnceLock,
//};
};
//use tempfile::NamedTempFile;
use tempfile::NamedTempFile;
//

//extern "C" {
extern "C" {
//    fn LLVMStructType_getFieldTypeAt(ty_ptr: *const c_void, index: u32) -> *const c_void;
    fn LLVMStructType_getFieldTypeAt(ty_ptr: *const c_void, index: u32) -> *const c_void;
//

//    /// Translate operation that satisfies LLVM dialect module requirements into an LLVM IR module living in the given context.
    /// Translate operation that satisfies LLVM dialect module requirements into an LLVM IR module living in the given context.
//    /// This translates operations from any dilalect that has a registered implementation of LLVMTranslationDialectInterface.
    /// This translates operations from any dilalect that has a registered implementation of LLVMTranslationDialectInterface.
//    fn mlirTranslateModuleToLLVMIR(
    fn mlirTranslateModuleToLLVMIR(
//        module_operation_ptr: MlirOperation,
        module_operation_ptr: MlirOperation,
//        llvm_context: LLVMContextRef,
        llvm_context: LLVMContextRef,
//    ) -> LLVMModuleRef;
    ) -> LLVMModuleRef;
//}
}
//

///// For any `!llvm.struct<...>` type, return the MLIR type of the field at the requested index.
/// For any `!llvm.struct<...>` type, return the MLIR type of the field at the requested index.
//pub fn get_struct_field_type_at<'c>(r#type: &Type<'c>, index: usize) -> Type<'c> {
pub fn get_struct_field_type_at<'c>(r#type: &Type<'c>, index: usize) -> Type<'c> {
//    let mut ty_ptr = r#type.to_raw();
    let mut ty_ptr = r#type.to_raw();
//

//    ty_ptr.ptr = unsafe { LLVMStructType_getFieldTypeAt(ty_ptr.ptr, index as u32) };
    ty_ptr.ptr = unsafe { LLVMStructType_getFieldTypeAt(ty_ptr.ptr, index as u32) };
//    unsafe { Type::from_raw(ty_ptr) }
    unsafe { Type::from_raw(ty_ptr) }
//}
}
//

///// A error from the LLVM API.
/// A error from the LLVM API.
//#[derive(Debug, Clone)]
#[derive(Debug, Clone)]
//pub struct LLVMCompileError(String);
pub struct LLVMCompileError(String);
//

//impl Error for LLVMCompileError {}
impl Error for LLVMCompileError {}
//

//impl Display for LLVMCompileError {
impl Display for LLVMCompileError {
//    #[inline]
    #[inline]
//    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//        f.write_str(&self.0)
        f.write_str(&self.0)
//    }
    }
//}
}
//

///// Optimization levels.
/// Optimization levels.
//#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
//pub enum OptLevel {
pub enum OptLevel {
//    None,
    None,
//    Less,
    Less,
//    #[default]
    #[default]
//    Default,
    Default,
//    Aggressive,
    Aggressive,
//}
}
//

//impl From<usize> for OptLevel {
impl From<usize> for OptLevel {
//    fn from(value: usize) -> Self {
    fn from(value: usize) -> Self {
//        match value {
        match value {
//            0 => OptLevel::None,
            0 => OptLevel::None,
//            1 => OptLevel::Less,
            1 => OptLevel::Less,
//            2 => OptLevel::Default,
            2 => OptLevel::Default,
//            _ => OptLevel::Aggressive,
            _ => OptLevel::Aggressive,
//        }
        }
//    }
    }
//}
}
//

//impl From<OptLevel> for usize {
impl From<OptLevel> for usize {
//    fn from(val: OptLevel) -> Self {
    fn from(val: OptLevel) -> Self {
//        match val {
        match val {
//            OptLevel::None => 0,
            OptLevel::None => 0,
//            OptLevel::Less => 1,
            OptLevel::Less => 1,
//            OptLevel::Default => 2,
            OptLevel::Default => 2,
//            OptLevel::Aggressive => 3,
            OptLevel::Aggressive => 3,
//        }
        }
//    }
    }
//}
}
//

//impl From<u8> for OptLevel {
impl From<u8> for OptLevel {
//    fn from(value: u8) -> Self {
    fn from(value: u8) -> Self {
//        match value {
        match value {
//            0 => OptLevel::None,
            0 => OptLevel::None,
//            1 => OptLevel::Less,
            1 => OptLevel::Less,
//            2 => OptLevel::Default,
            2 => OptLevel::Default,
//            _ => OptLevel::Aggressive,
            _ => OptLevel::Aggressive,
//        }
        }
//    }
    }
//}
}
//

///// Converts a MLIR module to a compile object, that can be linked with a linker.
/// Converts a MLIR module to a compile object, that can be linked with a linker.
//pub fn module_to_object(
pub fn module_to_object(
//    module: &Module<'_>,
    module: &Module<'_>,
//    opt_level: OptLevel,
    opt_level: OptLevel,
//) -> Result<Vec<u8>, LLVMCompileError> {
) -> Result<Vec<u8>, LLVMCompileError> {
//    static INITIALIZED: OnceLock<()> = OnceLock::new();
    static INITIALIZED: OnceLock<()> = OnceLock::new();
//

//    INITIALIZED.get_or_init(|| unsafe {
    INITIALIZED.get_or_init(|| unsafe {
//        LLVM_InitializeAllTargets();
        LLVM_InitializeAllTargets();
//        LLVM_InitializeAllTargetInfos();
        LLVM_InitializeAllTargetInfos();
//        LLVM_InitializeAllTargetMCs();
        LLVM_InitializeAllTargetMCs();
//        LLVM_InitializeAllAsmPrinters();
        LLVM_InitializeAllAsmPrinters();
//        LLVM_InitializeAllAsmParsers();
        LLVM_InitializeAllAsmParsers();
//    });
    });
//

//    unsafe {
    unsafe {
//        let llvm_context = LLVMContextCreate();
        let llvm_context = LLVMContextCreate();
//

//        let op = module.as_operation().to_raw();
        let op = module.as_operation().to_raw();
//

//        let llvm_module = mlirTranslateModuleToLLVMIR(op, llvm_context);
        let llvm_module = mlirTranslateModuleToLLVMIR(op, llvm_context);
//

//        let mut null = null_mut();
        let mut null = null_mut();
//        let mut error_buffer = addr_of_mut!(null);
        let mut error_buffer = addr_of_mut!(null);
//

//        let target_triple = LLVMGetDefaultTargetTriple();
        let target_triple = LLVMGetDefaultTargetTriple();
//        let target_cpu = LLVMGetHostCPUName();
        let target_cpu = LLVMGetHostCPUName();
//        let target_cpu_features = LLVMGetHostCPUFeatures();
        let target_cpu_features = LLVMGetHostCPUFeatures();
//

//        let mut target: MaybeUninit<LLVMTargetRef> = MaybeUninit::uninit();
        let mut target: MaybeUninit<LLVMTargetRef> = MaybeUninit::uninit();
//

//        if LLVMGetTargetFromTriple(target_triple, target.as_mut_ptr(), error_buffer) != 0 {
        if LLVMGetTargetFromTriple(target_triple, target.as_mut_ptr(), error_buffer) != 0 {
//            let error = CStr::from_ptr(*error_buffer);
            let error = CStr::from_ptr(*error_buffer);
//            let err = error.to_string_lossy().to_string();
            let err = error.to_string_lossy().to_string();
//            LLVMDisposeMessage(*error_buffer);
            LLVMDisposeMessage(*error_buffer);
//            Err(LLVMCompileError(err))?;
            Err(LLVMCompileError(err))?;
//        } else if !(*error_buffer).is_null() {
        } else if !(*error_buffer).is_null() {
//            LLVMDisposeMessage(*error_buffer);
            LLVMDisposeMessage(*error_buffer);
//            error_buffer = addr_of_mut!(null);
            error_buffer = addr_of_mut!(null);
//        }
        }
//

//        let target = target.assume_init();
        let target = target.assume_init();
//

//        let machine = LLVMCreateTargetMachine(
        let machine = LLVMCreateTargetMachine(
//            target,
            target,
//            target_triple.cast(),
            target_triple.cast(),
//            target_cpu.cast(),
            target_cpu.cast(),
//            target_cpu_features.cast(),
            target_cpu_features.cast(),
//            match opt_level {
            match opt_level {
//                OptLevel::None => LLVMCodeGenOptLevel::LLVMCodeGenLevelNone,
                OptLevel::None => LLVMCodeGenOptLevel::LLVMCodeGenLevelNone,
//                OptLevel::Less => LLVMCodeGenOptLevel::LLVMCodeGenLevelLess,
                OptLevel::Less => LLVMCodeGenOptLevel::LLVMCodeGenLevelLess,
//                OptLevel::Default => LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault,
                OptLevel::Default => LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault,
//                OptLevel::Aggressive => LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
                OptLevel::Aggressive => LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
//            },
            },
//            LLVMRelocMode::LLVMRelocDynamicNoPic,
            LLVMRelocMode::LLVMRelocDynamicNoPic,
//            LLVMCodeModel::LLVMCodeModelDefault,
            LLVMCodeModel::LLVMCodeModelDefault,
//        );
        );
//

//        let mut out_buf: MaybeUninit<LLVMMemoryBufferRef> = MaybeUninit::uninit();
        let mut out_buf: MaybeUninit<LLVMMemoryBufferRef> = MaybeUninit::uninit();
//

//        let ok = LLVMTargetMachineEmitToMemoryBuffer(
        let ok = LLVMTargetMachineEmitToMemoryBuffer(
//            machine,
            machine,
//            llvm_module,
            llvm_module,
//            LLVMCodeGenFileType::LLVMObjectFile,
            LLVMCodeGenFileType::LLVMObjectFile,
//            error_buffer,
            error_buffer,
//            out_buf.as_mut_ptr(),
            out_buf.as_mut_ptr(),
//        );
        );
//

//        if ok != 0 {
        if ok != 0 {
//            let error = CStr::from_ptr(*error_buffer);
            let error = CStr::from_ptr(*error_buffer);
//            let err = error.to_string_lossy().to_string();
            let err = error.to_string_lossy().to_string();
//            LLVMDisposeMessage(*error_buffer);
            LLVMDisposeMessage(*error_buffer);
//            Err(LLVMCompileError(err))?;
            Err(LLVMCompileError(err))?;
//        } else if !(*error_buffer).is_null() {
        } else if !(*error_buffer).is_null() {
//            LLVMDisposeMessage(*error_buffer);
            LLVMDisposeMessage(*error_buffer);
//        }
        }
//

//        let out_buf = out_buf.assume_init();
        let out_buf = out_buf.assume_init();
//

//        let out_buf_start: *const u8 = LLVMGetBufferStart(out_buf).cast();
        let out_buf_start: *const u8 = LLVMGetBufferStart(out_buf).cast();
//        let out_buf_size = LLVMGetBufferSize(out_buf);
        let out_buf_size = LLVMGetBufferSize(out_buf);
//

//        // keep it in rust side
        // keep it in rust side
//        let data = std::slice::from_raw_parts(out_buf_start, out_buf_size).to_vec();
        let data = std::slice::from_raw_parts(out_buf_start, out_buf_size).to_vec();
//

//        LLVMDisposeMemoryBuffer(out_buf);
        LLVMDisposeMemoryBuffer(out_buf);
//        LLVMDisposeTargetMachine(machine);
        LLVMDisposeTargetMachine(machine);
//        LLVMDisposeModule(llvm_module);
        LLVMDisposeModule(llvm_module);
//        LLVMContextDispose(llvm_context);
        LLVMContextDispose(llvm_context);
//

//        Ok(data)
        Ok(data)
//    }
    }
//}
}
//

///// Links the passed object into a shared library, stored on the given path.
/// Links the passed object into a shared library, stored on the given path.
//pub fn object_to_shared_lib(object: &[u8], output_filename: &Path) -> Result<(), std::io::Error> {
pub fn object_to_shared_lib(object: &[u8], output_filename: &Path) -> Result<(), std::io::Error> {
//    // linker seems to need a file and doesn't accept stdin
    // linker seems to need a file and doesn't accept stdin
//    let mut file = NamedTempFile::new()?;
    let mut file = NamedTempFile::new()?;
//    file.write_all(object)?;
    file.write_all(object)?;
//    let file = file.into_temp_path();
    let file = file.into_temp_path();
//

//    let file_path = file.display().to_string();
    let file_path = file.display().to_string();
//    let output_path = output_filename.display().to_string();
    let output_path = output_filename.display().to_string();
//

//    let args: Vec<Cow<'static, str>> = {
    let args: Vec<Cow<'static, str>> = {
//        #[cfg(target_os = "macos")]
        #[cfg(target_os = "macos")]
//        {
        {
//            let mut args: Vec<Cow<'static, str>> = vec![
            let mut args: Vec<Cow<'static, str>> = vec![
//                "-demangle".into(),
                "-demangle".into(),
//                "-no_deduplicate".into(),
                "-no_deduplicate".into(),
//                "-dynamic".into(),
                "-dynamic".into(),
//                "-dylib".into(),
                "-dylib".into(),
//                "-L/usr/local/lib".into(),
                "-L/usr/local/lib".into(),
//                "-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib".into(),
                "-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib".into(),
//            ];
            ];
//

//            args.extend([
            args.extend([
//                Cow::from(file_path),
                Cow::from(file_path),
//                "-o".into(),
                "-o".into(),
//                Cow::from(output_path),
                Cow::from(output_path),
//                "-lSystem".into(),
                "-lSystem".into(),
//            ]);
            ]);
//

//            if let Ok(extra_dir) = std::env::var("CAIRO_NATIVE_RUNTIME_LIBRARY") {
            if let Ok(extra_dir) = std::env::var("CAIRO_NATIVE_RUNTIME_LIBRARY") {
//                args.extend([Cow::from(extra_dir)]);
                args.extend([Cow::from(extra_dir)]);
//            } else {
            } else {
//                args.extend(["libcairo_native_runtime.a".into()]);
                args.extend(["libcairo_native_runtime.a".into()]);
//            }
            }
//

//            args
            args
//        }
        }
//        #[cfg(target_os = "linux")]
        #[cfg(target_os = "linux")]
//        {
        {
//            let mut args: Vec<Cow<'static, str>> = vec![
            let mut args: Vec<Cow<'static, str>> = vec![
//                "--hash-style=gnu".into(),
                "--hash-style=gnu".into(),
//                "--eh-frame-hdr".into(),
                "--eh-frame-hdr".into(),
//                "-shared".into(),
                "-shared".into(),
//                "-L/lib/../lib64".into(),
                "-L/lib/../lib64".into(),
//                "-L/usr/lib/../lib64".into(),
                "-L/usr/lib/../lib64".into(),
//            ];
            ];
//

//            args.extend([
            args.extend([
//                "-o".into(),
                "-o".into(),
//                Cow::from(output_path),
                Cow::from(output_path),
//                "-lc".into(),
                "-lc".into(),
//                //"-lcairo_native_runtime".into(),
                //"-lcairo_native_runtime".into(),
//                Cow::from(file_path),
                Cow::from(file_path),
//            ]);
            ]);
//

//            if let Ok(extra_dir) = std::env::var("CAIRO_NATIVE_RUNTIME_LIBRARY") {
            if let Ok(extra_dir) = std::env::var("CAIRO_NATIVE_RUNTIME_LIBRARY") {
//                args.extend([Cow::from(extra_dir)]);
                args.extend([Cow::from(extra_dir)]);
//            } else {
            } else {
//                args.extend(["libcairo_native_runtime.a".into()]);
                args.extend(["libcairo_native_runtime.a".into()]);
//            }
            }
//

//            args
            args
//        }
        }
//        #[cfg(target_os = "windows")]
        #[cfg(target_os = "windows")]
//        {
        {
//            unimplemented!()
            unimplemented!()
//        }
        }
//    };
    };
//

//    let mut linker = std::process::Command::new("ld");
    let mut linker = std::process::Command::new("ld");
//    let proc = linker.args(args.iter().map(|x| x.as_ref())).output()?;
    let proc = linker.args(args.iter().map(|x| x.as_ref())).output()?;
//    if proc.status.success() {
    if proc.status.success() {
//        Ok(())
        Ok(())
//    } else {
    } else {
//        let msg = String::from_utf8_lossy(&proc.stderr);
        let msg = String::from_utf8_lossy(&proc.stderr);
//        panic!("error linking:\n{}", msg);
        panic!("error linking:\n{}", msg);
//    }
    }
//}
}
//

///// Gets the target triple, which identifies the platform and ABI.
/// Gets the target triple, which identifies the platform and ABI.
//pub fn get_target_triple() -> String {
pub fn get_target_triple() -> String {
//    let target_triple = unsafe {
    let target_triple = unsafe {
//        let value = LLVMGetDefaultTargetTriple();
        let value = LLVMGetDefaultTargetTriple();
//        CStr::from_ptr(value).to_string_lossy().into_owned()
        CStr::from_ptr(value).to_string_lossy().into_owned()
//    };
    };
//    target_triple
    target_triple
//}
}
//

///// Gets the data layout reprrsentation as a string, to be given to the MLIR module.
/// Gets the data layout reprrsentation as a string, to be given to the MLIR module.
///// LLVM uses this to know the proper alignments for the given sizes, etc.
/// LLVM uses this to know the proper alignments for the given sizes, etc.
///// This function gets the data layout of the host target triple.
/// This function gets the data layout of the host target triple.
//pub fn get_data_layout_rep() -> Result<String, CompileError> {
pub fn get_data_layout_rep() -> Result<String, CompileError> {
//    unsafe {
    unsafe {
//        let mut null = null_mut();
        let mut null = null_mut();
//        let error_buffer = addr_of_mut!(null);
        let error_buffer = addr_of_mut!(null);
//

//        let target_triple = LLVMGetDefaultTargetTriple();
        let target_triple = LLVMGetDefaultTargetTriple();
//

//        let target_cpu = LLVMGetHostCPUName();
        let target_cpu = LLVMGetHostCPUName();
//

//        let target_cpu_features = LLVMGetHostCPUFeatures();
        let target_cpu_features = LLVMGetHostCPUFeatures();
//

//        let mut target: MaybeUninit<LLVMTargetRef> = MaybeUninit::uninit();
        let mut target: MaybeUninit<LLVMTargetRef> = MaybeUninit::uninit();
//

//        if LLVMGetTargetFromTriple(target_triple, target.as_mut_ptr(), error_buffer) != 0 {
        if LLVMGetTargetFromTriple(target_triple, target.as_mut_ptr(), error_buffer) != 0 {
//            let error = CStr::from_ptr(*error_buffer);
            let error = CStr::from_ptr(*error_buffer);
//            let err = error.to_string_lossy().to_string();
            let err = error.to_string_lossy().to_string();
//            tracing::error!("error getting target triple: {}", err);
            tracing::error!("error getting target triple: {}", err);
//            LLVMDisposeMessage(*error_buffer);
            LLVMDisposeMessage(*error_buffer);
//            Err(CompileError::LLVMCompileError(err))?;
            Err(CompileError::LLVMCompileError(err))?;
//        }
        }
//        if !(*error_buffer).is_null() {
        if !(*error_buffer).is_null() {
//            LLVMDisposeMessage(*error_buffer);
            LLVMDisposeMessage(*error_buffer);
//        }
        }
//

//        let target = target.assume_init();
        let target = target.assume_init();
//

//        let machine = LLVMCreateTargetMachine(
        let machine = LLVMCreateTargetMachine(
//            target,
            target,
//            target_triple.cast(),
            target_triple.cast(),
//            target_cpu.cast(),
            target_cpu.cast(),
//            target_cpu_features.cast(),
            target_cpu_features.cast(),
//            LLVMCodeGenOptLevel::LLVMCodeGenLevelNone,
            LLVMCodeGenOptLevel::LLVMCodeGenLevelNone,
//            LLVMRelocMode::LLVMRelocDynamicNoPic,
            LLVMRelocMode::LLVMRelocDynamicNoPic,
//            LLVMCodeModel::LLVMCodeModelDefault,
            LLVMCodeModel::LLVMCodeModelDefault,
//        );
        );
//

//        let data_layout = llvm_sys::target_machine::LLVMCreateTargetDataLayout(machine);
        let data_layout = llvm_sys::target_machine::LLVMCreateTargetDataLayout(machine);
//        let data_layout_str =
        let data_layout_str =
//            CStr::from_ptr(llvm_sys::target::LLVMCopyStringRepOfTargetData(data_layout));
            CStr::from_ptr(llvm_sys::target::LLVMCopyStringRepOfTargetData(data_layout));
//        Ok(data_layout_str.to_string_lossy().into_owned())
        Ok(data_layout_str.to_string_lossy().into_owned())
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
//

//    #[test]
    #[test]
//    fn test_opt_level_default() {
    fn test_opt_level_default() {
//        // Asserts that the default implementation of `OptLevel` returns `OptLevel::Default`.
        // Asserts that the default implementation of `OptLevel` returns `OptLevel::Default`.
//        assert_eq!(OptLevel::default(), OptLevel::Default);
        assert_eq!(OptLevel::default(), OptLevel::Default);
//

//        // Asserts that converting from usize value 2 returns `OptLevel::Default`.
        // Asserts that converting from usize value 2 returns `OptLevel::Default`.
//        assert_eq!(OptLevel::from(2usize), OptLevel::Default);
        assert_eq!(OptLevel::from(2usize), OptLevel::Default);
//

//        // Asserts that converting from u8 value 2 returns `OptLevel::Default`.
        // Asserts that converting from u8 value 2 returns `OptLevel::Default`.
//        assert_eq!(OptLevel::from(2u8), OptLevel::Default);
        assert_eq!(OptLevel::from(2u8), OptLevel::Default);
//    }
    }
//

//    #[test]
    #[test]
//    fn test_opt_level_conversion() {
    fn test_opt_level_conversion() {
//        // Test conversion from usize to OptLevel
        // Test conversion from usize to OptLevel
//        assert_eq!(OptLevel::from(0usize), OptLevel::None);
        assert_eq!(OptLevel::from(0usize), OptLevel::None);
//        assert_eq!(OptLevel::from(1usize), OptLevel::Less);
        assert_eq!(OptLevel::from(1usize), OptLevel::Less);
//        assert_eq!(OptLevel::from(2usize), OptLevel::Default);
        assert_eq!(OptLevel::from(2usize), OptLevel::Default);
//        assert_eq!(OptLevel::from(3usize), OptLevel::Aggressive);
        assert_eq!(OptLevel::from(3usize), OptLevel::Aggressive);
//        assert_eq!(OptLevel::from(30usize), OptLevel::Aggressive);
        assert_eq!(OptLevel::from(30usize), OptLevel::Aggressive);
//

//        // Test conversion from OptLevel to usize
        // Test conversion from OptLevel to usize
//        assert_eq!(usize::from(OptLevel::None), 0usize);
        assert_eq!(usize::from(OptLevel::None), 0usize);
//        assert_eq!(usize::from(OptLevel::Less), 1usize);
        assert_eq!(usize::from(OptLevel::Less), 1usize);
//        assert_eq!(usize::from(OptLevel::Default), 2usize);
        assert_eq!(usize::from(OptLevel::Default), 2usize);
//        assert_eq!(usize::from(OptLevel::Aggressive), 3usize);
        assert_eq!(usize::from(OptLevel::Aggressive), 3usize);
//

//        // Test conversion from u8 to OptLevel
        // Test conversion from u8 to OptLevel
//        assert_eq!(OptLevel::from(0u8), OptLevel::None);
        assert_eq!(OptLevel::from(0u8), OptLevel::None);
//        assert_eq!(OptLevel::from(1u8), OptLevel::Less);
        assert_eq!(OptLevel::from(1u8), OptLevel::Less);
//        assert_eq!(OptLevel::from(2u8), OptLevel::Default);
        assert_eq!(OptLevel::from(2u8), OptLevel::Default);
//        assert_eq!(OptLevel::from(3u8), OptLevel::Aggressive);
        assert_eq!(OptLevel::from(3u8), OptLevel::Aggressive);
//        assert_eq!(OptLevel::from(30u8), OptLevel::Aggressive);
        assert_eq!(OptLevel::from(30u8), OptLevel::Aggressive);
//    }
    }
//}
}
