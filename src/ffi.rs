//! # FFI Wrappers
//!
//! This is a "hotfix" for missing Rust interfaces to the C/C++ libraries we use, namely LLVM/MLIR
//! APIs that are missing from melior.

use crate::error::Error as CompileError;
use llvm_sys::{
    core::{
        LLVMContextCreate, LLVMContextDispose, LLVMDisposeMemoryBuffer, LLVMDisposeMessage,
        LLVMDisposeModule, LLVMGetBufferSize, LLVMGetBufferStart,
    },
    prelude::{LLVMContextRef, LLVMMemoryBufferRef, LLVMModuleRef},
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
};
use melior::ir::{Module, Type, TypeLike};
use mlir_sys::{MlirAttribute, MlirContext, MlirModule, MlirOperation};
use std::{
    borrow::Cow,
    error::Error,
    ffi::{c_void, CStr},
    fmt::Display,
    io::Write,
    mem::MaybeUninit,
    path::Path,
    ptr::{addr_of_mut, null_mut},
    sync::OnceLock,
};
use tempfile::NamedTempFile;

#[repr(C)]
#[allow(unused)]
pub enum DiEmissionKind {
    None,
    Full,
    LineTablesOnly,
    DebugDirectivesOnly,
}

#[repr(C)]
#[allow(unused)]
pub enum MlirLLVMTypeEncoding {
    Address = 0x1,
    Boolean = 0x2,
    ComplexFloat = 0x31,
    FloatT = 0x4,
    Signed = 0x5,
    SignedChar = 0x6,
    Unsigned = 0x7,
    UnsignedChar = 0x08,
    ImaginaryFloat = 0x09,
    PackedDecimal = 0x0a,
    NumericString = 0x0b,
    Edited = 0x0c,
    SignedFixed = 0x0d,
    UnsignedFixed = 0x0e,
    DecimalFloat = 0x0f,
    Utf = 0x10,
    Ucs = 0x11,
    Ascii = 0x12,
    LoUser = 0x80,
    HiUser = 0xff,
}

#[repr(C)]
#[allow(unused)]
#[allow(non_camel_case_types)]
pub enum MlirLLVMDWTag {
    DW_TAG_null = 0x00,

    DW_TAG_array_type = 0x01,
    DW_TAG_class_type = 0x02,
    DW_TAG_entry_point = 0x03,
    DW_TAG_enumeration_type = 0x04,
    DW_TAG_formal_parameter = 0x05,
    DW_TAG_imported_declaration = 0x08,
    DW_TAG_label = 0x0a,
    DW_TAG_lexical_block = 0x0b,
    DW_TAG_member = 0x0d,
    DW_TAG_pointer_type = 0x0f,
    DW_TAG_reference_type = 0x10,
    DW_TAG_compile_unit = 0x11,
    DW_TAG_string_type = 0x12,
    DW_TAG_structure_type = 0x13,
    DW_TAG_subroutine_type = 0x15,
    DW_TAG_typedef = 0x16,
    DW_TAG_union_type = 0x17,
    DW_TAG_unspecified_parameters = 0x18,
    DW_TAG_variant = 0x19,
    DW_TAG_common_block = 0x1a,
    DW_TAG_common_inclusion = 0x1b,
    DW_TAG_inheritance = 0x1c,
    DW_TAG_inlined_subroutine = 0x1d,
    DW_TAG_module = 0x1e,
    DW_TAG_ptr_to_member_type = 0x1f,
    DW_TAG_set_type = 0x20,
    DW_TAG_subrange_type = 0x21,
    DW_TAG_with_stmt = 0x22,
    DW_TAG_access_declaration = 0x23,
    DW_TAG_base_type = 0x24,
    DW_TAG_catch_block = 0x25,
    DW_TAG_const_type = 0x26,
    DW_TAG_constant = 0x27,
    DW_TAG_enumerator = 0x28,
    DW_TAG_file_type = 0x29,
    DW_TAG_friend = 0x2a,
    DW_TAG_namelist = 0x2b,
    DW_TAG_namelist_item = 0x2c,
    DW_TAG_packed_type = 0x2d,
    DW_TAG_subprogram = 0x2e,
    DW_TAG_template_type_parameter = 0x2f,
    DW_TAG_template_value_parameter = 0x30,
    DW_TAG_thrown_type = 0x31,
    DW_TAG_try_block = 0x32,
    DW_TAG_variant_part = 0x33,
    DW_TAG_variable = 0x34,
    DW_TAG_volatile_type = 0x35,
    // DWARF 3.
    DW_TAG_dwarf_procedure = 0x36,
    DW_TAG_restrict_type = 0x37,
    DW_TAG_interface_type = 0x38,
    DW_TAG_namespace = 0x39,
    DW_TAG_imported_module = 0x3a,
    DW_TAG_unspecified_type = 0x3b,
    DW_TAG_partial_unit = 0x3c,
    DW_TAG_imported_unit = 0x3d,
    DW_TAG_condition = 0x3f,
    DW_TAG_shared_type = 0x40,
    // DWARF 4.
    DW_TAG_type_unit = 0x41,
    DW_TAG_rvalue_reference_type = 0x42,
    DW_TAG_template_alias = 0x43,
    // DWARF 5.
    DW_TAG_coarray_type = 0x44,
    DW_TAG_generic_subrange = 0x45,
    DW_TAG_dynamic_type = 0x46,
    DW_TAG_atomic_type = 0x47,
    DW_TAG_call_site = 0x48,
    DW_TAG_call_site_parameter = 0x49,
    DW_TAG_skeleton_unit = 0x4a,
    DW_TAG_immutable_type = 0x4b,
}

extern "C" {
    fn LLVMStructType_getFieldTypeAt(ty_ptr: *const c_void, index: u32) -> *const c_void;

    /// Translate operation that satisfies LLVM dialect module requirements into an LLVM IR module living in the given context.
    /// This translates operations from any dilalect that has a registered implementation of LLVMTranslationDialectInterface.
    fn mlirTranslateModuleToLLVMIR(
        module_operation_ptr: MlirOperation,
        llvm_context: LLVMContextRef,
    ) -> LLVMModuleRef;

    pub fn mlirLLVMDistinctAttrCreate(attr: MlirAttribute) -> MlirAttribute;

    pub fn mlirLLVMDICompileUnitAttrGet(
        mlir_context: MlirContext,
        id: MlirAttribute,
        source_lang: u32,
        file: MlirAttribute,
        producer: MlirAttribute,
        is_optimized: bool,
        emission_kind: DiEmissionKind,
    ) -> MlirAttribute;

    pub fn mlirLLVMDIFileAttrGet(
        mlir_context: MlirContext,
        name: MlirAttribute,
        dir: MlirAttribute,
    ) -> MlirAttribute;

    pub fn mlirLLVMDISubprogramAttrGet(
        mlir_context: MlirContext,
        id: MlirAttribute,
        compile_unit: MlirAttribute,
        scope: MlirAttribute,
        name: MlirAttribute,
        linkage_name: MlirAttribute,
        file: MlirAttribute,
        line: u32,
        scope_line: u32,
        subprogram_flags: i32,
        ty: MlirAttribute,
    ) -> MlirAttribute;

    pub fn mlirLLVMDISubroutineTypeAttrGet(
        mlir_context: MlirContext,
        cconv: u32,
        ntypes: usize,
        types: *const MlirAttribute,
    ) -> MlirAttribute;

    pub fn mlirLLVMDIBasicTypeAttrGet(
        mlir_context: MlirContext,
        tag: u32,
        name: MlirAttribute,
        size_in_bits: u64,
        encoding: MlirLLVMTypeEncoding,
    ) -> MlirAttribute;

    pub fn mlirLLVMDIModuleAttrGet(
        mlir_context: MlirContext,
        file: MlirAttribute,
        scope: MlirAttribute,
        name: MlirAttribute,
        configMacros: MlirAttribute,
        includePath: MlirAttribute,
        apinotes: MlirAttribute,
        line: u32,
        is_decl: bool,
    ) -> MlirAttribute;

    pub fn mlirLLVMDIModuleAttrGetScope(di_module: MlirAttribute) -> MlirAttribute;

    pub fn mlirLLVMDILexicalBlockAttrGet(
        mlir_context: MlirContext,
        scope: MlirAttribute,
        file: MlirAttribute,
        line: u32,
        column: u32,
    ) -> MlirAttribute;

    pub fn mlirModuleCleanup(module: MlirModule);

    pub fn mlirLLVMDIDerivedTypeAttrGet(
        mlir_context: MlirContext,
        tag: MlirLLVMDWTag,
        name: MlirAttribute,
        base_type: MlirAttribute,
        size_in_bits: u64,
        align_in_bits: u64,
        offset_in_bits: u64,
    ) -> MlirAttribute;

    pub fn mlirLLVMDICompositeTypeAttrGet(
        mlir_context: MlirContext,
        tag: MlirLLVMDWTag,
        name: MlirAttribute,
        file: MlirAttribute,
        line: u32,
        scope: MlirAttribute,
        base_type: MlirAttribute,
        flags: u64,
        size_in_bits: u64,
        align_in_bits: u64,
        n_elements: usize,
        elements: *const MlirAttribute,
    ) -> MlirAttribute;

    pub fn mlirLLVMDINullTypeAttrGet(mlir_context: MlirContext) -> MlirAttribute;
}

/// For any `!llvm.struct<...>` type, return the MLIR type of the field at the requested index.
pub fn get_struct_field_type_at<'c>(r#type: &Type<'c>, index: usize) -> Type<'c> {
    let mut ty_ptr = r#type.to_raw();

    ty_ptr.ptr = unsafe { LLVMStructType_getFieldTypeAt(ty_ptr.ptr, index as u32) };
    unsafe { Type::from_raw(ty_ptr) }
}

/// A error from the LLVM API.
#[derive(Debug, Clone)]
pub struct LLVMCompileError(String);

impl Error for LLVMCompileError {}

impl Display for LLVMCompileError {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
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
pub fn module_to_object(
    module: &Module<'_>,
    opt_level: OptLevel,
) -> Result<Vec<u8>, LLVMCompileError> {
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

        let llvm_module = mlirTranslateModuleToLLVMIR(op, llvm_context);

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
            Err(LLVMCompileError(err))?;
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
            Err(LLVMCompileError(err))?;
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
pub fn object_to_shared_lib(object: &[u8], output_filename: &Path) -> Result<(), std::io::Error> {
    // linker seems to need a file and doesn't accept stdin
    let mut file = NamedTempFile::new()?;
    file.write_all(object)?;
    let file = file.into_temp_path();

    let file_path = file.display().to_string();
    let output_path = output_filename.display().to_string();

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
            ]);

            if let Ok(extra_dir) = std::env::var("CAIRO_NATIVE_RUNTIME_LIBRARY") {
                args.extend([Cow::from(extra_dir)]);
            } else {
                args.extend(["libcairo_native_runtime.a".into()]);
            }

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
                //"-lcairo_native_runtime".into(),
                Cow::from(file_path),
            ]);

            if let Ok(extra_dir) = std::env::var("CAIRO_NATIVE_RUNTIME_LIBRARY") {
                args.extend([Cow::from(extra_dir)]);
            } else {
                args.extend(["libcairo_native_runtime.a".into()]);
            }

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
        panic!("error linking:\n{}", msg);
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
pub fn get_data_layout_rep() -> Result<String, CompileError> {
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
            Err(CompileError::LLVMCompileError(err))?;
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
