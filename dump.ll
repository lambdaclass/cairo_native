; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin23.5.0"

define i32 @"program::program::main(f0)"() !dbg !3 {
  ret i32 2, !dbg !7
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !2, producer: "cairo-native", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!2 = !DIFile(filename: "program.sierra", directory: "")
!3 = distinct !DISubprogram(name: "program::program::main(f0)", linkageName: "program::program::main(f0)", scope: !2, file: !2, line: 1, type: !4, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !1)
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DIBasicType(name: "mytype", size: 64, encoding: DW_ATE_signed)
!7 = !DILocation(line: 0, scope: !3)
