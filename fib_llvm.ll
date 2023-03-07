; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

define { i256, i256 } @fib(i256 %0, i256 %1, i256 %2) !dbg !3 {
  %4 = icmp eq i256 %2, 0, !dbg !7
  br i1 %4, label %5, label %6, !dbg !9

5:                                                ; preds = %3
  br label %13, !dbg !10

6:                                                ; preds = %3
  %7 = add i256 %0, %1, !dbg !11
  %8 = sub i256 %0, 1, !dbg !12
  %9 = call { i256, i256 } @fib(i256 %1, i256 %7, i256 %8), !dbg !13
  %10 = extractvalue { i256, i256 } %9, 0, !dbg !14
  %11 = extractvalue { i256, i256 } %9, 1, !dbg !15
  %12 = add i256 %11, 1, !dbg !16
  br label %13, !dbg !17

13:                                               ; preds = %5, %6
  %14 = phi i256 [ %10, %6 ], [ %0, %5 ]
  %15 = phi i256 [ %12, %6 ], [ 0, %5 ]
  br label %16, !dbg !18

16:                                               ; preds = %13
  %17 = insertvalue { i256, i256 } undef, i256 %14, 0, !dbg !19
  %18 = insertvalue { i256, i256 } %17, i256 %15, 1, !dbg !20
  ret { i256, i256 } %18, !dbg !21
}

define i32 @main() !dbg !22 {
  %1 = call { i256, i256 } @fib(i256 1, i256 1, i256 20), !dbg !23
  %2 = extractvalue { i256, i256 } %1, 0, !dbg !25
  %3 = extractvalue { i256, i256 } %1, 1, !dbg !26
  ret i32 0, !dbg !27
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "fib", linkageName: "fib", scope: null, file: !4, line: 2, type: !5, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "fib2.mlir", directory: "/data1/edgar/work/cairo_sierra_2_MLIR")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 4, column: 10, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!9 = !DILocation(line: 6, column: 5, scope: !8)
!10 = !DILocation(line: 8, column: 5, scope: !8)
!11 = !DILocation(line: 10, column: 10, scope: !8)
!12 = !DILocation(line: 11, column: 10, scope: !8)
!13 = !DILocation(line: 12, column: 10, scope: !8)
!14 = !DILocation(line: 13, column: 10, scope: !8)
!15 = !DILocation(line: 14, column: 10, scope: !8)
!16 = !DILocation(line: 15, column: 10, scope: !8)
!17 = !DILocation(line: 16, column: 5, scope: !8)
!18 = !DILocation(line: 18, column: 5, scope: !8)
!19 = !DILocation(line: 21, column: 11, scope: !8)
!20 = !DILocation(line: 22, column: 11, scope: !8)
!21 = !DILocation(line: 23, column: 5, scope: !8)
!22 = distinct !DISubprogram(name: "main", linkageName: "main", scope: null, file: !4, line: 25, type: !5, scopeLine: 25, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!23 = !DILocation(line: 29, column: 10, scope: !24)
!24 = !DILexicalBlockFile(scope: !22, file: !4, discriminator: 0)
!25 = !DILocation(line: 30, column: 10, scope: !24)
!26 = !DILocation(line: 31, column: 10, scope: !24)
!27 = !DILocation(line: 33, column: 5, scope: !24)
