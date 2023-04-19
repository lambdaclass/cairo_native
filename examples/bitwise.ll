; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare ptr @realloc(ptr, i64)

declare ptr @memmove(ptr, ptr, i64)

declare i32 @dprintf(i32, ptr, ...)

; Function Attrs: alwaysinline norecurse nounwind
define internal { i128, i128, i128 } @bitwise(i128 %0, i128 %1) #0 {
  %3 = and i128 %0, %1
  %4 = xor i128 %0, %1
  %5 = or i128 %0, %1
  %6 = insertvalue { i128, i128, i128 } undef, i128 %3, 0
  %7 = insertvalue { i128, i128, i128 } %6, i128 %4, 1
  %8 = insertvalue { i128, i128, i128 } %7, i128 %5, 2
  ret { i128, i128, i128 } %8
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{}> @"struct_construct<Unit>"() #0 {
  ret <{}> undef
}

; Function Attrs: norecurse nounwind
define internal void @print_Unit(<{}> %0) #1 {
  ret void
}

define void @main() {
  %1 = call <{}> @"bitwise::bitwise::main"()
  call void @print_Unit(<{}> %1)
  ret void
}

define void @_mlir_ciface_main() {
  call void @main()
  ret void
}

define <{}> @"bitwise::bitwise::main"() {
  br label %1

1:                                                ; preds = %0
  %2 = call { i128, i128, i128 } @bitwise(i128 1234, i128 5678)
  %3 = extractvalue { i128, i128, i128 } %2, 0
  %4 = extractvalue { i128, i128, i128 } %2, 1
  %5 = extractvalue { i128, i128, i128 } %2, 2
  %6 = call { i128, i128, i128 } @bitwise(i128 1234, i128 5678)
  %7 = extractvalue { i128, i128, i128 } %6, 0
  %8 = extractvalue { i128, i128, i128 } %6, 1
  %9 = extractvalue { i128, i128, i128 } %6, 2
  %10 = call { i128, i128, i128 } @bitwise(i128 1234, i128 5678)
  %11 = extractvalue { i128, i128, i128 } %10, 0
  %12 = extractvalue { i128, i128, i128 } %10, 1
  %13 = extractvalue { i128, i128, i128 } %10, 2
  %14 = call <{}> @"struct_construct<Unit>"()
  ret <{}> %14
}

define void @"_mlir_ciface_bitwise::bitwise::main"(ptr %0) {
  %2 = call <{}> @"bitwise::bitwise::main"()
  store <{}> %2, ptr %0, align 1
  ret void
}

attributes #0 = { alwaysinline norecurse nounwind }
attributes #1 = { norecurse nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
