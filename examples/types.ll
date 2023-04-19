; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare ptr @realloc(ptr, i64)

declare ptr @memmove(ptr, ptr, i64)

declare i32 @dprintf(i32, ptr, ...)

; Function Attrs: alwaysinline norecurse nounwind
define internal <{}> @"struct_construct<Unit>"() #0 {
  ret <{}> undef
}

; Function Attrs: norecurse nounwind
define internal void @print_Unit(<{}> %0) #1 {
  ret void
}

define void @main() {
  %1 = call <{}> @"types::types::main"()
  call void @print_Unit(<{}> %1)
  ret void
}

define void @_mlir_ciface_main() {
  call void @main()
  ret void
}

define <{}> @"types::types::main"() {
  br label %1

1:                                                ; preds = %0
  %2 = call <{}> @"struct_construct<Unit>"()
  ret <{}> %2
}

define void @"_mlir_ciface_types::types::main"(ptr %0) {
  %2 = call <{}> @"types::types::main"()
  store <{}> %2, ptr %0, align 1
  ret void
}

attributes #0 = { alwaysinline norecurse nounwind }
attributes #1 = { norecurse nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
