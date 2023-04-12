; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare i32 @dprintf(i32, ptr, ...)

define internal {} @"struct_construct<Unit>"() {
  ret {} undef
}

define internal void @print_Unit({} %0) {
  ret void
}

define void @main() {
  %1 = call {} @"types::types::main"()
  call void @print_Unit({} %1)
  ret void
}

define void @_mlir_ciface_main() {
  call void @main()
  ret void
}

define {} @"types::types::main"() {
  br label %1

1:                                                ; preds = %0
  %2 = call {} @"struct_construct<Unit>"()
  ret {} %2
}

define void @"_mlir_ciface_types::types::main"(ptr %0) {
  %2 = call {} @"types::types::main"()
  store {} %2, ptr %0, align 1
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
