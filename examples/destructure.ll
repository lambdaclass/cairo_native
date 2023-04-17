; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare ptr @realloc(ptr, i64)

declare ptr @memmove(ptr, ptr, i64)

declare i32 @dprintf(i32, ptr, ...)

; Function Attrs: alwaysinline norecurse nounwind
define internal { i256, i256, i256 } @"struct_construct<destructure::destructure::MyStruct>"(i256 %0, i256 %1, i256 %2) #0 {
  %4 = insertvalue { i256, i256, i256 } undef, i256 %0, 0
  %5 = insertvalue { i256, i256, i256 } %4, i256 %1, 1
  %6 = insertvalue { i256, i256, i256 } %5, i256 %2, 2
  ret { i256, i256, i256 } %6
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i256, i256, i256 } @"struct_deconstruct<destructure::destructure::MyStruct>"({ i256, i256, i256 } %0) #0 {
  %2 = extractvalue { i256, i256, i256 } %0, 0
  %3 = extractvalue { i256, i256, i256 } %0, 1
  %4 = extractvalue { i256, i256, i256 } %0, 2
  %5 = insertvalue { i256, i256, i256 } undef, i256 %2, 0
  %6 = insertvalue { i256, i256, i256 } %5, i256 %3, 1
  %7 = insertvalue { i256, i256, i256 } %6, i256 %4, 2
  ret { i256, i256, i256 } %7
}

; Function Attrs: alwaysinline norecurse nounwind
define internal {} @"struct_construct<Unit>"() #0 {
  ret {} undef
}

; Function Attrs: norecurse nounwind
define internal void @print_Unit({} %0) #1 {
  ret void
}

define void @main() {
  %1 = call {} @"destructure::destructure::main"()
  call void @print_Unit({} %1)
  ret void
}

define void @_mlir_ciface_main() {
  call void @main()
  ret void
}

define {} @"destructure::destructure::main"() {
  br label %1

1:                                                ; preds = %0
  %2 = call { i256, i256, i256 } @"struct_construct<destructure::destructure::MyStruct>"(i256 12, i256 34, i256 56)
  %3 = call { i256, i256, i256 } @"struct_deconstruct<destructure::destructure::MyStruct>"({ i256, i256, i256 } %2)
  %4 = extractvalue { i256, i256, i256 } %3, 0
  %5 = extractvalue { i256, i256, i256 } %3, 1
  %6 = extractvalue { i256, i256, i256 } %3, 2
  %7 = call {} @"struct_construct<Unit>"()
  ret {} %7
}

define void @"_mlir_ciface_destructure::destructure::main"(ptr %0) {
  %2 = call {} @"destructure::destructure::main"()
  store {} %2, ptr %0, align 1
  ret void
}

attributes #0 = { alwaysinline norecurse nounwind }
attributes #1 = { norecurse nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
