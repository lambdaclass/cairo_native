; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare i32 @dprintf(i32, ptr, ...)

define internal {} @"struct_construct<Unit>"() {
  ret {} undef
}

define internal { i16, [0 x i8] } @"enum_init<core_bool, 1>"({} %0) {
  %2 = alloca { i16, [0 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [0 x i8] }, ptr %2, i32 0, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [0 x i8] }, ptr %2, i32 0, i32 1
  store {} %0, ptr %4, align 1
  %5 = load { i16, [0 x i8] }, ptr %2, align 2
  ret { i16, [0 x i8] } %5
}

define internal { i16, [0 x i8] } @"store_temp<core_bool>"({ i16, [0 x i8] } %0) {
  ret { i16, [0 x i8] } %0
}

define internal void @print_Unit({} %0) {
  ret void
}

define internal void @"print_core::bool"({ i16, [0 x i8] } %0) {
  %2 = extractvalue { i16, [0 x i8] } %0, 0
  %3 = alloca i8, i64 4, align 1
  store [4 x i8] c"%X\0A\00", ptr %3, align 1
  %4 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %3, i16 %2)
  %5 = alloca { i16, [0 x i8] }, i64 1, align 8
  store { i16, [0 x i8] } %0, ptr %5, align 2
  %6 = getelementptr inbounds { i16, [0 x i8] }, ptr %5, i32 0, i32 1
  switch i16 %2, label %11 [
    i16 0, label %7
    i16 1, label %9
  ]

7:                                                ; preds = %1
  %8 = load {}, ptr %6, align 1
  call void @print_Unit({} %8)
  ret void

9:                                                ; preds = %1
  %10 = load {}, ptr %6, align 1
  call void @print_Unit({} %10)
  ret void

11:                                               ; preds = %1
  ret void
}

define void @main() {
  %1 = call { i16, [0 x i8] } @boolean_boolean_main()
  call void @"print_core::bool"({ i16, [0 x i8] } %1)
  ret void
}

define void @_mlir_ciface_main() {
  call void @main()
  ret void
}

define { i16, [0 x i8] } @boolean_boolean_main() {
  br label %1

1:                                                ; preds = %0
  %2 = call {} @"struct_construct<Unit>"()
  %3 = call { i16, [0 x i8] } @"enum_init<core_bool, 1>"({} %2)
  %4 = call { i16, [0 x i8] } @"store_temp<core_bool>"({ i16, [0 x i8] } %3)
  ret { i16, [0 x i8] } %4
}

define void @_mlir_ciface_boolean_boolean_main(ptr %0) {
  %2 = call { i16, [0 x i8] } @boolean_boolean_main()
  store { i16, [0 x i8] } %2, ptr %0, align 2
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
