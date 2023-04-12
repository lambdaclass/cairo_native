; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare i32 @dprintf(i32, ptr, ...)

define internal { i16, [4 x i8] } @"enum_init<enum_match::enum_match::MyEnum, 1>"(i16 %0) {
  %2 = alloca { i16, [4 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [4 x i8] }, ptr %2, i32 0, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [4 x i8] }, ptr %2, i32 0, i32 1
  store i16 %0, ptr %4, align 2
  %5 = load { i16, [4 x i8] }, ptr %2, align 2
  ret { i16, [4 x i8] } %5
}

define internal void @print_u16(i16 %0) {
  %2 = zext i16 %0 to i32
  %3 = alloca i8, i64 4, align 1
  store [4 x i8] c"%X\0A\00", ptr %3, align 1
  %4 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %3, i32 %2)
  ret void
}

define void @main() {
  %1 = call i16 @"enum_match::enum_match::main"()
  call void @print_u16(i16 %1)
  ret void
}

define void @_mlir_ciface_main() {
  call void @main()
  ret void
}

define i16 @"enum_match::enum_match::get_my_enum_b"({ i16, [4 x i8] } %0) {
  br label %7

2:                                                ; preds = %7
  br label %12

3:                                                ; preds = %7
  %4 = load i16, ptr %11, align 2
  br label %13

5:                                                ; preds = %7
  br label %15

6:                                                ; preds = %7
  unreachable

7:                                                ; preds = %1
  %8 = phi { i16, [4 x i8] } [ %0, %1 ]
  %9 = extractvalue { i16, [4 x i8] } %8, 0
  %10 = extractvalue { i16, [4 x i8] } %8, 1
  %11 = alloca [4 x i8], i64 1, align 1
  store [4 x i8] %10, ptr %11, align 1
  switch i16 %9, label %6 [
    i16 0, label %2
    i16 1, label %3
    i16 2, label %5
  ]

12:                                               ; preds = %2
  br label %16

13:                                               ; preds = %3
  %14 = phi i16 [ %4, %3 ]
  br label %16

15:                                               ; preds = %5
  br label %16

16:                                               ; preds = %12, %13, %15
  %17 = phi i16 [ 0, %15 ], [ %14, %13 ], [ 1, %12 ]
  ret i16 %17
}

define i16 @"_mlir_ciface_enum_match::enum_match::get_my_enum_b"({ i16, [4 x i8] } %0) {
  %2 = call i16 @"enum_match::enum_match::get_my_enum_b"({ i16, [4 x i8] } %0)
  ret i16 %2
}

define i16 @"enum_match::enum_match::main"() {
  br label %1

1:                                                ; preds = %0
  %2 = call { i16, [4 x i8] } @"enum_init<enum_match::enum_match::MyEnum, 1>"(i16 16)
  %3 = call i16 @"enum_match::enum_match::get_my_enum_b"({ i16, [4 x i8] } %2)
  ret i16 %3
}

define i16 @"_mlir_ciface_enum_match::enum_match::main"() {
  %1 = call i16 @"enum_match::enum_match::main"()
  ret i16 %1
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
