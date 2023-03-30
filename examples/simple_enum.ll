; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare i32 @dprintf(i32, ptr, ...)

define internal { i16, [8 x i8] } @"enum_init<simple_enum_simple_enum_MyEnum, 0>"(i8 %0) {
  %2 = alloca { i16, [8 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [8 x i8] }, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [8 x i8] }, ptr %2, i32 0, i32 1
  store i8 %0, ptr %4, align 1
  %5 = load { i16, [8 x i8] }, ptr %2, align 2
  ret { i16, [8 x i8] } %5
}

define internal { i16, [8 x i8] } @"store_temp<simple_enum_simple_enum_MyEnum>"({ i16, [8 x i8] } %0) {
  ret { i16, [8 x i8] } %0
}

define internal { i16, [8 x i8] } @"enum_init<simple_enum_simple_enum_MyEnum, 1>"(i16 %0) {
  %2 = alloca { i16, [8 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [8 x i8] }, ptr %2, i32 0, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [8 x i8] }, ptr %2, i32 0, i32 1
  store i16 %0, ptr %4, align 2
  %5 = load { i16, [8 x i8] }, ptr %2, align 2
  ret { i16, [8 x i8] } %5
}

define internal { i16, [8 x i8] } @"enum_init<simple_enum_simple_enum_MyEnum, 2>"(i32 %0) {
  %2 = alloca { i16, [8 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [8 x i8] }, ptr %2, i32 0, i32 0
  store i16 2, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [8 x i8] }, ptr %2, i32 0, i32 1
  store i32 %0, ptr %4, align 4
  %5 = load { i16, [8 x i8] }, ptr %2, align 2
  ret { i16, [8 x i8] } %5
}

define internal { i16, [8 x i8] } @"enum_init<simple_enum_simple_enum_MyEnum, 3>"(i64 %0) {
  %2 = alloca { i16, [8 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [8 x i8] }, ptr %2, i32 0, i32 0
  store i16 3, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [8 x i8] }, ptr %2, i32 0, i32 1
  store i64 %0, ptr %4, align 4
  %5 = load { i16, [8 x i8] }, ptr %2, align 2
  ret { i16, [8 x i8] } %5
}

define internal void @print_u8(i8 %0) {
  %2 = alloca i8, i64 4, align 1
  store [4 x i8] c"%X\0A\00", ptr %2, align 1
  %3 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %2, i8 %0)
  ret void
}

define internal void @print_u16(i16 %0) {
  %2 = alloca i8, i64 4, align 1
  store [4 x i8] c"%X\0A\00", ptr %2, align 1
  %3 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %2, i16 %0)
  ret void
}

define internal void @print_u32(i32 %0) {
  %2 = alloca i8, i64 4, align 1
  store [4 x i8] c"%X\0A\00", ptr %2, align 1
  %3 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %2, i32 %0)
  ret void
}

define internal void @print_u64(i64 %0) {
  %2 = alloca i8, i64 4, align 1
  store [4 x i8] c"%X\0A\00", ptr %2, align 1
  %3 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %2, i64 %0)
  ret void
}

define internal void @"print_simple_enum::simple_enum::MyEnum"({ i16, [8 x i8] } %0) {
  %2 = extractvalue { i16, [8 x i8] } %0, 0
  %3 = alloca i8, i64 4, align 1
  store [4 x i8] c"%X\0A\00", ptr %3, align 1
  %4 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %3, i16 %2)
  %5 = alloca { i16, [8 x i8] }, i64 1, align 8
  store { i16, [8 x i8] } %0, ptr %5, align 2
  %6 = getelementptr inbounds { i16, [8 x i8] }, ptr %5, i32 0, i32 1
  switch i16 %2, label %15 [
    i16 0, label %7
    i16 1, label %9
    i16 2, label %11
    i16 3, label %13
  ]

7:                                                ; preds = %1
  %8 = load i8, ptr %6, align 1
  call void @print_u8(i8 %8)
  ret void

9:                                                ; preds = %1
  %10 = load i16, ptr %6, align 2
  call void @print_u16(i16 %10)
  ret void

11:                                               ; preds = %1
  %12 = load i32, ptr %6, align 4
  call void @print_u32(i32 %12)
  ret void

13:                                               ; preds = %1
  %14 = load i64, ptr %6, align 4
  call void @print_u64(i64 %14)
  ret void

15:                                               ; preds = %1
  ret void
}

define void @main() {
  %1 = call { i16, [8 x i8] } @simple_enum_simple_enum_main()
  call void @"print_simple_enum::simple_enum::MyEnum"({ i16, [8 x i8] } %1)
  ret void
}

define void @_mlir_ciface_main() {
  call void @main()
  ret void
}

define { i16, [8 x i8] } @simple_enum_simple_enum_my_enum_a() {
  br label %1

1:                                                ; preds = %0
  %2 = call { i16, [8 x i8] } @"enum_init<simple_enum_simple_enum_MyEnum, 0>"(i8 4)
  %3 = call { i16, [8 x i8] } @"store_temp<simple_enum_simple_enum_MyEnum>"({ i16, [8 x i8] } %2)
  ret { i16, [8 x i8] } %3
}

define void @_mlir_ciface_simple_enum_simple_enum_my_enum_a(ptr %0) {
  %2 = call { i16, [8 x i8] } @simple_enum_simple_enum_my_enum_a()
  store { i16, [8 x i8] } %2, ptr %0, align 2
  ret void
}

define { i16, [8 x i8] } @simple_enum_simple_enum_my_enum_b() {
  br label %1

1:                                                ; preds = %0
  %2 = call { i16, [8 x i8] } @"enum_init<simple_enum_simple_enum_MyEnum, 1>"(i16 8)
  %3 = call { i16, [8 x i8] } @"store_temp<simple_enum_simple_enum_MyEnum>"({ i16, [8 x i8] } %2)
  ret { i16, [8 x i8] } %3
}

define void @_mlir_ciface_simple_enum_simple_enum_my_enum_b(ptr %0) {
  %2 = call { i16, [8 x i8] } @simple_enum_simple_enum_my_enum_b()
  store { i16, [8 x i8] } %2, ptr %0, align 2
  ret void
}

define { i16, [8 x i8] } @simple_enum_simple_enum_my_enum_c() {
  br label %1

1:                                                ; preds = %0
  %2 = call { i16, [8 x i8] } @"enum_init<simple_enum_simple_enum_MyEnum, 2>"(i32 16)
  %3 = call { i16, [8 x i8] } @"store_temp<simple_enum_simple_enum_MyEnum>"({ i16, [8 x i8] } %2)
  ret { i16, [8 x i8] } %3
}

define void @_mlir_ciface_simple_enum_simple_enum_my_enum_c(ptr %0) {
  %2 = call { i16, [8 x i8] } @simple_enum_simple_enum_my_enum_c()
  store { i16, [8 x i8] } %2, ptr %0, align 2
  ret void
}

define { i16, [8 x i8] } @simple_enum_simple_enum_my_enum_d() {
  br label %1

1:                                                ; preds = %0
  %2 = call { i16, [8 x i8] } @"enum_init<simple_enum_simple_enum_MyEnum, 3>"(i64 16)
  %3 = call { i16, [8 x i8] } @"store_temp<simple_enum_simple_enum_MyEnum>"({ i16, [8 x i8] } %2)
  ret { i16, [8 x i8] } %3
}

define void @_mlir_ciface_simple_enum_simple_enum_my_enum_d(ptr %0) {
  %2 = call { i16, [8 x i8] } @simple_enum_simple_enum_my_enum_d()
  store { i16, [8 x i8] } %2, ptr %0, align 2
  ret void
}

define { i16, [8 x i8] } @simple_enum_simple_enum_main() {
  br label %1

1:                                                ; preds = %0
  %2 = call { i16, [8 x i8] } @"enum_init<simple_enum_simple_enum_MyEnum, 3>"(i64 16)
  %3 = call { i16, [8 x i8] } @"store_temp<simple_enum_simple_enum_MyEnum>"({ i16, [8 x i8] } %2)
  ret { i16, [8 x i8] } %3
}

define void @_mlir_ciface_simple_enum_simple_enum_main(ptr %0) {
  %2 = call { i16, [8 x i8] } @simple_enum_simple_enum_main()
  store { i16, [8 x i8] } %2, ptr %0, align 2
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
