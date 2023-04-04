; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare i32 @dprintf(i32, ptr, ...)

define internal i256 @"store_temp<felt252>"(i256 %0) {
  ret i256 %0
}

define internal void @print_felt252(i256 %0) {
  %2 = ashr i256 %0, 224
  %3 = trunc i256 %2 to i32
  %4 = alloca i8, i64 5, align 1
  store [5 x i8] c"%08X\00", ptr %4, align 1
  %5 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %4, i32 %3)
  %6 = ashr i256 %0, 192
  %7 = trunc i256 %6 to i32
  %8 = alloca i8, i64 5, align 1
  store [5 x i8] c"%08X\00", ptr %8, align 1
  %9 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %8, i32 %7)
  %10 = ashr i256 %0, 160
  %11 = trunc i256 %10 to i32
  %12 = alloca i8, i64 5, align 1
  store [5 x i8] c"%08X\00", ptr %12, align 1
  %13 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %12, i32 %11)
  %14 = ashr i256 %0, 128
  %15 = trunc i256 %14 to i32
  %16 = alloca i8, i64 5, align 1
  store [5 x i8] c"%08X\00", ptr %16, align 1
  %17 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %16, i32 %15)
  %18 = ashr i256 %0, 96
  %19 = trunc i256 %18 to i32
  %20 = alloca i8, i64 5, align 1
  store [5 x i8] c"%08X\00", ptr %20, align 1
  %21 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %20, i32 %19)
  %22 = ashr i256 %0, 64
  %23 = trunc i256 %22 to i32
  %24 = alloca i8, i64 5, align 1
  store [5 x i8] c"%08X\00", ptr %24, align 1
  %25 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %24, i32 %23)
  %26 = ashr i256 %0, 32
  %27 = trunc i256 %26 to i32
  %28 = alloca i8, i64 5, align 1
  store [5 x i8] c"%08X\00", ptr %28, align 1
  %29 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %28, i32 %27)
  %30 = trunc i256 %0 to i32
  %31 = alloca i8, i64 5, align 1
  store [5 x i8] c"%08X\00", ptr %31, align 1
  %32 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %31, i32 %30)
  %33 = alloca i8, i64 2, align 1
  store [2 x i8] c"\0A\00", ptr %33, align 1
  %34 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %33)
  ret void
}

define void @main() {
  %1 = call i256 @"print_test::print_test::main"()
  call void @print_felt252(i256 %1)
  ret void
}

define void @_mlir_ciface_main() {
  call void @main()
  ret void
}

define i256 @"print_test::print_test::main"() {
  br label %1

1:                                                ; preds = %0
  %2 = call i256 @"store_temp<felt252>"(i256 24)
  ret i256 %2
}

define i256 @"_mlir_ciface_print_test::print_test::main"() {
  %1 = call i256 @"print_test::print_test::main"()
  ret i256 %1
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
