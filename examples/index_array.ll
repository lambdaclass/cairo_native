; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare ptr @realloc(ptr, i64)

declare ptr @memmove(ptr, ptr, i64)

declare i32 @dprintf(i32, ptr, ...)

; Function Attrs: alwaysinline norecurse nounwind
define internal { i32, i32, ptr } @"array_new<u8>"() #0 {
  %1 = call ptr @realloc(ptr null, i64 8)
  %2 = insertvalue { i32, i32, ptr } { i32 0, i32 8, ptr undef }, ptr %1, 2
  ret { i32, i32, ptr } %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i32, i32, ptr } @"array_append<u8>"({ i32, i32, ptr } %0, i8 %1) #0 {
  %3 = extractvalue { i32, i32, ptr } %0, 0
  %4 = extractvalue { i32, i32, ptr } %0, 1
  %5 = icmp ult i32 %3, %4
  br i1 %5, label %13, label %6

6:                                                ; preds = %2
  %7 = mul i32 %4, 2
  %8 = zext i32 %7 to i64
  %9 = extractvalue { i32, i32, ptr } %0, 2
  %10 = call ptr @realloc(ptr %9, i64 %8)
  %11 = insertvalue { i32, i32, ptr } %0, ptr %10, 2
  %12 = insertvalue { i32, i32, ptr } %11, i32 %7, 1
  br label %13

13:                                               ; preds = %6, %2
  %14 = phi { i32, i32, ptr } [ %12, %6 ], [ %0, %2 ]
  %15 = extractvalue { i32, i32, ptr } %14, 2
  %16 = getelementptr i8, ptr %15, i32 %3
  store i8 %1, ptr %16, align 1
  %17 = add i32 %3, 1
  %18 = insertvalue { i32, i32, ptr } %14, i32 %17, 0
  ret { i32, i32, ptr } %18
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i8 @"struct_deconstruct<Tuple<u8>>"({ i8 } %0) #0 {
  %2 = extractvalue { i8 } %0, 0
  ret i8 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i8 } @"struct_construct<Tuple<u8>>"(i8 %0) #0 {
  %2 = insertvalue { i8 } undef, i8 %0, 0
  ret { i8 } %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u8)>, 0>"({ i8 } %0) #0 {
  %2 = alloca { i16, [16 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 1
  store { i8 } %0, ptr %4, align 1
  %5 = load { i16, [16 x i8] }, ptr %2, align 2
  ret { i16, [16 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u8)>, 1>"({ i32, i32, ptr } %0) #0 {
  %2 = alloca i8, i64 13, align 1
  store [13 x i8] c"trap reached\00", ptr %2, align 1
  %3 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %2)
  call void @llvm.trap()
  unreachable
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i8 @"struct_deconstruct<Tuple<Box<u8>>>"({ i8 } %0) #0 {
  %2 = extractvalue { i8 } %0, 0
  ret i8 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [16 x i8] } @"enum_init<core::PanicResult::<(@core::integer::u8)>, 0>"({ i8 } %0) #0 {
  %2 = alloca { i16, [16 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 1
  store { i8 } %0, ptr %4, align 1
  %5 = load { i16, [16 x i8] }, ptr %2, align 2
  ret { i16, [16 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [16 x i8] } @"enum_init<core::PanicResult::<(@core::integer::u8)>, 1>"({ i32, i32, ptr } %0) #0 {
  %2 = alloca i8, i64 13, align 1
  store [13 x i8] c"trap reached\00", ptr %2, align 1
  %3 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %2)
  call void @llvm.trap()
  unreachable
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i8 } @"struct_construct<Tuple<Box<u8>>>"(i8 %0) #0 {
  %2 = insertvalue { i8 } undef, i8 %0, 0
  ret { i8 } %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u8>)>, 0>"({ i8 } %0) #0 {
  %2 = alloca { i16, [16 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 1
  store { i8 } %0, ptr %4, align 1
  %5 = load { i16, [16 x i8] }, ptr %2, align 2
  ret { i16, [16 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i32, i32, ptr } @"array_new<felt252>"() #0 {
  %1 = call ptr @realloc(ptr null, i64 256)
  %2 = insertvalue { i32, i32, ptr } { i32 0, i32 8, ptr undef }, ptr %1, 2
  ret { i32, i32, ptr } %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i32, i32, ptr } @"array_append<felt252>"({ i32, i32, ptr } %0, i256 %1) #0 {
  %3 = extractvalue { i32, i32, ptr } %0, 0
  %4 = extractvalue { i32, i32, ptr } %0, 1
  %5 = icmp ult i32 %3, %4
  br i1 %5, label %13, label %6

6:                                                ; preds = %2
  %7 = mul i32 %4, 2
  %8 = zext i32 %7 to i64
  %9 = extractvalue { i32, i32, ptr } %0, 2
  %10 = call ptr @realloc(ptr %9, i64 %8)
  %11 = insertvalue { i32, i32, ptr } %0, ptr %10, 2
  %12 = insertvalue { i32, i32, ptr } %11, i32 %7, 1
  br label %13

13:                                               ; preds = %6, %2
  %14 = phi { i32, i32, ptr } [ %12, %6 ], [ %0, %2 ]
  %15 = extractvalue { i32, i32, ptr } %14, 2
  %16 = getelementptr i256, ptr %15, i32 %3
  store i256 %1, ptr %16, align 4
  %17 = add i32 %3, 1
  %18 = insertvalue { i32, i32, ptr } %14, i32 %17, 0
  ret { i32, i32, ptr } %18
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u8>)>, 1>"({ i32, i32, ptr } %0) #0 {
  %2 = alloca i8, i64 13, align 1
  store [13 x i8] c"trap reached\00", ptr %2, align 1
  %3 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %2)
  call void @llvm.trap()
  unreachable
}

define { i16, [16 x i8] } @"index_array::index_array::main"() {
  br label %6

1:                                                ; preds = %6
  %2 = load { i8 }, ptr %20, align 1
  br label %21

3:                                                ; preds = %6
  %4 = load { i32, i32, ptr }, ptr %20, align 8
  br label %26

5:                                                ; preds = %6
  unreachable

6:                                                ; preds = %0
  %7 = call { i32, i32, ptr } @"array_new<u8>"()
  %8 = call { i32, i32, ptr } @"array_append<u8>"({ i32, i32, ptr } %7, i8 4)
  %9 = call { i32, i32, ptr } @"array_append<u8>"({ i32, i32, ptr } %8, i8 5)
  %10 = call { i32, i32, ptr } @"array_append<u8>"({ i32, i32, ptr } %9, i8 4)
  %11 = call { i32, i32, ptr } @"array_append<u8>"({ i32, i32, ptr } %10, i8 4)
  %12 = call { i32, i32, ptr } @"array_append<u8>"({ i32, i32, ptr } %11, i8 4)
  %13 = call { i32, i32, ptr } @"array_append<u8>"({ i32, i32, ptr } %12, i8 1)
  %14 = call { i32, i32, ptr } @"array_append<u8>"({ i32, i32, ptr } %13, i8 1)
  %15 = call { i32, i32, ptr } @"array_append<u8>"({ i32, i32, ptr } %14, i8 1)
  %16 = call { i32, i32, ptr } @"array_append<u8>"({ i32, i32, ptr } %15, i8 2)
  %17 = call { i16, [16 x i8] } @"core::array::ArrayIndex::<core::integer::u8>::index"({ i32, i32, ptr } %16, i32 0)
  %18 = extractvalue { i16, [16 x i8] } %17, 0
  %19 = extractvalue { i16, [16 x i8] } %17, 1
  %20 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %19, ptr %20, align 1
  switch i16 %18, label %5 [
    i16 0, label %1
    i16 1, label %3
  ]

21:                                               ; preds = %1
  %22 = phi { i8 } [ %2, %1 ]
  %23 = call i8 @"struct_deconstruct<Tuple<u8>>"({ i8 } %22)
  %24 = call { i8 } @"struct_construct<Tuple<u8>>"(i8 %23)
  %25 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u8)>, 0>"({ i8 } %24)
  ret { i16, [16 x i8] } %25

26:                                               ; preds = %3
  %27 = phi { i32, i32, ptr } [ %4, %3 ]
  %28 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u8)>, 1>"({ i32, i32, ptr } %27)
  ret { i16, [16 x i8] } %28
}

define void @"_mlir_ciface_index_array::index_array::main"(ptr %0) {
  %2 = call { i16, [16 x i8] } @"index_array::index_array::main"()
  store { i16, [16 x i8] } %2, ptr %0, align 2
  ret void
}

define { i16, [16 x i8] } @"core::array::ArrayIndex::<core::integer::u8>::index"({ i32, i32, ptr } %0, i32 %1) {
  br label %8

3:                                                ; preds = %8
  %4 = load { i8 }, ptr %14, align 1
  br label %15

5:                                                ; preds = %8
  %6 = load { i32, i32, ptr }, ptr %14, align 8
  br label %20

7:                                                ; preds = %8
  unreachable

8:                                                ; preds = %2
  %9 = phi { i32, i32, ptr } [ %0, %2 ]
  %10 = phi i32 [ %1, %2 ]
  %11 = call { i16, [16 x i8] } @"core::array::array_at::<core::integer::u8>"({ i32, i32, ptr } %9, i32 %10)
  %12 = extractvalue { i16, [16 x i8] } %11, 0
  %13 = extractvalue { i16, [16 x i8] } %11, 1
  %14 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %13, ptr %14, align 1
  switch i16 %12, label %7 [
    i16 0, label %3
    i16 1, label %5
  ]

15:                                               ; preds = %3
  %16 = phi { i8 } [ %4, %3 ]
  %17 = call i8 @"struct_deconstruct<Tuple<Box<u8>>>"({ i8 } %16)
  %18 = call { i8 } @"struct_construct<Tuple<u8>>"(i8 %17)
  %19 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(@core::integer::u8)>, 0>"({ i8 } %18)
  ret { i16, [16 x i8] } %19

20:                                               ; preds = %5
  %21 = phi { i32, i32, ptr } [ %6, %5 ]
  %22 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(@core::integer::u8)>, 1>"({ i32, i32, ptr } %21)
  ret { i16, [16 x i8] } %22
}

define void @"_mlir_ciface_core::array::ArrayIndex::<core::integer::u8>::index"(ptr %0, { i32, i32, ptr } %1, i32 %2) {
  %4 = call { i16, [16 x i8] } @"core::array::ArrayIndex::<core::integer::u8>::index"({ i32, i32, ptr } %1, i32 %2)
  store { i16, [16 x i8] } %4, ptr %0, align 2
  ret void
}

define { i16, [16 x i8] } @"core::array::array_at::<core::integer::u8>"({ i32, i32, ptr } %0, i32 %1) {
  br label %7

3:                                                ; preds = %7
  %4 = extractvalue { i32, i32, ptr } %8, 2
  %5 = getelementptr i8, ptr %4, i32 %9
  %6 = load i8, ptr %5, align 1
  br label %12

7:                                                ; preds = %2
  %8 = phi { i32, i32, ptr } [ %0, %2 ]
  %9 = phi i32 [ %1, %2 ]
  %10 = extractvalue { i32, i32, ptr } %8, 0
  %11 = icmp ult i32 %9, %10
  br i1 %11, label %3, label %16

12:                                               ; preds = %3
  %13 = phi i8 [ %6, %3 ]
  %14 = call { i8 } @"struct_construct<Tuple<Box<u8>>>"(i8 %13)
  %15 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u8>)>, 0>"({ i8 } %14)
  ret { i16, [16 x i8] } %15

16:                                               ; preds = %7
  %17 = call { i32, i32, ptr } @"array_new<felt252>"()
  %18 = call { i32, i32, ptr } @"array_append<felt252>"({ i32, i32, ptr } %17, i256 1637570914057682275393755530660268060279989363)
  %19 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u8>)>, 1>"({ i32, i32, ptr } %18)
  ret { i16, [16 x i8] } %19
}

define void @"_mlir_ciface_core::array::array_at::<core::integer::u8>"(ptr %0, { i32, i32, ptr } %1, i32 %2) {
  %4 = call { i16, [16 x i8] } @"core::array::array_at::<core::integer::u8>"({ i32, i32, ptr } %1, i32 %2)
  store { i16, [16 x i8] } %4, ptr %0, align 2
  ret void
}

; Function Attrs: cold noreturn nounwind
declare void @llvm.trap() #1

attributes #0 = { alwaysinline norecurse nounwind }
attributes #1 = { cold noreturn nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
