; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare ptr @realloc(ptr, i64)

declare ptr @memmove(ptr, ptr, i64)

declare i32 @dprintf(i32, ptr, ...)

; Function Attrs: alwaysinline norecurse nounwind
define internal i16 @"struct_deconstruct<Tuple<u16>>"({ i16 } %0) #0 {
  %2 = extractvalue { i16 } %0, 0
  ret i16 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i32 @"struct_deconstruct<Tuple<u32>>"({ i32 } %0) #0 {
  %2 = extractvalue { i32 } %0, 0
  ret i32 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i64 @"struct_deconstruct<Tuple<u64>>"({ i64 } %0) #0 {
  %2 = extractvalue { i64 } %0, 0
  ret i64 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i128 @"struct_deconstruct<Tuple<u128>>"({ i128 } %0) #0 {
  %2 = extractvalue { i128 } %0, 0
  ret i128 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, i16, i16 } @"struct_construct<Tuple<u16, u16, u16>>"(i16 %0, i16 %1, i16 %2) #0 {
  %4 = insertvalue { i16, i16, i16 } undef, i16 %0, 0
  %5 = insertvalue { i16, i16, i16 } %4, i16 %1, 1
  %6 = insertvalue { i16, i16, i16 } %5, i16 %2, 2
  ret { i16, i16, i16 } %6
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i32, i32, i32 } @"struct_construct<Tuple<u32, u32, u32>>"(i32 %0, i32 %1, i32 %2) #0 {
  %4 = insertvalue { i32, i32, i32 } undef, i32 %0, 0
  %5 = insertvalue { i32, i32, i32 } %4, i32 %1, 1
  %6 = insertvalue { i32, i32, i32 } %5, i32 %2, 2
  ret { i32, i32, i32 } %6
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i64, i64, i64 } @"struct_construct<Tuple<u64, u64, u64>>"(i64 %0, i64 %1, i64 %2) #0 {
  %4 = insertvalue { i64, i64, i64 } undef, i64 %0, 0
  %5 = insertvalue { i64, i64, i64 } %4, i64 %1, 1
  %6 = insertvalue { i64, i64, i64 } %5, i64 %2, 2
  ret { i64, i64, i64 } %6
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i128, i128, i128 } @"struct_construct<Tuple<u128, u128, u128>>"(i128 %0, i128 %1, i128 %2) #0 {
  %4 = insertvalue { i128, i128, i128 } undef, i128 %0, 0
  %5 = insertvalue { i128, i128, i128 } %4, i128 %1, 1
  %6 = insertvalue { i128, i128, i128 } %5, i128 %2, 2
  ret { i128, i128, i128 } %6
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { { i16, i16, i16 }, { i32, i32, i32 }, { i64, i64, i64 }, { i128, i128, i128 } } @"struct_construct<Tuple<Tuple<u16, u16, u16>, Tuple<u32, u32, u32>, Tuple<u64, u64, u64>, Tuple<u128, u128, u128>>>"({ i16, i16, i16 } %0, { i32, i32, i32 } %1, { i64, i64, i64 } %2, { i128, i128, i128 } %3) #0 {
  %5 = insertvalue { { i16, i16, i16 }, { i32, i32, i32 }, { i64, i64, i64 }, { i128, i128, i128 } } undef, { i16, i16, i16 } %0, 0
  %6 = insertvalue { { i16, i16, i16 }, { i32, i32, i32 }, { i64, i64, i64 }, { i128, i128, i128 } } %5, { i32, i32, i32 } %1, 1
  %7 = insertvalue { { i16, i16, i16 }, { i32, i32, i32 }, { i64, i64, i64 }, { i128, i128, i128 } } %6, { i64, i64, i64 } %2, 2
  %8 = insertvalue { { i16, i16, i16 }, { i32, i32, i32 }, { i64, i64, i64 }, { i128, i128, i128 } } %7, { i128, i128, i128 } %3, 3
  ret { { i16, i16, i16 }, { i32, i32, i32 }, { i64, i64, i64 }, { i128, i128, i128 } } %8
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { { { i16, i16, i16 }, { i32, i32, i32 }, { i64, i64, i64 }, { i128, i128, i128 } } } @"struct_construct<Tuple<Tuple<Tuple<u16, u16, u16>, Tuple<u32, u32, u32>, Tuple<u64, u64, u64>, Tuple<u128, u128, u128>>>>"({ { i16, i16, i16 }, { i32, i32, i32 }, { i64, i64, i64 }, { i128, i128, i128 } } %0) #0 {
  %2 = insertvalue { { { i16, i16, i16 }, { i32, i32, i32 }, { i64, i64, i64 }, { i128, i128, i128 } } } undef, { { i16, i16, i16 }, { i32, i32, i32 }, { i64, i64, i64 }, { i128, i128, i128 } } %0, 0
  ret { { { i16, i16, i16 }, { i32, i32, i32 }, { i64, i64, i64 }, { i128, i128, i128 } } } %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [90 x i8] } @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 0>"({ { { i16, i16, i16 }, { i32, i32, i32 }, { i64, i64, i64 }, { i128, i128, i128 } } } %0) #0 {
  %2 = alloca { i16, [90 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [90 x i8] }, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [90 x i8] }, ptr %2, i32 0, i32 1
  store { { { i16, i16, i16 }, { i32, i32, i32 }, { i64, i64, i64 }, { i128, i128, i128 } } } %0, ptr %4, align 4
  %5 = load { i16, [90 x i8] }, ptr %2, align 2
  ret { i16, [90 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [90 x i8] } @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"({ i32, i32, ptr } %0) #0 {
  %2 = alloca i8, i64 13, align 1
  store [13 x i8] c"trap reached\00", ptr %2, align 1
  %3 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %2)
  call void @llvm.trap()
  unreachable
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [2 x i8] } @"enum_init<core::result::Result::<core::integer::u16, core::integer::u16>, 0>"(i16 %0) #0 {
  %2 = alloca { i16, [2 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [2 x i8] }, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [2 x i8] }, ptr %2, i32 0, i32 1
  store i16 %0, ptr %4, align 2
  %5 = load { i16, [2 x i8] }, ptr %2, align 2
  ret { i16, [2 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [2 x i8] } @"enum_init<core::result::Result::<core::integer::u16, core::integer::u16>, 1>"(i16 %0) #0 {
  %2 = alloca { i16, [2 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [2 x i8] }, ptr %2, i32 0, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [2 x i8] }, ptr %2, i32 0, i32 1
  store i16 %0, ptr %4, align 2
  %5 = load { i16, [2 x i8] }, ptr %2, align 2
  ret { i16, [2 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16 } @"struct_construct<Tuple<u16>>"(i16 %0) #0 {
  %2 = insertvalue { i16 } undef, i16 %0, 0
  ret { i16 } %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u16)>, 0>"({ i16 } %0) #0 {
  %2 = alloca { i16, [16 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 1
  store { i16 } %0, ptr %4, align 2
  %5 = load { i16, [16 x i8] }, ptr %2, align 2
  ret { i16, [16 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u16)>, 1>"({ i32, i32, ptr } %0) #0 {
  %2 = alloca i8, i64 13, align 1
  store [13 x i8] c"trap reached\00", ptr %2, align 1
  %3 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %2)
  call void @llvm.trap()
  unreachable
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [4 x i8] } @"enum_init<core::result::Result::<core::integer::u32, core::integer::u32>, 0>"(i32 %0) #0 {
  %2 = alloca { i16, [4 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [4 x i8] }, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [4 x i8] }, ptr %2, i32 0, i32 1
  store i32 %0, ptr %4, align 4
  %5 = load { i16, [4 x i8] }, ptr %2, align 2
  ret { i16, [4 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [4 x i8] } @"enum_init<core::result::Result::<core::integer::u32, core::integer::u32>, 1>"(i32 %0) #0 {
  %2 = alloca { i16, [4 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [4 x i8] }, ptr %2, i32 0, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [4 x i8] }, ptr %2, i32 0, i32 1
  store i32 %0, ptr %4, align 4
  %5 = load { i16, [4 x i8] }, ptr %2, align 2
  ret { i16, [4 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i32 } @"struct_construct<Tuple<u32>>"(i32 %0) #0 {
  %2 = insertvalue { i32 } undef, i32 %0, 0
  ret { i32 } %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u32)>, 0>"({ i32 } %0) #0 {
  %2 = alloca { i16, [16 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 1
  store { i32 } %0, ptr %4, align 4
  %5 = load { i16, [16 x i8] }, ptr %2, align 2
  ret { i16, [16 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u32)>, 1>"({ i32, i32, ptr } %0) #0 {
  %2 = alloca i8, i64 13, align 1
  store [13 x i8] c"trap reached\00", ptr %2, align 1
  %3 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %2)
  call void @llvm.trap()
  unreachable
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [8 x i8] } @"enum_init<core::result::Result::<core::integer::u64, core::integer::u64>, 0>"(i64 %0) #0 {
  %2 = alloca { i16, [8 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [8 x i8] }, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [8 x i8] }, ptr %2, i32 0, i32 1
  store i64 %0, ptr %4, align 4
  %5 = load { i16, [8 x i8] }, ptr %2, align 2
  ret { i16, [8 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [8 x i8] } @"enum_init<core::result::Result::<core::integer::u64, core::integer::u64>, 1>"(i64 %0) #0 {
  %2 = alloca { i16, [8 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [8 x i8] }, ptr %2, i32 0, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [8 x i8] }, ptr %2, i32 0, i32 1
  store i64 %0, ptr %4, align 4
  %5 = load { i16, [8 x i8] }, ptr %2, align 2
  ret { i16, [8 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i64 } @"struct_construct<Tuple<u64>>"(i64 %0) #0 {
  %2 = insertvalue { i64 } undef, i64 %0, 0
  ret { i64 } %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u64)>, 0>"({ i64 } %0) #0 {
  %2 = alloca { i16, [16 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 1
  store { i64 } %0, ptr %4, align 4
  %5 = load { i16, [16 x i8] }, ptr %2, align 2
  ret { i16, [16 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u64)>, 1>"({ i32, i32, ptr } %0) #0 {
  %2 = alloca i8, i64 13, align 1
  store [13 x i8] c"trap reached\00", ptr %2, align 1
  %3 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %2)
  call void @llvm.trap()
  unreachable
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [16 x i8] } @"enum_init<core::result::Result::<core::integer::u128, core::integer::u128>, 0>"(i128 %0) #0 {
  %2 = alloca { i16, [16 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 1
  store i128 %0, ptr %4, align 4
  %5 = load { i16, [16 x i8] }, ptr %2, align 2
  ret { i16, [16 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [16 x i8] } @"enum_init<core::result::Result::<core::integer::u128, core::integer::u128>, 1>"(i128 %0) #0 {
  %2 = alloca { i16, [16 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 1
  store i128 %0, ptr %4, align 4
  %5 = load { i16, [16 x i8] }, ptr %2, align 2
  ret { i16, [16 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i128 } @"struct_construct<Tuple<u128>>"(i128 %0) #0 {
  %2 = insertvalue { i128 } undef, i128 %0, 0
  ret { i128 } %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u128)>, 0>"({ i128 } %0) #0 {
  %2 = alloca { i16, [16 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 1
  store { i128 } %0, ptr %4, align 4
  %5 = load { i16, [16 x i8] }, ptr %2, align 2
  ret { i16, [16 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u128)>, 1>"({ i32, i32, ptr } %0) #0 {
  %2 = alloca i8, i64 13, align 1
  store [13 x i8] c"trap reached\00", ptr %2, align 1
  %3 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %2)
  call void @llvm.trap()
  unreachable
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

define { i16, [90 x i8] } @"uint_addition::uint_addition::main"() {
  br label %61

1:                                                ; preds = %61
  %2 = load { i16 }, ptr %65, align 2
  br label %66

3:                                                ; preds = %61
  %4 = load { i32, i32, ptr }, ptr %65, align 8
  br label %252

5:                                                ; preds = %61
  unreachable

6:                                                ; preds = %66
  %7 = load { i16 }, ptr %72, align 2
  br label %73

8:                                                ; preds = %66
  %9 = load { i32, i32, ptr }, ptr %72, align 8
  br label %249

10:                                               ; preds = %66
  unreachable

11:                                               ; preds = %73
  %12 = load { i16 }, ptr %80, align 2
  br label %81

13:                                               ; preds = %73
  %14 = load { i32, i32, ptr }, ptr %80, align 8
  br label %246

15:                                               ; preds = %73
  unreachable

16:                                               ; preds = %81
  %17 = load { i32 }, ptr %89, align 4
  br label %90

18:                                               ; preds = %81
  %19 = load { i32, i32, ptr }, ptr %89, align 8
  br label %243

20:                                               ; preds = %81
  unreachable

21:                                               ; preds = %90
  %22 = load { i32 }, ptr %99, align 4
  br label %100

23:                                               ; preds = %90
  %24 = load { i32, i32, ptr }, ptr %99, align 8
  br label %240

25:                                               ; preds = %90
  unreachable

26:                                               ; preds = %100
  %27 = load { i32 }, ptr %110, align 4
  br label %111

28:                                               ; preds = %100
  %29 = load { i32, i32, ptr }, ptr %110, align 8
  br label %237

30:                                               ; preds = %100
  unreachable

31:                                               ; preds = %111
  %32 = load { i64 }, ptr %122, align 4
  br label %123

33:                                               ; preds = %111
  %34 = load { i32, i32, ptr }, ptr %122, align 8
  br label %234

35:                                               ; preds = %111
  unreachable

36:                                               ; preds = %123
  %37 = load { i64 }, ptr %135, align 4
  br label %136

38:                                               ; preds = %123
  %39 = load { i32, i32, ptr }, ptr %135, align 8
  br label %231

40:                                               ; preds = %123
  unreachable

41:                                               ; preds = %136
  %42 = load { i64 }, ptr %149, align 4
  br label %150

43:                                               ; preds = %136
  %44 = load { i32, i32, ptr }, ptr %149, align 8
  br label %228

45:                                               ; preds = %136
  unreachable

46:                                               ; preds = %150
  %47 = load { i128 }, ptr %164, align 4
  br label %165

48:                                               ; preds = %150
  %49 = load { i32, i32, ptr }, ptr %164, align 8
  br label %225

50:                                               ; preds = %150
  unreachable

51:                                               ; preds = %165
  %52 = load { i128 }, ptr %180, align 4
  br label %181

53:                                               ; preds = %165
  %54 = load { i32, i32, ptr }, ptr %180, align 8
  br label %222

55:                                               ; preds = %165
  unreachable

56:                                               ; preds = %181
  %57 = load { i128 }, ptr %197, align 4
  br label %198

58:                                               ; preds = %181
  %59 = load { i32, i32, ptr }, ptr %197, align 8
  br label %219

60:                                               ; preds = %181
  unreachable

61:                                               ; preds = %0
  %62 = call { i16, [16 x i8] } @"core::integer::U16Add::add"(i16 4, i16 6)
  %63 = extractvalue { i16, [16 x i8] } %62, 0
  %64 = extractvalue { i16, [16 x i8] } %62, 1
  %65 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %64, ptr %65, align 1
  switch i16 %63, label %5 [
    i16 0, label %1
    i16 1, label %3
  ]

66:                                               ; preds = %1
  %67 = phi { i16 } [ %2, %1 ]
  %68 = call i16 @"struct_deconstruct<Tuple<u16>>"({ i16 } %67)
  %69 = call { i16, [16 x i8] } @"core::integer::U16Add::add"(i16 2, i16 10)
  %70 = extractvalue { i16, [16 x i8] } %69, 0
  %71 = extractvalue { i16, [16 x i8] } %69, 1
  %72 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %71, ptr %72, align 1
  switch i16 %70, label %10 [
    i16 0, label %6
    i16 1, label %8
  ]

73:                                               ; preds = %6
  %74 = phi i16 [ %68, %6 ]
  %75 = phi { i16 } [ %7, %6 ]
  %76 = call i16 @"struct_deconstruct<Tuple<u16>>"({ i16 } %75)
  %77 = call { i16, [16 x i8] } @"core::integer::U16Add::add"(i16 50, i16 2)
  %78 = extractvalue { i16, [16 x i8] } %77, 0
  %79 = extractvalue { i16, [16 x i8] } %77, 1
  %80 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %79, ptr %80, align 1
  switch i16 %78, label %15 [
    i16 0, label %11
    i16 1, label %13
  ]

81:                                               ; preds = %11
  %82 = phi i16 [ %74, %11 ]
  %83 = phi i16 [ %76, %11 ]
  %84 = phi { i16 } [ %12, %11 ]
  %85 = call i16 @"struct_deconstruct<Tuple<u16>>"({ i16 } %84)
  %86 = call { i16, [16 x i8] } @"core::integer::U32Add::add"(i32 4, i32 6)
  %87 = extractvalue { i16, [16 x i8] } %86, 0
  %88 = extractvalue { i16, [16 x i8] } %86, 1
  %89 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %88, ptr %89, align 1
  switch i16 %87, label %20 [
    i16 0, label %16
    i16 1, label %18
  ]

90:                                               ; preds = %16
  %91 = phi i16 [ %82, %16 ]
  %92 = phi i16 [ %83, %16 ]
  %93 = phi i16 [ %85, %16 ]
  %94 = phi { i32 } [ %17, %16 ]
  %95 = call i32 @"struct_deconstruct<Tuple<u32>>"({ i32 } %94)
  %96 = call { i16, [16 x i8] } @"core::integer::U32Add::add"(i32 2, i32 10)
  %97 = extractvalue { i16, [16 x i8] } %96, 0
  %98 = extractvalue { i16, [16 x i8] } %96, 1
  %99 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %98, ptr %99, align 1
  switch i16 %97, label %25 [
    i16 0, label %21
    i16 1, label %23
  ]

100:                                              ; preds = %21
  %101 = phi i16 [ %91, %21 ]
  %102 = phi i16 [ %92, %21 ]
  %103 = phi i16 [ %93, %21 ]
  %104 = phi i32 [ %95, %21 ]
  %105 = phi { i32 } [ %22, %21 ]
  %106 = call i32 @"struct_deconstruct<Tuple<u32>>"({ i32 } %105)
  %107 = call { i16, [16 x i8] } @"core::integer::U32Add::add"(i32 50, i32 2)
  %108 = extractvalue { i16, [16 x i8] } %107, 0
  %109 = extractvalue { i16, [16 x i8] } %107, 1
  %110 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %109, ptr %110, align 1
  switch i16 %108, label %30 [
    i16 0, label %26
    i16 1, label %28
  ]

111:                                              ; preds = %26
  %112 = phi i16 [ %101, %26 ]
  %113 = phi i16 [ %102, %26 ]
  %114 = phi i16 [ %103, %26 ]
  %115 = phi i32 [ %104, %26 ]
  %116 = phi i32 [ %106, %26 ]
  %117 = phi { i32 } [ %27, %26 ]
  %118 = call i32 @"struct_deconstruct<Tuple<u32>>"({ i32 } %117)
  %119 = call { i16, [16 x i8] } @"core::integer::U64Add::add"(i64 4, i64 6)
  %120 = extractvalue { i16, [16 x i8] } %119, 0
  %121 = extractvalue { i16, [16 x i8] } %119, 1
  %122 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %121, ptr %122, align 1
  switch i16 %120, label %35 [
    i16 0, label %31
    i16 1, label %33
  ]

123:                                              ; preds = %31
  %124 = phi i16 [ %112, %31 ]
  %125 = phi i16 [ %113, %31 ]
  %126 = phi i16 [ %114, %31 ]
  %127 = phi i32 [ %115, %31 ]
  %128 = phi i32 [ %116, %31 ]
  %129 = phi i32 [ %118, %31 ]
  %130 = phi { i64 } [ %32, %31 ]
  %131 = call i64 @"struct_deconstruct<Tuple<u64>>"({ i64 } %130)
  %132 = call { i16, [16 x i8] } @"core::integer::U64Add::add"(i64 2, i64 10)
  %133 = extractvalue { i16, [16 x i8] } %132, 0
  %134 = extractvalue { i16, [16 x i8] } %132, 1
  %135 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %134, ptr %135, align 1
  switch i16 %133, label %40 [
    i16 0, label %36
    i16 1, label %38
  ]

136:                                              ; preds = %36
  %137 = phi i16 [ %124, %36 ]
  %138 = phi i16 [ %125, %36 ]
  %139 = phi i16 [ %126, %36 ]
  %140 = phi i32 [ %127, %36 ]
  %141 = phi i32 [ %128, %36 ]
  %142 = phi i32 [ %129, %36 ]
  %143 = phi i64 [ %131, %36 ]
  %144 = phi { i64 } [ %37, %36 ]
  %145 = call i64 @"struct_deconstruct<Tuple<u64>>"({ i64 } %144)
  %146 = call { i16, [16 x i8] } @"core::integer::U64Add::add"(i64 50, i64 2)
  %147 = extractvalue { i16, [16 x i8] } %146, 0
  %148 = extractvalue { i16, [16 x i8] } %146, 1
  %149 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %148, ptr %149, align 1
  switch i16 %147, label %45 [
    i16 0, label %41
    i16 1, label %43
  ]

150:                                              ; preds = %41
  %151 = phi i16 [ %137, %41 ]
  %152 = phi i16 [ %138, %41 ]
  %153 = phi i16 [ %139, %41 ]
  %154 = phi i32 [ %140, %41 ]
  %155 = phi i32 [ %141, %41 ]
  %156 = phi i32 [ %142, %41 ]
  %157 = phi i64 [ %143, %41 ]
  %158 = phi i64 [ %145, %41 ]
  %159 = phi { i64 } [ %42, %41 ]
  %160 = call i64 @"struct_deconstruct<Tuple<u64>>"({ i64 } %159)
  %161 = call { i16, [16 x i8] } @"core::integer::U128Add::add"(i128 4, i128 6)
  %162 = extractvalue { i16, [16 x i8] } %161, 0
  %163 = extractvalue { i16, [16 x i8] } %161, 1
  %164 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %163, ptr %164, align 1
  switch i16 %162, label %50 [
    i16 0, label %46
    i16 1, label %48
  ]

165:                                              ; preds = %46
  %166 = phi i16 [ %151, %46 ]
  %167 = phi i16 [ %152, %46 ]
  %168 = phi i16 [ %153, %46 ]
  %169 = phi i32 [ %154, %46 ]
  %170 = phi i32 [ %155, %46 ]
  %171 = phi i32 [ %156, %46 ]
  %172 = phi i64 [ %157, %46 ]
  %173 = phi i64 [ %158, %46 ]
  %174 = phi i64 [ %160, %46 ]
  %175 = phi { i128 } [ %47, %46 ]
  %176 = call i128 @"struct_deconstruct<Tuple<u128>>"({ i128 } %175)
  %177 = call { i16, [16 x i8] } @"core::integer::U128Add::add"(i128 2, i128 10)
  %178 = extractvalue { i16, [16 x i8] } %177, 0
  %179 = extractvalue { i16, [16 x i8] } %177, 1
  %180 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %179, ptr %180, align 1
  switch i16 %178, label %55 [
    i16 0, label %51
    i16 1, label %53
  ]

181:                                              ; preds = %51
  %182 = phi i16 [ %166, %51 ]
  %183 = phi i16 [ %167, %51 ]
  %184 = phi i16 [ %168, %51 ]
  %185 = phi i32 [ %169, %51 ]
  %186 = phi i32 [ %170, %51 ]
  %187 = phi i32 [ %171, %51 ]
  %188 = phi i64 [ %172, %51 ]
  %189 = phi i64 [ %173, %51 ]
  %190 = phi i64 [ %174, %51 ]
  %191 = phi i128 [ %176, %51 ]
  %192 = phi { i128 } [ %52, %51 ]
  %193 = call i128 @"struct_deconstruct<Tuple<u128>>"({ i128 } %192)
  %194 = call { i16, [16 x i8] } @"core::integer::U128Add::add"(i128 50, i128 2)
  %195 = extractvalue { i16, [16 x i8] } %194, 0
  %196 = extractvalue { i16, [16 x i8] } %194, 1
  %197 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %196, ptr %197, align 1
  switch i16 %195, label %60 [
    i16 0, label %56
    i16 1, label %58
  ]

198:                                              ; preds = %56
  %199 = phi i16 [ %182, %56 ]
  %200 = phi i16 [ %183, %56 ]
  %201 = phi i16 [ %184, %56 ]
  %202 = phi i32 [ %185, %56 ]
  %203 = phi i32 [ %186, %56 ]
  %204 = phi i32 [ %187, %56 ]
  %205 = phi i64 [ %188, %56 ]
  %206 = phi i64 [ %189, %56 ]
  %207 = phi i64 [ %190, %56 ]
  %208 = phi i128 [ %191, %56 ]
  %209 = phi i128 [ %193, %56 ]
  %210 = phi { i128 } [ %57, %56 ]
  %211 = call i128 @"struct_deconstruct<Tuple<u128>>"({ i128 } %210)
  %212 = call { i16, i16, i16 } @"struct_construct<Tuple<u16, u16, u16>>"(i16 %199, i16 %200, i16 %201)
  %213 = call { i32, i32, i32 } @"struct_construct<Tuple<u32, u32, u32>>"(i32 %202, i32 %203, i32 %204)
  %214 = call { i64, i64, i64 } @"struct_construct<Tuple<u64, u64, u64>>"(i64 %205, i64 %206, i64 %207)
  %215 = call { i128, i128, i128 } @"struct_construct<Tuple<u128, u128, u128>>"(i128 %208, i128 %209, i128 %211)
  %216 = call { { i16, i16, i16 }, { i32, i32, i32 }, { i64, i64, i64 }, { i128, i128, i128 } } @"struct_construct<Tuple<Tuple<u16, u16, u16>, Tuple<u32, u32, u32>, Tuple<u64, u64, u64>, Tuple<u128, u128, u128>>>"({ i16, i16, i16 } %212, { i32, i32, i32 } %213, { i64, i64, i64 } %214, { i128, i128, i128 } %215)
  %217 = call { { { i16, i16, i16 }, { i32, i32, i32 }, { i64, i64, i64 }, { i128, i128, i128 } } } @"struct_construct<Tuple<Tuple<Tuple<u16, u16, u16>, Tuple<u32, u32, u32>, Tuple<u64, u64, u64>, Tuple<u128, u128, u128>>>>"({ { i16, i16, i16 }, { i32, i32, i32 }, { i64, i64, i64 }, { i128, i128, i128 } } %216)
  %218 = call { i16, [90 x i8] } @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 0>"({ { { i16, i16, i16 }, { i32, i32, i32 }, { i64, i64, i64 }, { i128, i128, i128 } } } %217)
  ret { i16, [90 x i8] } %218

219:                                              ; preds = %58
  %220 = phi { i32, i32, ptr } [ %59, %58 ]
  %221 = call { i16, [90 x i8] } @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"({ i32, i32, ptr } %220)
  ret { i16, [90 x i8] } %221

222:                                              ; preds = %53
  %223 = phi { i32, i32, ptr } [ %54, %53 ]
  %224 = call { i16, [90 x i8] } @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"({ i32, i32, ptr } %223)
  ret { i16, [90 x i8] } %224

225:                                              ; preds = %48
  %226 = phi { i32, i32, ptr } [ %49, %48 ]
  %227 = call { i16, [90 x i8] } @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"({ i32, i32, ptr } %226)
  ret { i16, [90 x i8] } %227

228:                                              ; preds = %43
  %229 = phi { i32, i32, ptr } [ %44, %43 ]
  %230 = call { i16, [90 x i8] } @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"({ i32, i32, ptr } %229)
  ret { i16, [90 x i8] } %230

231:                                              ; preds = %38
  %232 = phi { i32, i32, ptr } [ %39, %38 ]
  %233 = call { i16, [90 x i8] } @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"({ i32, i32, ptr } %232)
  ret { i16, [90 x i8] } %233

234:                                              ; preds = %33
  %235 = phi { i32, i32, ptr } [ %34, %33 ]
  %236 = call { i16, [90 x i8] } @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"({ i32, i32, ptr } %235)
  ret { i16, [90 x i8] } %236

237:                                              ; preds = %28
  %238 = phi { i32, i32, ptr } [ %29, %28 ]
  %239 = call { i16, [90 x i8] } @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"({ i32, i32, ptr } %238)
  ret { i16, [90 x i8] } %239

240:                                              ; preds = %23
  %241 = phi { i32, i32, ptr } [ %24, %23 ]
  %242 = call { i16, [90 x i8] } @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"({ i32, i32, ptr } %241)
  ret { i16, [90 x i8] } %242

243:                                              ; preds = %18
  %244 = phi { i32, i32, ptr } [ %19, %18 ]
  %245 = call { i16, [90 x i8] } @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"({ i32, i32, ptr } %244)
  ret { i16, [90 x i8] } %245

246:                                              ; preds = %13
  %247 = phi { i32, i32, ptr } [ %14, %13 ]
  %248 = call { i16, [90 x i8] } @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"({ i32, i32, ptr } %247)
  ret { i16, [90 x i8] } %248

249:                                              ; preds = %8
  %250 = phi { i32, i32, ptr } [ %9, %8 ]
  %251 = call { i16, [90 x i8] } @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"({ i32, i32, ptr } %250)
  ret { i16, [90 x i8] } %251

252:                                              ; preds = %3
  %253 = phi { i32, i32, ptr } [ %4, %3 ]
  %254 = call { i16, [90 x i8] } @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"({ i32, i32, ptr } %253)
  ret { i16, [90 x i8] } %254
}

define void @"_mlir_ciface_uint_addition::uint_addition::main"(ptr %0) {
  %2 = call { i16, [90 x i8] } @"uint_addition::uint_addition::main"()
  store { i16, [90 x i8] } %2, ptr %0, align 2
  ret void
}

define { i16, [16 x i8] } @"core::integer::U16Add::add"(i16 %0, i16 %1) {
  br label %8

3:                                                ; preds = %20
  %4 = load { i16 }, ptr %25, align 2
  br label %26

5:                                                ; preds = %20
  %6 = load { i32, i32, ptr }, ptr %25, align 8
  br label %31

7:                                                ; preds = %20
  unreachable

8:                                                ; preds = %2
  %9 = phi i16 [ %0, %2 ]
  %10 = phi i16 [ %1, %2 ]
  %11 = call { i16, i1 } @llvm.uadd.with.overflow.i16(i16 %9, i16 %10)
  %12 = extractvalue { i16, i1 } %11, 0
  %13 = extractvalue { i16, i1 } %11, 1
  br i1 %13, label %17, label %14

14:                                               ; preds = %8
  %15 = phi i16 [ %12, %8 ]
  %16 = call { i16, [2 x i8] } @"enum_init<core::result::Result::<core::integer::u16, core::integer::u16>, 0>"(i16 %15)
  br label %20

17:                                               ; preds = %8
  %18 = phi i16 [ %12, %8 ]
  %19 = call { i16, [2 x i8] } @"enum_init<core::result::Result::<core::integer::u16, core::integer::u16>, 1>"(i16 %18)
  br label %20

20:                                               ; preds = %17, %14
  %21 = phi { i16, [2 x i8] } [ %19, %17 ], [ %16, %14 ]
  %22 = call { i16, [16 x i8] } @"core::result::ResultTraitImpl::<core::integer::u16, core::integer::u16>::expect::<core::integer::u16Drop>"({ i16, [2 x i8] } %21, i256 155775200859838811096160292336445452151)
  %23 = extractvalue { i16, [16 x i8] } %22, 0
  %24 = extractvalue { i16, [16 x i8] } %22, 1
  %25 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %24, ptr %25, align 1
  switch i16 %23, label %7 [
    i16 0, label %3
    i16 1, label %5
  ]

26:                                               ; preds = %3
  %27 = phi { i16 } [ %4, %3 ]
  %28 = call i16 @"struct_deconstruct<Tuple<u16>>"({ i16 } %27)
  %29 = call { i16 } @"struct_construct<Tuple<u16>>"(i16 %28)
  %30 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u16)>, 0>"({ i16 } %29)
  ret { i16, [16 x i8] } %30

31:                                               ; preds = %5
  %32 = phi { i32, i32, ptr } [ %6, %5 ]
  %33 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u16)>, 1>"({ i32, i32, ptr } %32)
  ret { i16, [16 x i8] } %33
}

define void @"_mlir_ciface_core::integer::U16Add::add"(ptr %0, i16 %1, i16 %2) {
  %4 = call { i16, [16 x i8] } @"core::integer::U16Add::add"(i16 %1, i16 %2)
  store { i16, [16 x i8] } %4, ptr %0, align 2
  ret void
}

define { i16, [16 x i8] } @"core::integer::U32Add::add"(i32 %0, i32 %1) {
  br label %8

3:                                                ; preds = %20
  %4 = load { i32 }, ptr %25, align 4
  br label %26

5:                                                ; preds = %20
  %6 = load { i32, i32, ptr }, ptr %25, align 8
  br label %31

7:                                                ; preds = %20
  unreachable

8:                                                ; preds = %2
  %9 = phi i32 [ %0, %2 ]
  %10 = phi i32 [ %1, %2 ]
  %11 = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %9, i32 %10)
  %12 = extractvalue { i32, i1 } %11, 0
  %13 = extractvalue { i32, i1 } %11, 1
  br i1 %13, label %17, label %14

14:                                               ; preds = %8
  %15 = phi i32 [ %12, %8 ]
  %16 = call { i16, [4 x i8] } @"enum_init<core::result::Result::<core::integer::u32, core::integer::u32>, 0>"(i32 %15)
  br label %20

17:                                               ; preds = %8
  %18 = phi i32 [ %12, %8 ]
  %19 = call { i16, [4 x i8] } @"enum_init<core::result::Result::<core::integer::u32, core::integer::u32>, 1>"(i32 %18)
  br label %20

20:                                               ; preds = %17, %14
  %21 = phi { i16, [4 x i8] } [ %19, %17 ], [ %16, %14 ]
  %22 = call { i16, [16 x i8] } @"core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>"({ i16, [4 x i8] } %21, i256 155785504323917466144735657540098748279)
  %23 = extractvalue { i16, [16 x i8] } %22, 0
  %24 = extractvalue { i16, [16 x i8] } %22, 1
  %25 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %24, ptr %25, align 1
  switch i16 %23, label %7 [
    i16 0, label %3
    i16 1, label %5
  ]

26:                                               ; preds = %3
  %27 = phi { i32 } [ %4, %3 ]
  %28 = call i32 @"struct_deconstruct<Tuple<u32>>"({ i32 } %27)
  %29 = call { i32 } @"struct_construct<Tuple<u32>>"(i32 %28)
  %30 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u32)>, 0>"({ i32 } %29)
  ret { i16, [16 x i8] } %30

31:                                               ; preds = %5
  %32 = phi { i32, i32, ptr } [ %6, %5 ]
  %33 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u32)>, 1>"({ i32, i32, ptr } %32)
  ret { i16, [16 x i8] } %33
}

define void @"_mlir_ciface_core::integer::U32Add::add"(ptr %0, i32 %1, i32 %2) {
  %4 = call { i16, [16 x i8] } @"core::integer::U32Add::add"(i32 %1, i32 %2)
  store { i16, [16 x i8] } %4, ptr %0, align 2
  ret void
}

define { i16, [16 x i8] } @"core::integer::U64Add::add"(i64 %0, i64 %1) {
  br label %8

3:                                                ; preds = %20
  %4 = load { i64 }, ptr %25, align 4
  br label %26

5:                                                ; preds = %20
  %6 = load { i32, i32, ptr }, ptr %25, align 8
  br label %31

7:                                                ; preds = %20
  unreachable

8:                                                ; preds = %2
  %9 = phi i64 [ %0, %2 ]
  %10 = phi i64 [ %1, %2 ]
  %11 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %9, i64 %10)
  %12 = extractvalue { i64, i1 } %11, 0
  %13 = extractvalue { i64, i1 } %11, 1
  br i1 %13, label %17, label %14

14:                                               ; preds = %8
  %15 = phi i64 [ %12, %8 ]
  %16 = call { i16, [8 x i8] } @"enum_init<core::result::Result::<core::integer::u64, core::integer::u64>, 0>"(i64 %15)
  br label %20

17:                                               ; preds = %8
  %18 = phi i64 [ %12, %8 ]
  %19 = call { i16, [8 x i8] } @"enum_init<core::result::Result::<core::integer::u64, core::integer::u64>, 1>"(i64 %18)
  br label %20

20:                                               ; preds = %17, %14
  %21 = phi { i16, [8 x i8] } [ %19, %17 ], [ %16, %14 ]
  %22 = call { i16, [16 x i8] } @"core::result::ResultTraitImpl::<core::integer::u64, core::integer::u64>::expect::<core::integer::u64Drop>"({ i16, [8 x i8] } %21, i256 155801121779312277930962096923588980599)
  %23 = extractvalue { i16, [16 x i8] } %22, 0
  %24 = extractvalue { i16, [16 x i8] } %22, 1
  %25 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %24, ptr %25, align 1
  switch i16 %23, label %7 [
    i16 0, label %3
    i16 1, label %5
  ]

26:                                               ; preds = %3
  %27 = phi { i64 } [ %4, %3 ]
  %28 = call i64 @"struct_deconstruct<Tuple<u64>>"({ i64 } %27)
  %29 = call { i64 } @"struct_construct<Tuple<u64>>"(i64 %28)
  %30 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u64)>, 0>"({ i64 } %29)
  ret { i16, [16 x i8] } %30

31:                                               ; preds = %5
  %32 = phi { i32, i32, ptr } [ %6, %5 ]
  %33 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u64)>, 1>"({ i32, i32, ptr } %32)
  ret { i16, [16 x i8] } %33
}

define void @"_mlir_ciface_core::integer::U64Add::add"(ptr %0, i64 %1, i64 %2) {
  %4 = call { i16, [16 x i8] } @"core::integer::U64Add::add"(i64 %1, i64 %2)
  store { i16, [16 x i8] } %4, ptr %0, align 2
  ret void
}

define { i16, [16 x i8] } @"core::integer::U128Add::add"(i128 %0, i128 %1) {
  br label %8

3:                                                ; preds = %20
  %4 = load { i128 }, ptr %25, align 4
  br label %26

5:                                                ; preds = %20
  %6 = load { i32, i32, ptr }, ptr %25, align 8
  br label %31

7:                                                ; preds = %20
  unreachable

8:                                                ; preds = %2
  %9 = phi i128 [ %0, %2 ]
  %10 = phi i128 [ %1, %2 ]
  %11 = call { i128, i1 } @llvm.uadd.with.overflow.i128(i128 %9, i128 %10)
  %12 = extractvalue { i128, i1 } %11, 0
  %13 = extractvalue { i128, i1 } %11, 1
  br i1 %13, label %17, label %14

14:                                               ; preds = %8
  %15 = phi i128 [ %12, %8 ]
  %16 = call { i16, [16 x i8] } @"enum_init<core::result::Result::<core::integer::u128, core::integer::u128>, 0>"(i128 %15)
  br label %20

17:                                               ; preds = %8
  %18 = phi i128 [ %12, %8 ]
  %19 = call { i16, [16 x i8] } @"enum_init<core::result::Result::<core::integer::u128, core::integer::u128>, 1>"(i128 %18)
  br label %20

20:                                               ; preds = %17, %14
  %21 = phi { i16, [16 x i8] } [ %19, %17 ], [ %16, %14 ]
  %22 = call { i16, [16 x i8] } @"core::result::ResultTraitImpl::<core::integer::u128, core::integer::u128>::expect::<core::integer::u128Drop>"({ i16, [16 x i8] } %21, i256 39878429859757942499084499860145094553463)
  %23 = extractvalue { i16, [16 x i8] } %22, 0
  %24 = extractvalue { i16, [16 x i8] } %22, 1
  %25 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %24, ptr %25, align 1
  switch i16 %23, label %7 [
    i16 0, label %3
    i16 1, label %5
  ]

26:                                               ; preds = %3
  %27 = phi { i128 } [ %4, %3 ]
  %28 = call i128 @"struct_deconstruct<Tuple<u128>>"({ i128 } %27)
  %29 = call { i128 } @"struct_construct<Tuple<u128>>"(i128 %28)
  %30 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u128)>, 0>"({ i128 } %29)
  ret { i16, [16 x i8] } %30

31:                                               ; preds = %5
  %32 = phi { i32, i32, ptr } [ %6, %5 ]
  %33 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u128)>, 1>"({ i32, i32, ptr } %32)
  ret { i16, [16 x i8] } %33
}

define void @"_mlir_ciface_core::integer::U128Add::add"(ptr %0, i128 %1, i128 %2) {
  %4 = call { i16, [16 x i8] } @"core::integer::U128Add::add"(i128 %1, i128 %2)
  store { i16, [16 x i8] } %4, ptr %0, align 2
  ret void
}

define { i16, [16 x i8] } @"core::result::ResultTraitImpl::<core::integer::u16, core::integer::u16>::expect::<core::integer::u16Drop>"({ i16, [2 x i8] } %0, i256 %1) {
  br label %7

3:                                                ; preds = %7
  %4 = load i16, ptr %12, align 2
  br label %13

5:                                                ; preds = %7
  br label %17

6:                                                ; preds = %7
  unreachable

7:                                                ; preds = %2
  %8 = phi { i16, [2 x i8] } [ %0, %2 ]
  %9 = phi i256 [ %1, %2 ]
  %10 = extractvalue { i16, [2 x i8] } %8, 0
  %11 = extractvalue { i16, [2 x i8] } %8, 1
  %12 = alloca [2 x i8], i64 1, align 1
  store [2 x i8] %11, ptr %12, align 1
  switch i16 %10, label %6 [
    i16 0, label %3
    i16 1, label %5
  ]

13:                                               ; preds = %3
  %14 = phi i16 [ %4, %3 ]
  %15 = call { i16 } @"struct_construct<Tuple<u16>>"(i16 %14)
  %16 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u16)>, 0>"({ i16 } %15)
  ret { i16, [16 x i8] } %16

17:                                               ; preds = %5
  %18 = phi i256 [ %9, %5 ]
  %19 = call { i32, i32, ptr } @"array_new<felt252>"()
  %20 = call { i32, i32, ptr } @"array_append<felt252>"({ i32, i32, ptr } %19, i256 %18)
  %21 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u16)>, 1>"({ i32, i32, ptr } %20)
  ret { i16, [16 x i8] } %21
}

define void @"_mlir_ciface_core::result::ResultTraitImpl::<core::integer::u16, core::integer::u16>::expect::<core::integer::u16Drop>"(ptr %0, { i16, [2 x i8] } %1, i256 %2) {
  %4 = call { i16, [16 x i8] } @"core::result::ResultTraitImpl::<core::integer::u16, core::integer::u16>::expect::<core::integer::u16Drop>"({ i16, [2 x i8] } %1, i256 %2)
  store { i16, [16 x i8] } %4, ptr %0, align 2
  ret void
}

define { i16, [16 x i8] } @"core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>"({ i16, [4 x i8] } %0, i256 %1) {
  br label %7

3:                                                ; preds = %7
  %4 = load i32, ptr %12, align 4
  br label %13

5:                                                ; preds = %7
  br label %17

6:                                                ; preds = %7
  unreachable

7:                                                ; preds = %2
  %8 = phi { i16, [4 x i8] } [ %0, %2 ]
  %9 = phi i256 [ %1, %2 ]
  %10 = extractvalue { i16, [4 x i8] } %8, 0
  %11 = extractvalue { i16, [4 x i8] } %8, 1
  %12 = alloca [4 x i8], i64 1, align 1
  store [4 x i8] %11, ptr %12, align 1
  switch i16 %10, label %6 [
    i16 0, label %3
    i16 1, label %5
  ]

13:                                               ; preds = %3
  %14 = phi i32 [ %4, %3 ]
  %15 = call { i32 } @"struct_construct<Tuple<u32>>"(i32 %14)
  %16 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u32)>, 0>"({ i32 } %15)
  ret { i16, [16 x i8] } %16

17:                                               ; preds = %5
  %18 = phi i256 [ %9, %5 ]
  %19 = call { i32, i32, ptr } @"array_new<felt252>"()
  %20 = call { i32, i32, ptr } @"array_append<felt252>"({ i32, i32, ptr } %19, i256 %18)
  %21 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u32)>, 1>"({ i32, i32, ptr } %20)
  ret { i16, [16 x i8] } %21
}

define void @"_mlir_ciface_core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>"(ptr %0, { i16, [4 x i8] } %1, i256 %2) {
  %4 = call { i16, [16 x i8] } @"core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>"({ i16, [4 x i8] } %1, i256 %2)
  store { i16, [16 x i8] } %4, ptr %0, align 2
  ret void
}

define { i16, [16 x i8] } @"core::result::ResultTraitImpl::<core::integer::u64, core::integer::u64>::expect::<core::integer::u64Drop>"({ i16, [8 x i8] } %0, i256 %1) {
  br label %7

3:                                                ; preds = %7
  %4 = load i64, ptr %12, align 4
  br label %13

5:                                                ; preds = %7
  br label %17

6:                                                ; preds = %7
  unreachable

7:                                                ; preds = %2
  %8 = phi { i16, [8 x i8] } [ %0, %2 ]
  %9 = phi i256 [ %1, %2 ]
  %10 = extractvalue { i16, [8 x i8] } %8, 0
  %11 = extractvalue { i16, [8 x i8] } %8, 1
  %12 = alloca [8 x i8], i64 1, align 1
  store [8 x i8] %11, ptr %12, align 1
  switch i16 %10, label %6 [
    i16 0, label %3
    i16 1, label %5
  ]

13:                                               ; preds = %3
  %14 = phi i64 [ %4, %3 ]
  %15 = call { i64 } @"struct_construct<Tuple<u64>>"(i64 %14)
  %16 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u64)>, 0>"({ i64 } %15)
  ret { i16, [16 x i8] } %16

17:                                               ; preds = %5
  %18 = phi i256 [ %9, %5 ]
  %19 = call { i32, i32, ptr } @"array_new<felt252>"()
  %20 = call { i32, i32, ptr } @"array_append<felt252>"({ i32, i32, ptr } %19, i256 %18)
  %21 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u64)>, 1>"({ i32, i32, ptr } %20)
  ret { i16, [16 x i8] } %21
}

define void @"_mlir_ciface_core::result::ResultTraitImpl::<core::integer::u64, core::integer::u64>::expect::<core::integer::u64Drop>"(ptr %0, { i16, [8 x i8] } %1, i256 %2) {
  %4 = call { i16, [16 x i8] } @"core::result::ResultTraitImpl::<core::integer::u64, core::integer::u64>::expect::<core::integer::u64Drop>"({ i16, [8 x i8] } %1, i256 %2)
  store { i16, [16 x i8] } %4, ptr %0, align 2
  ret void
}

define { i16, [16 x i8] } @"core::result::ResultTraitImpl::<core::integer::u128, core::integer::u128>::expect::<core::integer::u128Drop>"({ i16, [16 x i8] } %0, i256 %1) {
  br label %7

3:                                                ; preds = %7
  %4 = load i128, ptr %12, align 4
  br label %13

5:                                                ; preds = %7
  br label %17

6:                                                ; preds = %7
  unreachable

7:                                                ; preds = %2
  %8 = phi { i16, [16 x i8] } [ %0, %2 ]
  %9 = phi i256 [ %1, %2 ]
  %10 = extractvalue { i16, [16 x i8] } %8, 0
  %11 = extractvalue { i16, [16 x i8] } %8, 1
  %12 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %11, ptr %12, align 1
  switch i16 %10, label %6 [
    i16 0, label %3
    i16 1, label %5
  ]

13:                                               ; preds = %3
  %14 = phi i128 [ %4, %3 ]
  %15 = call { i128 } @"struct_construct<Tuple<u128>>"(i128 %14)
  %16 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u128)>, 0>"({ i128 } %15)
  ret { i16, [16 x i8] } %16

17:                                               ; preds = %5
  %18 = phi i256 [ %9, %5 ]
  %19 = call { i32, i32, ptr } @"array_new<felt252>"()
  %20 = call { i32, i32, ptr } @"array_append<felt252>"({ i32, i32, ptr } %19, i256 %18)
  %21 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::integer::u128)>, 1>"({ i32, i32, ptr } %20)
  ret { i16, [16 x i8] } %21
}

define void @"_mlir_ciface_core::result::ResultTraitImpl::<core::integer::u128, core::integer::u128>::expect::<core::integer::u128Drop>"(ptr %0, { i16, [16 x i8] } %1, i256 %2) {
  %4 = call { i16, [16 x i8] } @"core::result::ResultTraitImpl::<core::integer::u128, core::integer::u128>::expect::<core::integer::u128Drop>"({ i16, [16 x i8] } %1, i256 %2)
  store { i16, [16 x i8] } %4, ptr %0, align 2
  ret void
}

; Function Attrs: cold noreturn nounwind
declare void @llvm.trap() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare { i16, i1 } @llvm.uadd.with.overflow.i16(i16, i16) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare { i128, i1 } @llvm.uadd.with.overflow.i128(i128, i128) #2

attributes #0 = { alwaysinline norecurse nounwind }
attributes #1 = { cold noreturn nounwind }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
