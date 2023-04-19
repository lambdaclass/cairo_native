; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare ptr @realloc(ptr, i64)

declare ptr @memmove(ptr, ptr, i64)

declare i32 @dprintf(i32, ptr, ...)

; Function Attrs: alwaysinline norecurse nounwind
define internal i16 @"struct_deconstruct<Tuple<u16>>"(<{ i16 }> %0) #0 {
  %2 = extractvalue <{ i16 }> %0, 0
  ret i16 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i32 @"struct_deconstruct<Tuple<u32>>"(<{ i32 }> %0) #0 {
  %2 = extractvalue <{ i32 }> %0, 0
  ret i32 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i64 @"struct_deconstruct<Tuple<u64>>"(<{ i64 }> %0) #0 {
  %2 = extractvalue <{ i64 }> %0, 0
  ret i64 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i128 @"struct_deconstruct<Tuple<u128>>"(<{ i128 }> %0) #0 {
  %2 = extractvalue <{ i128 }> %0, 0
  ret i128 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, i16, i16 }> @"struct_construct<Tuple<u16, u16, u16>>"(i16 %0, i16 %1, i16 %2) #0 {
  %4 = insertvalue <{ i16, i16, i16 }> undef, i16 %0, 0
  %5 = insertvalue <{ i16, i16, i16 }> %4, i16 %1, 1
  %6 = insertvalue <{ i16, i16, i16 }> %5, i16 %2, 2
  ret <{ i16, i16, i16 }> %6
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i32, i32, i32 }> @"struct_construct<Tuple<u32, u32, u32>>"(i32 %0, i32 %1, i32 %2) #0 {
  %4 = insertvalue <{ i32, i32, i32 }> undef, i32 %0, 0
  %5 = insertvalue <{ i32, i32, i32 }> %4, i32 %1, 1
  %6 = insertvalue <{ i32, i32, i32 }> %5, i32 %2, 2
  ret <{ i32, i32, i32 }> %6
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i64, i64, i64 }> @"struct_construct<Tuple<u64, u64, u64>>"(i64 %0, i64 %1, i64 %2) #0 {
  %4 = insertvalue <{ i64, i64, i64 }> undef, i64 %0, 0
  %5 = insertvalue <{ i64, i64, i64 }> %4, i64 %1, 1
  %6 = insertvalue <{ i64, i64, i64 }> %5, i64 %2, 2
  ret <{ i64, i64, i64 }> %6
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i128, i128, i128 }> @"struct_construct<Tuple<u128, u128, u128>>"(i128 %0, i128 %1, i128 %2) #0 {
  %4 = insertvalue <{ i128, i128, i128 }> undef, i128 %0, 0
  %5 = insertvalue <{ i128, i128, i128 }> %4, i128 %1, 1
  %6 = insertvalue <{ i128, i128, i128 }> %5, i128 %2, 2
  ret <{ i128, i128, i128 }> %6
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ <{ i16, i16, i16 }>, <{ i32, i32, i32 }>, <{ i64, i64, i64 }>, <{ i128, i128, i128 }> }> @"struct_construct<Tuple<Tuple<u16, u16, u16>, Tuple<u32, u32, u32>, Tuple<u64, u64, u64>, Tuple<u128, u128, u128>>>"(<{ i16, i16, i16 }> %0, <{ i32, i32, i32 }> %1, <{ i64, i64, i64 }> %2, <{ i128, i128, i128 }> %3) #0 {
  %5 = insertvalue <{ <{ i16, i16, i16 }>, <{ i32, i32, i32 }>, <{ i64, i64, i64 }>, <{ i128, i128, i128 }> }> undef, <{ i16, i16, i16 }> %0, 0
  %6 = insertvalue <{ <{ i16, i16, i16 }>, <{ i32, i32, i32 }>, <{ i64, i64, i64 }>, <{ i128, i128, i128 }> }> %5, <{ i32, i32, i32 }> %1, 1
  %7 = insertvalue <{ <{ i16, i16, i16 }>, <{ i32, i32, i32 }>, <{ i64, i64, i64 }>, <{ i128, i128, i128 }> }> %6, <{ i64, i64, i64 }> %2, 2
  %8 = insertvalue <{ <{ i16, i16, i16 }>, <{ i32, i32, i32 }>, <{ i64, i64, i64 }>, <{ i128, i128, i128 }> }> %7, <{ i128, i128, i128 }> %3, 3
  ret <{ <{ i16, i16, i16 }>, <{ i32, i32, i32 }>, <{ i64, i64, i64 }>, <{ i128, i128, i128 }> }> %8
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ <{ <{ i16, i16, i16 }>, <{ i32, i32, i32 }>, <{ i64, i64, i64 }>, <{ i128, i128, i128 }> }> }> @"struct_construct<Tuple<Tuple<Tuple<u16, u16, u16>, Tuple<u32, u32, u32>, Tuple<u64, u64, u64>, Tuple<u128, u128, u128>>>>"(<{ <{ i16, i16, i16 }>, <{ i32, i32, i32 }>, <{ i64, i64, i64 }>, <{ i128, i128, i128 }> }> %0) #0 {
  %2 = insertvalue <{ <{ <{ i16, i16, i16 }>, <{ i32, i32, i32 }>, <{ i64, i64, i64 }>, <{ i128, i128, i128 }> }> }> undef, <{ <{ i16, i16, i16 }>, <{ i32, i32, i32 }>, <{ i64, i64, i64 }>, <{ i128, i128, i128 }> }> %0, 0
  ret <{ <{ <{ i16, i16, i16 }>, <{ i32, i32, i32 }>, <{ i64, i64, i64 }>, <{ i128, i128, i128 }> }> }> %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [90 x i8] }> @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 0>"(<{ <{ <{ i16, i16, i16 }>, <{ i32, i32, i32 }>, <{ i64, i64, i64 }>, <{ i128, i128, i128 }> }> }> %0) #0 {
  %2 = alloca <{ i16, [90 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [90 x i8] }>, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, <{ <{ <{ i16, i16, i16 }>, <{ i32, i32, i32 }>, <{ i64, i64, i64 }>, <{ i128, i128, i128 }> }> }> }>, ptr %2, i32 0, i32 1
  store <{ <{ <{ i16, i16, i16 }>, <{ i32, i32, i32 }>, <{ i64, i64, i64 }>, <{ i128, i128, i128 }> }> }> %0, ptr %4, align 1
  %5 = load <{ i16, [90 x i8] }>, ptr %2, align 1
  ret <{ i16, [90 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [90 x i8] }> @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(<{ i32, i32, ptr }> %0) #0 {
  call void @llvm.trap()
  unreachable
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [2 x i8] }> @"enum_init<core::result::Result::<core::integer::u16, core::integer::u16>, 0>"(i16 %0) #0 {
  %2 = alloca <{ i16, [2 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [2 x i8] }>, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, i16 }>, ptr %2, i32 0, i32 1
  store i16 %0, ptr %4, align 2
  %5 = load <{ i16, [2 x i8] }>, ptr %2, align 1
  ret <{ i16, [2 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [2 x i8] }> @"enum_init<core::result::Result::<core::integer::u16, core::integer::u16>, 1>"(i16 %0) #0 {
  %2 = alloca <{ i16, [2 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [2 x i8] }>, ptr %2, i32 0, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, i16 }>, ptr %2, i32 0, i32 1
  store i16 %0, ptr %4, align 2
  %5 = load <{ i16, [2 x i8] }>, ptr %2, align 1
  ret <{ i16, [2 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16 }> @"struct_construct<Tuple<u16>>"(i16 %0) #0 {
  %2 = insertvalue <{ i16 }> undef, i16 %0, 0
  ret <{ i16 }> %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u16)>, 0>"(<{ i16 }> %0) #0 {
  %2 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, <{ i16 }> }>, ptr %2, i32 0, i32 1
  store <{ i16 }> %0, ptr %4, align 1
  %5 = load <{ i16, [16 x i8] }>, ptr %2, align 1
  ret <{ i16, [16 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u16)>, 1>"(<{ i32, i32, ptr }> %0) #0 {
  call void @llvm.trap()
  unreachable
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [4 x i8] }> @"enum_init<core::result::Result::<core::integer::u32, core::integer::u32>, 0>"(i32 %0) #0 {
  %2 = alloca <{ i16, [4 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [4 x i8] }>, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, i32 }>, ptr %2, i32 0, i32 1
  store i32 %0, ptr %4, align 4
  %5 = load <{ i16, [4 x i8] }>, ptr %2, align 1
  ret <{ i16, [4 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [4 x i8] }> @"enum_init<core::result::Result::<core::integer::u32, core::integer::u32>, 1>"(i32 %0) #0 {
  %2 = alloca <{ i16, [4 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [4 x i8] }>, ptr %2, i32 0, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, i32 }>, ptr %2, i32 0, i32 1
  store i32 %0, ptr %4, align 4
  %5 = load <{ i16, [4 x i8] }>, ptr %2, align 1
  ret <{ i16, [4 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i32 }> @"struct_construct<Tuple<u32>>"(i32 %0) #0 {
  %2 = insertvalue <{ i32 }> undef, i32 %0, 0
  ret <{ i32 }> %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u32)>, 0>"(<{ i32 }> %0) #0 {
  %2 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, <{ i32 }> }>, ptr %2, i32 0, i32 1
  store <{ i32 }> %0, ptr %4, align 1
  %5 = load <{ i16, [16 x i8] }>, ptr %2, align 1
  ret <{ i16, [16 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u32)>, 1>"(<{ i32, i32, ptr }> %0) #0 {
  call void @llvm.trap()
  unreachable
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [8 x i8] }> @"enum_init<core::result::Result::<core::integer::u64, core::integer::u64>, 0>"(i64 %0) #0 {
  %2 = alloca <{ i16, [8 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [8 x i8] }>, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, i64 }>, ptr %2, i32 0, i32 1
  store i64 %0, ptr %4, align 4
  %5 = load <{ i16, [8 x i8] }>, ptr %2, align 1
  ret <{ i16, [8 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [8 x i8] }> @"enum_init<core::result::Result::<core::integer::u64, core::integer::u64>, 1>"(i64 %0) #0 {
  %2 = alloca <{ i16, [8 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [8 x i8] }>, ptr %2, i32 0, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, i64 }>, ptr %2, i32 0, i32 1
  store i64 %0, ptr %4, align 4
  %5 = load <{ i16, [8 x i8] }>, ptr %2, align 1
  ret <{ i16, [8 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i64 }> @"struct_construct<Tuple<u64>>"(i64 %0) #0 {
  %2 = insertvalue <{ i64 }> undef, i64 %0, 0
  ret <{ i64 }> %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u64)>, 0>"(<{ i64 }> %0) #0 {
  %2 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, <{ i64 }> }>, ptr %2, i32 0, i32 1
  store <{ i64 }> %0, ptr %4, align 1
  %5 = load <{ i16, [16 x i8] }>, ptr %2, align 1
  ret <{ i16, [16 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u64)>, 1>"(<{ i32, i32, ptr }> %0) #0 {
  call void @llvm.trap()
  unreachable
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [16 x i8] }> @"enum_init<core::result::Result::<core::integer::u128, core::integer::u128>, 0>"(i128 %0) #0 {
  %2 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, i128 }>, ptr %2, i32 0, i32 1
  store i128 %0, ptr %4, align 4
  %5 = load <{ i16, [16 x i8] }>, ptr %2, align 1
  ret <{ i16, [16 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [16 x i8] }> @"enum_init<core::result::Result::<core::integer::u128, core::integer::u128>, 1>"(i128 %0) #0 {
  %2 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %2, i32 0, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, i128 }>, ptr %2, i32 0, i32 1
  store i128 %0, ptr %4, align 4
  %5 = load <{ i16, [16 x i8] }>, ptr %2, align 1
  ret <{ i16, [16 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i128 }> @"struct_construct<Tuple<u128>>"(i128 %0) #0 {
  %2 = insertvalue <{ i128 }> undef, i128 %0, 0
  ret <{ i128 }> %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u128)>, 0>"(<{ i128 }> %0) #0 {
  %2 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, <{ i128 }> }>, ptr %2, i32 0, i32 1
  store <{ i128 }> %0, ptr %4, align 1
  %5 = load <{ i16, [16 x i8] }>, ptr %2, align 1
  ret <{ i16, [16 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u128)>, 1>"(<{ i32, i32, ptr }> %0) #0 {
  call void @llvm.trap()
  unreachable
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i32, i32, ptr }> @"array_new<felt252>"() #0 {
  %1 = call ptr @realloc(ptr null, i64 256)
  %2 = insertvalue <{ i32, i32, ptr }> <{ i32 0, i32 8, ptr undef }>, ptr %1, 2
  ret <{ i32, i32, ptr }> %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i32, i32, ptr }> @"array_append<felt252>"(<{ i32, i32, ptr }> %0, i256 %1) #0 {
  %3 = extractvalue <{ i32, i32, ptr }> %0, 0
  %4 = extractvalue <{ i32, i32, ptr }> %0, 1
  %5 = icmp ult i32 %3, %4
  br i1 %5, label %13, label %6

6:                                                ; preds = %2
  %7 = mul i32 %4, 2
  %8 = zext i32 %7 to i64
  %9 = extractvalue <{ i32, i32, ptr }> %0, 2
  %10 = call ptr @realloc(ptr %9, i64 %8)
  %11 = insertvalue <{ i32, i32, ptr }> %0, ptr %10, 2
  %12 = insertvalue <{ i32, i32, ptr }> %11, i32 %7, 1
  br label %13

13:                                               ; preds = %6, %2
  %14 = phi <{ i32, i32, ptr }> [ %12, %6 ], [ %0, %2 ]
  %15 = extractvalue <{ i32, i32, ptr }> %14, 2
  %16 = getelementptr i256, ptr %15, i32 %3
  store i256 %1, ptr %16, align 4
  %17 = add i32 %3, 1
  %18 = insertvalue <{ i32, i32, ptr }> %14, i32 %17, 0
  ret <{ i32, i32, ptr }> %18
}

define <{ i16, [90 x i8] }> @"uint_addition::uint_addition::main"() {
  br label %85

1:                                                ; preds = %85
  %2 = getelementptr inbounds <{ i16, <{ i16 }> }>, ptr %87, i32 0, i32 1
  %3 = load <{ i16 }>, ptr %2, align 1
  br label %90

4:                                                ; preds = %85
  %5 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %87, i32 0, i32 1
  %6 = load <{ i32, i32, ptr }>, ptr %5, align 1
  br label %276

7:                                                ; preds = %85
  unreachable

8:                                                ; preds = %90
  %9 = getelementptr inbounds <{ i16, <{ i16 }> }>, ptr %94, i32 0, i32 1
  %10 = load <{ i16 }>, ptr %9, align 1
  br label %97

11:                                               ; preds = %90
  %12 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %94, i32 0, i32 1
  %13 = load <{ i32, i32, ptr }>, ptr %12, align 1
  br label %273

14:                                               ; preds = %90
  unreachable

15:                                               ; preds = %97
  %16 = getelementptr inbounds <{ i16, <{ i16 }> }>, ptr %102, i32 0, i32 1
  %17 = load <{ i16 }>, ptr %16, align 1
  br label %105

18:                                               ; preds = %97
  %19 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %102, i32 0, i32 1
  %20 = load <{ i32, i32, ptr }>, ptr %19, align 1
  br label %270

21:                                               ; preds = %97
  unreachable

22:                                               ; preds = %105
  %23 = getelementptr inbounds <{ i16, <{ i32 }> }>, ptr %111, i32 0, i32 1
  %24 = load <{ i32 }>, ptr %23, align 1
  br label %114

25:                                               ; preds = %105
  %26 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %111, i32 0, i32 1
  %27 = load <{ i32, i32, ptr }>, ptr %26, align 1
  br label %267

28:                                               ; preds = %105
  unreachable

29:                                               ; preds = %114
  %30 = getelementptr inbounds <{ i16, <{ i32 }> }>, ptr %121, i32 0, i32 1
  %31 = load <{ i32 }>, ptr %30, align 1
  br label %124

32:                                               ; preds = %114
  %33 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %121, i32 0, i32 1
  %34 = load <{ i32, i32, ptr }>, ptr %33, align 1
  br label %264

35:                                               ; preds = %114
  unreachable

36:                                               ; preds = %124
  %37 = getelementptr inbounds <{ i16, <{ i32 }> }>, ptr %132, i32 0, i32 1
  %38 = load <{ i32 }>, ptr %37, align 1
  br label %135

39:                                               ; preds = %124
  %40 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %132, i32 0, i32 1
  %41 = load <{ i32, i32, ptr }>, ptr %40, align 1
  br label %261

42:                                               ; preds = %124
  unreachable

43:                                               ; preds = %135
  %44 = getelementptr inbounds <{ i16, <{ i64 }> }>, ptr %144, i32 0, i32 1
  %45 = load <{ i64 }>, ptr %44, align 1
  br label %147

46:                                               ; preds = %135
  %47 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %144, i32 0, i32 1
  %48 = load <{ i32, i32, ptr }>, ptr %47, align 1
  br label %258

49:                                               ; preds = %135
  unreachable

50:                                               ; preds = %147
  %51 = getelementptr inbounds <{ i16, <{ i64 }> }>, ptr %157, i32 0, i32 1
  %52 = load <{ i64 }>, ptr %51, align 1
  br label %160

53:                                               ; preds = %147
  %54 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %157, i32 0, i32 1
  %55 = load <{ i32, i32, ptr }>, ptr %54, align 1
  br label %255

56:                                               ; preds = %147
  unreachable

57:                                               ; preds = %160
  %58 = getelementptr inbounds <{ i16, <{ i64 }> }>, ptr %171, i32 0, i32 1
  %59 = load <{ i64 }>, ptr %58, align 1
  br label %174

60:                                               ; preds = %160
  %61 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %171, i32 0, i32 1
  %62 = load <{ i32, i32, ptr }>, ptr %61, align 1
  br label %252

63:                                               ; preds = %160
  unreachable

64:                                               ; preds = %174
  %65 = getelementptr inbounds <{ i16, <{ i128 }> }>, ptr %186, i32 0, i32 1
  %66 = load <{ i128 }>, ptr %65, align 1
  br label %189

67:                                               ; preds = %174
  %68 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %186, i32 0, i32 1
  %69 = load <{ i32, i32, ptr }>, ptr %68, align 1
  br label %249

70:                                               ; preds = %174
  unreachable

71:                                               ; preds = %189
  %72 = getelementptr inbounds <{ i16, <{ i128 }> }>, ptr %202, i32 0, i32 1
  %73 = load <{ i128 }>, ptr %72, align 1
  br label %205

74:                                               ; preds = %189
  %75 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %202, i32 0, i32 1
  %76 = load <{ i32, i32, ptr }>, ptr %75, align 1
  br label %246

77:                                               ; preds = %189
  unreachable

78:                                               ; preds = %205
  %79 = getelementptr inbounds <{ i16, <{ i128 }> }>, ptr %219, i32 0, i32 1
  %80 = load <{ i128 }>, ptr %79, align 1
  br label %222

81:                                               ; preds = %205
  %82 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %219, i32 0, i32 1
  %83 = load <{ i32, i32, ptr }>, ptr %82, align 1
  br label %243

84:                                               ; preds = %205
  unreachable

85:                                               ; preds = %0
  %86 = call <{ i16, [16 x i8] }> @"core::integer::U16Add::add"(i16 4, i16 6)
  %87 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %86, ptr %87, align 1
  %88 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %87, i32 0, i32 0
  %89 = load i16, ptr %88, align 2
  switch i16 %89, label %7 [
    i16 0, label %1
    i16 1, label %4
  ]

90:                                               ; preds = %1
  %91 = phi <{ i16 }> [ %3, %1 ]
  %92 = call i16 @"struct_deconstruct<Tuple<u16>>"(<{ i16 }> %91)
  %93 = call <{ i16, [16 x i8] }> @"core::integer::U16Add::add"(i16 2, i16 10)
  %94 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %93, ptr %94, align 1
  %95 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %94, i32 0, i32 0
  %96 = load i16, ptr %95, align 2
  switch i16 %96, label %14 [
    i16 0, label %8
    i16 1, label %11
  ]

97:                                               ; preds = %8
  %98 = phi i16 [ %92, %8 ]
  %99 = phi <{ i16 }> [ %10, %8 ]
  %100 = call i16 @"struct_deconstruct<Tuple<u16>>"(<{ i16 }> %99)
  %101 = call <{ i16, [16 x i8] }> @"core::integer::U16Add::add"(i16 50, i16 2)
  %102 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %101, ptr %102, align 1
  %103 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %102, i32 0, i32 0
  %104 = load i16, ptr %103, align 2
  switch i16 %104, label %21 [
    i16 0, label %15
    i16 1, label %18
  ]

105:                                              ; preds = %15
  %106 = phi i16 [ %98, %15 ]
  %107 = phi i16 [ %100, %15 ]
  %108 = phi <{ i16 }> [ %17, %15 ]
  %109 = call i16 @"struct_deconstruct<Tuple<u16>>"(<{ i16 }> %108)
  %110 = call <{ i16, [16 x i8] }> @"core::integer::U32Add::add"(i32 4, i32 6)
  %111 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %110, ptr %111, align 1
  %112 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %111, i32 0, i32 0
  %113 = load i16, ptr %112, align 2
  switch i16 %113, label %28 [
    i16 0, label %22
    i16 1, label %25
  ]

114:                                              ; preds = %22
  %115 = phi i16 [ %106, %22 ]
  %116 = phi i16 [ %107, %22 ]
  %117 = phi i16 [ %109, %22 ]
  %118 = phi <{ i32 }> [ %24, %22 ]
  %119 = call i32 @"struct_deconstruct<Tuple<u32>>"(<{ i32 }> %118)
  %120 = call <{ i16, [16 x i8] }> @"core::integer::U32Add::add"(i32 2, i32 10)
  %121 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %120, ptr %121, align 1
  %122 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %121, i32 0, i32 0
  %123 = load i16, ptr %122, align 2
  switch i16 %123, label %35 [
    i16 0, label %29
    i16 1, label %32
  ]

124:                                              ; preds = %29
  %125 = phi i16 [ %115, %29 ]
  %126 = phi i16 [ %116, %29 ]
  %127 = phi i16 [ %117, %29 ]
  %128 = phi i32 [ %119, %29 ]
  %129 = phi <{ i32 }> [ %31, %29 ]
  %130 = call i32 @"struct_deconstruct<Tuple<u32>>"(<{ i32 }> %129)
  %131 = call <{ i16, [16 x i8] }> @"core::integer::U32Add::add"(i32 50, i32 2)
  %132 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %131, ptr %132, align 1
  %133 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %132, i32 0, i32 0
  %134 = load i16, ptr %133, align 2
  switch i16 %134, label %42 [
    i16 0, label %36
    i16 1, label %39
  ]

135:                                              ; preds = %36
  %136 = phi i16 [ %125, %36 ]
  %137 = phi i16 [ %126, %36 ]
  %138 = phi i16 [ %127, %36 ]
  %139 = phi i32 [ %128, %36 ]
  %140 = phi i32 [ %130, %36 ]
  %141 = phi <{ i32 }> [ %38, %36 ]
  %142 = call i32 @"struct_deconstruct<Tuple<u32>>"(<{ i32 }> %141)
  %143 = call <{ i16, [16 x i8] }> @"core::integer::U64Add::add"(i64 4, i64 6)
  %144 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %143, ptr %144, align 1
  %145 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %144, i32 0, i32 0
  %146 = load i16, ptr %145, align 2
  switch i16 %146, label %49 [
    i16 0, label %43
    i16 1, label %46
  ]

147:                                              ; preds = %43
  %148 = phi i16 [ %136, %43 ]
  %149 = phi i16 [ %137, %43 ]
  %150 = phi i16 [ %138, %43 ]
  %151 = phi i32 [ %139, %43 ]
  %152 = phi i32 [ %140, %43 ]
  %153 = phi i32 [ %142, %43 ]
  %154 = phi <{ i64 }> [ %45, %43 ]
  %155 = call i64 @"struct_deconstruct<Tuple<u64>>"(<{ i64 }> %154)
  %156 = call <{ i16, [16 x i8] }> @"core::integer::U64Add::add"(i64 2, i64 10)
  %157 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %156, ptr %157, align 1
  %158 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %157, i32 0, i32 0
  %159 = load i16, ptr %158, align 2
  switch i16 %159, label %56 [
    i16 0, label %50
    i16 1, label %53
  ]

160:                                              ; preds = %50
  %161 = phi i16 [ %148, %50 ]
  %162 = phi i16 [ %149, %50 ]
  %163 = phi i16 [ %150, %50 ]
  %164 = phi i32 [ %151, %50 ]
  %165 = phi i32 [ %152, %50 ]
  %166 = phi i32 [ %153, %50 ]
  %167 = phi i64 [ %155, %50 ]
  %168 = phi <{ i64 }> [ %52, %50 ]
  %169 = call i64 @"struct_deconstruct<Tuple<u64>>"(<{ i64 }> %168)
  %170 = call <{ i16, [16 x i8] }> @"core::integer::U64Add::add"(i64 50, i64 2)
  %171 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %170, ptr %171, align 1
  %172 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %171, i32 0, i32 0
  %173 = load i16, ptr %172, align 2
  switch i16 %173, label %63 [
    i16 0, label %57
    i16 1, label %60
  ]

174:                                              ; preds = %57
  %175 = phi i16 [ %161, %57 ]
  %176 = phi i16 [ %162, %57 ]
  %177 = phi i16 [ %163, %57 ]
  %178 = phi i32 [ %164, %57 ]
  %179 = phi i32 [ %165, %57 ]
  %180 = phi i32 [ %166, %57 ]
  %181 = phi i64 [ %167, %57 ]
  %182 = phi i64 [ %169, %57 ]
  %183 = phi <{ i64 }> [ %59, %57 ]
  %184 = call i64 @"struct_deconstruct<Tuple<u64>>"(<{ i64 }> %183)
  %185 = call <{ i16, [16 x i8] }> @"core::integer::U128Add::add"(i128 4, i128 6)
  %186 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %185, ptr %186, align 1
  %187 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %186, i32 0, i32 0
  %188 = load i16, ptr %187, align 2
  switch i16 %188, label %70 [
    i16 0, label %64
    i16 1, label %67
  ]

189:                                              ; preds = %64
  %190 = phi i16 [ %175, %64 ]
  %191 = phi i16 [ %176, %64 ]
  %192 = phi i16 [ %177, %64 ]
  %193 = phi i32 [ %178, %64 ]
  %194 = phi i32 [ %179, %64 ]
  %195 = phi i32 [ %180, %64 ]
  %196 = phi i64 [ %181, %64 ]
  %197 = phi i64 [ %182, %64 ]
  %198 = phi i64 [ %184, %64 ]
  %199 = phi <{ i128 }> [ %66, %64 ]
  %200 = call i128 @"struct_deconstruct<Tuple<u128>>"(<{ i128 }> %199)
  %201 = call <{ i16, [16 x i8] }> @"core::integer::U128Add::add"(i128 2, i128 10)
  %202 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %201, ptr %202, align 1
  %203 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %202, i32 0, i32 0
  %204 = load i16, ptr %203, align 2
  switch i16 %204, label %77 [
    i16 0, label %71
    i16 1, label %74
  ]

205:                                              ; preds = %71
  %206 = phi i16 [ %190, %71 ]
  %207 = phi i16 [ %191, %71 ]
  %208 = phi i16 [ %192, %71 ]
  %209 = phi i32 [ %193, %71 ]
  %210 = phi i32 [ %194, %71 ]
  %211 = phi i32 [ %195, %71 ]
  %212 = phi i64 [ %196, %71 ]
  %213 = phi i64 [ %197, %71 ]
  %214 = phi i64 [ %198, %71 ]
  %215 = phi i128 [ %200, %71 ]
  %216 = phi <{ i128 }> [ %73, %71 ]
  %217 = call i128 @"struct_deconstruct<Tuple<u128>>"(<{ i128 }> %216)
  %218 = call <{ i16, [16 x i8] }> @"core::integer::U128Add::add"(i128 50, i128 2)
  %219 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %218, ptr %219, align 1
  %220 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %219, i32 0, i32 0
  %221 = load i16, ptr %220, align 2
  switch i16 %221, label %84 [
    i16 0, label %78
    i16 1, label %81
  ]

222:                                              ; preds = %78
  %223 = phi i16 [ %206, %78 ]
  %224 = phi i16 [ %207, %78 ]
  %225 = phi i16 [ %208, %78 ]
  %226 = phi i32 [ %209, %78 ]
  %227 = phi i32 [ %210, %78 ]
  %228 = phi i32 [ %211, %78 ]
  %229 = phi i64 [ %212, %78 ]
  %230 = phi i64 [ %213, %78 ]
  %231 = phi i64 [ %214, %78 ]
  %232 = phi i128 [ %215, %78 ]
  %233 = phi i128 [ %217, %78 ]
  %234 = phi <{ i128 }> [ %80, %78 ]
  %235 = call i128 @"struct_deconstruct<Tuple<u128>>"(<{ i128 }> %234)
  %236 = call <{ i16, i16, i16 }> @"struct_construct<Tuple<u16, u16, u16>>"(i16 %223, i16 %224, i16 %225)
  %237 = call <{ i32, i32, i32 }> @"struct_construct<Tuple<u32, u32, u32>>"(i32 %226, i32 %227, i32 %228)
  %238 = call <{ i64, i64, i64 }> @"struct_construct<Tuple<u64, u64, u64>>"(i64 %229, i64 %230, i64 %231)
  %239 = call <{ i128, i128, i128 }> @"struct_construct<Tuple<u128, u128, u128>>"(i128 %232, i128 %233, i128 %235)
  %240 = call <{ <{ i16, i16, i16 }>, <{ i32, i32, i32 }>, <{ i64, i64, i64 }>, <{ i128, i128, i128 }> }> @"struct_construct<Tuple<Tuple<u16, u16, u16>, Tuple<u32, u32, u32>, Tuple<u64, u64, u64>, Tuple<u128, u128, u128>>>"(<{ i16, i16, i16 }> %236, <{ i32, i32, i32 }> %237, <{ i64, i64, i64 }> %238, <{ i128, i128, i128 }> %239)
  %241 = call <{ <{ <{ i16, i16, i16 }>, <{ i32, i32, i32 }>, <{ i64, i64, i64 }>, <{ i128, i128, i128 }> }> }> @"struct_construct<Tuple<Tuple<Tuple<u16, u16, u16>, Tuple<u32, u32, u32>, Tuple<u64, u64, u64>, Tuple<u128, u128, u128>>>>"(<{ <{ i16, i16, i16 }>, <{ i32, i32, i32 }>, <{ i64, i64, i64 }>, <{ i128, i128, i128 }> }> %240)
  %242 = call <{ i16, [90 x i8] }> @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 0>"(<{ <{ <{ i16, i16, i16 }>, <{ i32, i32, i32 }>, <{ i64, i64, i64 }>, <{ i128, i128, i128 }> }> }> %241)
  ret <{ i16, [90 x i8] }> %242

243:                                              ; preds = %81
  %244 = phi <{ i32, i32, ptr }> [ %83, %81 ]
  %245 = call <{ i16, [90 x i8] }> @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(<{ i32, i32, ptr }> %244)
  ret <{ i16, [90 x i8] }> %245

246:                                              ; preds = %74
  %247 = phi <{ i32, i32, ptr }> [ %76, %74 ]
  %248 = call <{ i16, [90 x i8] }> @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(<{ i32, i32, ptr }> %247)
  ret <{ i16, [90 x i8] }> %248

249:                                              ; preds = %67
  %250 = phi <{ i32, i32, ptr }> [ %69, %67 ]
  %251 = call <{ i16, [90 x i8] }> @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(<{ i32, i32, ptr }> %250)
  ret <{ i16, [90 x i8] }> %251

252:                                              ; preds = %60
  %253 = phi <{ i32, i32, ptr }> [ %62, %60 ]
  %254 = call <{ i16, [90 x i8] }> @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(<{ i32, i32, ptr }> %253)
  ret <{ i16, [90 x i8] }> %254

255:                                              ; preds = %53
  %256 = phi <{ i32, i32, ptr }> [ %55, %53 ]
  %257 = call <{ i16, [90 x i8] }> @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(<{ i32, i32, ptr }> %256)
  ret <{ i16, [90 x i8] }> %257

258:                                              ; preds = %46
  %259 = phi <{ i32, i32, ptr }> [ %48, %46 ]
  %260 = call <{ i16, [90 x i8] }> @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(<{ i32, i32, ptr }> %259)
  ret <{ i16, [90 x i8] }> %260

261:                                              ; preds = %39
  %262 = phi <{ i32, i32, ptr }> [ %41, %39 ]
  %263 = call <{ i16, [90 x i8] }> @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(<{ i32, i32, ptr }> %262)
  ret <{ i16, [90 x i8] }> %263

264:                                              ; preds = %32
  %265 = phi <{ i32, i32, ptr }> [ %34, %32 ]
  %266 = call <{ i16, [90 x i8] }> @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(<{ i32, i32, ptr }> %265)
  ret <{ i16, [90 x i8] }> %266

267:                                              ; preds = %25
  %268 = phi <{ i32, i32, ptr }> [ %27, %25 ]
  %269 = call <{ i16, [90 x i8] }> @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(<{ i32, i32, ptr }> %268)
  ret <{ i16, [90 x i8] }> %269

270:                                              ; preds = %18
  %271 = phi <{ i32, i32, ptr }> [ %20, %18 ]
  %272 = call <{ i16, [90 x i8] }> @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(<{ i32, i32, ptr }> %271)
  ret <{ i16, [90 x i8] }> %272

273:                                              ; preds = %11
  %274 = phi <{ i32, i32, ptr }> [ %13, %11 ]
  %275 = call <{ i16, [90 x i8] }> @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(<{ i32, i32, ptr }> %274)
  ret <{ i16, [90 x i8] }> %275

276:                                              ; preds = %4
  %277 = phi <{ i32, i32, ptr }> [ %6, %4 ]
  %278 = call <{ i16, [90 x i8] }> @"enum_init<core::PanicResult::<(((core::integer::u16, core::integer::u16, core::integer::u16), (core::integer::u32, core::integer::u32, core::integer::u32), (core::integer::u64, core::integer::u64, core::integer::u64), (core::integer::u128, core::integer::u128, core::integer::u128)))>, 1>"(<{ i32, i32, ptr }> %277)
  ret <{ i16, [90 x i8] }> %278
}

define void @"_mlir_ciface_uint_addition::uint_addition::main"(ptr %0) {
  %2 = call <{ i16, [90 x i8] }> @"uint_addition::uint_addition::main"()
  store <{ i16, [90 x i8] }> %2, ptr %0, align 1
  ret void
}

define <{ i16, [16 x i8] }> @"core::integer::U16Add::add"(i16 %0, i16 %1) {
  br label %10

3:                                                ; preds = %22
  %4 = getelementptr inbounds <{ i16, <{ i16 }> }>, ptr %25, i32 0, i32 1
  %5 = load <{ i16 }>, ptr %4, align 1
  br label %28

6:                                                ; preds = %22
  %7 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %25, i32 0, i32 1
  %8 = load <{ i32, i32, ptr }>, ptr %7, align 1
  br label %33

9:                                                ; preds = %22
  unreachable

10:                                               ; preds = %2
  %11 = phi i16 [ %0, %2 ]
  %12 = phi i16 [ %1, %2 ]
  %13 = call { i16, i1 } @llvm.uadd.with.overflow.i16(i16 %11, i16 %12)
  %14 = extractvalue { i16, i1 } %13, 0
  %15 = extractvalue { i16, i1 } %13, 1
  br i1 %15, label %19, label %16

16:                                               ; preds = %10
  %17 = phi i16 [ %14, %10 ]
  %18 = call <{ i16, [2 x i8] }> @"enum_init<core::result::Result::<core::integer::u16, core::integer::u16>, 0>"(i16 %17)
  br label %22

19:                                               ; preds = %10
  %20 = phi i16 [ %14, %10 ]
  %21 = call <{ i16, [2 x i8] }> @"enum_init<core::result::Result::<core::integer::u16, core::integer::u16>, 1>"(i16 %20)
  br label %22

22:                                               ; preds = %19, %16
  %23 = phi <{ i16, [2 x i8] }> [ %21, %19 ], [ %18, %16 ]
  %24 = call <{ i16, [16 x i8] }> @"core::result::ResultTraitImpl::<core::integer::u16, core::integer::u16>::expect::<core::integer::u16Drop>"(<{ i16, [2 x i8] }> %23, i256 155775200859838811096160292336445452151)
  %25 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %24, ptr %25, align 1
  %26 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %25, i32 0, i32 0
  %27 = load i16, ptr %26, align 2
  switch i16 %27, label %9 [
    i16 0, label %3
    i16 1, label %6
  ]

28:                                               ; preds = %3
  %29 = phi <{ i16 }> [ %5, %3 ]
  %30 = call i16 @"struct_deconstruct<Tuple<u16>>"(<{ i16 }> %29)
  %31 = call <{ i16 }> @"struct_construct<Tuple<u16>>"(i16 %30)
  %32 = call <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u16)>, 0>"(<{ i16 }> %31)
  ret <{ i16, [16 x i8] }> %32

33:                                               ; preds = %6
  %34 = phi <{ i32, i32, ptr }> [ %8, %6 ]
  %35 = call <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u16)>, 1>"(<{ i32, i32, ptr }> %34)
  ret <{ i16, [16 x i8] }> %35
}

define void @"_mlir_ciface_core::integer::U16Add::add"(ptr %0, i16 %1, i16 %2) {
  %4 = call <{ i16, [16 x i8] }> @"core::integer::U16Add::add"(i16 %1, i16 %2)
  store <{ i16, [16 x i8] }> %4, ptr %0, align 1
  ret void
}

define <{ i16, [16 x i8] }> @"core::integer::U32Add::add"(i32 %0, i32 %1) {
  br label %10

3:                                                ; preds = %22
  %4 = getelementptr inbounds <{ i16, <{ i32 }> }>, ptr %25, i32 0, i32 1
  %5 = load <{ i32 }>, ptr %4, align 1
  br label %28

6:                                                ; preds = %22
  %7 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %25, i32 0, i32 1
  %8 = load <{ i32, i32, ptr }>, ptr %7, align 1
  br label %33

9:                                                ; preds = %22
  unreachable

10:                                               ; preds = %2
  %11 = phi i32 [ %0, %2 ]
  %12 = phi i32 [ %1, %2 ]
  %13 = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %11, i32 %12)
  %14 = extractvalue { i32, i1 } %13, 0
  %15 = extractvalue { i32, i1 } %13, 1
  br i1 %15, label %19, label %16

16:                                               ; preds = %10
  %17 = phi i32 [ %14, %10 ]
  %18 = call <{ i16, [4 x i8] }> @"enum_init<core::result::Result::<core::integer::u32, core::integer::u32>, 0>"(i32 %17)
  br label %22

19:                                               ; preds = %10
  %20 = phi i32 [ %14, %10 ]
  %21 = call <{ i16, [4 x i8] }> @"enum_init<core::result::Result::<core::integer::u32, core::integer::u32>, 1>"(i32 %20)
  br label %22

22:                                               ; preds = %19, %16
  %23 = phi <{ i16, [4 x i8] }> [ %21, %19 ], [ %18, %16 ]
  %24 = call <{ i16, [16 x i8] }> @"core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>"(<{ i16, [4 x i8] }> %23, i256 155785504323917466144735657540098748279)
  %25 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %24, ptr %25, align 1
  %26 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %25, i32 0, i32 0
  %27 = load i16, ptr %26, align 2
  switch i16 %27, label %9 [
    i16 0, label %3
    i16 1, label %6
  ]

28:                                               ; preds = %3
  %29 = phi <{ i32 }> [ %5, %3 ]
  %30 = call i32 @"struct_deconstruct<Tuple<u32>>"(<{ i32 }> %29)
  %31 = call <{ i32 }> @"struct_construct<Tuple<u32>>"(i32 %30)
  %32 = call <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u32)>, 0>"(<{ i32 }> %31)
  ret <{ i16, [16 x i8] }> %32

33:                                               ; preds = %6
  %34 = phi <{ i32, i32, ptr }> [ %8, %6 ]
  %35 = call <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u32)>, 1>"(<{ i32, i32, ptr }> %34)
  ret <{ i16, [16 x i8] }> %35
}

define void @"_mlir_ciface_core::integer::U32Add::add"(ptr %0, i32 %1, i32 %2) {
  %4 = call <{ i16, [16 x i8] }> @"core::integer::U32Add::add"(i32 %1, i32 %2)
  store <{ i16, [16 x i8] }> %4, ptr %0, align 1
  ret void
}

define <{ i16, [16 x i8] }> @"core::integer::U64Add::add"(i64 %0, i64 %1) {
  br label %10

3:                                                ; preds = %22
  %4 = getelementptr inbounds <{ i16, <{ i64 }> }>, ptr %25, i32 0, i32 1
  %5 = load <{ i64 }>, ptr %4, align 1
  br label %28

6:                                                ; preds = %22
  %7 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %25, i32 0, i32 1
  %8 = load <{ i32, i32, ptr }>, ptr %7, align 1
  br label %33

9:                                                ; preds = %22
  unreachable

10:                                               ; preds = %2
  %11 = phi i64 [ %0, %2 ]
  %12 = phi i64 [ %1, %2 ]
  %13 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %11, i64 %12)
  %14 = extractvalue { i64, i1 } %13, 0
  %15 = extractvalue { i64, i1 } %13, 1
  br i1 %15, label %19, label %16

16:                                               ; preds = %10
  %17 = phi i64 [ %14, %10 ]
  %18 = call <{ i16, [8 x i8] }> @"enum_init<core::result::Result::<core::integer::u64, core::integer::u64>, 0>"(i64 %17)
  br label %22

19:                                               ; preds = %10
  %20 = phi i64 [ %14, %10 ]
  %21 = call <{ i16, [8 x i8] }> @"enum_init<core::result::Result::<core::integer::u64, core::integer::u64>, 1>"(i64 %20)
  br label %22

22:                                               ; preds = %19, %16
  %23 = phi <{ i16, [8 x i8] }> [ %21, %19 ], [ %18, %16 ]
  %24 = call <{ i16, [16 x i8] }> @"core::result::ResultTraitImpl::<core::integer::u64, core::integer::u64>::expect::<core::integer::u64Drop>"(<{ i16, [8 x i8] }> %23, i256 155801121779312277930962096923588980599)
  %25 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %24, ptr %25, align 1
  %26 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %25, i32 0, i32 0
  %27 = load i16, ptr %26, align 2
  switch i16 %27, label %9 [
    i16 0, label %3
    i16 1, label %6
  ]

28:                                               ; preds = %3
  %29 = phi <{ i64 }> [ %5, %3 ]
  %30 = call i64 @"struct_deconstruct<Tuple<u64>>"(<{ i64 }> %29)
  %31 = call <{ i64 }> @"struct_construct<Tuple<u64>>"(i64 %30)
  %32 = call <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u64)>, 0>"(<{ i64 }> %31)
  ret <{ i16, [16 x i8] }> %32

33:                                               ; preds = %6
  %34 = phi <{ i32, i32, ptr }> [ %8, %6 ]
  %35 = call <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u64)>, 1>"(<{ i32, i32, ptr }> %34)
  ret <{ i16, [16 x i8] }> %35
}

define void @"_mlir_ciface_core::integer::U64Add::add"(ptr %0, i64 %1, i64 %2) {
  %4 = call <{ i16, [16 x i8] }> @"core::integer::U64Add::add"(i64 %1, i64 %2)
  store <{ i16, [16 x i8] }> %4, ptr %0, align 1
  ret void
}

define <{ i16, [16 x i8] }> @"core::integer::U128Add::add"(i128 %0, i128 %1) {
  br label %10

3:                                                ; preds = %22
  %4 = getelementptr inbounds <{ i16, <{ i128 }> }>, ptr %25, i32 0, i32 1
  %5 = load <{ i128 }>, ptr %4, align 1
  br label %28

6:                                                ; preds = %22
  %7 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %25, i32 0, i32 1
  %8 = load <{ i32, i32, ptr }>, ptr %7, align 1
  br label %33

9:                                                ; preds = %22
  unreachable

10:                                               ; preds = %2
  %11 = phi i128 [ %0, %2 ]
  %12 = phi i128 [ %1, %2 ]
  %13 = call { i128, i1 } @llvm.uadd.with.overflow.i128(i128 %11, i128 %12)
  %14 = extractvalue { i128, i1 } %13, 0
  %15 = extractvalue { i128, i1 } %13, 1
  br i1 %15, label %19, label %16

16:                                               ; preds = %10
  %17 = phi i128 [ %14, %10 ]
  %18 = call <{ i16, [16 x i8] }> @"enum_init<core::result::Result::<core::integer::u128, core::integer::u128>, 0>"(i128 %17)
  br label %22

19:                                               ; preds = %10
  %20 = phi i128 [ %14, %10 ]
  %21 = call <{ i16, [16 x i8] }> @"enum_init<core::result::Result::<core::integer::u128, core::integer::u128>, 1>"(i128 %20)
  br label %22

22:                                               ; preds = %19, %16
  %23 = phi <{ i16, [16 x i8] }> [ %21, %19 ], [ %18, %16 ]
  %24 = call <{ i16, [16 x i8] }> @"core::result::ResultTraitImpl::<core::integer::u128, core::integer::u128>::expect::<core::integer::u128Drop>"(<{ i16, [16 x i8] }> %23, i256 39878429859757942499084499860145094553463)
  %25 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %24, ptr %25, align 1
  %26 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %25, i32 0, i32 0
  %27 = load i16, ptr %26, align 2
  switch i16 %27, label %9 [
    i16 0, label %3
    i16 1, label %6
  ]

28:                                               ; preds = %3
  %29 = phi <{ i128 }> [ %5, %3 ]
  %30 = call i128 @"struct_deconstruct<Tuple<u128>>"(<{ i128 }> %29)
  %31 = call <{ i128 }> @"struct_construct<Tuple<u128>>"(i128 %30)
  %32 = call <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u128)>, 0>"(<{ i128 }> %31)
  ret <{ i16, [16 x i8] }> %32

33:                                               ; preds = %6
  %34 = phi <{ i32, i32, ptr }> [ %8, %6 ]
  %35 = call <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u128)>, 1>"(<{ i32, i32, ptr }> %34)
  ret <{ i16, [16 x i8] }> %35
}

define void @"_mlir_ciface_core::integer::U128Add::add"(ptr %0, i128 %1, i128 %2) {
  %4 = call <{ i16, [16 x i8] }> @"core::integer::U128Add::add"(i128 %1, i128 %2)
  store <{ i16, [16 x i8] }> %4, ptr %0, align 1
  ret void
}

define <{ i16, [16 x i8] }> @"core::result::ResultTraitImpl::<core::integer::u16, core::integer::u16>::expect::<core::integer::u16Drop>"(<{ i16, [2 x i8] }> %0, i256 %1) {
  br label %8

3:                                                ; preds = %8
  %4 = getelementptr inbounds <{ i16, i16 }>, ptr %11, i32 0, i32 1
  %5 = load i16, ptr %4, align 2
  br label %14

6:                                                ; preds = %8
  br label %18

7:                                                ; preds = %8
  unreachable

8:                                                ; preds = %2
  %9 = phi <{ i16, [2 x i8] }> [ %0, %2 ]
  %10 = phi i256 [ %1, %2 ]
  %11 = alloca <{ i16, [2 x i8] }>, i64 1, align 8
  store <{ i16, [2 x i8] }> %9, ptr %11, align 1
  %12 = getelementptr inbounds <{ i16, [2 x i8] }>, ptr %11, i32 0, i32 0
  %13 = load i16, ptr %12, align 2
  switch i16 %13, label %7 [
    i16 0, label %3
    i16 1, label %6
  ]

14:                                               ; preds = %3
  %15 = phi i16 [ %5, %3 ]
  %16 = call <{ i16 }> @"struct_construct<Tuple<u16>>"(i16 %15)
  %17 = call <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u16)>, 0>"(<{ i16 }> %16)
  ret <{ i16, [16 x i8] }> %17

18:                                               ; preds = %6
  %19 = phi i256 [ %10, %6 ]
  %20 = call <{ i32, i32, ptr }> @"array_new<felt252>"()
  %21 = call <{ i32, i32, ptr }> @"array_append<felt252>"(<{ i32, i32, ptr }> %20, i256 %19)
  %22 = call <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u16)>, 1>"(<{ i32, i32, ptr }> %21)
  ret <{ i16, [16 x i8] }> %22
}

define void @"_mlir_ciface_core::result::ResultTraitImpl::<core::integer::u16, core::integer::u16>::expect::<core::integer::u16Drop>"(ptr %0, <{ i16, [2 x i8] }> %1, i256 %2) {
  %4 = call <{ i16, [16 x i8] }> @"core::result::ResultTraitImpl::<core::integer::u16, core::integer::u16>::expect::<core::integer::u16Drop>"(<{ i16, [2 x i8] }> %1, i256 %2)
  store <{ i16, [16 x i8] }> %4, ptr %0, align 1
  ret void
}

define <{ i16, [16 x i8] }> @"core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>"(<{ i16, [4 x i8] }> %0, i256 %1) {
  br label %8

3:                                                ; preds = %8
  %4 = getelementptr inbounds <{ i16, i32 }>, ptr %11, i32 0, i32 1
  %5 = load i32, ptr %4, align 4
  br label %14

6:                                                ; preds = %8
  br label %18

7:                                                ; preds = %8
  unreachable

8:                                                ; preds = %2
  %9 = phi <{ i16, [4 x i8] }> [ %0, %2 ]
  %10 = phi i256 [ %1, %2 ]
  %11 = alloca <{ i16, [4 x i8] }>, i64 1, align 8
  store <{ i16, [4 x i8] }> %9, ptr %11, align 1
  %12 = getelementptr inbounds <{ i16, [4 x i8] }>, ptr %11, i32 0, i32 0
  %13 = load i16, ptr %12, align 2
  switch i16 %13, label %7 [
    i16 0, label %3
    i16 1, label %6
  ]

14:                                               ; preds = %3
  %15 = phi i32 [ %5, %3 ]
  %16 = call <{ i32 }> @"struct_construct<Tuple<u32>>"(i32 %15)
  %17 = call <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u32)>, 0>"(<{ i32 }> %16)
  ret <{ i16, [16 x i8] }> %17

18:                                               ; preds = %6
  %19 = phi i256 [ %10, %6 ]
  %20 = call <{ i32, i32, ptr }> @"array_new<felt252>"()
  %21 = call <{ i32, i32, ptr }> @"array_append<felt252>"(<{ i32, i32, ptr }> %20, i256 %19)
  %22 = call <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u32)>, 1>"(<{ i32, i32, ptr }> %21)
  ret <{ i16, [16 x i8] }> %22
}

define void @"_mlir_ciface_core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>"(ptr %0, <{ i16, [4 x i8] }> %1, i256 %2) {
  %4 = call <{ i16, [16 x i8] }> @"core::result::ResultTraitImpl::<core::integer::u32, core::integer::u32>::expect::<core::integer::u32Drop>"(<{ i16, [4 x i8] }> %1, i256 %2)
  store <{ i16, [16 x i8] }> %4, ptr %0, align 1
  ret void
}

define <{ i16, [16 x i8] }> @"core::result::ResultTraitImpl::<core::integer::u64, core::integer::u64>::expect::<core::integer::u64Drop>"(<{ i16, [8 x i8] }> %0, i256 %1) {
  br label %8

3:                                                ; preds = %8
  %4 = getelementptr inbounds <{ i16, i64 }>, ptr %11, i32 0, i32 1
  %5 = load i64, ptr %4, align 4
  br label %14

6:                                                ; preds = %8
  br label %18

7:                                                ; preds = %8
  unreachable

8:                                                ; preds = %2
  %9 = phi <{ i16, [8 x i8] }> [ %0, %2 ]
  %10 = phi i256 [ %1, %2 ]
  %11 = alloca <{ i16, [8 x i8] }>, i64 1, align 8
  store <{ i16, [8 x i8] }> %9, ptr %11, align 1
  %12 = getelementptr inbounds <{ i16, [8 x i8] }>, ptr %11, i32 0, i32 0
  %13 = load i16, ptr %12, align 2
  switch i16 %13, label %7 [
    i16 0, label %3
    i16 1, label %6
  ]

14:                                               ; preds = %3
  %15 = phi i64 [ %5, %3 ]
  %16 = call <{ i64 }> @"struct_construct<Tuple<u64>>"(i64 %15)
  %17 = call <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u64)>, 0>"(<{ i64 }> %16)
  ret <{ i16, [16 x i8] }> %17

18:                                               ; preds = %6
  %19 = phi i256 [ %10, %6 ]
  %20 = call <{ i32, i32, ptr }> @"array_new<felt252>"()
  %21 = call <{ i32, i32, ptr }> @"array_append<felt252>"(<{ i32, i32, ptr }> %20, i256 %19)
  %22 = call <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u64)>, 1>"(<{ i32, i32, ptr }> %21)
  ret <{ i16, [16 x i8] }> %22
}

define void @"_mlir_ciface_core::result::ResultTraitImpl::<core::integer::u64, core::integer::u64>::expect::<core::integer::u64Drop>"(ptr %0, <{ i16, [8 x i8] }> %1, i256 %2) {
  %4 = call <{ i16, [16 x i8] }> @"core::result::ResultTraitImpl::<core::integer::u64, core::integer::u64>::expect::<core::integer::u64Drop>"(<{ i16, [8 x i8] }> %1, i256 %2)
  store <{ i16, [16 x i8] }> %4, ptr %0, align 1
  ret void
}

define <{ i16, [16 x i8] }> @"core::result::ResultTraitImpl::<core::integer::u128, core::integer::u128>::expect::<core::integer::u128Drop>"(<{ i16, [16 x i8] }> %0, i256 %1) {
  br label %8

3:                                                ; preds = %8
  %4 = getelementptr inbounds <{ i16, i128 }>, ptr %11, i32 0, i32 1
  %5 = load i128, ptr %4, align 4
  br label %14

6:                                                ; preds = %8
  br label %18

7:                                                ; preds = %8
  unreachable

8:                                                ; preds = %2
  %9 = phi <{ i16, [16 x i8] }> [ %0, %2 ]
  %10 = phi i256 [ %1, %2 ]
  %11 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %9, ptr %11, align 1
  %12 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %11, i32 0, i32 0
  %13 = load i16, ptr %12, align 2
  switch i16 %13, label %7 [
    i16 0, label %3
    i16 1, label %6
  ]

14:                                               ; preds = %3
  %15 = phi i128 [ %5, %3 ]
  %16 = call <{ i128 }> @"struct_construct<Tuple<u128>>"(i128 %15)
  %17 = call <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u128)>, 0>"(<{ i128 }> %16)
  ret <{ i16, [16 x i8] }> %17

18:                                               ; preds = %6
  %19 = phi i256 [ %10, %6 ]
  %20 = call <{ i32, i32, ptr }> @"array_new<felt252>"()
  %21 = call <{ i32, i32, ptr }> @"array_append<felt252>"(<{ i32, i32, ptr }> %20, i256 %19)
  %22 = call <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::integer::u128)>, 1>"(<{ i32, i32, ptr }> %21)
  ret <{ i16, [16 x i8] }> %22
}

define void @"_mlir_ciface_core::result::ResultTraitImpl::<core::integer::u128, core::integer::u128>::expect::<core::integer::u128Drop>"(ptr %0, <{ i16, [16 x i8] }> %1, i256 %2) {
  %4 = call <{ i16, [16 x i8] }> @"core::result::ResultTraitImpl::<core::integer::u128, core::integer::u128>::expect::<core::integer::u128Drop>"(<{ i16, [16 x i8] }> %1, i256 %2)
  store <{ i16, [16 x i8] }> %4, ptr %0, align 1
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
