; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

define internal { i16, [2 x i8] } @"enum_init<simple_enum_simple_enum_MyEnum, 0>"(i8 %0) {
  %2 = alloca { i16, [2 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds ptr, ptr %2, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds ptr, ptr %2, i32 1
  store i8 %0, ptr %4, align 1
  %5 = load { i16, [2 x i8] }, ptr %2, align 2
  ret { i16, [2 x i8] } %5
}

define internal { i16, [2 x i8] } @"store_temp<simple_enum_simple_enum_MyEnum>"({ i16, [2 x i8] } %0) {
  ret { i16, [2 x i8] } %0
}

define internal { i16, [2 x i8] } @"enum_init<simple_enum_simple_enum_MyEnum, 1>"(i16 %0) {
  %2 = alloca { i16, [2 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds ptr, ptr %2, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds ptr, ptr %2, i32 1
  store i16 %0, ptr %4, align 2
  %5 = load { i16, [2 x i8] }, ptr %2, align 2
  ret { i16, [2 x i8] } %5
}

define internal { i16, [2 x i8] } @simple_enum_simple_enum_my_enum() {
  br label %1

1:                                                ; preds = %0
  %2 = call { i16, [2 x i8] } @"enum_init<simple_enum_simple_enum_MyEnum, 0>"(i8 4)
  %3 = call { i16, [2 x i8] } @"store_temp<simple_enum_simple_enum_MyEnum>"({ i16, [2 x i8] } %2)
  ret { i16, [2 x i8] } %3
}

define internal { i16, [2 x i8] } @simple_enum_simple_enum_my_enum2() {
  br label %1

1:                                                ; preds = %0
  %2 = call { i16, [2 x i8] } @"enum_init<simple_enum_simple_enum_MyEnum, 1>"(i16 8)
  %3 = call { i16, [2 x i8] } @"store_temp<simple_enum_simple_enum_MyEnum>"({ i16, [2 x i8] } %2)
  ret { i16, [2 x i8] } %3
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
