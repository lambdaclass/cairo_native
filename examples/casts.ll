; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare ptr @realloc(ptr, i64)

declare ptr @memmove(ptr, ptr, i64)

declare i32 @dprintf(i32, ptr, ...)

; Function Attrs: alwaysinline norecurse nounwind
define internal i16 @"upcast<u8, u16>"(i8 %0) #0 {
  %2 = zext i8 %0 to i16
  ret i16 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i32 @"upcast<u8, u32>"(i8 %0) #0 {
  %2 = zext i8 %0 to i32
  ret i32 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i64 @"upcast<u8, u64>"(i8 %0) #0 {
  %2 = zext i8 %0 to i64
  ret i64 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i128 @"upcast<u8, u128>"(i8 %0) #0 {
  %2 = zext i8 %0 to i128
  ret i128 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i32 @"upcast<u16, u32>"(i16 %0) #0 {
  %2 = zext i16 %0 to i32
  ret i32 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i64 @"upcast<u16, u64>"(i16 %0) #0 {
  %2 = zext i16 %0 to i64
  ret i64 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i128 @"upcast<u16, u128>"(i16 %0) #0 {
  %2 = zext i16 %0 to i128
  ret i128 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i64 @"upcast<u32, u64>"(i32 %0) #0 {
  %2 = zext i32 %0 to i64
  ret i64 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i128 @"upcast<u32, u128>"(i32 %0) #0 {
  %2 = zext i32 %0 to i128
  ret i128 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i128 @"upcast<u64, u128>"(i64 %0) #0 {
  %2 = zext i64 %0 to i128
  ret i128 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [1 x i8] }> @"enum_init<core::option::Option::<core::integer::u8>, 0>"(i8 %0) #0 {
  %2 = alloca <{ i16, [1 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [1 x i8] }>, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, i8 }>, ptr %2, i32 0, i32 1
  store i8 %0, ptr %4, align 1
  %5 = load <{ i16, [1 x i8] }>, ptr %2, align 1
  ret <{ i16, [1 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{}> @"struct_construct<Unit>"() #0 {
  ret <{}> undef
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [1 x i8] }> @"enum_init<core::option::Option::<core::integer::u8>, 1>"(<{}> %0) #0 {
  %2 = alloca <{ i16, [1 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [1 x i8] }>, ptr %2, i32 0, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, <{}> }>, ptr %2, i32 0, i32 1
  store <{}> %0, ptr %4, align 1
  %5 = load <{ i16, [1 x i8] }>, ptr %2, align 1
  ret <{ i16, [1 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [2 x i8] }> @"enum_init<core::option::Option::<core::integer::u16>, 0>"(i16 %0) #0 {
  %2 = alloca <{ i16, [2 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [2 x i8] }>, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, i16 }>, ptr %2, i32 0, i32 1
  store i16 %0, ptr %4, align 2
  %5 = load <{ i16, [2 x i8] }>, ptr %2, align 1
  ret <{ i16, [2 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [2 x i8] }> @"enum_init<core::option::Option::<core::integer::u16>, 1>"(<{}> %0) #0 {
  %2 = alloca <{ i16, [2 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [2 x i8] }>, ptr %2, i32 0, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, <{}> }>, ptr %2, i32 0, i32 1
  store <{}> %0, ptr %4, align 1
  %5 = load <{ i16, [2 x i8] }>, ptr %2, align 1
  ret <{ i16, [2 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [4 x i8] }> @"enum_init<core::option::Option::<core::integer::u32>, 0>"(i32 %0) #0 {
  %2 = alloca <{ i16, [4 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [4 x i8] }>, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, i32 }>, ptr %2, i32 0, i32 1
  store i32 %0, ptr %4, align 4
  %5 = load <{ i16, [4 x i8] }>, ptr %2, align 1
  ret <{ i16, [4 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [4 x i8] }> @"enum_init<core::option::Option::<core::integer::u32>, 1>"(<{}> %0) #0 {
  %2 = alloca <{ i16, [4 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [4 x i8] }>, ptr %2, i32 0, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, <{}> }>, ptr %2, i32 0, i32 1
  store <{}> %0, ptr %4, align 1
  %5 = load <{ i16, [4 x i8] }>, ptr %2, align 1
  ret <{ i16, [4 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [8 x i8] }> @"enum_init<core::option::Option::<core::integer::u64>, 0>"(i64 %0) #0 {
  %2 = alloca <{ i16, [8 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [8 x i8] }>, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, i64 }>, ptr %2, i32 0, i32 1
  store i64 %0, ptr %4, align 4
  %5 = load <{ i16, [8 x i8] }>, ptr %2, align 1
  ret <{ i16, [8 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [8 x i8] }> @"enum_init<core::option::Option::<core::integer::u64>, 1>"(<{}> %0) #0 {
  %2 = alloca <{ i16, [8 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [8 x i8] }>, ptr %2, i32 0, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, <{}> }>, ptr %2, i32 0, i32 1
  store <{}> %0, ptr %4, align 1
  %5 = load <{ i16, [8 x i8] }>, ptr %2, align 1
  ret <{ i16, [8 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [16 x i8] }> @"enum_init<core::option::Option::<core::integer::u128>, 0>"(i128 %0) #0 {
  %2 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, i128 }>, ptr %2, i32 0, i32 1
  store i128 %0, ptr %4, align 4
  %5 = load <{ i16, [16 x i8] }>, ptr %2, align 1
  ret <{ i16, [16 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [16 x i8] }> @"enum_init<core::option::Option::<core::integer::u128>, 1>"(<{}> %0) #0 {
  %2 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %2, i32 0, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, <{}> }>, ptr %2, i32 0, i32 1
  store <{}> %0, ptr %4, align 1
  %5 = load <{ i16, [16 x i8] }>, ptr %2, align 1
  ret <{ i16, [16 x i8] }> %5
}

; Function Attrs: norecurse nounwind
define internal void @print_Unit(<{}> %0) #1 {
  ret void
}

define void @main() {
  %1 = call <{}> @"casts::casts::main"()
  call void @print_Unit(<{}> %1)
  ret void
}

define void @_mlir_ciface_main() {
  call void @main()
  ret void
}

define <{}> @"casts::casts::main"() {
  br label %26

1:                                                ; preds = %26
  br label %37

2:                                                ; preds = %51
  %3 = trunc i16 %52 to i8
  br label %57

4:                                                ; preds = %71
  br label %76

5:                                                ; preds = %88
  %6 = trunc i32 %89 to i8
  br label %93

7:                                                ; preds = %105
  %8 = trunc i32 %106 to i16
  br label %110

9:                                                ; preds = %122
  br label %126

10:                                               ; preds = %136
  %11 = trunc i64 %137 to i8
  br label %140

12:                                               ; preds = %150
  %13 = trunc i64 %151 to i16
  br label %154

14:                                               ; preds = %164
  %15 = trunc i64 %165 to i32
  br label %168

16:                                               ; preds = %178
  br label %181

17:                                               ; preds = %189
  %18 = trunc i128 %190 to i8
  br label %192

19:                                               ; preds = %200
  %20 = trunc i128 %201 to i16
  br label %203

21:                                               ; preds = %211
  %22 = trunc i128 %212 to i32
  br label %214

23:                                               ; preds = %222
  %24 = trunc i128 %223 to i64
  br label %225

25:                                               ; preds = %233
  br label %235

26:                                               ; preds = %0
  %27 = call i16 @"upcast<u8, u16>"(i8 0)
  %28 = call i32 @"upcast<u8, u32>"(i8 0)
  %29 = call i64 @"upcast<u8, u64>"(i8 0)
  %30 = call i128 @"upcast<u8, u128>"(i8 0)
  %31 = call i32 @"upcast<u16, u32>"(i16 0)
  %32 = call i64 @"upcast<u16, u64>"(i16 0)
  %33 = call i128 @"upcast<u16, u128>"(i16 0)
  %34 = call i64 @"upcast<u32, u64>"(i32 0)
  %35 = call i128 @"upcast<u32, u128>"(i32 0)
  %36 = call i128 @"upcast<u64, u128>"(i64 0)
  br i1 true, label %1, label %44

37:                                               ; preds = %1
  %38 = phi i16 [ 0, %1 ]
  %39 = phi i32 [ 0, %1 ]
  %40 = phi i64 [ 0, %1 ]
  %41 = phi i128 [ 0, %1 ]
  %42 = phi i8 [ 0, %1 ]
  %43 = call <{ i16, [1 x i8] }> @"enum_init<core::option::Option::<core::integer::u8>, 0>"(i8 %42)
  br label %51

44:                                               ; preds = %26
  %45 = phi i16 [ 0, %26 ]
  %46 = phi i32 [ 0, %26 ]
  %47 = phi i64 [ 0, %26 ]
  %48 = phi i128 [ 0, %26 ]
  %49 = call <{}> @"struct_construct<Unit>"()
  %50 = call <{ i16, [1 x i8] }> @"enum_init<core::option::Option::<core::integer::u8>, 1>"(<{}> %49)
  br label %51

51:                                               ; preds = %37, %44
  %52 = phi i16 [ %45, %44 ], [ %38, %37 ]
  %53 = phi i32 [ %46, %44 ], [ %39, %37 ]
  %54 = phi i64 [ %47, %44 ], [ %40, %37 ]
  %55 = phi i128 [ %48, %44 ], [ %41, %37 ]
  %56 = icmp ult i16 %52, 256
  br i1 %56, label %2, label %64

57:                                               ; preds = %2
  %58 = phi i16 [ %52, %2 ]
  %59 = phi i32 [ %53, %2 ]
  %60 = phi i64 [ %54, %2 ]
  %61 = phi i128 [ %55, %2 ]
  %62 = phi i8 [ %3, %2 ]
  %63 = call <{ i16, [1 x i8] }> @"enum_init<core::option::Option::<core::integer::u8>, 0>"(i8 %62)
  br label %71

64:                                               ; preds = %51
  %65 = phi i16 [ %52, %51 ]
  %66 = phi i32 [ %53, %51 ]
  %67 = phi i64 [ %54, %51 ]
  %68 = phi i128 [ %55, %51 ]
  %69 = call <{}> @"struct_construct<Unit>"()
  %70 = call <{ i16, [1 x i8] }> @"enum_init<core::option::Option::<core::integer::u8>, 1>"(<{}> %69)
  br label %71

71:                                               ; preds = %57, %64
  %72 = phi i16 [ %65, %64 ], [ %58, %57 ]
  %73 = phi i32 [ %66, %64 ], [ %59, %57 ]
  %74 = phi i64 [ %67, %64 ], [ %60, %57 ]
  %75 = phi i128 [ %68, %64 ], [ %61, %57 ]
  br i1 true, label %4, label %82

76:                                               ; preds = %4
  %77 = phi i32 [ %73, %4 ]
  %78 = phi i64 [ %74, %4 ]
  %79 = phi i128 [ %75, %4 ]
  %80 = phi i16 [ %72, %4 ]
  %81 = call <{ i16, [2 x i8] }> @"enum_init<core::option::Option::<core::integer::u16>, 0>"(i16 %80)
  br label %88

82:                                               ; preds = %71
  %83 = phi i32 [ %73, %71 ]
  %84 = phi i64 [ %74, %71 ]
  %85 = phi i128 [ %75, %71 ]
  %86 = call <{}> @"struct_construct<Unit>"()
  %87 = call <{ i16, [2 x i8] }> @"enum_init<core::option::Option::<core::integer::u16>, 1>"(<{}> %86)
  br label %88

88:                                               ; preds = %76, %82
  %89 = phi i32 [ %83, %82 ], [ %77, %76 ]
  %90 = phi i64 [ %84, %82 ], [ %78, %76 ]
  %91 = phi i128 [ %85, %82 ], [ %79, %76 ]
  %92 = icmp ult i32 %89, 256
  br i1 %92, label %5, label %99

93:                                               ; preds = %5
  %94 = phi i32 [ %89, %5 ]
  %95 = phi i64 [ %90, %5 ]
  %96 = phi i128 [ %91, %5 ]
  %97 = phi i8 [ %6, %5 ]
  %98 = call <{ i16, [1 x i8] }> @"enum_init<core::option::Option::<core::integer::u8>, 0>"(i8 %97)
  br label %105

99:                                               ; preds = %88
  %100 = phi i32 [ %89, %88 ]
  %101 = phi i64 [ %90, %88 ]
  %102 = phi i128 [ %91, %88 ]
  %103 = call <{}> @"struct_construct<Unit>"()
  %104 = call <{ i16, [1 x i8] }> @"enum_init<core::option::Option::<core::integer::u8>, 1>"(<{}> %103)
  br label %105

105:                                              ; preds = %93, %99
  %106 = phi i32 [ %100, %99 ], [ %94, %93 ]
  %107 = phi i64 [ %101, %99 ], [ %95, %93 ]
  %108 = phi i128 [ %102, %99 ], [ %96, %93 ]
  %109 = icmp ult i32 %106, 65536
  br i1 %109, label %7, label %116

110:                                              ; preds = %7
  %111 = phi i32 [ %106, %7 ]
  %112 = phi i64 [ %107, %7 ]
  %113 = phi i128 [ %108, %7 ]
  %114 = phi i16 [ %8, %7 ]
  %115 = call <{ i16, [2 x i8] }> @"enum_init<core::option::Option::<core::integer::u16>, 0>"(i16 %114)
  br label %122

116:                                              ; preds = %105
  %117 = phi i32 [ %106, %105 ]
  %118 = phi i64 [ %107, %105 ]
  %119 = phi i128 [ %108, %105 ]
  %120 = call <{}> @"struct_construct<Unit>"()
  %121 = call <{ i16, [2 x i8] }> @"enum_init<core::option::Option::<core::integer::u16>, 1>"(<{}> %120)
  br label %122

122:                                              ; preds = %110, %116
  %123 = phi i32 [ %117, %116 ], [ %111, %110 ]
  %124 = phi i64 [ %118, %116 ], [ %112, %110 ]
  %125 = phi i128 [ %119, %116 ], [ %113, %110 ]
  br i1 true, label %9, label %131

126:                                              ; preds = %9
  %127 = phi i64 [ %124, %9 ]
  %128 = phi i128 [ %125, %9 ]
  %129 = phi i32 [ %123, %9 ]
  %130 = call <{ i16, [4 x i8] }> @"enum_init<core::option::Option::<core::integer::u32>, 0>"(i32 %129)
  br label %136

131:                                              ; preds = %122
  %132 = phi i64 [ %124, %122 ]
  %133 = phi i128 [ %125, %122 ]
  %134 = call <{}> @"struct_construct<Unit>"()
  %135 = call <{ i16, [4 x i8] }> @"enum_init<core::option::Option::<core::integer::u32>, 1>"(<{}> %134)
  br label %136

136:                                              ; preds = %126, %131
  %137 = phi i64 [ %132, %131 ], [ %127, %126 ]
  %138 = phi i128 [ %133, %131 ], [ %128, %126 ]
  %139 = icmp ult i64 %137, 256
  br i1 %139, label %10, label %145

140:                                              ; preds = %10
  %141 = phi i64 [ %137, %10 ]
  %142 = phi i128 [ %138, %10 ]
  %143 = phi i8 [ %11, %10 ]
  %144 = call <{ i16, [1 x i8] }> @"enum_init<core::option::Option::<core::integer::u8>, 0>"(i8 %143)
  br label %150

145:                                              ; preds = %136
  %146 = phi i64 [ %137, %136 ]
  %147 = phi i128 [ %138, %136 ]
  %148 = call <{}> @"struct_construct<Unit>"()
  %149 = call <{ i16, [1 x i8] }> @"enum_init<core::option::Option::<core::integer::u8>, 1>"(<{}> %148)
  br label %150

150:                                              ; preds = %140, %145
  %151 = phi i64 [ %146, %145 ], [ %141, %140 ]
  %152 = phi i128 [ %147, %145 ], [ %142, %140 ]
  %153 = icmp ult i64 %151, 65536
  br i1 %153, label %12, label %159

154:                                              ; preds = %12
  %155 = phi i64 [ %151, %12 ]
  %156 = phi i128 [ %152, %12 ]
  %157 = phi i16 [ %13, %12 ]
  %158 = call <{ i16, [2 x i8] }> @"enum_init<core::option::Option::<core::integer::u16>, 0>"(i16 %157)
  br label %164

159:                                              ; preds = %150
  %160 = phi i64 [ %151, %150 ]
  %161 = phi i128 [ %152, %150 ]
  %162 = call <{}> @"struct_construct<Unit>"()
  %163 = call <{ i16, [2 x i8] }> @"enum_init<core::option::Option::<core::integer::u16>, 1>"(<{}> %162)
  br label %164

164:                                              ; preds = %154, %159
  %165 = phi i64 [ %160, %159 ], [ %155, %154 ]
  %166 = phi i128 [ %161, %159 ], [ %156, %154 ]
  %167 = icmp ult i64 %165, 4294967296
  br i1 %167, label %14, label %173

168:                                              ; preds = %14
  %169 = phi i64 [ %165, %14 ]
  %170 = phi i128 [ %166, %14 ]
  %171 = phi i32 [ %15, %14 ]
  %172 = call <{ i16, [4 x i8] }> @"enum_init<core::option::Option::<core::integer::u32>, 0>"(i32 %171)
  br label %178

173:                                              ; preds = %164
  %174 = phi i64 [ %165, %164 ]
  %175 = phi i128 [ %166, %164 ]
  %176 = call <{}> @"struct_construct<Unit>"()
  %177 = call <{ i16, [4 x i8] }> @"enum_init<core::option::Option::<core::integer::u32>, 1>"(<{}> %176)
  br label %178

178:                                              ; preds = %168, %173
  %179 = phi i64 [ %174, %173 ], [ %169, %168 ]
  %180 = phi i128 [ %175, %173 ], [ %170, %168 ]
  br i1 true, label %16, label %185

181:                                              ; preds = %16
  %182 = phi i128 [ %180, %16 ]
  %183 = phi i64 [ %179, %16 ]
  %184 = call <{ i16, [8 x i8] }> @"enum_init<core::option::Option::<core::integer::u64>, 0>"(i64 %183)
  br label %189

185:                                              ; preds = %178
  %186 = phi i128 [ %180, %178 ]
  %187 = call <{}> @"struct_construct<Unit>"()
  %188 = call <{ i16, [8 x i8] }> @"enum_init<core::option::Option::<core::integer::u64>, 1>"(<{}> %187)
  br label %189

189:                                              ; preds = %181, %185
  %190 = phi i128 [ %186, %185 ], [ %182, %181 ]
  %191 = icmp ult i128 %190, 256
  br i1 %191, label %17, label %196

192:                                              ; preds = %17
  %193 = phi i128 [ %190, %17 ]
  %194 = phi i8 [ %18, %17 ]
  %195 = call <{ i16, [1 x i8] }> @"enum_init<core::option::Option::<core::integer::u8>, 0>"(i8 %194)
  br label %200

196:                                              ; preds = %189
  %197 = phi i128 [ %190, %189 ]
  %198 = call <{}> @"struct_construct<Unit>"()
  %199 = call <{ i16, [1 x i8] }> @"enum_init<core::option::Option::<core::integer::u8>, 1>"(<{}> %198)
  br label %200

200:                                              ; preds = %192, %196
  %201 = phi i128 [ %197, %196 ], [ %193, %192 ]
  %202 = icmp ult i128 %201, 65536
  br i1 %202, label %19, label %207

203:                                              ; preds = %19
  %204 = phi i128 [ %201, %19 ]
  %205 = phi i16 [ %20, %19 ]
  %206 = call <{ i16, [2 x i8] }> @"enum_init<core::option::Option::<core::integer::u16>, 0>"(i16 %205)
  br label %211

207:                                              ; preds = %200
  %208 = phi i128 [ %201, %200 ]
  %209 = call <{}> @"struct_construct<Unit>"()
  %210 = call <{ i16, [2 x i8] }> @"enum_init<core::option::Option::<core::integer::u16>, 1>"(<{}> %209)
  br label %211

211:                                              ; preds = %203, %207
  %212 = phi i128 [ %208, %207 ], [ %204, %203 ]
  %213 = icmp ult i128 %212, 4294967296
  br i1 %213, label %21, label %218

214:                                              ; preds = %21
  %215 = phi i128 [ %212, %21 ]
  %216 = phi i32 [ %22, %21 ]
  %217 = call <{ i16, [4 x i8] }> @"enum_init<core::option::Option::<core::integer::u32>, 0>"(i32 %216)
  br label %222

218:                                              ; preds = %211
  %219 = phi i128 [ %212, %211 ]
  %220 = call <{}> @"struct_construct<Unit>"()
  %221 = call <{ i16, [4 x i8] }> @"enum_init<core::option::Option::<core::integer::u32>, 1>"(<{}> %220)
  br label %222

222:                                              ; preds = %214, %218
  %223 = phi i128 [ %219, %218 ], [ %215, %214 ]
  %224 = icmp ult i128 %223, 18446744073709551616
  br i1 %224, label %23, label %229

225:                                              ; preds = %23
  %226 = phi i128 [ %223, %23 ]
  %227 = phi i64 [ %24, %23 ]
  %228 = call <{ i16, [8 x i8] }> @"enum_init<core::option::Option::<core::integer::u64>, 0>"(i64 %227)
  br label %233

229:                                              ; preds = %222
  %230 = phi i128 [ %223, %222 ]
  %231 = call <{}> @"struct_construct<Unit>"()
  %232 = call <{ i16, [8 x i8] }> @"enum_init<core::option::Option::<core::integer::u64>, 1>"(<{}> %231)
  br label %233

233:                                              ; preds = %225, %229
  %234 = phi i128 [ %230, %229 ], [ %226, %225 ]
  br i1 true, label %25, label %238

235:                                              ; preds = %25
  %236 = phi i128 [ %234, %25 ]
  %237 = call <{ i16, [16 x i8] }> @"enum_init<core::option::Option::<core::integer::u128>, 0>"(i128 %236)
  br label %241

238:                                              ; preds = %233
  %239 = call <{}> @"struct_construct<Unit>"()
  %240 = call <{ i16, [16 x i8] }> @"enum_init<core::option::Option::<core::integer::u128>, 1>"(<{}> %239)
  br label %241

241:                                              ; preds = %235, %238
  %242 = call <{}> @"struct_construct<Unit>"()
  ret <{}> %242
}

define void @"_mlir_ciface_casts::casts::main"(ptr %0) {
  %2 = call <{}> @"casts::casts::main"()
  store <{}> %2, ptr %0, align 1
  ret void
}

attributes #0 = { alwaysinline norecurse nounwind }
attributes #1 = { norecurse nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
