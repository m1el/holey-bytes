# HERE SHALL THE DOCUMENTATION RESIDE

## Enforced Political Views

- worse is better
- less is more
- embrace `unsafe {}`
- adhere `macro_rules!`
- pessimization == death (put in `std::pin::Pin` and left with hungry crabs)
- importing external dependencies == death (`fn(dependencies) -> ExecutionStrategy`)
- above sell not be disputed, discussed, or questioned

## What hblang is

Holey-Bytes-Language (hblang for short) (*.hb) is the only true language targeting hbvm byte code. hblang is low level, manually managed, and procedural. Its rumored to be better then writing hbasm and you should probably use it for complex applications.

## What hblang isnt't

hblang knows what it isn't, because it knows what it is, hblang computes this by sub...


## Examples

Examples are also used in tests. To add an example that runs during testing add:
<pre>
#### &lt;name&gt
```hb
&lt;example&gt
```
</pre>
and also:
```rs
<name> => README;
```
to the `run_tests` macro at the bottom of the `src/codegen.rs`.

### Tour Examples

Following examples incrementally introduce language features and syntax.

#### main_fn
```hb
main := fn(): uint {
	return 1
}
```

#### arithmetic
```hb
main := fn(): uint {
	return 10 - 20 / 2 + 4 * (2 + 2) - 4 * 4 + (1 << 0) + -1
}
```

#### functions
```hb
main := fn(): uint {
	return add_one(10) + add_two(20)
}

add_two := fn(x: uint): uint {
	return x + 2
}

add_one := fn(x: uint): uint {
	return x + 1
}
```

#### comments
```hb
// commant is an item
main := fn(): uint {
	// comment is a statement

	foo(/* comment is an exprression /* if you are crazy */ */)
	return 0
}

foo := fn(comment: void): void return /* comment evaluates to void */

// comments might be formatted in the future
```

#### if_statements
```hb
main := fn(): uint {
	return fib(10)
}

fib := fn(x: uint): uint {
	if x <= 2 {
		return 1
	} else {
		return fib(x - 1) + fib(x - 2)
	}
}
```

#### variables
```hb
main := fn(): uint {
	ඞ := 1
	b := 2
	ඞ += 1
	return ඞ - b
}
```

#### loops
```hb
main := fn(): uint {
	return fib(10)
}

fib := fn(n: uint): uint {
	a := 0
	b := 1
	loop if n == 0 break else {
		c := a + b
		a = b
		b = c
		n -= 1
	}
	return a
}
```

#### pointers
```hb
main := fn(): uint {
	a := 1
	b := &a

	boundary := 1000

	b = b + boundary - 2
	b = b - (boundary - 2)

	modify(b)
	drop(a)
	return *b - 2
}

modify := fn(a: ^uint): void {
	*a = 2
	return
}

drop := fn(a: uint): void {
	return
}
```

#### structs
```hb
Ty := struct {
	// comment

	a: uint,
}

Ty2 := struct {
	ty: Ty,
	c: uint,
}

useless := struct {}

main := fn(): uint {
	// `packed` structs have no padding (all fields are alighred to 1)
	if @sizeof(packed struct {a: u8, b: u16}) != 3 {
		return 9001
	}

	finst := Ty2.{ty: .{a: 4}, c: 3}
	inst := odher_pass(finst)
	if inst.c == 3 {
		return pass(&inst.ty)
	}
	return 0
}

pass := fn(t: ^Ty): uint {
	return t.a
}

odher_pass := fn(t: Ty2): Ty2 {
	return t
}
```

#### hex_octal_binary_literals
```hb
main := fn(): uint {
	hex := 0xFF
	decimal := 255
	octal := 0o377
	binary := 0b11111111

	if hex == decimal & octal == decimal & binary == decimal {
		return 0
	}
	return 1
}
```

#### struct_operators
```hb
Point := struct {
	x: uint,
	y: uint,
}

Rect := struct {
	a: Point,
	b: Point,
}

Color := packed struct {b: u8, g: u8, r: u8, a: u8}

main := fn(): uint {
	i := Color.(0, 0, 0, 0)
	i += .(1, 1, 1, 1)
	if i.r + i.g + i.b + i.a != 4 {
		return 1008
	}

	a := Point.(1, 2)
	b := Point.(3, 4)

	d := Rect.(a + b, b - a)
	zp := Point.(0, 0)
	d2 := Rect.(zp - b, a)
	d2 += d

	c := d2.a + d2.b
	return c.x + c.y
}
```

#### global_variables
```hb
global_var := 10

complex_global_var := fib(global_var) - 5

fib := fn(n: uint): uint {
	if 2 > n {
		return n
	}
	return fib(n - 1) + fib(n - 2)
}

main := fn(): uint {
	complex_global_var += 5
	return complex_global_var
}
```
note: values of global variables are evaluated at compile time

#### directives
```hb
foo := @use("foo.hb")

main := fn(): uint {
	byte := @as(u8, 10)
	same_type_as_byte := @as(@TypeOf(byte), 30)
	wide_uint := @as(u32, 40)
	truncated_uint := @as(u8, @intcast(wide_uint))
	size_of_Type_in_bytes := @sizeof(foo.Type)
	align_of_Type_in_bytes := @alignof(foo.Type)
	hardcoded_pointer := @as(^u8, @bitcast(10))
	ecall_that_returns_uint := @as(uint, @eca(1, foo.Type.(10, 20), 5, 6))
	embedded_array := @as([u8; 15], @embed("text.txt"))
	return @inline(foo.foo)
}

// in module: foo.hb

Type := struct {
	brah: uint,
	blah: uint,
}

foo := fn(): uint return 0

// in module: text.txt
arbitrary text
```

- `@use(<string>)`: imports a module based of string, the string is passed to a loader that can be customized, default loader uses following syntax:
	- `((rel:|)(<path>)|git:<git-addr>:<path>)`: `rel:` and `''` prefixes both mean module is located at `path` relavive to the current file, `git:` takes a git url without `https://` passed as `git-addr`, `path` then refers to file within the repository
- `@TypeOf(<expr>)`: results into literal type of whatever the type of `<expr>` is, `<expr>` is not included in final binary
- `@as(<ty>, <expr>)`: huint to the compiler that  `@TypeOf(<expr>) == <ty>`
- `@intcast(<expr>)`: needs to be used when conversion of `@TypeOf(<expr>)` would loose precision (widening of integers is implicit)
- `@sizeof(<ty>), @alignof(<ty>)`: I think explaining this would insult your intelligence
- `@bitcast(<expr>)`: tell compiler to assume `@TypeOf(<expr>)` is whatever is inferred, so long as size and alignment did not change
- `@eca(<ty>, ...<expr>)`: invoke `eca` instruction, where `<ty>` is the type this will return and `<expr>...` are arguments passed to the call
- `@inline(<func>, ...<args>)`: equivalent to `<func>(...<args>)` but function is guaranteed to inline, compiler will otherwise never inline

#### c_strings
```hb
str_len := fn(str: ^u8): uint {
	len := 0
	loop if *str == 0 break else {
		len += 1
		str += 1
	}
	return len
}

main := fn(): uint {
	// when string ends with '\0' its a C string and thus type is '^u8'
	some_str := "abඞ\n\r\t\{35}\{36373839}\0"
	len := str_len(some_str)
	some_other_str := "fff\0"
	lep := str_len(some_other_str)
	return lep + len
}
```

#### struct_patterns
```hb
.{fib, fib_iter, Fiber} := @use("fibs.hb")

main := fn(): uint {
	.{a, b} := Fiber.{a: 10, b: 10}
	return fib(a) - fib_iter(b)
}

// in module: fibs.hb

Fiber := struct {a: u8, b: u8}

fib := fn(n: uint): uint if n < 2 {
	return n
} else {
	return fib(n - 1) + fib(n - 2)
}

fib_iter := fn(n: uint): uint {
	a := 0
	b := 1
	loop if n == 0 break else {
		c := a + b
		a = b
		b = c
		n -= 1
	}
	return a
}
```

#### arrays
```hb
main := fn(): uint {
	addr := @as(u16, 0x1FF)
	msg := [u8].(0, 0, @intcast(addr), @intcast(addr >> 8))
	_force_stack := &msg

	arr := [uint].(1, 2, 4)
	return pass(&arr) + msg[3]
}

pass := fn(arr: ^[uint; 3]): uint {
	return arr[0] + arr[1] + arr[arr[1]]
}
```

#### inline
```hb
main := fn(): uint {
	return @inline(foo, 1, 2, 3) - 6
}

gb := 0

foo := fn(a: uint, b: uint, c: uint): uint {
	if false | gb != 0 return 1
	return a + b + c
}
```

#### idk
```hb
_edge_case := @as(uint, idk)

main := fn(): uint {
	big_array := @as([u8; 128], idk)
	i := 0
	loop if i >= 128 break else {
		big_array[i] = 69
		i += 1
	}
	return big_array[42]
}
```

#### generic_functions
```hb
add := fn($T: type, a: T, b: T): T return a + b

main := fn(): uint {
	return add(u32, 2, 2) - add(uint, 1, 3)
}
```

### Incomplete Examples

#### comptime_pointers
```hb
main := fn(): uint {
	$integer := 7
	modify(&integer)
	return integer
}

modify := fn($num: ^uint): void {
	$: *num = 0
}
```

#### generic_types
```hb
MALLOC_SYS_CALL := 69
FREE_SYS_CALL := 96

malloc := fn(size: uint, align: uint): ^void return @eca(MALLOC_SYS_CALL, size, align)
free := fn(ptr: ^void, size: uint, align: uint): void return @eca(FREE_SYS_CALL, ptr, size, align)

Vec := fn($Elem: type): type {
	return struct {
		data: ^Elem,
		len: uint,
		cap: uint,
	}
}

new := fn($Elem: type): Vec(Elem) return Vec(Elem).{data: @bitcast(0), len: 0, cap: 0}

deinit := fn($Elem: type, vec: ^Vec(Elem)): void {
	free(@bitcast(vec.data), vec.cap * @sizeof(Elem), @alignof(Elem));
	*vec = new(Elem)
	return
}

push := fn($Elem: type, vec: ^Vec(Elem), value: Elem): ^Elem {
	if vec.len == vec.cap {
		if vec.cap == 0 {
			vec.cap = 1
		} else {
			vec.cap *= 2
		}

		new_alloc := @as(^Elem, @bitcast(malloc(vec.cap * @sizeof(Elem), @alignof(Elem))))
		if new_alloc == 0 return @bitcast(0)

		src_cursor := vec.data
		dst_cursor := new_alloc
		end := vec.data + vec.len

		loop if src_cursor == end break else {
			*dst_cursor = *src_cursor
			src_cursor += 1
			dst_cursor += 1
		}

		if vec.len != 0 {
			free(@bitcast(vec.data), vec.len * @sizeof(Elem), @alignof(Elem))
		}
		vec.data = new_alloc
	}

	slot := vec.data + vec.len;
	*slot = value
	vec.len += 1
	return slot
}

main := fn(): uint {
	vec := new(uint)
	_f := push(uint, &vec, 69)
	res := *vec.data
	deinit(uint, &vec)
	return res
}
```

#### fb_driver
```hb
arm_fb_ptr := fn(): uint return 100
x86_fb_ptr := fn(): uint return 100

check_platform := fn(): uint {
	return x86_fb_ptr()
}

set_pixel := fn(x: uint, y: uint, width: uint): uint {
	return y * width + x
}

main := fn(): uint {
	fb_ptr := check_platform()
	width := 100
	height := 30
	x := 0
	y := 0
	//t := 0
	i := 0

	loop {
		if x < height {
			//t += set_pixel(x, y, height)
			x += 1
			i += 1
		} else {
			x = 0
			y += 1
			if set_pixel(x, y, height) != i return 0
			if y == width break
		}
	}
	return i
}
```

### Purely Testing Examples

#### nonexistent_ident_import
```hb
main := @use("foo.hb").main
// in module: foo.hb
foo := fn(): void {
	return
}
foo := fn(): void {
	return
}
main := @use("bar.hb").mian
// in module: bar.hb
main := fn(): void {
	return
}
```

#### big_array_crash
```hb
SIN_TABLE := [int].(0, 174, 348, 523, 697, 871, 1045, 1218, 1391, 1564, 1736, 1908, 2079, 2249, 2419, 2588, 2756, 2923, 3090, 3255, 3420, 3583, 3746, 3907, 4067, 4226, 4384, 4540, 4695, 4848, 5000, 5150, 5299, 5446, 5591, 5735, 5877, 6018, 6156, 6293, 6427, 6560, 6691, 6819, 6946, 7071, 7193, 7313, 7431, 7547, 7660, 7771, 7880, 7986, 8090, 8191, 8290, 8386, 8480, 8571, 8660, 8746, 8829, 8910, 8987, 9063, 9135, 9205, 9271, 9335, 9396, 9455, 9510, 9563, 9612, 9659, 9702, 9743, 9781, 9816, 9848, 9877, 9902, 9925, 9945, 9961, 9975, 9986, 9993, 9998, 10000)

main := fn(): int return SIN_TABLE[10]
```

#### returning_global_struct
```hb
Color := struct {r: u8, g: u8, b: u8, a: u8}
white := Color.(255, 255, 255, 255)
random_color := fn(): Color {
	return white
}
main := fn(): uint {
	val := random_color()
	return @as(uint, val.r) + val.g + val.b + val.a
}
```

#### small_struct_bitcast
```hb
Color := struct {r: u8, g: u8, b: u8, a: u8}
white := Color.(255, 255, 255, 255)
u32_to_color := fn(v: u32): Color return @as(Color, @bitcast(u32_to_u32(@bitcast(v))))
u32_to_u32 := fn(v: u32): u32 return v
main := fn(): uint {
	return u32_to_color(@bitcast(white)).r
}
```

#### small_struct_assignment
```hb
Color := struct {r: u8, g: u8, b: u8, a: u8}
white := Color.(255, 255, 255, 255)
black := Color.(0, 0, 0, 0)
main := fn(): uint {
	f := black
	f = white
	return f.a
}
```

#### intcast_store
```hb
SetMsg := packed struct {a: u8, count: u32, size: u32, src: ^u8, dest: ^u8}
set := fn($Expr: type, src: ^Expr, dest: ^Expr, count: uint): u32 {
	l := SetMsg.(5, @intcast(count), @intcast(@sizeof(Expr)), @bitcast(src), @bitcast(dest))
	return l.count
}

main := fn(): uint {
	return set(uint, &0, &0, 1024)
}
```

#### string_flip
```hb
U := struct {u: uint}
main := fn(): uint {
	arr := @as([U; 2 * 2], idk)

	i := 0
	loop if i == 2 * 2 break else {
		arr[i] = .(i)
		i += 1
	}

	i = 0
	loop if i == 2 / 2 break else {
		j := 0
		loop if j == 2 break else {
			a := i * 2 + j
			b := (2 - i - 1) * 2 + j
			tmp := arr[a]
			arr[a] = arr[b]
			arr[b] = tmp
			j += 1
		}
		i += 1
	}

	return arr[0].u
}
```

#### wide_ret
```hb
OemIdent := struct {
	dos_version: [u8; 8],
	dos_version_name: [u8; 8],
}

Stru := struct {
	a: u16,
	b: u16,
}

small_struct := fn(): Stru {
	return .{a: 0, b: 0}
}

maina := fn(major: uint, minor: uint): OemIdent {
	_f := small_struct()
	ver := [u8].(0, 0, 0, 3, 1, 0, 0, 0)
	return OemIdent.(ver, ver)
}

main := fn(): uint {
	m := maina(0, 0)
	return m.dos_version[3] - m.dos_version_name[4]
}
```

#### comptime_min_reg_leak
```hb
a := @use("math.hb").min(100, 50)

main := fn(): uint {
	return a
}

// in module: math.hb

SIZEOF_uint := 32
SHIFT := SIZEOF_uint - 1
min := fn(a: uint, b: uint): uint {
	c := a - b
	return b + (c & c >> SHIFT)
}
```

#### different_types
```hb
Color := struct {
	r: u8,
	g: u8,
	b: u8,
	a: u8,
}

Point := struct {
	x: u32,
	y: u32,
}

Pixel := struct {
	color: Color,
	pouint: Point,
}

main := fn(): uint {
	pixel := Pixel.{
		color: Color.{
			r: 255,
			g: 0,
			b: 0,
			a: 255,
		},
		pouint: Point.{
			x: 0,
			y: 2,
		},
	}

	soupan := 1
	if *(&pixel.pouint.x + soupan) != 2 {
		return 0
	}

	if *(&pixel.pouint.y - 1) != 0 {
		return 64
	}

	return pixel.pouint.x + pixel.pouint.y + pixel.color.r
		+ pixel.color.g + pixel.color.b + pixel.color.a
}
```

#### struct_return_from_module_function
```hb
bar := @use("bar.hb")

main := fn(): uint {
	return 7 - bar.foo().x - bar.foo().y - bar.foo().z
}

// in module: bar.hb


foo := fn(): Foo {
	return .{x: 3, y: 2, z: 2}
}

Foo := struct {x: uint, y: u32, z: u32}
```

#### sort_something_viredly
```hb
main := fn(): uint {
	return sqrt(100)
}

sqrt := fn(x: uint): uint {
	temp := 0
	g := 0
	b := 32768
	bshift := 15
	loop if b == 0 {
		break
	} else {
		bshift -= 1
		temp = b + (g << 1)
		temp <<= bshift
		if x >= temp {
			g += b
			x -= temp
		}
		b >>= 1
	}
	return g
}
```

#### structs_in_registers
```hb
ColorBGRA := struct {b: u8, g: u8, r: u8, a: u8}
MAGENTA := ColorBGRA.{b: 205, g: 0, r: 205, a: 255}

main := fn(): uint {
	color := MAGENTA
	return color.r
}
```

#### comptime_function_from_another_file
```hb
stn := @use("stn.hb")

CONST_A := 100
CONST_B := 50
a := stn.math.min(CONST_A, CONST_B)

main := fn(): uint {
	return a
}

// in module: stn.hb
math := @use("math.hb")

// in module: math.hb
SIZEOF_uint := 32
SHIFT := SIZEOF_uint - 1
min := fn(a: uint, b: uint): uint {
	c := a - b
	return b + (c & c >> SHIFT)
}
```

#### inline_test
```hb
fna := fn(a: uint, b: uint, c: uint): uint return a + b + c

scalar_values := fn(): uint {
	return @inline(fna, 1, @inline(fna, 2, 3, 4), -10)
}

A := struct {a: uint}
AB := struct {a: A, b: uint}

mangle := fn(a: A, ab: AB): uint {
	return a.a + ab.a.a - ab.b
}

structs := fn(): uint {
	return @inline(mangle, .(0), .(.(@inline(mangle, .(20), .(.(5), 5))), 20))
}

main := fn(): uint {
	if scalar_values() != 0 return 1
	if structs() != 0 return structs()

	return 0
}
```

#### inlined_generic_functions
```hb
abs := fn($Expr: type, x: Expr): Expr {
	mask := x >> @bitcast(@sizeof(Expr) - 1)
	return (x ^ mask) - mask
}

main := fn(): uint {
	return @inline(abs, uint, -10)
}
```

#### some_generic_code
```hb
some_func := fn($Elem: type): void {
	return
}

main := fn(): void {
	some_func(u8)
	return
}
```

#### integer_inference_issues
```hb
.{integer_range} := @use("random.hb")
main := fn(): void {
	a := integer_range(0, 1000)
	return
}

// in module: random.hb
integer_range := fn(min: uint, max: uint): uint {
	return @eca(3, 4) % (@bitcast(max) - min + 1) + min
}
```

#### signed_to_unsigned_upcast
```hb
main := fn(): uint return @as(i32, 1)
```

#### writing_into_string
```hb
outl := fn(): void {
	msg := "whahaha\0"
	_u := @as(u8, 0)
	return
}

inl := fn(): void {
	msg := "luhahah\0"
	return
}

main := fn(): void {
	outl()
	inl()
	return
}
```

#### request_page
```hb
request_page := fn(page_count: u8): ^u8 {
	msg := "\{00}\{01}xxxxxxxx\0"
	msg_page_count := msg + 1;
	*msg_page_count = page_count
	return @eca(3, 2, msg, 12)
}

create_back_buffer := fn(total_pages: int): ^u32 {
	if total_pages <= 0xFF {
		return @bitcast(request_page(@intcast(total_pages)))
	}
	ptr := request_page(255)
	remaining := total_pages - 0xFF
	loop if remaining <= 0 break else {
		if remaining < 0xFF {
			_f := request_page(@intcast(remaining))
		} else {
			_f := request_page(0xFF)
		}
		remaining -= 0xFF
	}
	return @bitcast(ptr)
}

main := fn(): void {
	_f := create_back_buffer(400)
	return
}
```

#### tests_ptr_to_ptr_copy
```hb
main := fn(): uint {
	back_buffer := @as([u8; 1024 * 10], idk)

	n := 0
	loop if n >= 1024 break else {
		back_buffer[n] = 64
		n += 1
	}
	n = 1
	loop if n >= 10 break else {
		*(@as(^[u8; 1024], @bitcast(&back_buffer)) + n) = *@as(^[u8; 1024], @bitcast(&back_buffer))
		n += 1
	}
	return back_buffer[1024 * 2]
}
```

### Just Testing Optimizations

#### const_folding_with_arg
```hb
main := fn(arg: uint): uint {
	// reduces to 0
	return arg + 0 - arg * 1 + arg + 1 + arg + 2 + arg + 3 - arg * 3 - 6
}
```

#### branch_assignments
```hb
main := fn(arg: uint): uint {
	if arg == 1 {
		arg = 1
	} else if arg == 0 {
		arg = 2
	} else {
		arg = 3
	}
	return arg
}
```

#### exhaustive_loop_testing
```hb
main := fn(): uint {
	loop break

	x := 0
	loop {
		x += 1
		break
	}

	if multiple_breaks(0) != 3 {
		return 1
	}

	if multiple_breaks(4) != 10 {
		return 2
	}

	if state_change_in_break(0) != 0 {
		return 3
	}

	if state_change_in_break(4) != 10 {
		return 4
	}

	if continue_and_state_change(10) != 10 {
		return 5
	}

	if continue_and_state_change(3) != 0 {
		return 6
	}

	f := 0
	loop {
		if f == 1 {
			f = 0
		} else {
			f = 1
		}

		f = continue_and_state_change(0)
	}
}

multiple_breaks := fn(arg: uint): uint {
	loop if arg < 10 {
		arg += 1
		if arg == 3 break
	} else break
	return arg
}

state_change_in_break := fn(arg: uint): uint {
	loop if arg < 10 {
		if arg == 3 {
			arg = 0
			break
		}
		arg += 1
	} else break
	return arg
}

continue_and_state_change := fn(arg: uint): uint {
	loop if arg < 10 {
		if arg == 2 {
			arg = 4
			continue
		}
		if arg == 3 {
			arg = 0
			break
		}
		arg += 1
	} else break
	return arg
}
```

#### pointer_opts
```hb
main := fn(): uint {
	mem := &0;
	*mem = 1;
	*mem = 2

	b := *mem + *mem
	clobber(mem)

	b -= *mem
	return b
}

clobber := fn(cb: ^uint): void {
	*cb = 4
	return
}
```

#### conditional_stores
```hb
main := fn(): uint {
	cnd := cond()
	mem := &1

	if cnd == 0 {
		*mem = 0
	} else {
		*mem = 2
	}

	return *mem
}

cond := fn(): uint return 0
```

#### loop_stores
```hb
main := fn(): uint {
	mem := &10

	loop if *mem == 0 break else {
		*mem -= 1
	}

	return *mem
}
```

#### dead_code_in_loop
```hb
main := fn(): uint {
	n := 0

	loop if n < 10 {
		if n < 10 break
		n += 1
	} else break

	loop if n == 0 return n

	return 1
}
```
