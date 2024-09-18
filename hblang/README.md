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
main := fn(): int {
	return 1;
}
```

#### arithmetic
```hb
main := fn(): int {
	return 10 - 20 / 2 + 4 * (2 + 2) - 4 * 4 + (1 << 0) + -1
}
```

#### functions
```hb
main := fn(): int {
	return add_one(10) + add_two(20)
}

add_two := fn(x: int): int {
	return x + 2
}

add_one := fn(x: int): int {
	return x + 1
}
```

#### comments
```hb
// commant is an item
main := fn(): int {
	// comment is a statement

	foo(/* comment is an exprression /* if you are crazy */ */)
	return 0
}

foo := fn(comment: void): void return /* comment evaluates to void */

// comments might be formatted in the future
```

#### if_statements
```hb
main := fn(): int {
	return fib(10)
}

fib := fn(x: int): int {
	if x <= 2 {
		return 1
	} else {
		return fib(x - 1) + fib(x - 2)
	}
}
```

#### variables
```hb
main := fn(): int {
	ඞ := 1
	b := 2
	ඞ += 1
	return ඞ - b
}
```

#### loops
```hb
main := fn(): int {
	return fib(10)
}

fib := fn(n: int): int {
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
main := fn(): int {
	a := 1
	b := &a
	modify(b)
	drop(a)
	stack_reclamation_edge_case := 0
	return *b - 2
}

modify := fn(a: ^int): void {
	*a = 2
	return
}

drop := fn(a: int): void {
	return
}
```

#### structs
```hb
Ty := struct {
	// comment

	a: int,
	b: int,
}

Ty2 := struct {
	ty: Ty,
	c: int,
}

main := fn(): int {
	finst := Ty2.{ty: Ty.{a: 4, b: 1}, c: 3}
	inst := odher_pass(finst)
	if inst.c == 3 {
		return pass(&inst.ty)
	}
	return 0
}

pass := fn(t: ^Ty): int {
	.{a, b} := *t
	return a - b
}

odher_pass := fn(t: Ty2): Ty2 {
	return t
}
```

#### struct_operators
```hb
Point := struct {
	x: int,
	y: int,
}

Rect := struct {
	a: Point,
	b: Point,
}

main := fn(): int {
	a := Point.(1, 2)
	b := Point.(3, 4)

	d := Rect.(a + b, b - a)
	d2 := Rect.(Point.(0, 0) - b, a)
	d2 += d

	c := d2.a + d2.b
	return c.x + c.y
}
```

#### global_variables
```hb
global_var := 10

complex_global_var := fib(global_var) - 5

fib := fn(n: int): int {
	if 2 > n {
		return n
	}
	return fib(n - 1) + fib(n - 2)
}

main := fn(): int {
	complex_global_var += 5
	return complex_global_var
}
```
note: values of global variables are evaluated at compile time

#### directives
```hb
foo := @use("foo.hb")

main := fn(): int {
	byte := @as(u8, 10)
	same_type_as_byte := @as(@TypeOf(byte), 30)
	wide_uint := @as(u32, 40)
	truncated_uint := @as(u8, @intcast(wide_uint))
	size_of_Type_in_bytes := @sizeof(foo.Type)
	align_of_Type_in_bytes := @alignof(foo.Type)
	hardcoded_pointer := @as(^u8, @bitcast(10))
	ecall_that_returns_int := @as(int, @eca(1, foo.Type.(10, 20), 5, 6))
	return @inline(foo.foo)
}

// in module: fooasd.hb

Type := struct {
	brah: int,
	blah: int,
}

foo := fn(): int return 0
```

- `@use(<string>)`: imports a module based of string, the string is passed to a loader that can be customized, default loader uses following syntax:
	- `((rel:|)(<path>)|git:<git-addr>:<path>)`: `rel:` and `''` prefixes both mean module is located at `path` relavive to the current file, `git:` takes a git url without `https://` passed as `git-addr`, `path` then refers to file within the repository
- `@TypeOf(<expr>)`: results into literal type of whatever the type of `<expr>` is, `<expr>` is not included in final binary
- `@as(<ty>, <expr>)`: hint to the compiler that  `@TypeOf(<expr>) == <ty>`
- `@intcast(<expr>)`: needs to be used when conversion of `@TypeOf(<expr>)` would loose precision (widening of integers is implicit)
- `@sizeof(<ty>), @alignof(<ty>)`: I think explaining this would insult your intelligence
- `@bitcast(<expr>)`: tell compiler to assume `@TypeOf(<expr>)` is whatever is inferred, so long as size and alignment did not change
- `@eca(<ty>, ...<expr>)`: invoke `eca` instruction, where `<ty>` is the type this will return and `<expr>...` are arguments passed to the call
- `@inline(<func>, ...<args>)`: equivalent to `<func>(...<args>)` but function is guaranteed to inline, compiler will otherwise never inline

#### c_strings
```hb
str_len := fn(str: ^u8): int {
	len := 0
	loop if *str == 0 break else {
		len += 1
		str += 1
	}
	return len
}

main := fn(): int {
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

main := fn(): int {
	.{a, b} := Fiber.{a: 10, b: 10}
	return fib(a) - fib_iter(b)
}

// in module: fibs.hb

Fiber := struct {a: u8, b: u8}

fib := fn(n: int): int if n < 2 {
	return n
} else {
	return fib(n - 1) + fib(n - 2)
}

fib_iter := fn(n: int): int {
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
main := fn(): int {
	addr := @as(u16, 0x1FF)
	msg := [u8].(0, 0, @as(u8, addr & 0xFF), @as(u8, addr >> 8 & 0xFF))
	_force_stack := &msg

	arr := [int].(1, 2, 4)
	return pass(&arr) + msg[3]
}

pass := fn(arr: ^[int; 3]): int {
	return arr[0] + arr[1] + arr[arr[1]]
}
```

#### inline
```hb
main := fn(): int {
	return @inline(foo, 1, 2, 3) - 6
}

foo := fn(a: int, b: int, c: int): int {
	return a + b + c
}
```

#### idk
```hb
_edge_case := @as(int, idk)

main := fn(): int {
	big_array := @as([u8; 128], idk)
	i := 0
	loop if i >= 128 break else {
		big_array[i] = 69
		i += 1
	}
	return big_array[42]
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

main := fn(major: int, minor: int): OemIdent {
	small_struct()
	ver := [u8].(0, 0, 0, 0, 0, 0, 0, 0)
	return OemIdent.(ver, ver)
}
```

### Incomplete Examples

#### comptime_pointers
```hb
main := fn(): int {
	$integer := 7
	modify(&integer)
	return integer
}

modify := fn($num: ^int): void {
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
		if new_alloc == 0 return 0

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

main := fn(): int {
	vec := new(int)
	push(int, &vec, 69)
	res := *vec.data
	deinit(int, &vec)
	return res
}
```

#### generic_functions
```hb
add := fn($T: type, a: T, b: T): T return a + b

main := fn(): int {
	return add(u32, 2, 2) - add(int, 1, 3)
}
```

#### fb_driver
```hb
arm_fb_ptr := fn(): int return 100
x86_fb_ptr := fn(): int return 100

check_platform := fn(): int {
	return x86_fb_ptr()
}

set_pixel := fn(x: int, y: int, width: int): int {
	return y * width + x
}

main := fn(): int {
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

#### comptime_min_reg_leak
```hb
a := @use("math.hb").min(100, 50)

main := fn(): int {
	return a
}

// in module: math.hb

SIZEOF_INT := 32
SHIFT := SIZEOF_INT - 1
min := fn(a: int, b: int): int {
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
	point: Point,
}

main := fn(): int {
	pixel := Pixel.{
		color: Color.{
			r: 255,
			g: 0,
			b: 0,
			a: 255,
		},
		point: Point.{
			x: 0,
			y: 2,
		},
	}

	soupan := 1
	if *(&pixel.point.x + soupan) != 2 {
		return 0
	}

	if *(&pixel.point.y - 1) != 0 {
		return 64
	}

	return pixel.point.x + pixel.point.y + pixel.color.r
		+ pixel.color.g + pixel.color.b + pixel.color.a
}
```

#### struct_return_from_module_function
```hb
bar := @use("bar.hb")

main := fn(): int {
	return 7 - bar.foo().x - bar.foo().y - bar.foo().z
}

// in module: bar.hb


foo := fn(): Foo {
	return .{x: 3, y: 2, z: 2}
}

Foo := struct {x: int, y: u32, z: u32}
```

#### sort_something_viredly
```hb
main := fn(): int {
	foo := sqrt
	return 0
}

sqrt := fn(x: int): int {
	temp := 0
	g := 0
	b := 32768
	bshift := 15
	loop if b == 0 break else {
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

#### hex_octal_binary_literals
```hb
main := fn(): int {
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

#### structs_in_registers
```hb
ColorBGRA := struct {b: u8, g: u8, r: u8, a: u8}
MAGENTA := ColorBGRA.{b: 205, g: 0, r: 205, a: 255}

main := fn(): int {
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

main := fn(): int {
	return a
}

// in module: stn.hb
math := @use("math.hb")

// in module: math.hb
SIZEOF_INT := 32
SHIFT := SIZEOF_INT - 1
min := fn(a: int, b: int): int {
	c := a - b
	return b + (c & c >> SHIFT)
}
```

### Just Testing Optimizations

#### const_folding_with_arg
```hb
main := fn(arg: int): int {
	// reduces to 0
	return arg + 0 - arg * 1 + arg + 1 + arg + 2 + arg + 3 - arg * 3 - 6
}
```

#### branch_assignments
```hb
main := fn(arg: int): int {
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

#### inline_test
```hb
Point := struct {x: int, y: int}
Buffer := struct {}
Transform := Point
ColorBGRA := Point

line := fn(buffer: Buffer, p0: Point, p1: Point, color: ColorBGRA, thickness: int): void {
	if true {
		if p0.x > p1.x {
			@inline(line_low, buffer, p1, p0, color)
		} else {
			@inline(line_low, buffer, p0, p1, color)
		}
	} else {
		if p0.y > p1.y {
			// blah, test leading new line on directives

			@inline(line_high, buffer, p1, p0, color)
		} else {
			@inline(line_high, buffer, p0, p1, color)
		}
	}
	return
}

line_low := fn(buffer: Buffer, p0: Point, p1: Point, color: ColorBGRA): void {
	return
}

line_high := fn(buffer: Buffer, p0: Point, p1: Point, color: ColorBGRA): void {
	return
}

screenidx := @use("screen.hb").screenidx

rect_line := fn(buffer: Buffer, pos: Point, tr: Transform, color: ColorBGRA, thickness: int): void {
	t := 0
	y := 0
	x := 0
	loop if t == thickness break else {
		y = pos.y
		x = pos.x
		loop if y == pos.y + tr.x break else {
			a := 1 + @inline(screenidx, 10)
			a = 1 + @inline(screenidx, 2)
			y += 1
		}
		t += 1
	}
	return
}

random := @use("random.hb")

example := fn(): void {
	loop {
		random_x := @inline(random.integer, 0, 1024)
		random_y := random.integer(0, 768)
		a := @inline(screenidx, random_x)
		break
	}
	return
}

main := fn(): int {
	line(.(), .(0, 0), .(0, 0), .(0, 0), 10)
	rect_line(.(), .(0, 0), .(0, 0), .(0, 0), 10)
	example()
	return 0
}

// in module: screen.hb

screenidx := fn(orange: int): int {
	return orange
}

// in module: random.hb

integer := fn(min: int, max: int): int {
	rng := @as(int, @eca(3, 4))

	if min != 0 | max != 0 {
		return rng % (max - min + 1) + min
	}
	return rng
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
integer_range := fn(min: uint, max: int): uint {
	return @eca(3, 4) % (@bitcast(max) - min + 1) + min
}
```

#### exhaustive_loop_testing
```hb
main := fn(): int {
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

	return 0
}

multiple_breaks := fn(arg: int): int {
	loop if arg < 10 {
		arg += 1
		if arg == 3 break
	} else break
	return arg
}

state_change_in_break := fn(arg: int): int {
	loop if arg < 10 {
		if arg == 3 {
			arg = 0
			break
		}
		arg += 1
	} else break
	return arg
}

continue_and_state_change := fn(arg: int): int {
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

#### writing_into_string
```hb
outl := fn(): void {
	msg := "whahaha\0"
	@as(u8, 0)
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
		return @bitcast(request_page(total_pages))
	}
	ptr := request_page(255)
	remaining := total_pages - 0xFF
	loop if remaining <= 0 break else {
		if remaining < 0xFF {
			request_page(remaining)
		} else {
			request_page(0xFF)
		}
		remaining -= 0xFF
	}
	return @bitcast(ptr)
}

main := fn(): void {
	create_back_buffer(400)
	return
}
```

#### tests_ptr_to_ptr_copy
```hb
main := fn(): int {
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
