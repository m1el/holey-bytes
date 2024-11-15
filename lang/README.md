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
<name>;
```
to the `run_tests` macro at the bottom of the `src/son.rs`.

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

#### floating_point_arithmetic
```hb
main := fn(): f32 {
	return 10. - 20. / 2. + 4. * (2. + 2.) - 4. * 4. + -1.
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

#### nullable_types
```hb
main := fn(): uint {
	a := &1

	b := @as(?^uint, null)
	if decide() b = a

	if b == null return 9001

	c := @as(?uint, *b)
	if decide() c = null

	if c != null return 42

	d := @as(?u16, null)
	if decide() d = 1

	if d == null return 69

	sf := new_foo()
	if sf == null return 999
	str := "foo\0"
	use_foo(sf, str)

	f := no_foo()

	if decide() f = .(a, 1)

	if f == null return 34

	bar := new_bar(a)

	if decide() bar = null

	if bar != null return 420

	g := @as(?^uint, null)
	g = a

	_rd := *g

	return d - *f.a
}

Foo := struct {a: ^uint, b: uint}
no_foo := fn(): ?Foo return null
new_foo := fn(): ?Foo return .(&0, 0)
use_foo := fn(foo: Foo, str: ^u8): void {
}

Bar := struct {a: ?^uint, b: uint}
new_bar := fn(a: ?^uint): ?Bar return .(a, 1)

decide := fn(): bool return !false
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

#### wrong_dead_code_elimination
```hb
Color := struct {b: u8}
main := fn(): void {
	color := Color.(0)
	n := @as(u8, 1)
	loop {
		if color.b == 255 | color.b == 0 {
			n = -n
		}
		color.b += n
	}
}
```

#### inline_return_stack
```hb
$fun := fn(): [uint; 3] {
	res := [uint].(0, 1, 2)
	return res
}

main := fn(): uint {
	vl := fun()
	return vl[0]
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

#### constants
```hb
main := fn(): u32 {
	return some_const + 35
}

$some_const := 34
```

#### directives
```hb
foo := @use("foo.hb")

main := fn(): uint {
	byte := @as(u8, 10)
	same_type_as_byte := @as(@TypeOf(byte), 30)
	wide_uint := @as(u32, 40)
	truncated_uint := @as(u8, @intcast(wide_uint))
	widened_float := @as(f64, @floatcast(1.))
	int_from_float := @as(int, @fti(1.))
	float_from_int := @as(f64, @itf(@as(int, 1)))
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

- `@use(<string>)`: imports a module based on relative path, cycles are allowed when importing
- `@TypeOf(<expr>)`: results into literal type of whatever the type of `<expr>` is, `<expr>` is not included in final binary
- `@as(<ty>, <expr>)`: hint to the compiler that  `@TypeOf(<expr>) == <ty>`
- `@intcast(<expr>)`: needs to be used when conversion of `@TypeOf(<expr>)` would loose precision (widening of integers is implicit)
- `@sizeof(<ty>), @alignof(<ty>)`: get size and align of a type in bytes
- `@bitcast(<expr>)`: tell compiler to assume `@TypeOf(<expr>)` is whatever is inferred, so long as size matches
- `@eca(...<expr>)`: invoke `eca` instruction, where return type is inferred and `<expr>...` are arguments passed to the call in the standard call convention
- `@embed(<string>)`: include relative file as an array of bytes
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
	some_eca()
	return @inline(foo, 1, 2, 3) - bar(3)
}

$some_eca := fn(): void return @eca(8)

// only for functions with no control flow (if, loop)
$bar := fn(a: uint): uint return a * 2

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
note: this does not work on scalar values

#### generic_functions
```hb
add := fn($T: type, a: T, b: T): T return a + b

main := fn(): uint {
	return add(u32, 2, 2) - add(uint, 1, 3)
}
```

#### die
```hb
main := fn(): never {
	// simply emmits 'un' instruction that immediately terminates the execution
	// the expresion has similar properties to 'return' but does not accept a value
	die
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

malloc := fn(size: uint, align: uint): ?^void return @eca(MALLOC_SYS_CALL, size, align)
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

push := fn($Elem: type, vec: ^Vec(Elem), value: Elem): ?^Elem {
	if vec.len == vec.cap {
		if vec.cap == 0 {
			vec.cap = 1
		} else {
			vec.cap *= 2
		}

		new_alloc := @as(?^Elem, @bitcast(malloc(vec.cap * @sizeof(Elem), @alignof(Elem))))
		if new_alloc == null return null

		src_cursor := vec.data
		dst_cursor := @as(^Elem, new_alloc)
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

#### different_function_destinations
```hb
Stru := struct {a: uint, b: uint}
new_stru := fn(): Stru return .(0, 0)

glob_stru := Stru.(1, 1)

main := fn(): uint {
	glob_stru = new_stru()
	if glob_stru.a != 0 return 300
	glob_stru = .(1, 1)
	glob_stru = @inline(new_stru)
	if glob_stru.a != 0 return 200

	glob_stru = .(1, 1)
	strus := [Stru].(glob_stru, glob_stru, glob_stru)
	i := 0
	loop if i == 3 break else {
		strus[i] = new_stru()
		i += 1
	}
	if strus[2].a != 0 return 100

	strus = [Stru].(glob_stru, glob_stru, glob_stru)
	i = 0
	loop if i == 3 break else {
		strus[i] = @inline(new_stru)
		i += 1
	}
	if strus[2].a != 0 return 10

	return 0
}
```

#### triggering_store_in_divergent_branch
```hb
opaque := fn(): uint {
	return 1 << 31
}

main := fn(): void {
	a := 0
	loop if a >= opaque() break else {
		valid := true
		b := 0
		loop if b >= opaque() break else {
			if b == 1 << 16 {
				valid = false
				break
			}
			b += 1
		}
		if valid == false continue
		a += 1
	}
}
```

#### very_nested_loops
```hb
$W := 200
$H := 200
$MAX_ITER := 20
$ZOOM := 0.5

main := fn(): int {
	mv_x := 0.5
	mv_y := 0.0

	y := 0
	loop if y < H break else {
		x := 0
		loop if x < W break else {
			i := MAX_ITER - 1

			c_i := (2.0 * @floatcast(@itf(@as(i32, @intcast(x)))) - @floatcast(@itf(@as(i32, @intcast(W))))) / (W * ZOOM) + mv_x
			c_r := (2.0 * @floatcast(@itf(@as(i32, @intcast(y)))) - @floatcast(@itf(@as(i32, @intcast(H))))) / (H * ZOOM) + mv_y

			z_i := c_i
			z_r := c_r

			// iteration
			loop if (z_r + z_i) * (z_r + z_i) >= 4 | i == 0 break else {
				// z = z * z + c
				z_i = z_i * z_i + c_i
				z_r = z_r * z_r + c_r
				i -= 1
			}
			//   b g r
			put_pixel(.(x, y), .(@intcast(i), @intcast(i), @intcast(i), 255))
		}
	}

	return 0
}

Color := struct {r: u8, g: u8, b: u8, a: u8}
Vec := struct {x: uint, y: uint}

put_pixel := fn(pos: Vec, color: Color): void {
}
```

#### generic_type_mishap
```hb
opaque := fn($Expr: type, ptr: ^?Expr): void {
}

process := fn($Expr: type): void {
	optional := @as(?Expr, null)
	timer := 1000
	loop if timer > 0 {
		opaque(Expr, &optional)
		if optional != null return {
		}
		timer -= 1
	}
	return
}

main := fn(): void process(uint)
```

#### storing_into_nullable_struct
```hb
StructA := struct {b: StructB, c: ^uint, d: uint}

StructB := struct {g: ^uint, c: StructC}

StructC := struct {c: uint}

optionala := fn(): ?StructA {
	return .(.(&0, .(1)), &0, 0)
}

Struct := struct {inner: uint}

optional := fn(): ?Struct {
	return .(10)
}

do_stuff := fn(arg: uint): uint {
	return arg
}

just_read := fn(s: StructA): void {
}

main := fn(): uint {
	a := optionala()
	if a == null {
		return 10
	}
	a.b.c = .(0)
	just_read(a)
	innera := do_stuff(a.b.c.c)

	val := optional()
	if val == null {
		return 20
	}
	val.inner = 100
	inner := do_stuff(val.inner)
	return innera + inner
}
```

#### scheduling_block_did_dirty
```hb
Struct := struct {
	pad: uint,
	pad2: uint,
}

file := [u8].(255)

opaque := fn(x: uint): uint {
	return file[x]
}

constructor := fn(x: uint): Struct {
	a := opaque(x)
	return .(a, a)
}

main := fn(): void {
	something := constructor(0)
	return
}
```

#### null_check_returning_small_global
```hb
MAGIC := 127
get := fn(file: ^u8): ?uint {
	if *file == MAGIC {
		return MAGIC
	} else {
		return null
	}
}

some_file := [u8].(127, 255, 255, 255, 255, 255)

foo := fn(): ?uint {
	gotten := get(&some_file[0])
	if gotten == null {
		return null
	} else if gotten == 4 {
		return 2
	} else if gotten == MAGIC {
		return 0
	}

	return null
}

main := fn(): uint {
	f := foo()
	if f == null return 100
	return f
}
```

#### null_check_in_the_loop
```hb
A := struct {
	x_change: u8,
	y_change: u8,
	left: u8,
	middle: u8,
	right: u8,
}

return_fn := fn(): ?A {
	return A.(0, 0, 0, 0, 0)
}

main := fn(): int {
	loop {
		ret := return_fn()
		if ret != null {
			return 1
		}
	}
}
```

#### stack_provenance
```hb
main := fn(): uint {
	return *dangle()
}
dangle := fn(): ^uint return &0
```

#### advanced_floating_point_arithmetic
```hb
SIN_TABLE := [f32].(0.0, 0.02454122852291229, 0.04906767432741801, 0.07356456359966743, 0.0980171403295606, 0.1224106751992162, 0.1467304744553617, 0.1709618887603012, 0.1950903220161282, 0.2191012401568698, 0.2429801799032639, 0.2667127574748984, 0.2902846772544623, 0.3136817403988915, 0.3368898533922201, 0.3598950365349881, 0.3826834323650898, 0.4052413140049899, 0.4275550934302821, 0.4496113296546065, 0.4713967368259976, 0.492898192229784, 0.5141027441932217, 0.5349976198870972, 0.5555702330196022, 0.5758081914178453, 0.5956993044924334, 0.6152315905806268, 0.6343932841636455, 0.6531728429537768, 0.6715589548470183, 0.6895405447370668, 0.7071067811865475, 0.7242470829514669, 0.7409511253549591, 0.7572088465064845, 0.773010453362737, 0.7883464276266062, 0.8032075314806448, 0.8175848131515837, 0.8314696123025452, 0.844853565249707, 0.8577286100002721, 0.8700869911087113, 0.8819212643483549, 0.8932243011955153, 0.9039892931234433, 0.9142097557035307, 0.9238795325112867, 0.9329927988347388, 0.9415440651830208, 0.9495281805930367, 0.9569403357322089, 0.9637760657954398, 0.970031253194544, 0.9757021300385286, 0.9807852804032304, 0.9852776423889412, 0.989176509964781, 0.99247953459871, 0.9951847266721968, 0.9972904566786902, 0.9987954562051724, 0.9996988186962042, 1.0, 0.9996988186962042, 0.9987954562051724, 0.9972904566786902, 0.9951847266721969, 0.99247953459871, 0.989176509964781, 0.9852776423889412, 0.9807852804032304, 0.9757021300385286, 0.970031253194544, 0.9637760657954398, 0.9569403357322089, 0.9495281805930367, 0.9415440651830208, 0.9329927988347388, 0.9238795325112867, 0.9142097557035307, 0.9039892931234434, 0.8932243011955152, 0.881921264348355, 0.8700869911087115, 0.8577286100002721, 0.8448535652497072, 0.8314696123025455, 0.8175848131515837, 0.8032075314806449, 0.7883464276266063, 0.7730104533627371, 0.7572088465064847, 0.740951125354959, 0.7242470829514669, 0.7071067811865476, 0.6895405447370671, 0.6715589548470186, 0.6531728429537766, 0.6343932841636455, 0.6152315905806269, 0.5956993044924335, 0.5758081914178454, 0.5555702330196022, 0.5349976198870972, 0.5141027441932218, 0.4928981922297841, 0.4713967368259979, 0.4496113296546069, 0.427555093430282, 0.4052413140049899, 0.3826834323650899, 0.3598950365349883, 0.3368898533922203, 0.3136817403988914, 0.2902846772544624, 0.2667127574748985, 0.2429801799032641, 0.21910124015687, 0.1950903220161286, 0.1709618887603012, 0.1467304744553618, 0.1224106751992163, 0.09801714032956083, 0.07356456359966773, 0.04906767432741797, 0.02454122852291233, 0.0, -0.02454122852291208, -0.04906767432741772, -0.0735645635996675, -0.09801714032956059, -0.1224106751992161, -0.1467304744553616, -0.170961888760301, -0.1950903220161284, -0.2191012401568698, -0.2429801799032638, -0.2667127574748983, -0.2902846772544621, -0.3136817403988912, -0.3368898533922201, -0.3598950365349881, -0.3826834323650897, -0.4052413140049897, -0.4275550934302818, -0.4496113296546067, -0.4713967368259976, -0.4928981922297839, -0.5141027441932216, -0.5349976198870969, -0.555570233019602, -0.5758081914178453, -0.5956993044924332, -0.6152315905806267, -0.6343932841636453, -0.6531728429537765, -0.6715589548470184, -0.6895405447370668, -0.7071067811865475, -0.7242470829514668, -0.7409511253549589, -0.7572088465064842, -0.7730104533627367, -0.7883464276266059, -0.8032075314806451, -0.8175848131515838, -0.8314696123025452, -0.844853565249707, -0.857728610000272, -0.8700869911087113, -0.8819212643483549, -0.8932243011955152, -0.9039892931234431, -0.9142097557035305, -0.9238795325112865, -0.932992798834739, -0.9415440651830208, -0.9495281805930367, -0.9569403357322088, -0.9637760657954398, -0.970031253194544, -0.9757021300385285, -0.9807852804032303, -0.9852776423889411, -0.9891765099647809, -0.9924795345987101, -0.9951847266721969, -0.9972904566786902, -0.9987954562051724, -0.9996988186962042, -1.0, -0.9996988186962042, -0.9987954562051724, -0.9972904566786902, -0.9951847266721969, -0.9924795345987101, -0.9891765099647809, -0.9852776423889412, -0.9807852804032304, -0.9757021300385286, -0.970031253194544, -0.96377606579544, -0.9569403357322089, -0.9495281805930368, -0.9415440651830209, -0.9329927988347391, -0.9238795325112866, -0.9142097557035306, -0.9039892931234433, -0.8932243011955153, -0.881921264348355, -0.8700869911087115, -0.8577286100002722, -0.8448535652497072, -0.8314696123025455, -0.817584813151584, -0.8032075314806453, -0.7883464276266061, -0.7730104533627369, -0.7572088465064846, -0.7409511253549591, -0.724247082951467, -0.7071067811865477, -0.6895405447370672, -0.6715589548470187, -0.6531728429537771, -0.6343932841636459, -0.6152315905806274, -0.5956993044924332, -0.5758081914178452, -0.5555702330196022, -0.5349976198870973, -0.5141027441932219, -0.4928981922297843, -0.4713967368259979, -0.449611329654607, -0.4275550934302825, -0.4052413140049904, -0.3826834323650904, -0.359895036534988, -0.33688985339222, -0.3136817403988915, -0.2902846772544625, -0.2667127574748986, -0.2429801799032642, -0.2191012401568702, -0.1950903220161287, -0.1709618887603018, -0.1467304744553624, -0.122410675199216, -0.09801714032956051, -0.07356456359966741, -0.04906767432741809, -0.02454122852291245)

sin := fn(theta: f32): f32 {
	PI := 3.14159265358979323846
	TABLE_SIZE := @as(i32, 256)
	si := @fti(theta * 0.5 * @itf(TABLE_SIZE) / PI)
	d := theta - @floatcast(@itf(si)) * 2.0 * PI / @itf(TABLE_SIZE)
	ci := si + TABLE_SIZE / 4 & TABLE_SIZE - 1
	si &= TABLE_SIZE - 1
	return SIN_TABLE[@bitcast(si)] + (SIN_TABLE[@bitcast(ci)] - 0.5 * SIN_TABLE[@bitcast(si)] * d) * d
}

main := fn(): int {
	// expected result: 826
	return @fti(sin(1000.0) * 1000.0)
}
```

#### nullable_structure
```hb
Structure := struct {}

BigStructure := struct {a: uint, b: uint}

MidStructure := struct {a: u8}

returner_fn := fn(): ?Structure {
	return .()
}
returner_bn := fn(): ?BigStructure {
	return .(0, 0)
}
returner_cn := fn(): ?MidStructure {
	return .(0)
}

main := fn(): int {
	ret := returner_fn()
	bet := returner_bn()
	cet := returner_cn()
	if ret != null & bet != null & cet != null {
		return 1
	}

	return 0
}
```

#### needless_unwrap
```hb
main := fn(): uint {
	always_nn := @as(?^uint, &0)
	ptr1 := @unwrap(always_nn)
	always_n := @as(?^uint, null)
	ptr2 := @unwrap(always_n)
	return *ptr1 + *ptr2
}
```

#### optional_from_eca
```hb
main := fn(): uint {
	a := @as(?uint, @eca(0, 0, 0, 0))

	if a == null {
		die
	}
	return a
}
```

#### returning_optional_issues
```hb
BMP := 0

get_format := fn(): ?uint {
	return BMP
}

main := fn(): uint {
	fmt := get_format()
	if fmt == null {
		return 1
	} else {
		return fmt
	}
}
```

#### inlining_issues
```hb
main := fn(): void {
	@use("main.hb").put_filled_rect(.(&.(0), 100, 100), .(0, 0), .(0, 0), .(1))
}

// in module: memory.hb

SetMsg := packed struct {a: u8, count: u32, size: u32, src: ^u8, dest: ^u8}
set := fn($Expr: type, src: ^Expr, dest: ^Expr, count: uint): void {
	return @eca(8, 2, &SetMsg.(5, @intcast(count), @intcast(@sizeof(Expr)), @bitcast(src), @bitcast(dest)), @sizeof(SetMsg))
}

// in module: main.hb

Color := struct {r: u8}

Vec2 := fn($Ty: type): type return struct {x: Ty, y: Ty}

memory := @use("memory.hb")

Surface := struct {
	buf: ^Color,
	width: uint,
	height: uint,
}

indexptr := fn(surface: Surface, x: uint, y: uint): ^Color {
	return surface.buf + y * surface.width + x
}

put_filled_rect := fn(surface: Surface, pos: Vec2(uint), tr: Vec2(uint), color: Color): void {
	top_start_idx := @inline(indexptr, surface, pos.x, pos.y)
	bottom_start_idx := @inline(indexptr, surface, pos.x, pos.y + tr.y - 1)
	rows_to_fill := tr.y

	loop if rows_to_fill <= 1 break else {
		@inline(memory.set, Color, &color, top_start_idx, @bitcast(tr.x))
		@inline(memory.set, Color, &color, bottom_start_idx, @bitcast(tr.x))

		top_start_idx += surface.width
		bottom_start_idx -= surface.width
		rows_to_fill -= 2
	}

	if rows_to_fill == 1 {
		@inline(memory.set, Color, &color, top_start_idx, @bitcast(tr.x))
	}

	return
}
```

#### only_break_loop
```hb
memory := @use("memory.hb")

bar := fn(): int {
	loop if memory.inb(0x64) != 0 return 1
}

foo := fn(): void {
	loop if (memory.inb(0x64) & 2) == 0 break
	memory.outb(0x60, 0x0)
}

main := fn(): int {
	@inline(foo)
	return @inline(bar)
}

// in module: memory.hb
inb := fn(f: int): int return f
outb := fn(f: int, g: int): void {
}
```

#### reading_idk
```hb
main := fn(): int {
	a := @as(int, idk)
	return a
}
```

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
u32_to_color := fn(v: u32): Color return @bitcast(u32_to_u32(@bitcast(v)))
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

	return arr[2].u
}
```

#### memory_swap
```hb
U := struct {u: uint, v: uint, g: uint}
main := fn(): uint {
	a := decide(0)
	b := decide(1)

	c := a
	a = b
	b = c

	return b.u + a.u
}

decide := fn(u: uint): U return .(u, 0, 0)
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

#### struct_in_register
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
			_ = request_page(@intcast(remaining))
		} else {
			_ = request_page(0xFF)
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
		*(@as(^[u8; 1024], @bitcast(&back_buffer)) + n) = *@bitcast(&back_buffer)
		n += 1
	}
	return back_buffer[1024 * 2]
}
```

#### global_variable_wiredness
```hb
ports := false

inb := fn(): uint return 0

main := fn(): void {
	if ports {
		ports = inb() == 0x0
	}
}
```

### Just Testing Optimizations

#### null_check_test
```hb
get_ptr := fn(): ?^uint return null

main := fn(): uint {
	ptr := get_ptr()

	if ptr == null {
		return 0
	}

	loop if *ptr != 10 {
		*ptr += 1
	} else break

	return *ptr
}
```

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

	infinite_loop()
	return 0
}

infinite_loop := fn(): void {
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

#### infinite_loop_after_peephole
```hb
main := fn(): uint {
	n := 0
	f := 0
	loop if n != 0 break else {
		f += 1
	}
	return f
}
```

#### aliasing_overoptimization
```hb
Foo := struct {ptr: ^uint, rnd: uint}

main := fn(): uint {
	mem := &2
	stru := Foo.(mem, 0);
	*stru.ptr = 0
	return *mem
}
```

#### global_aliasing_overptimization
```hb
var := 0

main := fn(): uint {
	var = 2
	clobber()
	return var
}

clobber := fn(): void {
	var = 0
}
```

#### overwrite_aliasing_overoptimization
```hb
Foo := struct {a: int, b: int}
Bar := struct {f: Foo, b: int}

main := fn(): int {
	value := Bar.{b: 1, f: .(4, 1)}
	value.f = opaque()
	return value.f.a - value.f.b - value.b
}

opaque := fn(): Foo {
	return .(3, 2)
}
```

#### more_if_opts
```hb
main := fn(): uint {
	opq1 := opaque()
	opq2 := opaque()
	a := 0

	if opq1 == null {
	} else a = *opq1
	if opq1 != null a = *opq1
	//if opq1 == null | opq2 == null {
	//} else a = *opq1
	//if opq1 != null & opq2 != null a = *opq1

	return a
}

opaque := fn(): ?^uint return null
```
