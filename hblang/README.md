# HERE SHALL THE DOCUMENTATION RESIDE

### Examples

Examples are also used in tests, to and an example that runs during testing add:
<pre>
#### &ls;name&gt
```hb
&lt;example&gt
```
</pre>
and also:
```rs
<name> => README;
```
to the `run_tests` macro at the bottom of the `src/codegen.rs`.

#### directives
```hb
Type := struct {
	brah: int,
	blah: int,
}

main := fn(): int {
	byte := @as(u8, 10);
	same_type_as_byte := @as(@TypeOf(byte), 30);
	wide_uint := @as(u32, 40);
	truncated_uint := @as(u8, @intcast(wide_uint));
	size_of_Type_in_bytes := @sizeof(Type);
	align_of_Type_in_bytes := @alignof(Type);
	hardcoded_pointer := @as(^u8, @bitcast(10));
	ecall_that_returns_int := @eca(int, 1, Type.(10, 20), 5, 6);
	return 0;
}
```

#### if_statements
```hb
main := fn(): int {
	return fib(10);
}

fib := fn(x: int): int {
	if x <= 2 {
		return 1;
	} else {
		return fib(x - 1) + fib(x - 2);
	}
}
```

#### global_variables
```hb
global_var := 10;

complex_global_var := fib(global_var) - 5;

fib := fn(n: int): int {
	if 2 > n {
		return n;
	}
	return fib(n - 1) + fib(n - 2);
}

main := fn(): int {
	return complex_global_var;
}

```

#### generic_types
```hb
Vec := fn($Elem: type): type {
	return struct {
		data: ^Elem,
		len: uint,
		cap: uint,
	};
}

main := fn(): int {
	i := 69;
	vec := Vec(int).{
		data: &i,
		len: 1,
		cap: 1,
	};
	return *vec.data;
}
```

#### fb_driver
```hb
arm_fb_ptr := fn(): int return 100;
x86_fb_ptr := fn(): int return 100;


check_platform := fn(): int {
    return x86_fb_ptr();
}

set_pixel := fn(x: int, y: int, width: int): int {
    pix_offset := y * width + x;

    return 0;
}

main := fn(): int {
    fb_ptr := check_platform();
    width := 100;
    height := 30;
    x:= 0;
    y:= 0;

    loop {
        if x <= height + 1 {
            set_pixel(x,y,width);
            x = x + 1;
        } else {
            set_pixel(x,y,width);
            x = 0;
            y = y + 1;
        }
        if y == width {
            break;
        }
    }
    return 0;
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
	a := Point.(1, 2);
	b := Point.(3, 4);

	d := Rect.(a + b, b - a);
	d2 := Rect.(Point.(0, 0) - b, a);
	d2 = d2 + d;

	c := d2.a + d2.b;
	return c.x + c.y;
}
```

#### main_fn
```hb
main := fn(): int {
	return 1;
}
```

#### structs
```hb
Ty := struct {
	a: int,
	b: int,
}

Ty2 := struct {
	ty: Ty,
	c: int,
}

main := fn(): int {
	finst := Ty2.{ ty: Ty.{ a: 4, b: 1 }, c: 3 };
	inst := odher_pass(finst);
	if inst.c == 3 {
		return pass(&inst.ty);
	}
	return 0;
}

pass := fn(t: ^Ty): int {
	return t.a - t.b;
}

odher_pass := fn(t: Ty2): Ty2 {
	return t;
}
```

#### pointers
```hb
main := fn(): int {
	a := 1;
	b := &a;
	modify(b);
	drop(a);
	stack_reclamation_edge_case := 0;
	return *b - 2;
}

modify := fn(a: ^int): void {
	*a = 2;
	return;
}

drop := fn(a: int): void {
	return;
}
```

#### functions
```hb

main := fn(): int {
	return add_one(10) + add_two(20);
}

add_two := fn(x: int): int {
	return x + 2;
}

add_one := fn(x: int): int {
	return x + 1;
}


```

#### variables
```hb
main := fn(): int {
	a := 1;
	b := 2;
	a = a + 1;
	return a - b;
}
```

#### loops
```hb
main := fn(): int {
	return fib(10);
}

fib := fn(n: int): int {
	a := 0;
	b := 1;
	loop {
		if n == 0 {
			break;
		}
		c := a + b;
		a = b;
		b = c;
		n -= 1;

		stack_reclamation_edge_case := 0;

		continue;
	}
	return a;
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
	};

	if *(&pixel.point.x + 1) != 2 {
		return 0;
	}

	if *(&pixel.point.y - 1) != 0 {
		return 64;
	}

	return pixel.point.x + pixel.point.y + pixel.color.r
		+ pixel.color.g + pixel.color.b + pixel.color.a;
}
```

#### arithmetic
```hb
main := fn(): int {
	return 10 - 20 / 2 + 4 * (2 + 2) - 4 * 4 + 1;
}
```

