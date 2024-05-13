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

pass := fn(t: *Ty): int {
	return t.a - t.b;
}

odher_pass := fn(t: Ty2): Ty2 {
	return t;
}
