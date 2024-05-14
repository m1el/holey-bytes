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
