
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

	return pixel.point.x + pixel.point.y + pixel.color.r
		+ pixel.color.g + pixel.color.b + pixel.color.a;
}
