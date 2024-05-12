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
		n = n - 1;
		continue;
	}
	return a;
}
