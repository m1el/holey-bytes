global_var := 10;

complex_global_var := fib(global_var) - 5;

fib := fn(n: int): int {
	if n <= 2 {
		return n;
	}
	return fib(n - 1) + fib(n - 2);
}

main := fn(): int {
	return complex_global_var;
}
