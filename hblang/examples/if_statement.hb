
main := ||: int {
	return fib(10);
}

fib := |x: int|: int {
	if x <= 2 {
		return 1;
	} else {
		return fib(x - 1) + fib(x - 2);
	}
}
