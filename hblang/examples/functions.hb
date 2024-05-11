
main := fn(): int {
	return add_one(10) + add_two(20);
}

add_two := fn(x: int): int {
	return x + 2;
}

add_one := fn(x: int): int {
	return x + 1;
}


