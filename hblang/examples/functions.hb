
main := ||: int {
	return add_one(10) + add_two(20);
}

add_two := |x: int|: int {
	return x + 2;
}

add_one := |x: int|: int {
	return x + 1;
}


