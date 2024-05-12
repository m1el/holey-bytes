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
