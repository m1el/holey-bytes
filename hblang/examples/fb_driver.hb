arm_fb_ptr := ||:int return 100;
x86_fb_ptr := ||:int return 100;


check_platform:= ||: int {

    return x86_fb_ptr();
}

set_pixel := |x: int, y: int, r: u8, g: u8, b: u8|: int := {
    pix_offset := y * width + x;

    return 0;
}

main := ||: int {
	fb_ptr := check_platform();
    
    width := 1024;
    height := 768;
    x:= 0;
    y:= 0;

    loop {
        if x <= height + 1 {
            set_pixel(x,y,100,100,100);
            x= x + 1;
        } else {
            set_pixel(x,y,100,100,100);
            x := 0;
            y = y + 1;
        }
        if y == width {
            break;
        }
    }
    return 0;
}
