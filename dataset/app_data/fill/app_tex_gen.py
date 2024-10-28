import numpy as np


def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

def point_in_triangle(pt, v1, v2, v3):
    b1 = sign(pt, v1, v2) < 0.0
    b2 = sign(pt, v2, v3) < 0.0
    b3 = sign(pt, v3, v1) < 0.0
    return ((b1 == b2) and (b2 == b3))

def draw_filled_triangle(image, v1, v2, v3, color):
    min_x = int(min(v1[0], v2[0], v3[0]))
    max_x = int(max(v1[0], v2[0], v3[0]))
    min_y = int(min(v1[1], v2[1], v3[1]))
    max_y = int(max(v1[1], v2[1], v3[1]))

    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if point_in_triangle((x, y), v1, v2, v3):
                image[y, x] = color


# A-B
# |
# C
def tex_gen():
    corner = np.array([[0.7,0.6],[0.9,0.6],[0.7,0.4]])

    v1 = corner[1] - corner[0]
    v2 = corner[2] - corner[0]
    p4 = corner[0]+v1+v2


    test_im = np.zeros((1024,1024,3)).astype(np.uint8)

    corner_1024 = corner * 1024


    draw_filled_triangle(test_im, corner_1024[0], corner_1024[1], corner_1024[2], [255, 0, 0])
    draw_filled_triangle(test_im,  corner_1024[1], corner_1024[2], p4*1024, [255, 0, 0])

    # dir = 'app_tex'
    # imageio.imsave(os.path.join(dir,"test.png"),np.flip(test_im,0))

    res = 1024

    mx,my = np.meshgrid(np.arange(res),np.arange(res))

    div_x = 10
    div_y = 10

    mx = mx / res
    my = my / res


    mx_f, mx_i = np.modf((mx * div_x))
    my_f, my_i  = np.modf((my * div_y))
    # x = mx_f
    # y = my_f
    x = (-1*mx_f+1) * (mx_i % 2) + mx_f* ((mx_i+1) % 2)
    y = (-1*my_f+1) * (my_i % 2) + my_f* ((my_i+1) % 2)

    # breakpoint()
    c_val_r = (v1[0] * x) + (v2[0] * y) + corner[0][0]
    c_val_g = (v1[1] * x) + (v2[1] * y) + corner[0][1]

    c_val = np.stack([c_val_r,c_val_g,np.zeros_like(c_val_r)],-1)

    return c_val
# imageio.imsave(os.path.join(dir,'tex.png'),(c_val*255).astype(np.uint8))
# np.save(os.path.join(dir,'scale.npy'), np.stack([c_min,c_max]))
    