import numpy as np

import igl
res = 1024
range_res = np.arange(res)
tex_map = np.stack(np.meshgrid(range_res, range_res),-1)/1024 # (N,N,2)


tex_im = np.concatenate([tex_map,np.zeros((1024,1024,1))],-1)

im = (tex_im*65536).astype(np.uint16)

# breakpoint()

from numpngw import write_png
write_png('test.png', im)
# import cv2
# image = cv2.imread('result2.png', cv2.IMREAD_UNCHANGED)
# tc1
# ft1