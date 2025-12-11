import numpy as np
from PIL import Image

def preprocess_image(pil_img, target_size=(128,128)):
    # Accepts PIL image, returns numpy array shape (1,h,w,3) float32 scaled 0-1
    img = pil_img.resize(target_size)
    arr = np.array(img).astype('float32') / 255.0
    # if grayscale, convert to 3 channel
    if arr.ndim==2:
        arr = np.stack([arr]*3, axis=-1)
    arr = np.expand_dims(arr, 0)
    return arr


def postprocess_mask(mask):
    # mask can be (128,128,1) or (128,128)
    import numpy as np
    from PIL import Image
    mask_arr = np.array(mask)
    if mask_arr.ndim==3 and mask_arr.shape[-1]==1:
        mask_arr = mask_arr[...,0]
    # normalize
    mask_arr = (mask_arr - mask_arr.min()) / (mask_arr.max() - mask_arr.min() + 1e-8)
    mask_img = Image.fromarray((mask_arr*255).astype('uint8'))
    return mask_img