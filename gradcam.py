import cv2
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_heatmap(
    model,
    image_tensor,
    image_path,
    output_path
):

    target_layer = model.features[-1]

    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    grayscale_cam = cam(
        input_tensor=image_tensor
    )[0]

    rgb_img = cv2.imread(image_path)

    rgb_img = cv2.resize(
        rgb_img,
        (224,224)
    )

    rgb_img = np.float32(
        rgb_img
    ) / 255

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    cv2.imwrite(
        output_path,
        visualization
    )

    return output_path