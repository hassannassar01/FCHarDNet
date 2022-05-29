import torch
import argparse
import os
import numpy as np

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict

torch.backends.cudnn.benchmark = True
import cv2


def init_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = get_loader("cityscapes")
    loader = data_loader(
        root="dataset",
        split='test',
        is_transform=True,
        img_size=eval(args.size),
        test_mode=True
    )
    n_classes = 4

    # Setup Model
    model = get_model({"arch": "hardnet"}, n_classes)
    state = convert_state_dict(torch.load(args.model_path, map_location=device)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    return device, model, loader


def demo(args):
    device, model, loader = init_model(args)
    proc_size = eval(args.size)
    cap = cv2.VideoCapture(args.input)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_writer = cv2.VideoWriter(
        'demo.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(1024), int(1024))
    )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            img_raw, decoded = process_img(frame, proc_size, device, model, loader)
            blend = np.concatenate((img_raw, decoded), axis=1)
            vid_writer.write(decoded*75)

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    

def process_img(frame, size, device, model, loader):
    img_resized = cv2.resize(frame, (size[1], size[0]))  # uint8 with RGB mode
    img = img_resized.astype(np.float16)

    # norm
    value_scale = 255
    mean = [0.406, 0.456, 0.485]
    mean = [item * value_scale for item in mean]
    std = [0.225, 0.224, 0.229]
    std = [item * value_scale for item in std]
    img = (img - mean) / std

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    images = img.to(device)
    outputs = model(images)
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    decoded = loader.decode_segmap(pred)

    return img_resized, decoded.astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="icboard",
        help="Path to the saved model",
    )

    parser.add_argument(
        "--size",
        type=str,
        default="1024,1024",
        help="Inference size",
    )

    parser.add_argument(
        "--input", nargs="?", type=str, default=None, help="Path of the input video"
    )
    parser.add_argument(
        "--output", nargs="?", type=str, default="./", help="Path of the output directory"
    )
    args = parser.parse_args()
    demo(args)
