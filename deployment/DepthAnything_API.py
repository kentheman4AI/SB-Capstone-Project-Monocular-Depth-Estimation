import io
import os
import cv2
import numpy as np
import torch
import argparse
import glob
import matplotlib
from flask import Flask, render_template, send_file, request, jsonify, Response, stream_with_context

from depth_anything_v2.dpt import DepthAnythingV2


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("DepthAnything_FT_GUI.html")

# ----- Mapping for User Options -----
encoder_mapping = {
    "mirror": "vitlm",
    "indoor art": "vitli",
    "outdoor art": "vitlo"
}

# ----- Model configurations for each encoder -----
# (These are example configurations; adjust features and out_channels as needed.)
model_configs = {
    'vitlm': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitli': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitlo': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

# ----- Mapping for checkpoint files -----
# (Update the checkpoint paths to point to your actual fine-tuned model files.)
checkpoint_mapping = {
    "mirror": "checkpoints/depth_anything_v2_metric_hypersim_vitlm.pth",
    "indoor art": "checkpoints/depth_anything_v2_metric_hypersim_vitli.pth",
    "outdoor art": "checkpoints/depth_anything_v2_metric_vkitti_vitlo.pth",
    "vkitti original": "checkpoints/depth_anything_v2_metric_vkitti_vitl.pth",
    "hypersim original": "checkpoints/depth_anything_v2_metric_hypersim_vitl.pth"
}

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(option,user_pretrained):
    """
    Loads the DepthAnythingV2 model for the given option.
    Option should be one of: "mirror", "indoor art", "outdoor art".
    """
    encoder_key = encoder_mapping.get(option.lower(), "vitlm")
    config = model_configs[encoder_key]
    if user_pretrained=="true":
        if encoder_key=="vitlo":
            checkpoint_path = checkpoint_mapping.get("vkitti original")
        else:
            checkpoint_path = checkpoint_mapping.get("hypersim original")
    else:
        checkpoint_path = checkpoint_mapping.get(option.lower())

    MAX_DEPTH = 80 if encoder_key == 'vitlo' else 20
    # Initialize the model using the appropriate configuration and max_depth.
    model = DepthAnythingV2(**{**config, 'max_depth': MAX_DEPTH})

    # Load the checkpoint (assuming the state dict is stored at the top level)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE)
    model.eval()
    return model


def generate_image_chunks(buffer, chunk_size=4096):
    data = buffer.getvalue()
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]


@app.route("/infer", methods=["POST"])
def infer():
    # Check if a file was provided
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    # Get the uploaded file and the user option (default to "mirror")
    file = request.files["file"]
    user_augment = request.form.get("augment", "mirror").lower()
    user_otype = request.form.get("otype", "grayscale").lower()
    user_pretrained = request.form.get("pretrained", "false").lower()

    cmap = matplotlib.colormaps.get_cmap('Spectral')

    # Load the image from the uploaded file
    file_bytes = np.frombuffer(file.read(), np.uint8)
    raw_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if raw_image is None:
        return jsonify({"error": "Invalid image file"}), 400

    # Load the model based on user option
    try:
        model = load_model(user_augment,user_pretrained)
    except Exception as e:
        return jsonify({"error": f"Model loading failed: {str(e)}"}), 500

    # Set an input size; you can also allow this as an optional form parameter.
    input_size = 518

    # Run inference using the model's infer_image method.
    # (Assuming that infer_image takes (raw_image, input_size) and returns a numpy depth array.)
    try:
        depth = model.infer_image(raw_image, input_size)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    # Normalize the depth map to the 0-255 range and convert to uint8.
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_norm = depth_norm.astype(np.uint8)

    # For visualization, choose grayscale or colormap per user option user_otype
    if user_otype=='grayscale':
        depth_img = np.repeat(depth_norm[..., np.newaxis], 3, axis=-1)
    else:
        depth_img = (cmap(depth_norm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    # Create a white separator (50 pixels wide)
    separator = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
    # Combine the original image and the depth image side by side
    combined_result = cv2.hconcat([raw_image, separator, depth_img])

    # Encode the combined image as PNG in memory
    success, buffer = cv2.imencode(".png", combined_result)
    if not success:
        return jsonify({"error": "Failed to encode output image"}), 500

    # Create a BytesIO buffer from the encoded image
    io_buf = io.BytesIO(buffer.tobytes())
    # Return a streaming response using our generator function
    return Response(
        stream_with_context(generate_image_chunks(io_buf)),
        mimetype="image/png",
        headers={"Content-Disposition": "attachment; filename=output.png"}
    )


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)