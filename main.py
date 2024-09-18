import gradio as gr
import subprocess
import os
import tempfile
import shutil
import torch
import cv2

from PIL import Image
from torchvision import transforms

import torch.nn.functional as F
import numpy as np

class Config:
    CHECKPOINTS_DIR = os.path.join(".", "checkpoints")

class ModelManager:
    @staticmethod
    def load_model(checkpoint_name: str, cuda):
        if checkpoint_name is None:
            return None
        checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, checkpoint_name)
        model = torch.jit.load(checkpoint_path)
        model.eval()
        if cuda:
            model.to("cuda")
        return model

    @staticmethod
    @torch.inference_mode()
    def run_model(model, input_tensor, height, width):
        output = model(input_tensor)
        return F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)

class ImageProcessor:
    def __init__(self):
        self.transform_fn = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.5/255, 116.5/255, 103.5/255], std=[58.5/255, 57.0/255, 57.5/255]),
        ])
        self.cuda = torch.cuda.is_available()
        self.seg_model = ModelManager.load_model("sapiens_1b_seg_foreground_epoch_8_torchscript.pt2", self.cuda)

    def process_image(self, image: Image.Image):
        input_tensor = self.transform_fn(image).unsqueeze(0)
        if self.cuda:
            input_tensor.to("cuda")
            
        seg_output = ModelManager.run_model(self.seg_model, input_tensor, image.height, image.width)
        seg_mask = (seg_output.argmax(dim=1) > 0).float().cpu().numpy()[0]
            
        image_array = np.array(image)
        image_array[seg_mask == 0] = 0
        image_array[seg_mask != 0] = 255

        return image_array

def ffmpeg(cmd, progress, total_frames, process_start, process_end):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            
    for line in process.stdout:
        print(line.strip())
        if "frame=" in line:
            parts = line.split()
            try:
                current_frame = int(parts[1])
                progress(process_start + (current_frame / total_frames) * (process_end - process_start), desc="Converting")
            except ValueError:
                pass
    
    process.wait()
    
    if process.returncode != 0:
        print(cmd)
        return False
    
    return True


def process(video, projection, progress=gr.Progress()):
    if video is None:
        return None, "No video uploaded"
    
    progress(0, desc="Starting conversion")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        original_filename = os.path.basename(video.name)
        file_name, file_extension = os.path.splitext(original_filename)
        
        temp_input_path = os.path.join(temp_dir, original_filename)
        shutil.copy(video.name, temp_input_path)

        cap = cv2.VideoCapture(temp_input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        output_filename = f"{file_name}-fisheye.{file_extension}"
        output_path = os.path.join(temp_dir, output_filename)
        # final_output_path = os.path.join(os.getcwd(), output_filename)

        if str(projection) == "eq":
            cmd = [
                "ffmpeg",
                "-i", temp_input_path,
                "-filter_complex",
                "[0:v]split=2[left][right]; [left]crop=ih:ih:0:0[left_crop]; [right]crop=ih:ih:ih:0[right_crop]; [left_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[leftfisheye]; [right_crop]v360=hequirect:fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180[rightfisheye]; [leftfisheye][rightfisheye]hstack[v]",
                "-map", "[v]",
                "-c:a", "copy",
                "-crf", "16",
                output_path
            ]
            projection = "fisheye180"
            if not ffmpeg(cmd, progress, total_frames, 0.0, 0.2):
                return None, "Convertion 1 failed"
            
        else:
            output_path = temp_input_path

        progress(0.2, desc="Conversion 1 complete")


        image_processor = ImageProcessor()
        cap = cv2.VideoCapture(output_path)

        mask_video = file_name + "-alpha.avi"
        out = cv2.VideoWriter(
            mask_video,
            cv2.VideoWriter_fourcc(*'MJPG'),
            cap.get(cv2.CAP_PROP_FPS), 
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        )
        current_frame = 0
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break
            current_frame += 1
            progress(0.2 + (current_frame / total_frames) * 0.6, desc=f"Converting {current_frame}/{total_frames}")

            progress(1, desc="Conversion 1 complete")
            _, width = img.shape[:2]
            imgL = img[:, :int(width/2)]
            imgR = img[:, int(width/2):]
            imgL = image_processor.process_image(Image.fromarray(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)))
            imgR = image_processor.process_image(Image.fromarray(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)))
            combined_image = cv2.hconcat([imgL, imgR])
            _, binary = cv2.threshold(combined_image, 127, 255, cv2.THRESH_BINARY)
            out.write(binary)

        cap.release()
        out.release()

        del image_processor

        result_name = file_name + "_" + projection + "_alpha" + file_extension 
        cmd = [
            "ffmpeg",
            "-i", output_path,
            "-i", mask_video,
            "-i", "mask.png",
            "-i", temp_input_path,
            "-filter_complex",
            "[1]scale=iw*0.4:-1[alpha];[2][alpha]scale2ref[mask][alpha];[alpha][mask]alphamerge,split=2[masked_alpha1][masked_alpha2]; [masked_alpha1]crop=iw/2:ih:0:0,split=2[masked_alpha_l1][masked_alpha_l2]; [masked_alpha2]crop=iw/2:ih:iw/2:0,split=4[masked_alpha_r1][masked_alpha_r2][masked_alpha_r3][masked_alpha_r4]; [0][masked_alpha_l1]overlay=W*0.5-w*0.5:-0.5*h[out_lt];[out_lt][masked_alpha_l2]overlay=W*0.5-w*0.5:H-0.5*h[out_tb]; [out_tb][masked_alpha_r1]overlay=0-w*0.5:-0.5*h[out_l_lt];[out_l_lt][masked_alpha_r2]overlay=0-w*0.5:H-0.5*h[out_tb_ltb]; [out_tb_ltb][masked_alpha_r3]overlay=W-w*0.5:-0.5*h[out_r_lt];[out_r_lt][masked_alpha_r4]overlay=W-w*0.5:H-0.5*h",
            "-c:v", "libx265", 
            "-crf", "16",
            "-preset", "veryfast",
            "-map", "3:a:?",
            "-c:a", "copy",
            result_name,
            "-y"
        ]

        if not ffmpeg(cmd, progress, total_frames, 0.8, 1.0):
            return None, "Convertion 2 failed"

        progress(1, desc="Conversion 2 complete")

    return result_name, f"Conversion successful"

def process_video(video, projection):
    output_path, message = process(video, projection)
    if output_path:
        return gr.File.update(value=output_path, visible=True), message
    else:
        return gr.File.update(visible=False), message

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Video VR2AR Converter")
    with gr.Row():
        input_video = gr.File(label="Upload Video (MKV or MP4)", file_types=["mkv", "mp4"])
        projection_dropdown = gr.Dropdown(choices=["eq", "fisheye180", "fisheye190", "fisheye200"], label="VR Video Format", value="eq")
        output_video = gr.File(label="Download Converted Video", visible=False)
    convert_button = gr.Button("Convert")
    status = gr.Textbox(label="Status")
    
    convert_button.click(
        fn=process_video,
        inputs=[input_video, projection_dropdown],
        outputs=[output_video, status]
    )

if __name__ == "__main__":
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    demo.launch(server_name="0.0.0.0", server_port=7860)
