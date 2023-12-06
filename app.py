# Copyright 2023 ByteDance and/or its affiliates.
#
# Copyright (2023) MagicAnimate Authors
#
# ByteDance, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from ByteDance or
# its affiliates is strictly prohibited.
import argparse
import imageio
import numpy as np
import gradio as gr
from PIL import Image
from subprocess import PIPE, run

from demo.animate import MagicAnimate

from huggingface_hub import snapshot_download

# snapshot_download(repo_id="runwayml/stable-diffusion-v1-5", local_dir="./stable-diffusion-v1-5")
snapshot_download(repo_id="stabilityai/sd-vae-ft-mse", local_dir="./sd-vae-ft-mse")
snapshot_download(repo_id="zcxu-eric/MagicAnimate", local_dir="./MagicAnimate")

animator = MagicAnimate()

def animate(reference_image, motion_sequence_state, seed, steps, guidance_scale):
    return animator(reference_image, motion_sequence_state, seed, steps, guidance_scale)

with gr.Blocks() as demo:

    gr.HTML(
        """
        <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
        <h1 style="font-weight: 800; font-size: 2rem; margin: 0rem">
            MagicAnimate: Temporally Consistent Human Image Animation
        </h1>
        <br>
        <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
            <a href="https://showlab.github.io/magicanimate">Project page</a> | 
            <a href="https://github.com/magic-research/magic-animate"> GitHub </a> | 
            <a href="https://arxiv.org/abs/2311.16498"> arXiv </a>
        </h2>
        </div>
        """)
    animation = gr.Video(format="mp4", label="Animation Results", autoplay=True)
    
    with gr.Row():
        reference_image  = gr.Image(label="Reference Image")
        motion_sequence  = gr.Video(format="mp4", label="Motion Sequence")
        
        with gr.Column():
            random_seed         = gr.Textbox(label="Random seed", value=1, info="default: -1")
            sampling_steps      = gr.Textbox(label="Sampling steps", value=25, info="default: 25")
            guidance_scale      = gr.Textbox(label="Guidance scale", value=7.5, info="default: 7.5")
            submit              = gr.Button("Animate")

    def read_video(video, size=512):
        size = int(size)
        reader = imageio.get_reader(video)
        fps = reader.get_meta_data()['fps']
        assert fps == 25.0, f'Expected video fps: 25, but {fps} fps found'
        return video
    
    def read_image(image, size=512):
        return np.array(Image.fromarray(image).resize((size, size)))
    
    # when user uploads a new video
    motion_sequence.upload(
        read_video,
        motion_sequence,
        motion_sequence
    )
    # when `first_frame` is updated
    reference_image.upload(
        read_image,
        reference_image,
        reference_image
    )
    # when the `submit` button is clicked
    submit.click(
        animate,
        [reference_image, motion_sequence, random_seed, sampling_steps, guidance_scale], 
        animation
    )

    # Examples
    gr.Markdown("## Examples")
    gr.Examples(
        examples=[
            ["inputs/applications/source_image/monalisa.png", "inputs/applications/driving/densepose/running.mp4"],
            ["inputs/applications/source_image/demo4.png", "inputs/applications/driving/densepose/demo4.mp4"],
            ["inputs/applications/source_image/0002.png", "inputs/applications/driving/densepose/demo4.mp4"],
            ["inputs/applications/source_image/dalle2.jpeg", "inputs/applications/driving/densepose/running2.mp4"],
            ["inputs/applications/source_image/dalle8.jpeg", "inputs/applications/driving/densepose/dancing2.mp4"],
            ["inputs/applications/source_image/multi1_source.png", "inputs/applications/driving/densepose/multi_dancing.mp4"],
        ],
        inputs=[reference_image, motion_sequence],
        outputs=animation
    )


demo.launch(share=True)