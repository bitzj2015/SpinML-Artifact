# Data Generation with Hugging Face Diffusers

This README talks about the steps for data generation using the Hugging Face Diffusers library, focusing on fine-tuning the Stable Diffusion model.
**Note**: Parts of it are taken from the README under examples/dreambooth of the diffusers repository.

## Prerequisites

- **Python**: Version 3.10 or higher.
- **GPU**: A CUDA-compatible GPU is recommended for efficient training and inference.
- **NVIDIA Drivers**: Ensure that the appropriate NVIDIA drivers, CUDA Toolkit, and cuDNN are installed.

## Installation

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then cd in the example folder and run
```bash
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell e.g. a notebook

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

When running `accelerate config`, if we specify torch compile mode to True there can be dramatic speedups.
Note also that we use PEFT library as backend for LoRA training, make sure to have `peft>=0.6.0` installed in your environment.

For more detailed information, refer to the [DreamBooth documentation](https://huggingface.co/docs/diffusers/main/en/training/dreambooth).  

## Fine-Tuning Stable Diffusion
The `train_dreambooth.py` script shows how to implement the training procedure and adapt it for stable diffusion.
To fine-tune the Stable Diffusion model with your dataset:

1. **Prepare Your Dataset**:
   - Organize your training images in a directory (e.g., `dataset/dog`).
   - Ensure each image is appropriately labeled and in a supported format (e.g., PNG or JPEG).

2. **Configure the Training Script arguements**:
   - Assign the necessary training arguements for `train_dreambooth.py` scripts.
   - Adjust other hyperparameters such as batch size, learning rate, and the number of training steps as needed.

3. **Run the Fine-Tuning Script**:
    ```bash
    export MODEL_NAME="CompVis/stable-diffusion-v1-4"
    export INSTANCE_DIR="dataset/dog" # The dataset you prepared in Step 2.
    export OUTPUT_DIR="path-to-save-model"
    
    accelerate launch train_dreambooth.py \
      --pretrained_model_name_or_path=$MODEL_NAME  \
      --instance_data_dir=$INSTANCE_DIR \
      --output_dir=$OUTPUT_DIR \
      --instance_prompt="a photo of sks dog" \
      --resolution=512 \
      --train_batch_size=1 \
      --gradient_accumulation_steps=1 \
      --learning_rate=5e-6 \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --max_train_steps=400 \
      --push_to_hub
    ```
   *Note*: Adjust the `--pretrained_model_name_or_path` to the specific Stable Diffusion model version you intend to fine-tune. For our experiments, we used Stable Diffusion v1.4. 

4. **Monitor Training**:
   - Training logs and outputs will be saved in the specified `--output_dir`.
   - Utilize tools like TensorBoard to visualize training progress and metrics.

## Generating Images with the Fine-Tuned Model

Once you have fine-tuned a model using the above command, you can run inference simply using the `StableDiffusionPipeline`. Make sure to include the `identifier` (e.g. sks in above example) in your prompt.

1. **Load the Fine-Tuned Model**:
   ```python
   from diffusers import StableDiffusionPipeline
   import torch

   model_path = "./output"
   pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
   pipe.to("cuda")
   ```

2. **Generate Images**:
   ```python
   prompt = "A photo of sks dog eating."
   image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
   image.save("dog_eating.png")
   ```
   Experiment with different prompts to produce a variety of images.

## Additional Resources

- **Hugging Face Diffusers Documentation**: [https://huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers)
- **Stable Diffusion Fine-Tuning Guide**: [https://huggingface.co/docs/diffusers/training/text2image](https://huggingface.co/docs/diffusers/training/text2image)
