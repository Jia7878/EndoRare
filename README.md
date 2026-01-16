# EndoRare

## Hardware Configuration
- GPU: NVIDIA GeForce RTX 4090
- Recommended CUDA Version: 11.7 or higher

## Environment Dependencies

### Software Environment
- Python 3.8+
- PyTorch
- CLIP
- DeepFloyd
- PyYAML
- LDM

### Installation Dependencies
```bash  
conda create -n EndoRare python=3.8  
conda activate EndoRare  

# Install PyTorch   
pip install torch torchvision  
pip install clip pyyaml deepfloyd
```

## Data Processing
All images are resized to 256x256 pixels

## Training Encoder Script
```bash 
python scripts/train_clip_inversion.py \
    -c configs/train_deepfloyd_inversion.yaml \
    -t polyp_7 \
    training.optimizers.embeddings.kwargs.lr=0.0002 \
    shared_tokens=0 \
    gt_init=0 \
    fruit_blip_coeff=0.00001 \
    mat_blip_coeff=0.00001 \
    color_blip_coeff=0.00001 \
    blip_guidance=0 \
    num_placeholder_groups=3 \
    num_placeholder_words=215
```
##  PSE Script
```bash
python endorare_pse.py
```
