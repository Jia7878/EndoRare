# Import the PolypDiffusionPipeline class from the file where it is defined
from polypdiffusion.polyp_diffusion_pipeline import PolypDiffusionPipeline

# Now you can use PolypDiffusionPipeline in this file
def main():
    # Example usage of the pipeline
    config_path = '/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/configs/latent-diffusion/re_sd_mask.yaml'
    ckpt_path = '/home/hyl/yujia/conditional-polyp-diffusion/latent-diffusion/resume_sd/2024-11-21T00-12-28_stable-diffusion/checkpoints/trainstep_checkpoints/epoch=49-step=699.ckpt'
    
    pipeline = PolypDiffusionPipeline(config_path=config_path, ckpt_path=ckpt_path, batch_size=1, seed=55)
    
    # Run the pipeline and get results
    results = pipeline.get_results()

    # Print or use the results
    print(f"Real Image saved at: {results['real_image']}")
    print(f"Generated Image saved at: {results['fake_image']}")

if __name__ == "__main__":
    main()
