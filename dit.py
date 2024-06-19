# Short write-up on how Diffusion Transformers work


# DiT (diffusion transformer) is a new approach for generating images
# it uses both diffusion models and transformers to create high-quality images
# its key part is the diffusion process where a noisy image is denoised over several steps to produce a clean image
# this uses a model that predicts the noise at each step -> generates detailed and realistic images
# it also, since it uses a transformer, generates images in batches of patches


# why it's built different (literally)

# 1. GANs (generative aversarial networks) generate images in one go
#    DiT is a diffusion process that iteratively refines the image -> detailed results
#    also each diffusion step allows patches to communicate w each other -> more cohesive results

# 2. U-NET backbone is only really good at capturing local data
#    but DiTs have transformer backbones for better handling of long-range
#    dependencies and richer contextual information

# 3. the process of noise prediction each step of the diffusion process enables
#    more controlled and stable image generation compared to the adversarial training of GANs

# 4. ViTs use transformers to create the image autoregressively -> patch by patch
#    since each patch is more or less finalized there's not as much cohesiveness -> this trades off for high detail tho

# overall DiT offers superior image quality and versatility compared to previous methods
# tts combination of diffusion processes with transformer models is goated