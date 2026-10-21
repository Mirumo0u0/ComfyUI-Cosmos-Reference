# ComfyUI-Cosmos-Reference
Inspired by https://github.com/levzzz5154/ComfyUI/tree/ref-latents-support.  

  
Add an image reference feature to the Cosmos model or models based on it, such as the Anima model.  
Essentially, it concatenates the reference image to the latent space. This approach has poor performance, but it preserves the information of the reference image to the greatest extent.  
Thanks to [@pamparamm](https://github.com/pamparamm) for optimizing the node code

## Dynamic reference inputs

`Apply Cosmos Reference Latent` uses an Autogrow input and supports up to two reference latents:

- `ref_latent_1`: first reference latent.
- `ref_latent_2`: second reference latent. Autogrow exposes this input after the preceding input is used.

Temporal order is fixed as:

1. generation latent
2. `ref_latent_1`
3. `ref_latent_2`, when connected

Existing workflows using one reference retain the original two-slot layout.
`ref_latent_2` requires `ref_latent_1` to be connected.
