from typing import Any, Callable

import torch
import torch.nn.functional as F

import comfy.conds
import comfy.patcher_extension
from comfy.model_base import Anima
from comfy.model_patcher import ModelPatcher
from comfy_api.latest import io


# Autogrow interface while allowing up to two reference latent inputs.
# Inputs are processed in numeric slot order.
MAX_REF_LATENTS = 2
COND_REF_LATENTS_KEY = "ref_latents"
TEMPORAL_REFERENCE_WRAPPER_KEY = "cosmos_temporal_reference"


class ApplyCosmosReferenceLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ApplyCosmosReferenceLatent",
            search_aliases=["cosmos reference", "anima reference"],
            display_name="Apply Cosmos Reference Latent",
            category="conditioning",
            inputs=[
                io.Model.Input("model"),
                io.Autogrow.Input(
                    "ref_latents",
                    template=io.Autogrow.TemplateNames(
                        io.Latent.Input("ref_latent"),
                        names=[f"ref_latent_{i}" for i in range(1, MAX_REF_LATENTS + 1)],
                        min=0,
                    ),
                    tooltip=(
                        "Reference latent inputs for generation. "
                        "Autogrow exposes the next input after the preceding input is used. "
                        "Inputs are applied in numeric order, up to 2 latents."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        # Copy instead of mutating the mapping owned by the workflow/runtime.
        ref_latents: dict[str, dict[str, Any]] = dict(kwargs.get("ref_latents") or {})
        if "latent" in kwargs:
            ref_latents["latent_for_compatibility"] = kwargs["latent"]

        ordered_ref_latents = _ordered_reference_latents(ref_latents)

        m = model.clone()
        model_type = type(m.model)

        if issubclass(model_type, Anima):
            extra_conds = m.get_model_object("extra_conds")
            process_latent_in = m.get_model_object("process_latent_in")
            m.add_object_patch(
                "extra_conds",
                cosmos_extra_conds_reference(extra_conds, process_latent_in, ordered_ref_latents,),
            )
            m.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                TEMPORAL_REFERENCE_WRAPPER_KEY,
                cosmos_diffusion_reference_wrapper,
            )

        return io.NodeOutput(m)


def _ordered_reference_latents(
    ref_latents: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return references in stable temporal-slot order.

    Slots are processed as ref_latent_1 followed by ref_latent_2. This avoids
    depending on dictionary insertion order and prevents a later slot from being
    used when an earlier slot is absent.
    """
    first = ref_latents.get("ref_latent_1")
    second = ref_latents.get("ref_latent_2")
    compatibility = ref_latents.get("latent_for_compatibility")

    # Older workflows may provide one reference through the legacy "latent"
    # input. Use it as the first slot only when the named slot is absent.
    if first is None and compatibility is not None:
        first = compatibility

    if second is not None and first is None:
        raise ValueError(
            "ref_latent_2 requires ref_latent_1. Connect ref_latent_1 before "
            "using ref_latent_2."
        )

    ordered: list[dict[str, Any]] = []
    if first is not None:
        ordered.append(first)
    if second is not None:
        ordered.append(second)
    return ordered


def cosmos_extra_conds_reference(
    extra_conds: Callable[..., dict],
    process_latent_in: Callable[..., torch.Tensor],
    ref_latents: list[dict[str, Any]] | None = None,
):
    def _anima_extra_conds_reference(**kwargs):
        out = extra_conds(**kwargs)
        if ref_latents:
            latents = [process_latent_in(latent["samples"]) for latent in ref_latents]
            out[COND_REF_LATENTS_KEY] = comfy.conds.CONDList(latents)

        return out

    return _anima_extra_conds_reference


def _resize_reference_latent(
    ref: torch.Tensor,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Resize a reference latent to x's H x W without preserving aspect ratio."""
    if ref.ndim == 4:
        ref = ref.unsqueeze(2)
    elif ref.ndim != 5:
        raise ValueError(
            "Reference latent must be 4D [B, C, H, W] or 5D [B, C, T, H, W], "
            f"but got shape {tuple(ref.shape)}."
        )

    if ref.shape[-2:] == (target_height, target_width):
        return ref

    batch, channels, temporal, height, width = ref.shape
    frames = ref.permute(0, 2, 1, 3, 4).reshape(
        batch * temporal, channels, height, width
    )
    frames = F.interpolate(
        frames,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    return frames.reshape(
        batch, temporal, channels, target_height, target_width
    ).permute(0, 2, 1, 3, 4)


def cosmos_diffusion_reference_wrapper(executor, *args, **kwargs):
    x: torch.Tensor = args[0]
    x_temporal_dim = x.shape[2]
    target_height, target_width = x.shape[-2:]
    ref_latents: torch.Tensor | None = kwargs.get(COND_REF_LATENTS_KEY)

    newargs = list(args)

    if ref_latents is not None:
        for ref in ref_latents:
            ref = ref.to(dtype=x.dtype, device=x.device)
            ref = _resize_reference_latent(ref, target_height, target_width)
            x = torch.cat([x, ref], dim=2)

    newargs[0] = x

    result = executor(*newargs, **kwargs)[:, :, :x_temporal_dim]

    return result
