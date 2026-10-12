"""Explicit inference cache with bounded decoder evidence and error-safe abort."""

from collections.abc import Callable

import torch

from tools.clip_plain.policy import boot_id, evaluation_policy


def guard_device(device: torch.device, scope: dict | None) -> None:
    if device.type == "cpu":
        return
    if (
        device.type != "cuda"
        or scope is None
        or scope.get("approved") is not True
        or scope.get("boot_id") != boot_id()
        or scope.get("consumed", False)
    ):
        raise RuntimeError("Reviewed active GPU scope required before device operations")


def memory_snapshot(device: torch.device) -> dict:
    if device.type == "cpu":
        return {
            k: None
            for k in [
                "allocated_bytes",
                "reserved_bytes",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            ]
        }
    return dict(
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
    )


@torch.no_grad()
def generate_cached_text(
    language,
    visual: torch.Tensor,
    visual_mask: torch.Tensor,
    prompts: list[str],
    *,
    on_event: Callable[[dict], None],
    scope: dict | None = None,
) -> dict:
    guard_device(visual.device, scope)
    if (
        visual.ndim != 3
        or not len(visual)
        or len(prompts) != len(visual)
        or visual_mask.dtype != torch.bool
        or visual_mask.shape != visual.shape[:2]
        or visual_mask.device != visual.device
        or not visual_mask.any(1).all()
    ):
        raise ValueError("Visual tensor, boolean mask and prompts must match")
    if not torch.isfinite(visual[visual_mask]).all():
        raise ValueError("Nonfinite valid visual token")
    flags = [(m, m.training) for m in language.modules()]
    base = (
        language.model.get_base_model()
        if hasattr(language.model, "get_base_model")
        else language.model
    )
    step = [0]
    handles = []

    def before(module, args, kwargs):
        ids = kwargs.get("decoder_input_ids")
        step[0] += 1
        on_event(
            dict(
                phase="decoder_start",
                step=step[0],
                decoder_length=ids.shape[1],
                beam_rows=ids.shape[0],
            )
        )

    def after(module, args, output):
        logits = output.logits[:, -1, :]
        finite = bool(torch.isfinite(logits).all())
        on_event(
            dict(
                phase="decoder_logits",
                step=step[0],
                shape=list(logits.shape),
                dtype=str(logits.dtype),
                stride=list(logits.stride()),
                finite=finite,
            )
        )
        if not finite:
            raise FloatingPointError("Nonfinite raw decoder logits")

    try:
        language.eval()
        joint, mask = language._joint_inputs(visual, visual_mask, prompts)
        on_event(
            dict(
                phase="generation_start",
                joint_shape=list(joint.shape),
                joint_lengths=mask.sum(1).tolist(),
                effective_beam_sequences=len(prompts) * 5,
                decoding=evaluation_policy(),
            )
        )
        handles.append(base.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(base.register_forward_hook(after))
        ids = language.model.generate(
            inputs_embeds=joint,
            attention_mask=mask,
            num_beams=5,
            max_length=64,
            do_sample=False,
            use_cache=True,
        )
        texts = language.tokenizer.batch_decode(ids, skip_special_tokens=True)
        if len(texts) != len(prompts):
            raise ValueError("Prediction count mismatch")
        rows = ids.detach().cpu().tolist()
        return dict(
            texts=[t.lower() for t in texts],
            token_ids=rows,
            output_lengths=[len(r) for r in rows],
            decoding=evaluation_policy(),
        )
    except Exception as error:
        try:
            on_event(
                dict(
                    phase="generation_error",
                    step=step[0],
                    error_type=type(error).__name__,
                    error=str(error),
                )
            )
        except Exception as observer_error:
            error.add_note(f"Error telemetry failed: {observer_error}")
        raise
    finally:
        for handle in handles:
            handle.remove()
        for module, flag in flags:
            module.training = flag
