"""Select the matching training implementation when replaying a checkpoint."""

from __future__ import annotations


def evaluation_components(config):
    model = config["model"]
    if str(model.get("backbone", "")).startswith("dinov3_convnext_"):
        from scripts import train_dinov3_multilabel as backend

        if "decoder_mechanism" in model:
            from scripts.train_dino_decoder_mechanism import build_model

            return (backend.build_loader, lambda cfg: build_model(cfg, load_pretrained=False),
                    backend.build_loss, backend.run_epoch)
        return (backend.build_loader, lambda cfg: backend.build_model(cfg, load_pretrained=False),
                backend.build_loss, backend.run_epoch)
    from scripts import train_unet_multilabel as backend

    factory = backend.build_model
    if model.get("name") == "stair_pooling_controlled":
        from scripts.train_stair_pooling_multilabel import build_model

        factory = build_model
    elif model.get("name") == "depth_space_controlled":
        from scripts.train_depth_space_multilabel import build_model

        factory = build_model
    elif model.get("name") == "wavelet_decoder_controlled":
        from scripts.train_wavelet_decoder_multilabel import build_model

        factory = build_model
    elif model.get("name") == "deformable_bottleneck_controlled":
        from scripts.train_deformable_bottleneck_multilabel import build_model

        factory = build_model
    elif model.get("name") == "prototype_regularized_controlled":
        from scripts.train_prototype_regularized_multilabel import build_model

        factory = build_model
    elif model.get("name") == "component_supervised_controlled":
        from scripts.train_component_supervised_multilabel import build_model

        factory = build_model
    elif model.get("name") == "bidirectional_component_controlled":
        from scripts.train_bidirectional_component_multilabel import build_model

        factory = build_model
    elif model.get("name") == "cc_ground_truth_followup_controlled":
        from scripts.train_cc_ground_truth_followup import build_model

        factory = build_model
    return backend.build_loader, factory, backend.build_loss, backend.run_epoch
