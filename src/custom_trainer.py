# src/custom_trainer.py

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union, Any

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import is_sagemaker_mp_enabled
from transformers.file_utils import is_apex_available

if is_apex_available():
    from apex import amp

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel.torch.state_mod import smp_forward_only

from .stablemax import StableMax, LogStableMax
from .orthogonal_optimizer import with_orthogonal_gradient


##############################################################################
# Model Configuration Mapping
##############################################################################

MODEL_CONFIG_MAPPING = {
    "gpt2": {
        "stable_max_layer_name": "lm_head",
        "skip_orthogonal_param_types": ["bias", "LayerNorm"],
    },
    "LlamaForCausalLM": {
        "stable_max_layer_name": "lm_head",
        "skip_orthogonal_param_types": ["bias", "LayerNorm"],
    },
    "Qwen2ForCausalLM": {
        "stable_max_layer_name": "lm_head",
        "skip_orthogonal_param_types": ["bias", "LayerNorm"],
    },
    "MistralForCausalLM": {
        "stable_max_layer_name": "lm_head",
        "skip_orthogonal_param_types": ["bias", "LayerNorm"],
    },
    "MixtralForCausalLM": {
        "stable_max_layer_name": "lm_head",
        "skip_orthogonal_param_types": ["bias", "LayerNorm"],
    },
    "Phi3ForCausalLM": {
        "stable_max_layer_name": "lm_head",
        "skip_orthogonal_param_types": ["bias", "LayerNorm"],
    },
    "GemmaForCausalLM": {
        "stable_max_layer_name": "lm_head",
        "skip_orthogonal_param_types": ["bias", "LayerNorm"],
    },
    "StableLMEpochForCausalLM": {
        "stable_max_layer_name": "lm_head",
        "skip_orthogonal_param_types": ["bias", "LayerNorm"],
    },
    # Add more as needed...
}


##############################################################################
# Custom Training Arguments
##############################################################################

@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    Extends the default Hugging Face TrainingArguments to add flags for:
      - Use StableMax or LogStableMax,
      - Use orthogonal gradient decomposition,
      - Skip specific parameter types or 1D parameters,
      - Optionally name of final layer to apply StableMax/LogStableMax to ("auto" by default).
    """
    use_stable_max: bool = field(default=False, metadata={"help": "Use StableMax activation in the final layer."})
    use_log_stable_max: bool = field(default=False, metadata={"help": "Use LogStableMax activation in the final layer. Mutually exclusive with use_stable_max."})
    use_orthogonal_optimizer: bool = field(default=False, metadata={"help": "Use orthogonal gradient decomposition with the optimizer."})
    skip_orthogonal_param_types: List[str] = field(default_factory=list, metadata={"help": "List of parameter types (name substrings) to skip for orthogonal decomposition."})
    stable_max_layer_name: Optional[str] = field(default="auto", metadata={"help": "Name of the layer to apply StableMax/LogStableMax to. 'auto' finds the last nn.Linear."})

    def __post_init__(self):
        super().__post_init__()
        if self.use_stable_max and self.use_log_stable_max:
            raise ValueError("Cannot enable both use_stable_max and use_log_stable_max simultaneously!")


##############################################################################
# Custom Trainer
##############################################################################

class CustomTrainer(Trainer):
    """
    Custom trainer that:
      - Infers model-specific config from MODEL_CONFIG_MAPPING,
      - Optionally replaces the final layer's Softmax with StableMax/LogStableMax,
      - Integrates the orthogonal optimizer if requested.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Fetch any model-specific config
        self.model_config_params = self._get_model_config_params()

        # Update arguments from config if not manually set
        self.args.stable_max_layer_name = self.model_config_params.get(
            "stable_max_layer_name", self.args.stable_max_layer_name
        )
        self.args.skip_orthogonal_param_types = self.model_config_params.get(
            "skip_orthogonal_param_types", self.args.skip_orthogonal_param_types
        )

        # Apply StableMax or LogStableMax if requested
        if self.args.use_stable_max or self.args.use_log_stable_max:
            self._replace_final_activation_with_stablemax()

    def _get_model_config_params(self) -> Dict:
        """
        Attempt to detect the model type or architecture, then use
        MODEL_CONFIG_MAPPING for defaults. Falls back to empty dict if none found.
        """
        config = getattr(self.model, "config", None)
        if config:
            model_type = getattr(config, "model_type", None)
            architecture = getattr(config, "architectures", [None])[0]
            logging.info(f"Detected model type: {model_type}, architecture: {architecture}")
            config_params = MODEL_CONFIG_MAPPING.get(
                model_type, MODEL_CONFIG_MAPPING.get(architecture, {})
            )
            if not config_params:
                logging.warning(f"No specific config found for model_type={model_type} or architecture={architecture}. Using defaults.")
            return config_params
        return {}

    def _replace_final_activation_with_stablemax(self):
        """
        Finds the final linear layer and sets `self.final_activation`
        to StableMax or LogStableMax, without changing the layer dimension.
        """
        layer_name = self.args.stable_max_layer_name
        if layer_name == "auto":
            last_layer = self._find_last_linear_layer()
        else:
            last_layer = self._get_layer_by_name(layer_name)

        if last_layer is None:
            logging.warning(f"Could not find a suitable layer to apply StableMax to. Layer name: {layer_name}")
            return

        if isinstance(last_layer, nn.Linear):
            final_layer_name = layer_name if layer_name != "auto" else self._get_last_layer_name(last_layer)
            logging.info(f"Will apply StableMax/LogStableMax to layer: {final_layer_name}")
            logging.info(f"Layer shape: {last_layer.weight.shape}")

            if self.args.use_stable_max:
                self.final_activation = StableMax()
            else:
                self.final_activation = LogStableMax()
        else:
            logging.warning(f"Target layer '{layer_name}' is not an nn.Linear. Cannot apply StableMax/LogStableMax.")

    def _get_layer_by_name(self, layer_name: str) -> Optional[nn.Module]:
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        return None

    def _find_last_linear_layer(self):
        linear_layers = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        return linear_layers[-1] if linear_layers else None

    def _get_last_layer_name(self, last_layer):
        for name, module in self.model.named_modules():
            if module is last_layer:
                return name
        return None

    def get_optimizer(self):
        """
        Wraps the default AdamW optimizer with the orthogonal gradient decorator if requested.
        Groups parameters to skip based on `skip_orthogonal_param_types` logic.
        """
        if self.optimizer is not None:
            return self.optimizer

        # Base optimizer
        optimizer_class = torch.optim.AdamW

        # Wrap with orthogonal gradient if requested
        if self.args.use_orthogonal_optimizer:
            optimizer_class = with_orthogonal_gradient(optimizer_class)

        # Group parameters: those we do not skip vs. those we skip from orthogonal updates
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and not any(skip_t in n for skip_t in self.args.skip_orthogonal_param_types)
                ],
                "weight_decay": self.args.weight_decay,
                "name": "non_skipped"
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and any(skip_t in n for skip_t in self.args.skip_orthogonal_param_types)
                ],
                "weight_decay": 0.0,  # e.g., no weight decay for bias, LN
                "name": "skipped"
            }
        ]

        self.optimizer = optimizer_class(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            skip_orthogonal_param_types=self.args.skip_orthogonal_param_types,
            skip_orthogonal_1d=True
        )
        return self.optimizer

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Standard HF training step, with stablemax/log-stablemax and orthogonal gradient in place.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            # If using DataParallel, average the loss
            loss = loss.mean()

        # Mixed precision logic (apex or huggingface accelerate)
        if hasattr(self, "use_apex") and self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Apply final_activation (StableMax or LogStableMax) and compute the 
        appropriate loss function (CrossEntropy for probabilities, NLLLoss for log-probs).
        """
        labels = inputs.pop("labels", None) if "labels" in inputs else None

        outputs = model(**inputs)

        # If we set final_activation (StableMax or LogStableMax), apply it to logits
        if hasattr(self, "final_activation"):
            outputs.logits = self.final_activation(outputs.logits)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            # If using sagemaker mp
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_only(model, labels)
                loss_mb = self.accelerator.gather(loss_mb.reshape(-1)).mean().detach().item()
                loss = torch.tensor(loss_mb, device=self.args.device)
            else:
                # LogStableMax => NLLLoss, otherwise CrossEntropy
                if isinstance(self.final_activation, LogStableMax):
                    loss_fct = nn.NLLLoss()
                    loss = loss_fct(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        labels.view(-1)
                    )
                else:
                    # For probabilities from StableMax
                    loss = self.label_smoother(outputs, labels)
            
            # Check dimension matches vocab_size
            vocab_size = self.model.config.vocab_size
            assert outputs.logits.size(-1) == vocab_size, \
                f"Output dimension mismatch! Expected {vocab_size}, got {outputs.logits.size(-1)}"
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss. Keys: "
                    f"{','.join(outputs.keys())}. Inputs: {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
