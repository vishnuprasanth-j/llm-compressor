import inspect
from itertools import product
from typing import Any, Dict, Iterator, List, Literal, Optional, Set, Union

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    disable_quantization,
    is_preset_scheme,
    preset_name_to_scheme,
)
from compressed_tensors.utils import (
    align_modules,
    get_execution_device,
    get_lowest_common_ancestor_name,
    getattr_chain,
    match_modules_set,
    match_named_modules,
    update_offload_parameter,
)
from loguru import logger
from pydantic import ConfigDict, PrivateAttr, field_validator
from torch.nn import Module
from torch.utils._pytree import tree_leaves
from tqdm import tqdm

from llmcompressor.core import Event, EventType, State, active_session
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.awq.mappings import (
    AWQMapping,
    ResolvedMapping,
    get_layer_mappings_from_architecture,
)
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.modifiers.utils.pytorch_helpers import is_moe_model
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.sentinel import Sentinel
from llmcompressor.utils.helpers import calibration_forward_context
from llmcompressor.utils.pytorch.module import (
    get_module_to_name_dict,
)

__all__ = ["AWQModifier", "pseudo_quantize_tensor"]


def pseudo_quantize_tensor(
    weight: torch.Tensor,
    weight_args: QuantizationArgs,
) -> torch.Tensor:
    """
    Pure mathematical simulation of quantization without using PyTorch Observers.
    This simulates the rounding/clamping error that occurs during quantization
    based on the quantization args (bit_width, group_size, strategy, symmetric).

    :param weight: The weight tensor to pseudo-quantize
    :param weight_args: QuantizationArgs containing bit_width, group_size, strategy, etc.
    :return: Pseudo-quantized weight tensor (still in float, but with quantization error)
    """
    if weight_args is None:
        return weight

    orig_shape = weight.shape
    orig_dtype = weight.dtype

    # Get quantization parameters
    bit_width = weight_args.num_bits
    symmetric = weight_args.symmetric

    # Handle different quantization strategies
    strategy = weight_args.strategy

    if strategy == QuantizationStrategy.TENSOR:
        # Entire tensor quantized together
        chunk_size = weight.numel()
        weight = weight.reshape(-1, chunk_size)

    elif strategy == QuantizationStrategy.CHANNEL:
        # Per output channel quantization
        chunk_size = weight.size(1)
        weight = weight.reshape(weight.size(0), -1)

    elif strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        # Group quantization
        group_size = weight_args.group_size
        if group_size is None or group_size <= 0:
            # Fallback to per-tensor if group_size not specified
            group_size = weight.numel()
        chunk_size = group_size
        # Reshape to (num_groups, group_size)
        if weight.numel() % chunk_size != 0:
            # If not divisible, pad with zeros (will be removed after quantization)
            pad_size = chunk_size - (weight.numel() % chunk_size)
            weight = torch.nn.functional.pad(weight.reshape(-1), (0, pad_size))
        weight = weight.reshape(-1, chunk_size)

    elif strategy == QuantizationStrategy.BLOCK:
        # Block quantization
        block_height, block_width = weight_args.block_structure
        weight = (
            weight.unflatten(0, (-1, block_height))
            .unflatten(-1, (-1, block_width))
            .transpose(1, 2)
        )
        chunk_size = block_height * block_width
        weight = weight.reshape(-1, chunk_size)

    else:
        # Unknown strategy, return unchanged
        return weight

    # Compute scale and zero-point per chunk
    if symmetric:
        # Symmetric quantization: scale = max_abs / (2^(bits-1) - 1)
        max_abs = weight.abs().amax(dim=1, keepdim=True)
        scale = max_abs / (2 ** (bit_width - 1) - 1)
        scale = scale.clamp(min=1e-8)  # Avoid division by zero
        zero_point = torch.zeros_like(scale)
    else:
        # Asymmetric quantization
        min_val = weight.amin(dim=1, keepdim=True)
        max_val = weight.amax(dim=1, keepdim=True)
        qmin = 0
        qmax = 2**bit_width - 1
        scale = (max_val - min_val) / (qmax - qmin)
        scale = scale.clamp(min=1e-8)  # Avoid division by zero
        zero_point = qmin - min_val / scale
        zero_point = zero_point.round().clamp(qmin, qmax)

    # Quantize: q = round(w / scale) + zp
    q_weight = (weight / scale).round() + zero_point

    # Clamp to valid range
    if symmetric:
        qmin = -(2 ** (bit_width - 1))
        qmax = 2 ** (bit_width - 1) - 1
    else:
        qmin = 0
        qmax = 2**bit_width - 1
    q_weight = q_weight.clamp(qmin, qmax)

    # Dequantize: w' = (q - zp) * scale
    dequant_weight = (q_weight - zero_point) * scale

    # Reshape back to original shape
    if strategy == QuantizationStrategy.CHANNEL:
        dequant_weight = dequant_weight.reshape(orig_shape)
    elif strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        dequant_weight = dequant_weight.reshape(-1)[: orig_shape.numel()].reshape(
            orig_shape
        )
    elif strategy == QuantizationStrategy.BLOCK:
        # Reverse the block reshaping
        num_blocks_h = orig_shape[0] // weight_args.block_structure[0]
        num_blocks_w = orig_shape[1] // weight_args.block_structure[1]
        dequant_weight = dequant_weight.reshape(
            num_blocks_h, num_blocks_w, weight_args.block_structure[0], weight_args.block_structure[1]
        ).transpose(1, 2).reshape(orig_shape)
    else:
        dequant_weight = dequant_weight.reshape(orig_shape)

    return dequant_weight.to(orig_dtype)


class AWQModifier(Modifier):
    """
    Implements the AWQ (Activation-Weighted Quantization) algorithm,
    as described in https://arxiv.org/pdf/2306.00978. The algorithm
    significantly reduces quantization error by protecting only 1%
    of the most salient weight channels.

    Instead of relying on raw weight values, AWQ identifies important channels by
    analyzing activation patterns, focusing on the channels in the weight tensor that
    are most responsive to the input. To reduce quantization error, it scales these
    channels in a way that preserves the model's original behavior, using scaling
    factors computed offline from activation statistics.

    Because this modifier manipulates the weights of the model, it can only be used in
    in one-shot and not during training. Activation ranges are determined by running a
    small set of calibration data through the model.

    NOTE: AWQModifier performs smoothing only and should be used in combination with
    a quantization modifier (QuantizationModifier or GPTQModifier) for full
    quantization support. When used standalone, a warning will be logged.

    Example recipe (stacked with QuantizationModifier):
    ```yaml
    AWQModifier:
      mappings:
        - smooth_layer: "re:.*self_attn_layer_norm"
          balance_layers: ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"]
        - smooth_layer: "re:.*final_layer_norm"
          balance_layers: ["re:.*fc1"]
      ignore: ["lm_head"]
      duo_scaling: true
      scheme: W4A16_ASYM
    QuantizationModifier:
      targets: ["Linear"]
      scheme: W4A16_ASYM
      ignore: ["lm_head"]
    ```

    IMPORTANT: When stacking AWQModifier with QuantizationModifier or GPTQModifier,
    the `scheme` (or `config_groups`) must match exactly between the two modifiers.
    This is validated at recipe creation time.

    Lifecycle:

    - on_initialize
        - resolve mappings
        - resolve scheme to _weight_args for quantization simulation
        - capture kwargs needed for forward passes into modules
    - on_start
        - set up activation cache hooks to capture input activations
            to balance layers
    - on sequential epoch end
        - apply smoothing to each smoothing layer
            - consume cached activations across all batches
                - clear cached activations as they are used
            - find best smoothing scale for each smoothing layer via grid search
            - apply best scales to model weights
            - raise error if any unused activations remain
    - on_end
        - re-run logic of sequential epoch end (in case of basic pipeline)
        - remove activation hooks
    - on_finalize
        - clear resolved mappings and captured activations

    :param sequential_targets: list of module names to compress in
        the same calibration pass
    :param mappings: list activation layers to smooth, and which layers to
        scale the output such that activations are smoothed.
        Each entry of the mapping list should be a list itself, in which the first
        entry is a list of layers who share the same input activation (the one to be
        to smoothed) and the second entry is the layer whose output is scaled to
        achieve the smoothing.
        If regex is used, it matches layers with the largest overlap in module name.
        Each mapping may also include an ``activation_hook_target``: a dotted
        attribute path relative to the parent module (lowest common ancestor)
        specifying which submodule to hook for activation caching. This is useful
        for parallel transformer blocks where the default (hooking
        ``balance_layers[0]``) would capture the wrong activations.
    :param targets: list of layer types or names to consider for smoothing.
        Defaults to ["Linear"].
    :param ignore: list of layers to ignore during smoothing.
        It should match the name of layers whose outputs are scaled to achieve
        smoothing (the second entry of the mappings list).
    :param offload_device: offload cached args to this device, which reduces memory
        requirements but requires more time to move data between cpu and execution
        device. Defaults to None, so cached args are not offloaded. Consider setting
        to torch.device("cpu") if you are encountering OOM errors
    :param duo_scaling: whether to use duo scaling, which uses both input activations
        and weights to determine the scaling factor. Defaults to True
        If True, both activations and weights are used.
        If False, only activations are used.
        If "both", half the grid search is performed with duo_scaling=False and the
        other half is performed with duo_scaling=True.
    :param n_grid: when performing the best scales grid search for each mapping,
        this specifies how many grid points should be used. To decrease the runtime,
        at the possible cost of slightly worse scales, this can be decreased.
        Defaults to 20
    :param scheme: a quantization scheme to use for the grid search simulation.
        This should match the scheme used in the subsequent QuantizationModifier
        or GPTQModifier. Can be a preset scheme name (e.g., "W4A16_ASYM") or a
        dictionary specifying the scheme. If None, the grid search will use
        FP-only smoothing without quantization simulation.
    :param config_groups: alternative to scheme, allows specifying quantization
        config groups directly. Must match the config_groups of the subsequent
        quantization modifier if stacked.
    """

    # Allow arbitrary types because AWQMapping has fields of type torch.nn.Module
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    # User-provided vars
    sequential_targets: str | list[str] | None = None
    mappings: list[AWQMapping] | None = None
    targets: str | list[str] = "Linear"
    ignore: list[str] | None = None
    offload_device: torch.device | None | Sentinel = Sentinel("not_provided")
    duo_scaling: bool | Literal["both"] = True
    n_grid: int = 20
    # Quantization scheme for grid search simulation
    scheme: Optional[Union[str, Dict[str, Any]]] = None
    config_groups: Optional[Dict[str, QuantizationScheme]] = None

    # Private vars set during initialization, cleared during finalization
    _resolved_mappings: list[ResolvedMapping] = PrivateAttr(default_factory=list)
    # Cache list of forward input args for each parent module, one dict for each batch
    _parent_args_cache: dict[Module, IntermediatesCache] = PrivateAttr(
        default_factory=dict
    )
    # Dict[smooth layer name, (activation means, activation counts)]
    _smooth_activation_means: dict[str, tuple[torch.FloatTensor, int]] = PrivateAttr(
        default_factory=dict
    )
    # List to store error metrics for each layer
    _error_metrics: list[dict] = PrivateAttr(default_factory=list)
    # Resolved QuantizationArgs for weight quantization simulation
    _weight_args: Optional[QuantizationArgs] = PrivateAttr(None)

    @property
    def resolved_targets(self) -> Set[str]:
        """
        Return targets for smoothing.
        """
        if isinstance(self.targets, str):
            return {self.targets}
        return set(self.targets) if self.targets else {"Linear"}

    @field_validator("scheme", mode="before")
    @classmethod
    def validate_scheme(
        cls, value: Optional[Union[str, Dict[str, Any]]]
    ) -> Optional[Union[str, Dict[str, Any]]]:
        """Validate that scheme is either a preset name or a valid dict."""
        if value is None:
            return value

        if isinstance(value, str) and not is_preset_scheme(value):
            raise ValueError(
                f"`scheme` must either be a preset scheme name or a dictionary. "
                f"Got string '{value}' which is not a known preset. "
                f"Available presets include: W4A16, W4A16_ASYM, W8A8, etc."
            )

        if isinstance(value, dict):
            for scheme_name in value.keys():
                if not is_preset_scheme(scheme_name):
                    raise ValueError(
                        f"Scheme key '{scheme_name}' is not a known preset scheme. "
                        f"Available presets include: W4A16, W4A16_ASYM, W8A8, etc."
                    )

        return value

    @field_validator("duo_scaling")
    @classmethod
    def validate_duo_scaling(cls, v):
        """Validate that duo_scaling is either True, False, or 'both' (lowercase)"""
        if v not in (True, False, "both"):
            raise ValueError(f"duo_scaling must be True, False, or 'both', got {v!r}")
        return v

    def _resolve_weight_args(self) -> Optional[QuantizationArgs]:
        """
        Resolve the scheme or config_groups into a QuantizationArgs object
        for use in the grid search quantization simulation.

        Returns the weight QuantizationArgs from the first config group,
        or None if no scheme is specified.
        """
        if self.scheme is None and self.config_groups is None:
            return None

        if self.scheme is not None and self.config_groups is not None:
            raise ValueError("Please specify either `scheme` or `config_groups`, not both")

        # Resolve scheme to config_groups
        config_groups = self.config_groups
        if self.scheme is not None:
            scheme = self.scheme
            targets = [self.targets] if isinstance(self.targets, str) else self.targets

            if isinstance(scheme, str) and is_preset_scheme(scheme):
                scheme = {scheme: targets}

            config_groups = {}
            for idx, key in enumerate(scheme.keys() if isinstance(scheme, dict) else []):
                if is_preset_scheme(key):
                    scheme_obj = preset_name_to_scheme(key, scheme[key])
                else:
                    scheme_obj = QuantizationScheme.model_validate(
                        {"targets": scheme[key], **scheme}
                    )
                config_groups[f"group_{idx}"] = scheme_obj

        if config_groups is None or len(config_groups) == 0:
            return None

        # Get weight args from the first config group
        first_group = next(iter(config_groups.values()))
        return first_group.weights

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize AWQ on the given state
        Resolve mappings, resolve scheme to _weight_args, cache module kwargs

        :param state: state to run AWQ on
        :return: True on a successful run, False otherwise
        """

        if self.mappings is None:
            logger.info("No AWQModifier.mappings provided, inferring from model...")
            self.mappings = get_layer_mappings_from_architecture(
                architecture=state.model.__class__.__name__
            )

        # Set default offload_device
        if self.offload_device == Sentinel("not_provided"):
            # Check if we have a MoE model
            if is_moe_model(state.model):
                self.offload_device = torch.device("cpu")
                logger.info(
                    "MoE model detected: setting offload_device to 'cpu' by default "
                    "to reduce memory usage. You can override this by explicitly "
                    "setting offload_device in your recipe."
                )
            else:
                # For non-MoE models, convert sentinel to None
                # (no offloading by default)
                self.offload_device = None

        # Resolve scheme to _weight_args for quantization simulation
        self._weight_args = self._resolve_weight_args()

        if self._weight_args is not None:
            logger.info(
                f"AWQModifier will use quantization simulation with "
                f"num_bits={self._weight_args.num_bits}, "
                f"strategy={self._weight_args.strategy}, "
                f"group_size={self._weight_args.group_size}, "
                f"symmetric={self._weight_args.symmetric}"
            )
        else:
            logger.info(
                "AWQModifier running without quantization simulation. "
                "Grid search will use FP-only smoothing. "
                "For best results, provide a `scheme` argument matching "
                "your subsequent QuantizationModifier or GPTQModifier."
            )

        # Validate duo_scaling with strategy
        if self._weight_args is not None and self.duo_scaling is not False:
            if self._weight_args.strategy == QuantizationStrategy.TENSOR:
                raise ValueError(
                    "duo_scaling is only supported with per-channel "
                    "quantization strategies (group or channel), but found "
                    "TENSOR strategy. Please set duo_scaling=False or use a "
                    "per-channel quantization strategy."
                )

        self._set_resolved_mappings(state.model)

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

        # Check for unsupported token masking with MoE up_proj -> down_proj mappings
        if state.loss_masks is not None and self._has_moe_up_down_proj_mapping():
            raise ValueError(
                "Token masking (use_loss_mask=True) is not supported with "
                "up_proj -> down_proj mappings in MoE models. The MoE routing "
                "mechanism dispatches tokens to different experts, and the loss mask "
                "cannot be properly aligned with this dispatch. Please either "
                "disable token masking or exclude the up_proj -> down_proj mapping "
                "for MoE layers from the AWQ configuration."
            )

        # AWQ performs forward passes during _apply_smoothing
        # Quantization must be disabled during smoothing, otherwise NaNs will
        # appear in quantized forward method
        state.model.apply(disable_quantization)

        self._setup_activation_cache_hooks()

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        elif event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            # Run smoothing in case of sequential pipeline
            self._apply_smoothing(state.model)

        elif event.type_ == EventType.CALIBRATION_EPOCH_END:
            # Run smoothing in case of basic pipeline
            self._apply_smoothing(state.model)

            if not self.ended_:
                self.on_end(state, None)

    def on_end(self, state: State, event: Event, **kwargs):
        """
        Finish smoothing by removing activation hooks.
        Note: Scale and zero-point generation is now handled by the quantization
        modifier (QuantizationModifier or GPTQModifier) when stacked.
        """
        self._assert_all_activations_consumed()

        self.ended_ = True

        # remove activation hooks
        self.remove_hooks()

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        Clean up by clearing the activations and mapping data

        :param state: unused
        :return: True
        """
        if not self.ended_:
            self.on_end(state, None)

        self._log_error_metrics()

        self._parent_args_cache.clear()
        self._smooth_activation_means.clear()
        self._resolved_mappings.clear()
        self._error_metrics.clear()

        return True

    def _set_resolved_mappings(self, model: Module) -> None:
        """
        Transforms the list of activations to smooth and their corresponding weights
        into ResolvedMapping objects, resolving regular expressions.
        Result is stored in _resolved_mappings.

        For each activation in the mapping list, we find the corresponding weight to
        balance by searching for the longest substring. For instance, if our balance
        weight is ".*re:.*q_proj" and the activation is "re:.*self_attn_layer_norm" we
        would match model.layer.0.p_proj to model.layer.0.self_attn_layer_norm and
        repeat for model.layer.1 and so on
        """
        resolved_mappings: list[ResolvedMapping] = []
        module_to_name = get_module_to_name_dict(model)
        # Get names of modules targeted for smoothing (excludes ignored)
        targeted_names = set(
            name
            for name, _ in match_named_modules(
                model, self.resolved_targets, self.ignore
            )
        )
        for mapping in self.mappings:
            # we deliberately don't use the ignore list when matching mappings,
            # so that we can handle layers that need smoothing but not quantization
            # we only skip if no layers in mapping are targeted for smoothing.
            for smooth_layers, *nested_balance_layers in match_modules_set(
                model, (mapping.smooth_layer, *mapping.balance_layers)
            ):
                if len(smooth_layers) > 1:
                    raise ValueError(
                        "AWQ needs to match a single smoothlayer for each mapping but "
                        f"got {[module_to_name.get(s) for s in smooth_layers]}"
                        f" for mapping: {mapping}"
                    )
                smooth_layer = smooth_layers[0]
                smooth_name = module_to_name.get(smooth_layer)

                # [[b00, b01, b02...], [b10, b11, b12,...], ...] ↓
                #                             [b00, b01, b02, ..., b10, b11, b12, ...]
                balance_layers = tree_leaves(nested_balance_layers)
                balance_names = [
                    module_to_name.get(balance_layer)
                    for balance_layer in balance_layers
                ]

                # Check if at least one layer is targeted for smoothing
                any_targeted = smooth_name in targeted_names or any(
                    bn in targeted_names for bn in balance_names
                )

                all_compatible = _check_layers_are_compatible(
                    smooth_layer, smooth_name, balance_layers, balance_names
                )

                skip_message: str | None = None
                if not all_compatible:
                    skip_message = " because found incompatible balance layers"
                elif not any_targeted:
                    skip_message = " because no layers are targeted for smoothing"
                elif len(balance_layers) == 0:
                    skip_message = " because no balance layers were found"

                if skip_message:
                    logger.warning(
                        f"skipping AWQ for {smooth_name} for mapping {mapping}"
                        + skip_message
                    )

                    continue

                ancestor_name, ancestor = get_lowest_common_ancestor_with_avoid(
                    balance_names, model, torch.nn.ModuleList
                )

                activation_hook_target = None
                if mapping.activation_hook_target:
                    activation_hook_target = getattr_chain(
                        ancestor, mapping.activation_hook_target
                    )
                    if activation_hook_target is None:
                        raise ValueError(
                            f"activation_hook_target '{mapping.activation_hook_target}'"
                            f" not found on parent module '{ancestor_name}'"
                        )

                resolved_mappings.append(
                    ResolvedMapping(
                        smooth_name,
                        smooth_layer,
                        balance_layers,
                        balance_names=balance_names,
                        parent=ancestor,
                        parent_name=ancestor_name,
                        activation_hook_target=activation_hook_target,
                    )
                )
        self._resolved_mappings = resolved_mappings
        return

    def _setup_activation_cache_hooks(self) -> None:
        """
        Attach a forward hook to each activation we want to smooth. This allows us to
        calculate the dynamic range during calibration
        """

        def cache_parent_kwargs_hook(
            module: Module,
            args: tuple[torch.Tensor, ...],
            kwargs,
        ):
            values = inspect.signature(module.forward).bind(*args, **kwargs)
            self._parent_args_cache[module].append(values.arguments)

        def create_cache_smooth_activations_hook_fn(smooth_name):
            def cache_smooth_activations_hook(
                _module: Module,
                args: tuple[torch.Tensor, ...],
                _output: torch.Tensor,
            ):
                activations = args[0].abs().detach()

                # Get loss mask for current batch from state
                session = active_session()
                state = session.state
                loss_masks = state.loss_masks if state else None
                batch_idx = state.current_batch_idx if state else -1
                loss_mask = (
                    loss_masks[batch_idx] if loss_masks and batch_idx >= 0 else None
                )

                if loss_mask is not None:
                    # Mask: [batch, seq] -> [batch, seq, 1]
                    mask = loss_mask.to(activations.device).unsqueeze(-1)
                    flat_activations = activations.flatten(0, -2)  # [batch*seq, hidden]
                    flat_mask = mask.flatten(0, -2).squeeze(-1)
                    masked_activations = flat_activations[flat_mask.bool()]
                else:
                    masked_activations = activations.flatten(0, -2)

                act_mean, count = _accumulate_mean(
                    masked_activations,
                    self._smooth_activation_means.get(smooth_name, None),
                )
                self._smooth_activation_means[smooth_name] = (act_mean.cpu(), count)

            return cache_smooth_activations_hook

        for mapping in self._resolved_mappings:
            # parent kwargs needed for future forward passes
            # same parent may appear multiple times in resolved mappings
            if mapping.parent not in self._parent_args_cache:
                self._parent_args_cache[mapping.parent] = IntermediatesCache(
                    None,
                    self.offload_device,
                )
                self.register_hook(
                    mapping.parent,
                    cache_parent_kwargs_hook,
                    "forward_pre",
                    with_kwargs=True,
                )

            # input activations to balance layers needed for loss function
            # storing inputs to first balance layer is sufficient
            # other balance layers get the same input
            #
            # For parallel transformer blocks (e.g. Command A, Gemma 3) the first
            # balance layer may not receive the right activations.  When
            # activation_hook_target is set on the mapping, hook that module
            # instead of balance_layers[0].
            layer_to_hook = mapping.activation_hook_target or mapping.balance_layers[0]
            self.register_hook(
                layer_to_hook,
                create_cache_smooth_activations_hook_fn(mapping.smooth_name),
                "forward",
            )

    @torch.no_grad()
    def _apply_smoothing(self, model: Module) -> None:
        """
        Calculate the best scaling factors for each layer to smooth activations and
        apply the scaling factors to the weights of the next layer to offset the
        smoothing

        :param model: model to apply smoothing to
        """
        # NOTE: When using SequentialPipeline, not all the mappings
        # will have cached activations in the segment being updated
        mappings_to_smooth = [
            mapping
            for mapping in self._resolved_mappings
            if mapping.smooth_name in self._smooth_activation_means
        ]
        for mapping in tqdm(mappings_to_smooth, desc="Smoothing"):
            smooth_layer = mapping.smooth_layer
            balance_layers = mapping.balance_layers
            parent_module = mapping.parent

            with (
                align_modules([parent_module, smooth_layer, *balance_layers]),
                calibration_forward_context(model),
                HooksMixin.disable_hooks(),
            ):
                # Compute output of unquantized module
                fp16_outputs = self._run_samples(parent_module)
                if len(fp16_outputs) == 0 or all(f.numel() == 0 for f in fp16_outputs):
                    logger.info(
                        f"Skipping smooth_layer {mapping.smooth_name}, no activations "
                        "found to scale. This can occasionally occur in MoE models "
                        "when certain experts are not activated by calibration samples."
                    )
                    del self._smooth_activation_means[mapping.smooth_name]
                    continue
                if not all(
                    [fp16_output.isfinite().all() for fp16_output in fp16_outputs]
                ):
                    logger.warning(
                        f"Skipping smooth_layer {mapping.smooth_name}, NaN or inf "
                        "outputs found during forward pass of the parent module "
                        f"{mapping.parent_name}. The model is either generating NaN "
                        "output with provided calibration data set, or the mappings "
                        "are incorrectly set and modifying the model in undesired "
                        "ways. If you encounter this consistently, raise an issue at "
                        "https://github.com/vllm-project/llm-compressor/issues"
                    )
                    del self._smooth_activation_means[mapping.smooth_name]
                    continue

                orig_layer_weights = {
                    balance_layer: balance_layer.weight.clone()
                    for balance_layer in mapping.balance_layers
                }

                best_scales = self._compute_best_scale(
                    mapping, fp16_outputs, orig_layer_weights
                )

                @torch.no_grad()
                def _smooth(
                    module: Module, orig_layer_weights: dict[Module, torch.Tensor]
                ):
                    scales = best_scales.to(module.weight.device)
                    if module in balance_layers:
                        update_offload_parameter(
                            module,
                            "weight",
                            orig_layer_weights[module].to(module.weight.device)
                            * scales.view(1, -1),
                        )
                    elif module == smooth_layer:
                        if module.weight.ndim == 1:
                            update_offload_parameter(
                                module,
                                "weight",
                                module.weight.div_(scales),
                            )
                        else:
                            # NOTE: edge case when smooth layer number of out_features
                            # is not equal to balance layer number of in_features
                            # e.g. when fused qkv_proj is used to smooth o_proj
                            # in this case, default to scaling the last output features
                            # because the desired smooth layer is v_proj
                            # https://github.com/casper-hansen/AutoAWQ/blob/main/awq/quantize/scale.py#L123
                            weight = module.weight
                            weight[-scales.size(0) :].div_(scales.view(-1, 1))
                            update_offload_parameter(module, "weight", weight)
                        if hasattr(module, "bias") and module.bias is not None:
                            update_offload_parameter(
                                module,
                                "bias",
                                module.bias.div_(scales),
                            )

                for layer in balance_layers:
                    _smooth(layer, orig_layer_weights)
                _smooth(smooth_layer, orig_layer_weights)

                # remove caches needed to smooth this mapping
                del self._smooth_activation_means[mapping.smooth_name]
                del orig_layer_weights

        for v in self._parent_args_cache.values():
            v.batch_intermediates.clear()
        self._assert_all_activations_consumed()

    @torch.no_grad()
    def _run_samples(self, module: Module) -> list[torch.Tensor]:
        outputs = [
            module(**batch_kwargs) for batch_kwargs in self._parent_args_cache[module]
        ]
        return [
            # If tuple, assume that first argument is the input
            output[0] if isinstance(output, tuple) else output
            for output in outputs
        ]

    def _compute_best_scale(
        self,
        mapping: ResolvedMapping,
        fp16_outputs: list[torch.Tensor],
        orig_layer_weights: dict[torch.nn.Module, torch.Tensor],
    ) -> torch.Tensor:
        """
        Select best scales for a given mapping in a grid search.
        Best scales minimize MSE loss of (quantized) weight outputs vs fp16_outputs.

        When a scheme is provided to AWQModifier, uses pure mathematical quantization
        simulation (full AWQ objective). When no scheme is provided, falls back to
        FP-only smoothing with activation-only scaling.

        L(s) = || Q(W * s)(s^-1 * X) - WX ||   [with scheme]
        L(s) = || (W * s)(s^-1 * X) - WX ||     [without scheme]
        """
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")
        initial_error = None

        device = get_execution_device(mapping.parent)
        x_mean = self._smooth_activation_means[mapping.smooth_name][0].to(device)

        # Standalone mode: no quantization args provided to AWQModifier
        # Fall back to activation-only scaling (duo_scaling forced off)
        standalone_mode = self._weight_args is None
        if standalone_mode:
            logger.warning(
                f"No scheme provided to AWQModifier for {mapping.smooth_name}. "
                "Running grid search in FP-only mode (activation-only scaling). "
                "For best results, provide a `scheme` argument matching your "
                "subsequent QuantizationModifier or GPTQModifier."
            )
            effective_duo_scaling = False
        else:
            effective_duo_scaling = self.duo_scaling

        # Compute weight means only if duo_scaling is active
        if effective_duo_scaling is not False:
            w_mean = self._compute_layer_means(
                mapping.balance_layers, self._weight_args
            ).to(device)

        # Grid search configuration
        match effective_duo_scaling:
            case "both":
                n_grid = int(self.n_grid / 2)
                duo_scalings = [False, True]
            case _:
                n_grid = self.n_grid
                duo_scalings = [effective_duo_scaling]

        total_iterations = n_grid * len(duo_scalings)
        pbar = tqdm(
            product(range(n_grid), duo_scalings),
            total=total_iterations,
            desc=f"Grid search for {mapping.smooth_name}",
            leave=False,
        )

        for grid_idx, use_duo_scaling in pbar:
            ratio = grid_idx / n_grid

            # Compute candidate scales
            if use_duo_scaling:
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(
                    min=1e-4
                )
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)

            scales = scales / (scales.max() * scales.min()).sqrt()
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1
            _scalesview = scales.view(1, -1).to(device)

            # Apply scaled + (optionally) quantized weights
            for layer in mapping.balance_layers:
                scaled_weight = (
                    orig_layer_weights[layer].to(_scalesview.device) * _scalesview
                )

                if self._weight_args is not None:
                    # Full AWQ: simulate quantization error Q(W*s)/s using pure math
                    # No PyTorch Observers needed - just mathematical simulation
                    quantized_weight = pseudo_quantize_tensor(
                        scaled_weight, self._weight_args
                    )
                    layer.weight.data = (quantized_weight / _scalesview).to(
                        layer.weight.dtype
                    )
                else:
                    # Standalone fallback: FP smoothing only
                    layer.weight.data.copy_(scaled_weight)

            # Measure error
            int_w_outputs = self._run_samples(mapping.parent)
            loss = self._compute_loss(fp16_outputs, int_w_outputs)
            del int_w_outputs

            if initial_error is None:
                initial_error = loss

            history.append(
                {"ratio": ratio, "duo_scaling": use_duo_scaling, "error": loss}
            )
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()

            pbar.set_postfix({"best_error": f"{best_error:.3e}"})

        if best_ratio == -1:
            logger.debug(history)
            raise Exception(
                "No finite loss found in best scales grid search. "
                "https://github.com/vllm-project/llm-compressor/issues"
            )

        err_reduction = best_error / initial_error if initial_error > 0 else 1.0
        logger.debug(
            f"AWQ grid search for {mapping.smooth_name}: "
            f"initial={initial_error:.3e}, best={best_error:.3e}, "
            f"mode={'FP-only' if standalone_mode else 'quantized'}, "
            f"reduction={err_reduction * 100:.3f}%"
        )

        self._error_metrics.append(
            {
                "layer_name": mapping.smooth_name,
                "parent_name": mapping.parent_name,
                "initial_error": initial_error,
                "best_error": best_error,
                "reduction": err_reduction,
                "standalone_mode": standalone_mode,
            }
        )

        assert torch.isnan(best_scales).sum() == 0, f"NaN in scales: {best_scales}"
        return best_scales.detach().cpu()

    @torch.no_grad()
    def _compute_loss(
        self,
        fp16_outputs: list[torch.Tensor],
        int_w_outputs: list[torch.Tensor],
    ) -> float:
        session = active_session()
        loss_masks = session.state.loss_masks if session.state else None

        loss = 0.0
        num_elements = 0

        # Compute the MSE loss for each batch
        for batch_idx, (fp16_batch, int_w_batch) in enumerate(
            zip(fp16_outputs, int_w_outputs)
        ):
            loss_mask = loss_masks[batch_idx] if loss_masks else None

            if loss_mask is not None:
                token_mask = loss_mask.to(fp16_batch.device) == 1  # (batch, seq)
                fp16_masked = fp16_batch[token_mask]  # (num_masked_tokens, hidden)
                int_w_masked = int_w_batch.to(fp16_batch.device)[token_mask]
                loss += torch.nn.functional.mse_loss(
                    fp16_masked, int_w_masked, reduction="sum"
                )
                num_elements += fp16_masked.numel()
            else:
                loss += torch.nn.functional.mse_loss(
                    fp16_batch, int_w_batch.to(fp16_batch.device), reduction="sum"
                )
                num_elements += fp16_batch.numel()

        # Normalize the loss by the total number of elements
        return (loss / num_elements).item()

    def _log_error_metrics(self):
        """
        Log the error metrics (initial error, best error, reduction).
        """

        # Prepare data for saving
        metrics_data = {
            "quantization_config": {
                "duo_scaling": self.duo_scaling,
                "n_grid": self.n_grid,
            },
            "total_layers": len(self._error_metrics),
            "metrics": self._error_metrics,
        }

        # Save to disk
        logger.debug(f"AWQ per-mapping error metrics: {metrics_data}")

        # Also print summary statistics
        reductions = [m["reduction"] for m in self._error_metrics]
        avg_reduction = sum(reductions) / len(reductions)
        min_reduction = min(reductions)
        max_reduction = max(reductions)
        sorted_reductions = sorted(reductions)
        median_reduction = sorted_reductions[len(sorted_reductions) // 2]
        logger.debug(
            f"Error reduction statistics: "
            f"avg={avg_reduction:.4f}, median={median_reduction:.4f}, "
            f"min={min_reduction:.4f}, max={max_reduction:.4f}"
        )

    def _assert_all_activations_consumed(self):
        """
        Confirm all activations have been consumed
        If not, something has gone wrong
        """
        if len(self._smooth_activation_means) != 0:
            raise RuntimeError("Some cached activations were not used")

    def _has_moe_up_down_proj_mapping(self) -> bool:
        """
        Check if any resolved mapping is an up_proj -> down_proj mapping
        where the balance layers are MoE experts (indicated by '.experts.'
        in the name).

        Token masking is not supported for such mappings because the MoE
        routing mechanism dispatches tokens to different experts, and the
        loss mask cannot be properly aligned with this dispatch.
        """
        for mapping in self._resolved_mappings:
            # Check if this is an up_proj -> down_proj mapping
            if mapping.smooth_name.endswith("up_proj"):
                for balance_name in mapping.balance_names:
                    if (
                        balance_name.endswith("down_proj")
                        and ".experts." in balance_name
                    ):
                        return True
        return False

    @staticmethod
    def _compute_layer_means(
        layers: list[Module], weight_args: Optional[QuantizationArgs]
    ) -> torch.Tensor:
        """
        Compute per-channel/group/block/tensor mean of normalised weights
        for all passed in layers taking into account the quantization args.

        To minimize memory requirements, layers are reduced to a running total
            of sums and counts when calculating mean

        :param layers: List of layers to compute weight means for
        :param weight_args: QuantizationArgs to use for determining chunk size
        :return: Tensor of per-channel weight means
        """
        if weight_args is None:
            raise ValueError(
                "weight_args must be provided to _compute_layer_means. "
                "This should not happen when duo_scaling is enabled."
            )

        # to calculate mean without having to carry full population
        weight_total_count = 0
        weight_total_sum = 0

        for layer in layers:
            if not hasattr(layer, "weight"):
                logger.warning(
                    "Unable to find weight param for targeted"
                    f" layer {type(layer)}, skipping"
                )
                continue
            weight = layer.weight.clone()
            orig_shape = weight.shape

            match weight_args.strategy:
                # chunk size is the size of the size of the
                # set of elements that get quantized together
                case QuantizationStrategy.TENSOR:
                    chunk_size = weight.numel()
                case QuantizationStrategy.CHANNEL:
                    chunk_size = weight.size(1)
                case QuantizationStrategy.GROUP | QuantizationStrategy.TENSOR_GROUP:
                    chunk_size = weight_args.group_size
                case QuantizationStrategy.BLOCK:
                    block_height, block_width = weight_args.block_structure
                    weight = (  # (row, col) = (num_H*block_H, num_W*block_W)
                        weight.unflatten(0, (-1, block_height))
                        .unflatten(-1, (-1, block_width))
                        .transpose(1, 2)  # ↳ (num_H, num_W, block_H, block_W)
                    )
                    intermediate_shape = weight.shape
                    chunk_size = block_height * block_width

            # need to get to shape (num_chunks x chunk_size)
            weight = weight.reshape(-1, chunk_size)
            # normalize
            weight.abs_()
            weight.div_(weight.amax(dim=1, keepdim=True) + 1e-6)
            # Reshape back to original dimensions
            if weight_args.strategy == QuantizationStrategy.BLOCK:
                weight = weight.view(intermediate_shape).transpose(1, 2)

            # back to (rows, cols)
            weight = weight.reshape(orig_shape)
            # Gets the average rescaled magnitude for each output channel
            weight_total_count += weight.size(0)
            weight_sum = weight.sum(0, dtype=torch.float64)
            weight_total_sum += weight_sum

        return weight_total_sum / weight_total_count


def _check_layers_are_compatible(
    smooth_layer, smooth_name, balance_layers, balance_names
):
    """
    returns True if they are all compatible
    returns False if any smooth & balance layers are incompatible
    """
    for balance_layer, balance_name in zip(balance_layers, balance_names):
        # exclude v_proj->o_proj mappings whose shapes are incompatible
        # https://github.com/mit-han-lab/llm-awq/pull/67#issuecomment-1681632777
        if (
            isinstance(smooth_layer, torch.nn.Linear)
            and isinstance(balance_layer, torch.nn.Linear)
            and balance_name.endswith(".o_proj")
            and (
                (
                    smooth_name.endswith(".v_proj")
                    and smooth_layer.out_features != balance_layer.in_features
                )
                or (
                    smooth_name.endswith(".qkv_proj")
                    and smooth_layer.out_features != 3 * balance_layer.in_features
                )
            )
        ):
            return False
    return True


def get_lowest_common_ancestor_with_avoid(
    balance_names: Iterator[str], model: Module, avoid=torch.nn.ModuleList
):
    """
    Get the lowest ancestor that is not the avoided class/type.
    see compressed_tensors.utils.get_lowest_common_ancestor_name
    for detail on case handling.

    NOTE: primarily used to exclude parents of type ModuleList, which don't play
    nicely with hooks because their forward method is never directly
    called for MoE models. See Qwen3MoeSparseMoeBlock for example, experts
    are selected based on router output and their forward method is called.
    https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L233
    """
    ancestor_name = get_lowest_common_ancestor_name(balance_names)

    while True:
        if ancestor_name == "":
            return "", model
        ancestor = model.get_submodule(ancestor_name)
        if not isinstance(ancestor, avoid):
            return ancestor_name, ancestor
        ancestor_name = ".".join(ancestor_name.split(".")[:-1])


def _accumulate_mean(
    inp: torch.Tensor,
    prev_mean_and_count: tuple[torch.FloatTensor, int] | None,
) -> tuple[torch.FloatTensor, int]:
    sum_added = inp.sum(dim=0)
    num_added = inp.size(0)
    if prev_mean_and_count is None:
        return sum_added / num_added, num_added

    prev_mean, prev_count = prev_mean_and_count
    prev_mean = prev_mean.to(inp.device)

    prev_sum = prev_mean * prev_count
    new_count = prev_count + num_added

    return (prev_sum + sum_added) / new_count, new_count
