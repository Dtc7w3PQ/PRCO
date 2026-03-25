from __future__ import annotations

import json
import logging
import os
import warnings

import torch

from ..base import BaseModel
from .prompt import Qwen3VLPromptMixin
from ...smp import get_gpu_memory, listinstr


VLLM_MAX_IMAGE_INPUT_NUM = 24


def _to_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def is_moe_model(model_path: str) -> bool:
    """Check if the model is a Mixture of Experts model.

    Previous path-name-only heuristic could misclassify local checkpoints
    (e.g., custom folder names without "8B/4B/2B/32B"). We now prefer
    explicit MoE markers from model path / config.
    """
    path_lower = model_path.lower()

    # Explicit MoE markers in common Qwen3 naming.
    if any(tag in path_lower for tag in ['a22b', 'a3b', 'moe']):
        return True

    # Inspect local HF config when available.
    cfg_path = os.path.join(model_path, 'config.json')
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

        arch_text = ' '.join(cfg.get('architectures', [])).lower()
        if any(tag in arch_text for tag in ['moe', 'a22b', 'a3b']):
            return True

        for sub in (cfg, cfg.get('text_config', {}), cfg.get('vision_config', {})):
            if not isinstance(sub, dict):
                continue
            for k in ('num_experts', 'n_routed_experts'):
                v = sub.get(k)
                if isinstance(v, int) and v > 0:
                    return True
    except Exception:
        # Best-effort only; fallback to non-MoE.
        pass

    return False


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


class Qwen3VLChat(Qwen3VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,          # 默认 None（不限制最小像素）
        max_pixels: int | None = None,          # 默认 None（不限制最大像素）
        total_pixels: int | None = None,        # 默认 None（不限制总像素）
        max_new_tokens: int = 32768,            # 默认 32768
        top_p: float = 0.8,                     # 默认 0.8
        top_k: int = 20,                        # 默认 20
        temperature: float = 0.01,              # 默认 0.01
        repetition_penalty: float = 1.0,        # 默认 1.0
        presence_penalty: float = 1.5,          # 默认 1.5
        frequency_penalty: float = 0.0,         # 默认 0.0
        use_custom_prompt: bool = True,         # 默认 True（按数据集自定义 prompt）
        system_prompt: str | None = None,       # 默认 None（无 system prompt）
        post_process: bool = False,             # 默认 False（不做 \boxed{} 截取）
        verbose: bool = False,                  # 默认 False（不打印生成过程）
        use_audio_in_video: bool = True,        # 默认 True（视频时允许音频）
        max_num_seqs: int | None = None,        # 默认 None（下方回落 env 或 64）
        **kwargs,
    ) -> None:
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        if self.total_pixels and self.total_pixels > 24576 * 32 * 32:
            print('The total number of video tokens might too large, resulting in an overly long input sequence.')
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)
        self.FRAME_FACTOR = 2
        self.use_audio_in_video = use_audio_in_video

        assert model_path is not None
        self.model_path = model_path
        from transformers import AutoProcessor, AutoModelForImageTextToText
        # Use official Qwen3-Omni classes when model_path indicates omni
        if listinstr(['omni'], model_path.lower()):
            try:
                from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
            except Exception as err:
                logging.critical("pip install git+https://github.com/huggingface/transformers")
                raise err
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        else:
            self.processor = AutoProcessor.from_pretrained(model_path)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        self.use_vllm = kwargs.get('use_vllm', False)
        self.use_lmdeploy = kwargs.get('use_lmdeploy', False)
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        assert self.use_vllm + self.use_lmdeploy <= 1, "You can only set one flag `use_vllm` to True"
        # vLLM max concurrency
        env_max_num_seqs = os.environ.get('VLLM_MAX_NUM_SEQS')
        if max_num_seqs is None and env_max_num_seqs is not None:
            try:
                max_num_seqs = int(env_max_num_seqs)
            except Exception:
                max_num_seqs = None
        self.max_num_seqs = max_num_seqs if max_num_seqs is not None else 64
        # Allow forcing eager mode to reduce CUDA graph memory pressure.
        # Priority: explicit arg > env var VLLM_ENFORCE_EAGER > default False.
        enforce_eager = kwargs.get('enforce_eager', None)
        if enforce_eager is None:
            enforce_eager = os.environ.get('VLLM_ENFORCE_EAGER')
        self.enforce_eager = _to_bool(enforce_eager, default=False)

        if self.use_vllm:
            if listinstr(['omni'], self.model_path.lower()):
                os.environ['VLLM_USE_V1'] = '0'
            from vllm import LLM
            gpu_count = torch.cuda.device_count()
            tp_size = gpu_count if gpu_count > 0 else 1
            logging.info(
                f'Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})'
            )
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning(
                    "VLLM_WORKER_MULTIPROC_METHOD is not set to spawn. Use 'export VLLM_WORKER_MULTIPROC_METHOD=spawn'"
                )
            enable_expert_parallel = is_moe_model(self.model_path)
            # For Qwen3-Omni, vLLM engine v1 is not supported yet
            if listinstr(['omni'], self.model_path.lower()):
                limit_mm = {"image": 3, "video": 3, "audio": 3}
            else:
                limit_mm = {"image": self.limit_mm_per_prompt}
            self.llm = LLM(
                model=self.model_path,
                max_num_seqs=self.max_num_seqs,
                limit_mm_per_prompt=limit_mm,
                tensor_parallel_size=tp_size,
                enable_expert_parallel=enable_expert_parallel,
                seed=0,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
                enforce_eager=self.enforce_eager,
                trust_remote_code=True,
            )
        else:
            if listinstr(['omni'], model_path.lower()):
                self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    model_path, dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
                )
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
                )
            self.model.eval()

        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 32 * 32
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['min_pixels', 'max_pixels', 'total_pixels', 'resized_height', 'resized_width']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
            elif s['type'] == 'video':
                value = s['value']
                if isinstance(value, list):
                    item = {
                        'type': 'video',
                        'video': [ensure_image_url(v) for v in value],
                    }
                else:
                    item = {'type': 'video', 'video': ensure_video_url(value)}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['resized_height', 'resized_width', 'fps', 'nframes', 'sample_fps']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
                if not isinstance(value, list):
                    if self.fps is not None and 'fps' not in item:
                        item['fps'] = self.fps
                    elif self.nframe is not None and 'nframes' not in item:
                        import cv2
                        video = cv2.VideoCapture(s['value'])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        if frame_count < self.nframe:
                            new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                            print(f"use {new_frame_count} for {s['value']}")
                            item['nframes'] = new_frame_count
                        else:
                            item['nframes'] = self.nframe
            elif s['type'] == 'audio':
                item = {'type': 'audio', 'audio': s['value']}
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner_transformers(self, message, dataset=None):
        is_omni = listinstr(['omni'], self.model_path.lower())
        if is_omni:
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("Please install it via 'pip install qwen-omni-utils[decord]'")
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("Please install it via 'pip install qwen-vl-utils'")
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        if is_omni:
            # For Qwen3-Omni, messages is a list of dicts
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors='pt',
                padding=True,
                use_audio_in_video=self.use_audio_in_video,
            )
        else:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            try:
                images, videos, video_kwargs = process_vision_info(
                    messages,
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
            except TypeError:
                # Older qwen_vl_utils does not support image_patch_size argument
                images, videos, video_kwargs = process_vision_info(
                    messages,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )

            video_metadatas = None
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)

            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                do_resize=False,
                return_tensors='pt',
                **(video_kwargs or {}),
            )
        try:
            inputs = inputs.to(self.model.device)
            if hasattr(self.model, 'dtype'):
                inputs = inputs.to(self.model.dtype)
        except Exception:
            inputs = inputs.to('cuda')

        if is_omni:
            try:
                text_ids, _ = self.model.generate(
                    **inputs,
                    return_audio=False,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=self.use_audio_in_video,
                )
            except TypeError:
                text_ids, _ = self.model.generate(
                    **inputs,
                    return_audio=False,
                    use_audio_in_video=self.use_audio_in_video,
                )
            response = self.processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        else:
            generated_ids = self.model.generate(
                **inputs,
                **self.generate_kwargs,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            out = self.processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = out[0]
        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response

    def generate_inner_vllm(self, message, dataset=None):
        from vllm import SamplingParams
        is_omni = listinstr(['omni'], self.model_path.lower())
        if is_omni:
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, 'pip install qwen-omni-utils[decord]'")
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, 'pip install qwen-vl-utils'")
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if is_omni:
            audios, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
        else:
            try:
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages,
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
            except TypeError:
                # Older qwen_vl_utils does not support image_patch_size argument
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            stop_token_ids=None
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        if is_omni and 'audios' in locals() and audios is not None:
            mm_data['audio'] = audios

        req = {'prompt': text}
        if mm_data:
            req['multi_modal_data'] = mm_data
        if is_omni:
            req['mm_processor_kwargs'] = {"use_audio_in_video": self.use_audio_in_video}
        elif video_kwargs is not None:
            req['mm_processor_kwargs'] = video_kwargs

        outputs = self.llm.generate([req], sampling_params=sampling_params)

        for o in outputs:
            generated_text = o.outputs[0].text

        if self.post_process:
            resp = generated_text.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                generated_text = resp[:end]

        if self.verbose:
            print(f'\033[32m{generated_text}\033[0m')
        return generated_text

    def generate_batch_inner_vllm(self, messages, dataset=None):
        """Batched vLLM generation.

        vLLM only achieves high throughput when multiple requests are queued together.
        The original `generate_inner_vllm` path sends **one** request each time, which
        makes vLLM behave like a serial generator.

        Args:
            messages: A list of single-sample messages (each is the same format as `message`
                in :meth:`generate_inner_vllm`).
            dataset: Dataset name.

        Returns:
            A list of generated strings aligned with ``messages``.
        """
        from vllm import SamplingParams

        is_omni = listinstr(['omni'], self.model_path.lower())
        if is_omni:
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, 'pip install qwen-omni-utils[decord]'")
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, 'pip install qwen-vl-utils'")
                raise err

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            stop_token_ids=None,
        )

        reqs = []
        for message in messages:
            msgs = []
            if self.system_prompt is not None:
                msgs.append({'role': 'system', 'content': self.system_prompt})
            msgs.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})

            text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

            if is_omni:
                audios, image_inputs, video_inputs = process_mm_info(msgs, use_audio_in_video=self.use_audio_in_video)
                video_kwargs = None
            else:
                try:
                    image_inputs, video_inputs, video_kwargs = process_vision_info(
                        msgs,
                        image_patch_size=16,
                        return_video_kwargs=True,
                        return_video_metadata=True,
                    )
                except TypeError:
                    image_inputs, video_inputs, video_kwargs = process_vision_info(
                        msgs,
                        return_video_kwargs=True,
                        return_video_metadata=True,
                    )

            mm_data = {}
            if image_inputs is not None:
                mm_data['image'] = image_inputs
            if video_inputs is not None:
                mm_data['video'] = video_inputs
            if is_omni and audios is not None:
                mm_data['audio'] = audios

            req = {'prompt': text}
            if mm_data:
                req['multi_modal_data'] = mm_data
            if is_omni:
                req['mm_processor_kwargs'] = {"use_audio_in_video": self.use_audio_in_video}
            elif video_kwargs is not None:
                req['mm_processor_kwargs'] = video_kwargs

            reqs.append(req)

        outputs = self.llm.generate(reqs, sampling_params=sampling_params)

        generated_texts = []
        for o in outputs:
            generated_text = o.outputs[0].text
            if self.post_process:
                resp = generated_text.split('\\boxed{')[-1]
                lt = len(resp)
                counter, end = 1, None
                for i in range(lt):
                    if resp[i] == '{':
                        counter += 1
                    elif resp[i] == '}':
                        counter -= 1
                    if counter == 0:
                        end = i
                        break
                    elif i == lt - 1:
                        end = lt
                        break
                if end is not None:
                    generated_text = resp[:end]
            if self.verbose:
                print(f'\033[32m{generated_text}\033[0m')
            generated_texts.append(generated_text)

        return generated_texts

    def generate_batch(self, messages, dataset=None):
        """Batch wrapper for inference.

        For non-vLLM backends, this falls back to sequential generation.
        """
        if not self.use_vllm:
            # Default behavior (sequential) is good enough for transformers/lmdeploy backends.
            return super().generate_batch(messages, dataset=dataset)

        # Preprocess each message with BaseModel rules to keep behavior consistent with `generate`.
        pp_msgs = []
        for m in messages:
            assert self.check_content(m) in ['str', 'dict', 'liststr', 'listdict'], f'Invalid input type: {m}'
            m = self.preproc_content(m)
            assert m is not None and self.check_content(m) == 'listdict'
            for item in m:
                assert item['type'] in self.allowed_types, f'Invalid input type: {item["type"]}'
            pp_msgs.append(m)

        return self.generate_batch_inner_vllm(pp_msgs, dataset=dataset)

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)
