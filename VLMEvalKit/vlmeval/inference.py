import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *

FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'

    # To reuse records in MMBench_V11
    if dataset_name in ['MMBench', 'MMBench_CN']:
        pred_format = get_pred_file_format()
        v11_pred = f'{work_dir}/{model_name}_{dataset_name}_V11.{pred_format}'
        if osp.exists(v11_pred):
            try:
                reuse_inds = load('http://opencompass.openxlab.space/utils/mmb_reuse.pkl')
                data = load(v11_pred)
                ans_map = {x: y for x, y in zip(data['index'], data['prediction']) if x in reuse_inds}
                dump(ans_map, out_file)
            except Exception as err:
                print(type(err), err)

    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4, use_vllm=False):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    # (25.06.05) In newer version of transformers (after 4.50), with device_map='auto' and torchrun launcher,
    # Transformers automatically adopt TP parallelism, which leads to compatibility problems with VLMEvalKit
    # (In VLMEvalKit, we use torchrun to launch multiple model instances on a single node).
    # To bypass this problem, we unset `WORLD_SIZE` before building the model to not use TP parallel.
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)

    # NOTE: vLLM only achieves high throughput when multiple requests are queued together.
    uses_vllm = bool(use_vllm) or bool(getattr(model, 'use_vllm', False))
    if not uses_vllm:
        # Composite models may hold vLLM models internally (e.g., Observer/Solver).
        for _sub_name in ['observer_model', 'solver_model']:
            _sub = getattr(model, _sub_name, None)
            if _sub is not None and bool(getattr(_sub, 'use_vllm', False)):
                uses_vllm = True
                break

    # Allow overriding batch size via env; default to `api_nproc` for convenience.
    _env_bs = os.environ.get('VLLM_BATCH_SIZE')
    try:
        vllm_batch_size = int(_env_bs) if _env_bs is not None else int(api_nproc)
    except Exception:
        vllm_batch_size = int(api_nproc)

    # Respect model-side concurrency cap if present.
    if hasattr(model, 'max_num_seqs'):
        try:
            vllm_batch_size = min(vllm_batch_size, int(getattr(model, 'max_num_seqs')))
        except Exception:
            pass
    vllm_batch_size = max(1, vllm_batch_size)

    def _build_struct(i):
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            return model.build_prompt(data.iloc[i], dataset=dataset_name)
        return dataset.build_prompt(data.iloc[i])

    def _flush_batch(batch_indices, batch_structs):
        if not batch_indices:
            return
        try:
            if hasattr(model, 'generate_batch'):
                outputs = model.generate_batch(batch_structs, dataset=dataset_name)
            else:
                outputs = [model.generate(message=s, dataset=dataset_name) for s in batch_structs]
        except Exception as err:
            warnings.warn(f'Batch generation failed, fallback to sequential. {type(err)} {err}')
            outputs = []
            for s in batch_structs:
                if os.environ.get('SKIP_ERR', False) == '1':
                    _fail_msg = 'Failed to obtain answer'
                    try:
                        outputs.append(model.generate(message=s, dataset=dataset_name))
                    except RuntimeError as e:
                        torch.cuda.synchronize()
                        warnings.warn(f'{type(e)} {str(e)}')
                        outputs.append(f'{_fail_msg}: {type(e)} {str(e)}')
                else:
                    outputs.append(model.generate(message=s, dataset=dataset_name))

        # Defensive alignment: vLLM should return one output per input, but guard anyway.
        if len(outputs) != len(batch_indices):
            _default_fail = 'Failed to obtain answer'
            if len(outputs) < len(batch_indices):
                outputs = list(outputs) + [_default_fail] * (len(batch_indices) - len(outputs))
            else:
                outputs = list(outputs)[: len(batch_indices)]

        for idx, out in zip(batch_indices, outputs):
            if verbose:
                print(out, flush=True)
            res[idx] = out

        dump(res, out_file)
        torch.cuda.empty_cache()

    if uses_vllm and vllm_batch_size > 1:
        batch_structs, batch_indices = [], []
        for i in tqdm(range(lt), desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}'):
            idx = data.iloc[i]['index']
            if idx in res:
                continue
            struct = _build_struct(i)
            batch_indices.append(idx)
            batch_structs.append(struct)
            if len(batch_structs) >= vllm_batch_size:
                _flush_batch(batch_indices, batch_structs)
                batch_structs, batch_indices = [], []
        _flush_batch(batch_indices, batch_structs)
    else:
        for i in tqdm(range(lt), desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}'):
            idx = data.iloc[i]['index']
            if idx in res:
                continue

            struct = _build_struct(i)

            # If `SKIP_ERR` flag is set, the model will skip the generation if error is encountered
            if os.environ.get('SKIP_ERR', False) == '1':
                fail_msg = 'Failed to obtain answer'
                try:
                    response = model.generate(message=struct, dataset=dataset_name)
                except RuntimeError as err:
                    torch.cuda.synchronize()
                    warnings.warn(f'{type(err)} {str(err)}')
                    response = f'{fail_msg}: {type(err)} {str(err)}'
            else:
                response = model.generate(message=struct, dataset=dataset_name)
            torch.cuda.empty_cache()

            if verbose:
                print(response, flush=True)

            res[idx] = response
            if (i + 1) % 10 == 0:
                dump(res, out_file)

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model


# Add for agent evaluation
def _is_prediction_record(v):
    return isinstance(v, dict) and 'prediction' in v


def _collect_prediction_fields(records):
    predictions = []
    extra_records = []
    full_predictions = []
    has_extra_records = False
    has_full_prediction = False

    for record in records:
        if _is_prediction_record(record):
            prediction = str(record.get('prediction', ''))
            extra_record = record.get('extra_records', None)
            full_prediction = record.get('full_prediction', None)
            if full_prediction is not None:
                full_prediction = str(full_prediction)
        else:
            prediction = str(record)
            extra_record = None
            full_prediction = None

        predictions.append(prediction)
        extra_records.append(extra_record)
        full_predictions.append(full_prediction)
        has_extra_records = has_extra_records or extra_record is not None
        has_full_prediction = has_full_prediction or full_prediction is not None

    return (
        predictions,
        extra_records if has_extra_records else None,
        full_predictions if has_full_prediction else None,
    )


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(
    model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False, use_vllm=False
):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    # 使用环境变量控制的文件格式
    result_file = get_pred_file_path(work_dir, model_name, dataset_name, use_env_format=True)

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            # breakpoint()
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data(
        model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
        out_file=out_file, verbose=verbose, api_nproc=api_nproc, use_vllm=use_vllm)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        ordered_records = [data_all[x] for x in data['index']]
        prediction, extra_records, full_prediction = _collect_prediction_fields(ordered_records)
        if os.getenv('SPLIT_THINK', False):
            def split_thinking(s):
                if '</think>' in s:
                    splits = s.split('</think>')
                    prediction = splits[-1].strip()
                    if len(splits) == 2 and '<think>' in splits[0]:
                        thinking = splits[0].split('<think>')[1].strip()
                    else:
                        thinking = '</think>'.join(splits[:-1])
                        thinking += '</think>'
                        warnings.warn('Failed to parse thinking, multiple </think> tags or missing <think> tag.')
                else:
                    thinking = ''
                    prediction = s
                return (prediction, thinking)
            split_func = model.split_thinking if hasattr(model, 'split_thinking') else split_thinking
            print(f'Prediction format: {os.getenv("SPLIT_THINK")},splitting func: {split_func}')
            tups = [split_func(x) for x in prediction]
            data['prediction'] = [x[0] for x in tups]
            data['thinking'] = [x[1] for x in tups]
        else:
            data['prediction'] = prediction
        if extra_records is not None:
            data['extra_records'] = extra_records
        if full_prediction is not None:
            data['full_prediction'] = full_prediction
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    if world_size > 1:
        dist.barrier()
    return model
