"""Paper calibration: 128 examples from five local VLMEvalKit TSVs.

Uses near equal source counts with deterministic reservoir sampling.
No fallback sources, duplicated backfill, or automatic dataset downloads.
"""
import argparse
import ast
import base64
import csv
import io
import json
import os
from pathlib import Path
import random
from PIL import Image

SOURCES = ('DocVQA_VAL', 'InfoVQA_VAL', 'ScienceQA_VAL', 'TextVQA_VAL', 'ChartQA_TEST')


def rows(path):
    csv.field_size_limit(2**31 - 1)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream, delimiter='\t')


def qa(row):
    question = row.get('question', '').strip()
    answer = row.get('answer', '').strip()
    if answer.startswith('['):
        try:
            parsed = ast.literal_eval(answer)
            if isinstance(parsed, list):
                answer = str(parsed[0]) if parsed else ''
        except (ValueError, SyntaxError):
            pass
    for key in 'ABCDE':
        option = row.get(key, '').strip()
        if option and option.lower() != 'nan':
            question += f'\n{key}. {option}'
    if answer.lower() in ('nan', 'none'):
        answer = ''
    return question, answer


def build(lmudata, output, total_samples=128, seed=0):
    lmudata, output = Path(lmudata).resolve(), Path(output).resolve()
    if total_samples < 5:
        raise ValueError('At least five samples required for all paper sources')
    missing = [str(lmudata / (n + '.tsv')) for n in SOURCES if not (lmudata / (n + '.tsv')).is_file()]
    if missing:
        raise FileNotFoundError('Missing paper datasets: ' + ', '.join(missing))
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite calibration set: {output}')
    selected = []
    for index, source in enumerate(SOURCES):
        count = total_samples // 5 + (index < total_samples % 5)
        rng = random.Random(seed + index)
        pool, seen = [], 0
        tsv = lmudata / (source + '.tsv')
        # Reservoir sampling bounds memory for multi-GB embedded-image TSVs.
        for row in rows(tsv):
            question, answer = qa(row)
            if not question or not answer:
                continue
            seen += 1
            if len(pool) < count:
                pool.append(row)
            else:
                position = rng.randrange(seen)
                if position < count:
                    pool[position] = row
        if len(pool) < count:
            raise ValueError(f'{source}: need {count} valid QA rows, found {len(pool)}')
        refs = {r.get('image', '') for r in pool if r.get('image', '').isdigit()}
        resolved = {}
        while refs:
            pending = set()
            for row in rows(tsv):
                if row.get('index') in refs:
                    encoded = row.get('image', '')
                    resolved[row['index']] = encoded
                    if encoded.isdigit() and encoded not in resolved:
                        pending.add(encoded)
            unresolved = refs - resolved.keys()
            if unresolved:
                raise ValueError(f'{source}: unresolved image references {unresolved}')
            refs = pending - resolved.keys()
        for row in pool:
            raw_path = row.get('image_path', '')
            image_dir = lmudata / 'images' / source
            candidates = [Path(raw_path)] if raw_path else []
            candidates += [image_dir / (row.get('index', '') + ext) for ext in ('.jpg', '.png', '.jpeg', '.webp')]
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                encoded, visited = row.get('image', ''), set()
                while encoded.isdigit():
                    if encoded in visited:
                        raise ValueError(f'{source}: cyclic image reference {encoded}')
                    visited.add(encoded)
                    encoded = resolved[encoded]
                try:
                    image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
                    image.load()
                except Exception as exc:
                    raise ValueError(f'{source} row {row.get("index")}: missing/invalid image') from exc
                path = output.parent / (output.stem + '_images') / source / (str(row['index']) + '.png')
                path.parent.mkdir(parents=True, exist_ok=True)
                image.convert('RGB').save(path)
            with Image.open(path) as image:
                image.verify()
            question, answer = qa(row)
            selected.append(dict(image_path=str(path.resolve()), source=source, conversations=[
                dict(role='user', content='<image>\n' + question), dict(role='assistant', content=answer)]))
        print(f'{source}: {count}', flush=True)
    random.Random(seed).shuffle(selected)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'Saved {len(selected)} samples to {output}', flush=True)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmudata', default=os.environ.get('LMUData', str(Path.home() / 'LMUData')))
    parser.add_argument('--output', required=True)
    parser.add_argument('--total_samples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    build(args.lmudata, args.output, args.total_samples, args.seed)


if __name__ == '__main__':
    main()
