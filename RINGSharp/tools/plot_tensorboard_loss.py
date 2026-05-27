import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)
for path in (WORKSPACE_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


DEFAULT_TAGS = ['PR_Loss', 'Total_loss', 'Train_loss', 'Val_loss']


def parse_args():
    parser = argparse.ArgumentParser(description='Plot TensorBoard scalar loss curves from event files.')
    parser.add_argument(
        '--event_path',
        required=True,
        help='TensorBoard event file or directory containing events.out.tfevents.* files.',
    )
    parser.add_argument(
        '--tags',
        nargs='+',
        default=DEFAULT_TAGS,
        help='Scalar tags to plot. Missing tags are skipped.',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output image path. If omitted, saves loss_curve.png under event directory.',
    )
    parser.add_argument('--pdf', action='store_true', help='Also save a PDF next to the image output.')
    parser.add_argument('--title', default='Training Loss', help='Figure title.')
    parser.add_argument('--smooth', type=float, default=0.0, help='EMA smoothing factor in [0,1). 0 disables smoothing.')
    parser.add_argument('--list_tags', action='store_true', help='List available scalar tags and exit.')
    return parser.parse_args()


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def find_event_files(event_path: str) -> List[str]:
    event_path = expand_path(event_path)
    if os.path.isfile(event_path):
        return [event_path]
    if not os.path.isdir(event_path):
        raise FileNotFoundError(f'Cannot access event path: {event_path}')

    files = []
    for root, _, names in os.walk(event_path):
        for name in names:
            if name.startswith('events.out.tfevents'):
                files.append(os.path.join(root, name))
    files.sort(key=lambda p: os.path.getmtime(p))
    if not files:
        raise FileNotFoundError(f'No TensorBoard event files found under: {event_path}')
    return files


def load_scalar_events(event_files: List[str]) -> Tuple[Dict[str, List[Tuple[int, float, float]]], List[str]]:
    from tensorboard.backend.event_processing import event_accumulator

    scalar_events: Dict[str, List[Tuple[int, float, float]]] = {}
    for event_file in event_files:
        accumulator = event_accumulator.EventAccumulator(event_file, size_guidance={'scalars': 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get('scalars', []):
            scalar_events.setdefault(tag, [])
            for event in accumulator.Scalars(tag):
                scalar_events[tag].append((int(event.step), float(event.wall_time), float(event.value)))

    for tag, events in scalar_events.items():
        events.sort(key=lambda item: (item[0], item[1]))
        dedup = {}
        for step, wall_time, value in events:
            dedup[step] = (step, wall_time, value)
        scalar_events[tag] = [dedup[step] for step in sorted(dedup.keys())]

    return scalar_events, sorted(scalar_events.keys())


def smooth_values(values: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 0.0:
        return values
    if factor >= 1.0:
        raise ValueError('--smooth must be in [0,1)')
    smoothed = np.empty_like(values, dtype=np.float64)
    last = float(values[0])
    for idx, value in enumerate(values):
        last = factor * last + (1.0 - factor) * float(value)
        smoothed[idx] = last
    return smoothed


def default_output_path(event_path: str) -> str:
    event_path = expand_path(event_path)
    out_dir = event_path if os.path.isdir(event_path) else os.path.dirname(event_path)
    return os.path.join(out_dir, 'loss_curve.png')


def plot_loss_curves(
    scalar_events: Dict[str, List[Tuple[int, float, float]]],
    tags: List[str],
    output_path: str,
    title: str,
    smooth: float = 0.0,
    save_pdf: bool = False,
) -> List[str]:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plotted = []
    plt.figure(figsize=(10, 6))
    for tag in tags:
        if tag not in scalar_events or len(scalar_events[tag]) == 0:
            continue
        steps = np.asarray([event[0] for event in scalar_events[tag]], dtype=np.int64)
        values = np.asarray([event[2] for event in scalar_events[tag]], dtype=np.float64)
        values = smooth_values(values, smooth)
        plt.plot(steps, values, linewidth=1.8, label=tag)
        plotted.append(tag)

    if not plotted:
        raise ValueError(f'None of the requested tags were found: {tags}')

    plt.title(title)
    plt.xlabel('Step / Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.45)
    plt.legend()
    plt.tight_layout()

    output_path = expand_path(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    if save_pdf:
        pdf_path = os.path.splitext(output_path)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    return plotted


def main():
    args = parse_args()
    event_path = expand_path(args.event_path)
    event_files = find_event_files(event_path)
    scalar_events, available_tags = load_scalar_events(event_files)

    if args.list_tags:
        print('Available scalar tags:')
        for tag in available_tags:
            print(f'  {tag}')
        return

    output_path = args.output or default_output_path(event_path)
    plotted = plot_loss_curves(
        scalar_events,
        tags=args.tags,
        output_path=output_path,
        title=args.title,
        smooth=args.smooth,
        save_pdf=args.pdf,
    )

    print(f'Event files: {len(event_files)}')
    for event_file in event_files:
        print(f'  {event_file}')
    print(f'Plotted tags: {plotted}')
    print(f'Saved loss curve: {expand_path(output_path)}')
    if args.pdf:
        print(f'Saved PDF: {os.path.splitext(expand_path(output_path))[0] + ".pdf"}')


if __name__ == '__main__':
    main()
