import csv
from tensorboard.backend.event_processing import event_accumulator

EVENT_FILE = "/home/wyz/RINGSharp/RINGSharp/results/tensorboard/ring_sharp_vl_pr_nclt/events.out.tfevents.1777892743.autodl-container-7xvlkqa815-da99048f"
OUT_CSV = "/home/wyz/RINGSharp/RINGSharp/results/tensorboard/ring_sharp_vl_pr_nclt/loss_from_tensorboard.csv"

TAGS = ["PR_Loss", "Total_loss", "Train_loss"]

ea = event_accumulator.EventAccumulator(
    EVENT_FILE,
    size_guidance={"scalars": 0},
)
ea.Reload()

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["tag", "step", "wall_time", "value"])

    for tag in TAGS:
        events = ea.Scalars(tag)
        for e in events:
            writer.writerow([tag, e.step, e.wall_time, e.value])

print("saved:", OUT_CSV)
