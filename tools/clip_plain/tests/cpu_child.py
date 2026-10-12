"""Test-only actual subprocess harness; never invoked by production controller."""

import json
import random
import sys
from pathlib import Path

import torch
from torch import nn

from tools.benchmark.compact_checkpoint import frozen_fingerprint
from tools.clip_plain import policy, train
from tools.clip_plain.data import admit_data
from tools.clip_plain.evaluate import score_dev, validate_report
from tools.clip_plain.state import load_weights, restore_training
from tools.clip_s2_inference.tests.helpers import language
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.official_sampler import OfficialBatchSampler
from tools.dinov3.storage import atomic_json
from tools.followup.recipe import PromptPool


class Visual(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(1024, 32)

    def forward(self, s, sm, m, mm):
        return self.project(s), sm


def main():
    assert not torch.cuda.is_initialized()
    torch.cuda.is_available = lambda: False
    stage = sys.argv[1]
    c = json.loads(Path(sys.argv[2]).read_text())
    policy.COUNTS = dict(train=8, dev=3, test=2)
    a = admit_data(c)
    torch.manual_seed(17)
    m = nn.Module()
    m.visual_adapter = Visual()
    m.language_model = language()
    m.rng = random.Random(23)
    base = frozen_fingerprint(m)
    recipe = policy.recipe(c)
    i = dict(
        policy=policy.POLICY,
        seed=0,
        recipe=recipe,
        data=a["identity"],
        evaluation_policy_hash=digest(policy.evaluation_policy()),
    )
    if stage == "initialize":
        atomic_json(Path(sys.argv[3]) / "identity.json", i)
        return
    job = json.loads(Path(sys.argv[3]).read_text())
    run = Path(job["run_root"])
    pool = PromptPool(a["datasets"]["train"].records, 0)
    state = load_weights(Path(job["checkpoint"]), m, i, base) if job.get("checkpoint") else None
    if stage == "train":
        opt = torch.optim.AdamW(m.parameters(), lr=0.001)
        sampler = OfficialBatchSampler(a["ids"]["train"], seed=0, batch_size=8)
        if state is not None:
            restore_training(state, m, opt, sampler)

        def update(model, optimizer, batch, k, recipe, pool):
            optimizer.zero_grad()
            v = model.visual_adapter.project(batch.spatial)
            loss = v.square().mean() * model.rng.random()
            loss.backward()
            optimizer.step()
            return dict(translation_loss=None)

        train.update = update
        train.epoch_guard = lambda *args: None
        r = train.train_to_boundary(
            m,
            opt,
            sampler,
            a,
            pool,
            i,
            run,
            job["epoch"],
            base,
            torch.device("cpu"),
            lambda e: atomic_json(Path(job["progress_path"]), e),
        )
    else:
        report = score_dev(m, a, pool, job, on_event=lambda e: None)
        validate_report(report, job, a)
        report["checkpoint_path"] = job["checkpoint"]
        target = run / "dev" / f"dev-epoch-{job['epoch']}.json"
        atomic_json(target, report)
        r = dict(
            status="complete",
            stage="dev",
            completed_updates=job["epoch"],
            durable_updates=job["epoch"],
            report=str(target),
            report_sha256=sha256_file(target),
        )
    atomic_json(Path(job["result_path"]), r)
    assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    main()
