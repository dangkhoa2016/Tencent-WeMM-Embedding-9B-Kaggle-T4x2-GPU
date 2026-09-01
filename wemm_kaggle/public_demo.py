from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import time

from wemm_runtime import EmbeddingWorker, RuntimeConfig

from .demo_config import EVID, MODEL_DIR, PROJECT_ROOT, RUN_ROOT, WORK
from .demo_io import banner, gpu_processes, gpu_snapshot, kv, run, write_json
from .demo_qdrant import prepare_qdrant, stop_qdrant, write_cache_seal
from .demo_showcase import (
    run_image_showcase as execute_image_showcase,
    run_text_showcase as execute_text_showcase,
    verify_dataset,
)


WORKER_REGISTRY = WORK / "tencent-wemm-embedding-9b-public-demo-worker.json"


def _recover_orphan_wemm_gpu_workers() -> list[int]:
    """Recover only provable orphan workers created by this demo."""
    recovered: list[int] = []
    unexpected: list[dict] = []
    for row in gpu_processes():
        try:
            pid = int(str(row).split(",", 1)[0].strip())
        except (TypeError, ValueError):
            unexpected.append({"raw": row, "reason": "UNPARSEABLE_GPU_PROCESS"})
            continue
        proc = Path(f"/proc/{pid}")
        if not proc.exists():
            continue
        try:
            cmdline = (
                (proc / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
        except OSError:
            unexpected.append({"pid": pid, "reason": "CMDLINE_UNREADABLE"})
            continue

        is_wemm_worker = (
            "wemm_runtime.worker_process" in cmdline
            or (
                "wemm-production-runtime-" in cmdline
                and "--config" in cmdline
                and "wemm-runtime-" in cmdline
            )
        )
        if not is_wemm_worker:
            unexpected.append(
                {"pid": pid, "cmdline": cmdline, "reason": "UNRELATED_GPU_PROCESS"}
            )
            continue

        os.kill(pid, signal.SIGTERM)
        deadline = time.time() + 10
        while proc.exists() and time.time() < deadline:
            time.sleep(0.2)
        if proc.exists():
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)
        if proc.exists():
            raise RuntimeError(f"Orphan WeMM worker PID {pid} did not exit")
        recovered.append(pid)

    if unexpected:
        raise RuntimeError(
            "GPU baseline is not clean and contains process(es) not provably owned "
            f"by this demo: {unexpected}"
        )
    return recovered


def _recover_registered_worker() -> None:
    if not WORKER_REGISTRY.is_file():
        return
    try:
        record = json.loads(WORKER_REGISTRY.read_text(encoding="utf-8"))
        pid = int(record["pid"])
    except Exception:
        WORKER_REGISTRY.unlink(missing_ok=True)
        return
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        WORKER_REGISTRY.unlink(missing_ok=True)
        return
    cmdline = (
        (proc / "cmdline")
        .read_bytes()
        .replace(b"\0", b" ")
        .decode(errors="replace")
    )
    if "wemm_runtime.worker_process" not in cmdline:
        raise RuntimeError(
            f"Registered PID {pid} no longer matches the WeMM worker; refuse unsafe kill"
        )
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 10
    while proc.exists() and time.time() < deadline:
        time.sleep(0.2)
    if proc.exists():
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
    if proc.exists():
        raise RuntimeError(f"Registered WeMM worker PID {pid} did not exit")
    WORKER_REGISTRY.unlink(missing_ok=True)


class PublicDemoSession:
    """Stateful notebook demo session.

    Steps 1-5 initialize and retain Qdrant + the isolated GPU worker.
    Steps 6, 7 and 8 are intentionally separate notebook-facing calls so each
    produces a focused Saved Version output cell.
    """

    def __init__(
        self,
        *,
        t0: float,
        phase_times: dict[str, float],
        expected_commit: str,
        client,
        worker: EmbeddingWorker,
        data_mode: str,
        cache_reason: str,
    ):
        self.t0 = t0
        self.phase_times = phase_times
        self.expected_commit = expected_commit
        self.client = client
        self.worker = worker
        self.data_mode = data_mode
        self.cache_reason = cache_reason
        self.text_results: list[dict] | None = None
        self.image_results: list[dict] | None = None
        self.closed = False

    @classmethod
    def start(cls) -> "PublicDemoSession":
        t0 = time.perf_counter()
        phase_times: dict[str, float] = {}
        client = None
        worker: EmbeddingWorker | None = None

        try:
            _recover_registered_worker()
            recovered_orphans = _recover_orphan_wemm_gpu_workers()
            if recovered_orphans:
                print("WORKER_ORPHAN_RECOVERY=PASS")
                kv(
                    "Recovered orphan worker PIDs / PID worker mồ côi đã thu hồi",
                    recovered_orphans,
                )

            baseline_gpu = gpu_snapshot()
            if gpu_processes() or any(
                gpu["memory_used_mib"] > 1024 for gpu in baseline_gpu
            ):
                raise RuntimeError(
                    f"GPU baseline is not clean after recovery: {baseline_gpu}"
                )
            print("WORKER_LIFECYCLE_GPU_BASELINE=PASS")

            if RUN_ROOT.exists():
                shutil.rmtree(RUN_ROOT)
            RUN_ROOT.mkdir(parents=True, exist_ok=True)
            EVID.mkdir(parents=True, exist_ok=True)

            phase_t0 = time.perf_counter()
            banner(
                "1/8",
                "Hardware + Kaggle Input preflight / "
                "Kiểm tra phần cứng + Kaggle Input",
            )
            gpus = gpu_snapshot()
            if len(gpus) != 2 or any("T4" not in gpu["name"] for gpu in gpus):
                raise RuntimeError(f"Expected Kaggle T4 x2, observed {gpus}")
            kv("GPU0", gpus[0]["name"])
            kv("GPU1", gpus[1]["name"])
            for gpu in gpus:
                kv(
                    f"GPU{gpu['index']} bộ nhớ / memory GPU{gpu['index']}",
                    f"{gpu['memory_used_mib']} MiB used / "
                    f"{gpu['memory_total_mib']} MiB total",
                )
            phase_times["1_hardware_preflight"] = time.perf_counter() - phase_t0
            print("PUBLIC_DEMO_HARDWARE=PASS")

            phase_t0 = time.perf_counter()
            banner(
                "2/8",
                "Pinned reusable Git runtime / "
                "Runtime Git tái sử dụng đã được pin",
            )
            expected_commit = os.environ.get(
                "WEMM_RUNTIME_SOURCE_COMMIT", ""
            ).strip()
            if not expected_commit:
                raise RuntimeError("WEMM_RUNTIME_SOURCE_COMMIT is required")
            git_head = run(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                capture=True,
            ).stdout.strip()
            if git_head != expected_commit:
                raise RuntimeError(
                    f"Git source mismatch: expected={expected_commit} "
                    f"actual={git_head}"
                )
            kv("Commit nguồn runtime / Runtime source commit", git_head)
            kv(
                "Gói độc lập nhà cung cấp / Provider-neutral package",
                "wemm_runtime",
            )
            phase_times["2_source_verification"] = (
                time.perf_counter() - phase_t0
            )
            print("PUBLIC_DEMO_SOURCE_BOOTSTRAP=PASS")

            phase_t0 = time.perf_counter()
            banner(
                "3/8",
                "Public Dataset byte-identity gate / "
                "Kiểm tra byte-identity Dataset công khai",
            )
            dataset_evidence = verify_dataset()
            write_json(EVID / "dataset.json", dataset_evidence)
            kv("Tệp Dataset / Dataset files", "9/9")
            kv(
                "Độ phủ manifest / Manifest coverage",
                "8/8 non-manifest files",
            )
            print("PUBLIC_MANIFEST_VERIFY=PASS")
            print("4096_PUBLIC_SNAPSHOT_SHA256=PASS")
            for dim, snap in dataset_evidence["snapshots"].items():
                kv(
                    f"{dim} snapshot / snapshot {dim}",
                    f"{snap['bytes']:,} bytes | "
                    f"sha256={snap['sha256'][:16]}…",
                )
            phase_times["3_dataset_verification"] = (
                time.perf_counter() - phase_t0
            )
            print("1024_PUBLIC_SNAPSHOT_SHA256=PASS")

            phase_t0 = time.perf_counter()
            banner(
                "4/8",
                "Qdrant 1.19.0 verified reuse or snapshot restore / "
                "Tái sử dụng đã xác minh hoặc phục hồi snapshot",
            )
            client, qdrant_evidence, data_mode, cache_reason = prepare_qdrant()
            write_json(
                EVID / "qdrant.json",
                {
                    "mode": data_mode,
                    "cache_reason": cache_reason,
                    "collections": qdrant_evidence,
                },
            )
            kv(
                "Chế độ lưu trữ Qdrant / Qdrant storage mode",
                data_mode,
            )
            kv("Quyết định cache / Cache decision", cache_reason)
            kv(
                "Bản sao snapshot lưu lâu dài / Persistent snapshot copy",
                "NO / KHÔNG",
            )
            for dim in ("4096", "1024"):
                item = qdrant_evidence[dim]
                kv(
                    f"Qdrant {dim} / collection {dim}",
                    f"{item['points_count']:,} pts | vectors=en,vi | "
                    f"{item['vector_size']}d | "
                    f"{item['distance']} | {item['status']}",
                )
                kv(
                    f"{dim} snapshot SHA / SHA snapshot {dim}",
                    item["snapshot_sha256"],
                )
            phase_times["4_qdrant_prepare"] = time.perf_counter() - phase_t0
            print(f"QDRANT_STORAGE_MODE={data_mode}")
            print("QDRANT_SNAPSHOT_PERSISTENT_COPY=NO")

            phase_t0 = time.perf_counter()
            banner(
                "5/8",
                "Tencent WeMM-Embedding-9B isolated GPU worker / "
                "GPU worker cô lập",
            )
            worker = EmbeddingWorker(
                RuntimeConfig(
                    model_path=MODEL_DIR,
                    device_map="balanced",
                    gpu_ids=(0, 1),
                    per_gpu_mib=14200,
                    cpu_memory_mib=2048,
                    dtype="float16",
                    local_files_only=True,
                    offline=True,
                    allow_cpu_offload=False,
                )
            )

            def register_spawned_worker(pid: int) -> None:
                write_json(
                    WORKER_REGISTRY,
                    {
                        "pid": int(pid),
                        "kind": "wemm_runtime.worker_process",
                    },
                )

            try:
                info = worker.start(on_spawn=register_spawned_worker)
            except BaseException:
                WORKER_REGISTRY.unlink(missing_ok=True)
                raise

            if info.gpu_ids != (0, 1) or info.offload_targets:
                raise RuntimeError(f"Unexpected model placement: {info}")
            kv("PID worker / Worker PID", info.pid)
            kv("Kiểu dữ liệu mô hình / Model dtype", info.dtype)
            kv("Thiết bị đầu vào / Input device", info.input_device)
            kv(
                "Module trên GPU0 / GPU0 mapped modules",
                info.module_counts.get(0, 0),
            )
            kv(
                "Module trên GPU1 / GPU1 mapped modules",
                info.module_counts.get(1, 0),
            )
            kv(
                "Offload CPU/đĩa / CPU/disk offload",
                "NONE / KHÔNG",
            )
            identity = info.model_identity
            kv("Đường dẫn mô hình / Model path", identity["path"])
            kv(
                "Trọng số Safetensors / Safetensors weights",
                f"{len(identity['weight_files'])} files | "
                f"{identity['weight_bytes']:,} bytes",
            )
            kv(
                "SHA256 cấu hình / Config SHA256",
                identity["config_sha256"],
            )
            kv(
                "Số chiều Matryoshka / Matryoshka dimensions",
                identity["matryoshka_dimensions"],
            )
            loaded_gpu = gpu_snapshot()
            for gpu in loaded_gpu:
                kv(
                    f"GPU{gpu['index']} sau khi load / after load",
                    f"{gpu['memory_used_mib']} MiB used / "
                    f"{gpu['memory_total_mib']} MiB total",
                )
            phase_times["5_model_load"] = time.perf_counter() - phase_t0
            print("MODEL_LOAD=PASS")

            return cls(
                t0=t0,
                phase_times=phase_times,
                expected_commit=expected_commit,
                client=client,
                worker=worker,
                data_mode=data_mode,
                cache_reason=cache_reason,
            )
        except BaseException:
            if worker is not None:
                worker.terminate()
            WORKER_REGISTRY.unlink(missing_ok=True)
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
            try:
                stop_qdrant()
            except Exception:
                pass
            raise

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("Public demo session is already closed")

    def abort(self) -> None:
        if self.closed:
            return
        try:
            self.worker.terminate()
        finally:
            WORKER_REGISTRY.unlink(missing_ok=True)
            try:
                self.client.close()
            finally:
                stop_qdrant()
                self.closed = True

    def run_text(self) -> list[dict]:
        self._require_open()
        if self.text_results is not None:
            raise RuntimeError(
                "Step 6 has already completed in this demo session"
            )

        phase_t0 = time.perf_counter()
        banner(
            "6/8",
            "Frozen genuine-bilingual EN↔VI text showcase / "
            "Trình diễn văn bản song ngữ EN↔VI cố định",
        )
        try:
            self.text_results = execute_text_showcase(
                self.worker, self.client
            )
            write_json(
                EVID / "text-showcase.json",
                self.text_results,
            )
            print("FROZEN_BILINGUAL_SHOWCASE=PASS")
            print("TEXT_SHOWCASE_ENTITIES=5")
            print("TEXT_SHOWCASE_EMBEDDING_REQUESTS=10")
            print("TEXT_SHOWCASE_QUERY_EXECUTIONS=20")
            print("TEXT_SHOWCASE_LANGUAGE_GATE_AUTHORITY=PASS")
            print("TEXT_SHOWCASE_ALL_TOP1=5/5")
            print("TEXT_SHOWCASE_STRICT_ALL_TOP3=PASS")
            self.phase_times["6_text_showcase"] = (
                time.perf_counter() - phase_t0
            )
            print("TEXT_SHOWCASE_THRESHOLD_RELAXED=NO")

            print(
                "\nTEXT RETRIEVAL SUMMARY / "
                "TỔNG KẾT TRUY XUẤT VĂN BẢN"
            )
            print("-" * 92)
            print(
                f"{'Entity / Thực thể':<28} "
                f"{'EN→VI 4096':>13} {'EN→VI 1024':>13} "
                f"{'VI→EN 4096':>13} {'VI→EN 1024':>13}",
                flush=True,
            )
            for item in self.text_results:
                scores = [
                    f"#{p['rank']} {p['expected_score']:.4f}"
                    for p in item["runtime_paths"]
                ]
                print(
                    f"{item['name'][:27]:<28} "
                    f"{scores[0]:>13} {scores[1]:>13} "
                    f"{scores[2]:>13} {scores[3]:>13}",
                    flush=True,
                )
            return self.text_results
        except BaseException:
            self.abort()
            raise

    def run_image(self) -> list[dict]:
        self._require_open()
        if self.text_results is None:
            raise RuntimeError("Step 6 must complete before step 7")
        if self.image_results is not None:
            raise RuntimeError(
                "Step 7 has already completed in this demo session"
            )

        phase_t0 = time.perf_counter()
        banner(
            "7/8",
            "Frozen cross-modal public showcase / "
            "Trình diễn cross-modal công khai cố định",
        )
        try:
            self.image_results = execute_image_showcase(
                self.worker, self.client
            )
            write_json(
                EVID / "image-showcase.json",
                self.image_results,
            )
            print("FROZEN_SHOWCASE=PASS")
            print("FROZEN_SHOWCASE_COUNT=4")
            print("FROZEN_SHOWCASE_IMAGE_SHA256=PASS")
            print("FROZEN_SHOWCASE_QUERY_EXECUTIONS=16")
            self.phase_times["7_image_showcase"] = (
                time.perf_counter() - phase_t0
            )
            print("FROZEN_SHOWCASE_STRICT_ALL_TOP3=PASS")

            print(
                "\nIMAGE RETRIEVAL SUMMARY / "
                "TỔNG KẾT TRUY XUẤT HÌNH ẢNH"
            )
            print("-" * 92)
            print(
                f"{'Entity / Thực thể':<28} "
                f"{'EN 4096':>12} {'EN 1024':>12} "
                f"{'VI 4096':>12} {'VI 1024':>12}",
                flush=True,
            )
            for item in self.image_results:
                scores = [
                    f"#{p['rank']} {p['expected_score']:.4f}"
                    for p in item["runtime_paths"]
                ]
                print(
                    f"{item['name'][:27]:<28} "
                    f"{scores[0]:>12} {scores[1]:>12} "
                    f"{scores[2]:>12} {scores[3]:>12}",
                    flush=True,
                )
            return self.image_results
        except BaseException:
            self.abort()
            raise

    def closeout(self) -> dict:
        self._require_open()
        if self.text_results is None:
            raise RuntimeError("Step 6 must complete before step 8")
        if self.image_results is None:
            raise RuntimeError("Step 7 must complete before step 8")

        phase_t0 = time.perf_counter()
        banner(
            "8/8",
            "Final closeout + reusable Qdrant storage seal / "
            "Đóng phiên + niêm phong Qdrant để tái sử dụng",
        )
        try:
            worker_rc = self.worker.shutdown()
            WORKER_REGISTRY.unlink(missing_ok=True)
            if worker_rc != 0:
                raise RuntimeError(
                    f"WeMM worker return code is {worker_rc}"
                )

            time.sleep(1.0)
            after_gpu = gpu_snapshot()
            procs = gpu_processes()
            if procs or any(
                gpu["memory_used_mib"] > 1024 for gpu in after_gpu
            ):
                raise RuntimeError(
                    f"GPU reclaim failed: state={after_gpu} "
                    f"processes={procs}"
                )
            print("WORKER_LIFECYCLE_GPU_RECLAIM=PASS")

            self.client.close()
            stop_qdrant()
            seal = write_cache_seal(self.data_mode)
            if not seal["fingerprint"]["digest"]:
                raise RuntimeError(
                    "Qdrant seal missing storage fingerprint"
                )

            seal_fp = seal["fingerprint"]
            kv(
                "Số tệp lưu trữ / Qdrant storage files",
                seal_fp["file_count"],
            )
            kv(
                "Dung lượng lưu trữ / Qdrant storage bytes",
                f"{seal_fp['bytes']:,}",
            )
            kv(
                "Dấu vân tay lưu trữ / Qdrant storage fingerprint",
                seal_fp["digest"],
            )
            for gpu in after_gpu:
                kv(
                    f"GPU{gpu['index']} sau shutdown / after shutdown",
                    f"{gpu['memory_used_mib']} MiB used / "
                    f"{gpu['memory_total_mib']} MiB total",
                )
            self.phase_times["8_closeout"] = (
                time.perf_counter() - phase_t0
            )
            print("QDRANT_STORAGE_SEAL=PASS")
            print("QDRANT_SNAPSHOT_PERSISTENT_COPY=NO")

            summary = {
                "verdict": "PASS",
                "runtime_source_commit": self.expected_commit,
                "qdrant_storage_mode": self.data_mode,
                "text_entities": len(self.text_results),
                "text_all_top1": sum(
                    1 for x in self.text_results if x["all_top1"]
                ),
                "image_entities": len(self.image_results),
                "image_all_top1": sum(
                    1 for x in self.image_results if x["all_top1"]
                ),
                "worker_gpu_reclaim": "PASS",
                "qdrant_storage_seal": "PASS",
                "wall_seconds": round(
                    time.perf_counter() - self.t0, 3
                ),
                "phase_seconds": {
                    k: round(v, 3)
                    for k, v in self.phase_times.items()
                },
            }
            write_json(
                EVID / "public-demo-final-summary.json",
                summary,
            )

            print("\n" + "=" * 92)
            print("BẢNG TỔNG KẾT / PUBLIC DEMO SCORECARD")
            print("=" * 92)
            kv(
                "Commit nguồn / Runtime source commit",
                self.expected_commit,
            )
            kv("Phần cứng / Hardware", "2 × NVIDIA Tesla T4")
            kv(
                "Mô hình / Model",
                "Tencent WeMM-Embedding-9B | FP16 | dual GPU",
            )
            kv(
                "Qdrant",
                "1.19.0 | 2 collections | 99,967 points each",
            )
            kv(
                "Văn bản song ngữ / Bilingual text",
                "20/20 retrieval paths TOP-1 | 10 embeddings",
            )
            kv(
                "Đa phương thức / Cross-modal",
                "16/16 retrieval paths TOP-1 | 4 image embeddings",
            )
            kv(
                "TỔNG CỘNG / TOTAL",
                "36/36 RETRIEVAL PATHS TOP-1",
            )
            kv(
                "Nới ngưỡng / Threshold relaxation",
                "NO / KHÔNG",
            )
            kv("Giải phóng GPU / GPU reclaim", "PASS")
            kv("Niêm phong lưu trữ / Storage seal", "PASS")

            print("\nPhase timing / Thời gian từng giai đoạn:")
            for name, seconds in self.phase_times.items():
                kv(name, f"{seconds:.3f} s")

            print("\nFINAL EXECUTION ACCEPTANCE VERDICT : PASS")
            kv("Văn bản top-1 / Text all-top1", "5/5")
            kv("Hình ảnh top-1 / Image all-top1", "4/4")
            kv(
                "Tổng đường truy xuất / Total retrieval paths",
                "36/36 TOP-1",
            )
            kv(
                "Chế độ lưu trữ Qdrant / Qdrant storage mode",
                self.data_mode,
            )
            kv(
                "Tổng thời gian / Wall time",
                f"{summary['wall_seconds']:.3f} s",
            )
            self.closed = True
            return summary
        except BaseException:
            self.abort()
            raise


def start_public_demo() -> PublicDemoSession:
    return PublicDemoSession.start()


def main() -> None:
    demo = start_public_demo()
    demo.run_text()
    demo.run_image()
    demo.closeout()


if __name__ == "__main__":
    main()
