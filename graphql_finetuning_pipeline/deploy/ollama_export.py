from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from graphql_finetuning_pipeline.utils.io import ensure_dir

QUANT_LEVELS = ("f16", "Q8_0", "Q4_K_M")

_REQUIRED_ST_FILES = ("modules.json", "config_sentence_transformers.json")


def _find_transformer_subdir(model_dir: Path) -> Path:
    modules_path = model_dir / "modules.json"
    if not modules_path.exists():
        raise FileNotFoundError(f"Missing modules.json in {model_dir}")
    modules = json.loads(modules_path.read_text(encoding="utf-8"))
    for mod in modules:
        mod_type = mod.get("type", "")
        if mod_type.endswith(".Transformer"):
            sub = (model_dir / mod.get("path", "")).resolve()
            if (sub / "config.json").exists():
                return sub
    for fallback_name in ("1_Transformer", ""):
        fallback = model_dir / fallback_name
        if (fallback / "config.json").exists():
            return fallback
    raise FileNotFoundError(f"Could not locate HF transformer sub-directory under {model_dir}")


def _validate_model_dir(model_dir: Path) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir does not exist: {model_dir}")
    missing = [name for name in _REQUIRED_ST_FILES if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{model_dir} is not a SentenceTransformer directory — missing: {missing}")


def _resolve_llama_cpp_dir(llama_cpp_dir: str | Path | None) -> Path:
    candidate = llama_cpp_dir or os.environ.get("LLAMA_CPP_DIR")
    if not candidate:
        raise ValueError("llama.cpp directory not provided (pass --llama-cpp-dir or set LLAMA_CPP_DIR)")
    path = Path(candidate).expanduser().resolve()
    if not (path / "convert_hf_to_gguf.py").exists():
        raise FileNotFoundError(f"{path}/convert_hf_to_gguf.py not found — is this a llama.cpp checkout?")
    return path


def _find_quantize_binary(llama_cpp_dir: Path) -> Path:
    for rel in ("build/bin/llama-quantize", "build/bin/quantize", "llama-quantize", "quantize"):
        candidate = llama_cpp_dir / rel
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        f"Could not locate llama-quantize binary under {llama_cpp_dir}. "
        "Build llama.cpp (cmake --build build) before exporting quantizations beyond f16."
    )


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[ollama-export] $ {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(str(c) for c in cmd)}")


def _convert_to_gguf(transformer_dir: Path, out_file: Path, llama_cpp_dir: Path) -> None:
    script = llama_cpp_dir / "convert_hf_to_gguf.py"
    cmd = [
        sys.executable,
        str(script),
        str(transformer_dir),
        "--outfile",
        str(out_file),
        "--outtype",
        "f16",
    ]
    _run(cmd)


def _quantize(source_gguf: Path, target_gguf: Path, level: str, llama_cpp_dir: Path) -> None:
    binary = _find_quantize_binary(llama_cpp_dir)
    _run([str(binary), str(source_gguf), str(target_gguf), level])


def _write_modelfile(out_dir: Path, gguf_name: str, suffix: str) -> Path:
    path = out_dir / f"Modelfile.{suffix}"
    content = (
        f"FROM ./{gguf_name}\n"
        "PARAMETER embedding_only true\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def _write_readme(out_dir: Path, tag: str, default_suffix: str, produced: list[dict]) -> None:
    lines = [
        f"# Ollama export: {tag}",
        "",
        "## Create the model in Ollama",
        "",
        "```sh",
        f"ollama create {tag} -f Modelfile",
        "```",
        "",
        f"Default points at the `{default_suffix}` GGUF. To pick another quantization, use its Modelfile explicitly:",
        "",
        "```sh",
    ]
    for item in produced:
        lines.append(f"ollama create {tag}-{item['suffix']} -f Modelfile.{item['suffix']}")
    lines.append("```")
    lines.append("")
    lines.append("## Smoke test (OpenAI-compatible endpoint)")
    lines.append("")
    lines.append("```sh")
    lines.append("curl -s http://localhost:11434/v1/embeddings \\")
    lines.append("  -H 'Content-Type: application/json' \\")
    lines.append(f"  -d '{{\"model\":\"{tag}\",\"input\":\"users list\"}}' | jq '.data[0].embedding | length'")
    lines.append("```")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for item in produced:
        lines.append(f"- `{item['gguf']}` ({item['suffix']}) — Modelfile.{item['suffix']}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_to_ollama(
    model_dir: Path,
    out_dir: Path,
    *,
    tag: str = "graphql-embedder",
    quantizations: list[str] | None = None,
    llama_cpp_dir: str | Path | None = None,
) -> dict:
    model_dir = Path(model_dir).resolve()
    out_dir = Path(out_dir).resolve()
    _validate_model_dir(model_dir)

    wanted = list(quantizations) if quantizations else list(QUANT_LEVELS)
    normalized: list[str] = []
    for level in wanted:
        canonical = "f16" if level.lower() == "f16" else level.upper()
        if canonical not in QUANT_LEVELS:
            raise ValueError(f"Unsupported quantization {level!r}; valid: {QUANT_LEVELS}")
        if canonical not in normalized:
            normalized.append(canonical)
    if "f16" not in normalized:
        normalized.insert(0, "f16")

    llama_dir = _resolve_llama_cpp_dir(llama_cpp_dir)
    ensure_dir(out_dir)

    transformer_dir = _find_transformer_subdir(model_dir)
    f16_path = out_dir / "model-f16.gguf"
    _convert_to_gguf(transformer_dir, f16_path, llama_dir)

    produced: list[dict] = [{"suffix": "f16", "gguf": "model-f16.gguf", "path": str(f16_path)}]
    for level in normalized:
        if level == "f16":
            continue
        suffix = level.lower()
        target = out_dir / f"model-{suffix}.gguf"
        _quantize(f16_path, target, level, llama_dir)
        produced.append({"suffix": suffix, "gguf": target.name, "path": str(target)})

    for item in produced:
        _write_modelfile(out_dir, item["gguf"], item["suffix"])

    default_suffix = "f16"
    default_modelfile = out_dir / "Modelfile"
    default_modelfile.write_text(
        (out_dir / f"Modelfile.{default_suffix}").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    manifest_src = model_dir / "training_manifest.json"
    if manifest_src.exists():
        shutil.copy2(manifest_src, out_dir / "training_manifest.json")
    epoch_info_src = model_dir / "epoch_info.json"
    if epoch_info_src.exists():
        shutil.copy2(epoch_info_src, out_dir / "epoch_info.json")

    _write_readme(out_dir, tag, default_suffix, produced)

    summary = {
        "tag": tag,
        "out_dir": str(out_dir),
        "default_modelfile": str(default_modelfile),
        "quantizations": produced,
        "llama_cpp_dir": str(llama_dir),
        "source_model_dir": str(model_dir),
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
