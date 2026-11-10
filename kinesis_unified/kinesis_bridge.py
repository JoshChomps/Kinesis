#!/usr/bin/env python3
"""
File: kinesis_unified/kinesis_bridge.py
Native C++20 Shared-Library Bridge (ctypes) & SafeTensors / Hugging Face Sub-2-Bit Checkpoint Converter (Suite F2).

Provides:
  1. KinesisNativeBridge: Zero-copy ctypes wrapper around kinesis_unified/kinesis_c_abi.cpp
     (compiling libkinesis_unified.dll / libkinesis_unified.so automatically via g++ -std=c++20 -O3 -mavx2 -mfma).
  2. Spec-compliant Hugging Face SafeTensors binary serializer/deserializer (save_safetensors_bytes /
     load_safetensors_bytes) with zero external third-party dependencies.
  3. Hugging Face Transformer layer state-dict converter (convert_hf_transformer_layer_to_sub2bit) with
     offline Kronecker-Stiefel W_O absorption (W_O' = (R_P^T (x) I_16) W_O), GotoBLAS W1/W2/T-MAC packing,
     and lossless CPU GotoBLAS -> GPU Interleaved 16-Bit Split uint32 layout conversion.
"""

from __future__ import annotations

import ctypes
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
KINESIS_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR / "fused_w1_hadacore") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "fused_w1_hadacore"))

from spinquant_packer import (  # noqa: E402
    optimize_kronecker_stiefel_rotation,
    quantize_and_pack_w1_g128_interleaved,
    quantize_and_pack_w2_ternary_interleaved,
    unpack_w1_g128_lop3_reference,
    unpack_w2_ternary_lop3_reference,
)


def build_or_locate_kinesis_shared_lib(force_rebuild: bool = False) -> Path:
    ext = ".dll" if os.name == "nt" else ".so"
    lib_path = KINESIS_DIR / f"libkinesis_unified{ext}"
    src_path = KINESIS_DIR / "kinesis_c_abi.cpp"
    hdr_paths = [
        KINESIS_DIR / "sub2bit_gemm_and_smart_cache.hpp",
        KINESIS_DIR / "memory_primitives.hpp",
        KINESIS_DIR / "hardware_and_sandbox.hpp",
        KINESIS_DIR / "unified_sizing.h",
    ]

    need_build = force_rebuild or (not lib_path.exists())
    if not need_build:
        lib_mtime = lib_path.stat().st_mtime
        if src_path.stat().st_mtime > lib_mtime or any(h.stat().st_mtime > lib_mtime for h in hdr_paths):
            need_build = True

    if need_build:
        tmp_path = lib_path.with_name(f"{lib_path.stem}.{os.getpid()}.tmp{ext}")
        cmd = [
            "g++",
            "-std=c++20",
            "-O3",
            "-mavx2",
            "-mfma",
            "-mstackrealign",
            "-Wall",
            "-Wextra",
            "-shared",
        ]
        if os.name == "nt":
            # -Wa,-muse-unaligned-vector-move: F21 / GCC PR 54412, see docs/12 4.11
            cmd.extend(["-static-libgcc", "-static-libstdc++", "-static", "-Wa,-muse-unaligned-vector-move"])
        else:
            cmd.append("-fPIC")
        cmd.extend([
            f"-I{ROOT_DIR}",
            f"-I{KINESIS_DIR}",
            str(src_path),
            "-o",
            str(tmp_path),
        ])
        res = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
        if res.returncode != 0:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise RuntimeError(
                f"Failed to build {lib_path.name} (exit {res.returncode}):\n{res.stdout}\n{res.stderr}"
            )
        try:
            os.replace(str(tmp_path), str(lib_path))
        except PermissionError:
            # Another process already holds the freshly built DLL open on Windows
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
    return lib_path


class KinesisNativeBridge:
    """
    Zero-copy ctypes wrapper over the compiled C++20 AVX2 micro-kernels in kinesis_c_abi.cpp.
    """

    def __init__(self, lib_path: Optional[Path] = None) -> None:
        resolved = lib_path if lib_path is not None else build_or_locate_kinesis_shared_lib()
        self.lib_path = resolved
        self._lib = ctypes.CDLL(str(resolved))
        self._bind_signatures()

    def _bind_signatures(self) -> None:
        c_f32_p = ctypes.POINTER(ctypes.c_float)
        c_u8_p = ctypes.POINTER(ctypes.c_uint8)
        c_u16_p = ctypes.POINTER(ctypes.c_uint16)
        c_u32_p = ctypes.POINTER(ctypes.c_uint32)
        c_size_p = ctypes.POINTER(ctypes.c_size_t)

        sz = ctypes.c_size_t
        # Every packed buffer is followed by its capacity in elements (docs/12 4.5); the C side returns
        # -3 instead of writing or reading past a short buffer.
        self._lib.uha_kinesis_pack_w1_panels.argtypes = [
            c_f32_p, sz, sz, c_u8_p, sz, c_u16_p, sz
        ]
        self._lib.uha_kinesis_pack_w1_panels.restype = ctypes.c_int

        self._lib.uha_kinesis_pack_w2_panels.argtypes = [
            c_f32_p, sz, sz, c_u16_p, sz, c_u16_p, sz
        ]
        self._lib.uha_kinesis_pack_w2_panels.restype = ctypes.c_int

        self._lib.uha_kinesis_pack_tmac_w1_panels.argtypes = [
            c_f32_p, sz, sz, c_u32_p, sz, c_u16_p, sz
        ]
        self._lib.uha_kinesis_pack_tmac_w1_panels.restype = ctypes.c_int

        self._lib.uha_kinesis_convert_w1_to_gpu_interleaved_u32.argtypes = [
            c_u8_p, sz, c_u16_p, sz, sz, sz, c_u32_p, sz, c_u16_p, sz
        ]
        self._lib.uha_kinesis_convert_w1_to_gpu_interleaved_u32.restype = ctypes.c_int

        self._lib.uha_kinesis_gemm_fp32_4x8.argtypes = [
            c_f32_p, c_f32_p, c_f32_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t
        ]
        self._lib.uha_kinesis_gemm_fp32_4x8.restype = ctypes.c_int

        self._lib.uha_kinesis_smart_w1_gemm.argtypes = [
            c_f32_p, c_u8_p, sz, c_u16_p, sz, c_f32_p,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_float, c_size_p, c_size_p,
        ]
        self._lib.uha_kinesis_smart_w1_gemm.restype = ctypes.c_int

        self._lib.uha_kinesis_smart_w2_gemm.argtypes = [
            c_f32_p, c_u16_p, sz, c_u16_p, sz, c_f32_p,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_float, c_size_p, c_size_p,
        ]
        self._lib.uha_kinesis_smart_w2_gemm.restype = ctypes.c_int

        self._lib.uha_kinesis_tmac_lut_w1_gemm.argtypes = [
            c_f32_p, c_u32_p, sz, c_u16_p, sz, c_f32_p,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_float, c_size_p, c_size_p,
        ]
        self._lib.uha_kinesis_tmac_lut_w1_gemm.restype = ctypes.c_int

    def pack_w1_panels(self, w_kn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Packs a [K, N] FP32 weight matrix into W1GotoBlasPanelSet (uint8 bits, uint16 BF16 scales)."""
        w_c = w_kn.detach().to(device="cpu", dtype=torch.float32).contiguous()
        K, N = w_c.shape
        num_panels = (N + 7) // 8
        groups_per_col = (K + 127) // 128
        bits = torch.empty((num_panels * K,), dtype=torch.uint8, device="cpu")
        scales_u16 = torch.empty((num_panels * groups_per_col * 8,), dtype=torch.int16, device="cpu")
        rc = self._lib.uha_kinesis_pack_w1_panels(
            ctypes.cast(w_c.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            K,
            N,
            ctypes.cast(bits.data_ptr(), ctypes.POINTER(ctypes.c_uint8)),
            bits.numel(),
            ctypes.cast(scales_u16.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
            scales_u16.numel(),
        )
        if rc != 0:
            raise RuntimeError(f"uha_kinesis_pack_w1_panels failed with status {rc}")
        return bits, scales_u16

    def pack_w2_panels(self, w_kn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Packs a [K, N] FP32 weight matrix into W2GotoBlasPanelSet (uint16 trits, uint16 BF16 scales)."""
        w_c = w_kn.detach().to(device="cpu", dtype=torch.float32).contiguous()
        K, N = w_c.shape
        num_panels = (N + 7) // 8
        groups_per_col = (K + 127) // 128
        trits_u16 = torch.empty((num_panels * K,), dtype=torch.int16, device="cpu")
        scales_u16 = torch.empty((num_panels * groups_per_col * 8,), dtype=torch.int16, device="cpu")
        rc = self._lib.uha_kinesis_pack_w2_panels(
            ctypes.cast(w_c.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            K,
            N,
            ctypes.cast(trits_u16.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
            trits_u16.numel(),
            ctypes.cast(scales_u16.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
            scales_u16.numel(),
        )
        if rc != 0:
            raise RuntimeError(f"uha_kinesis_pack_w2_panels failed with status {rc}")
        return trits_u16, scales_u16

    def pack_tmac_w1_panels(self, w_kn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Packs a [K, N] FP32 weight matrix into W1TmacLutPanelSet (int32 nibbles, int16 BF16 scales)."""
        w_c = w_kn.detach().to(device="cpu", dtype=torch.float32).contiguous()
        K, N = w_c.shape
        assert K % 128 == 0, "T-MAC LUT packer requires K % 128 == 0"
        num_panels = (N + 7) // 8
        groups_per_col = K // 128
        chunks = K // 4
        nibbles_u32 = torch.empty((num_panels * chunks,), dtype=torch.int32, device="cpu")
        scales_u16 = torch.empty((num_panels * groups_per_col * 8,), dtype=torch.int16, device="cpu")
        rc = self._lib.uha_kinesis_pack_tmac_w1_panels(
            ctypes.cast(w_c.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            K,
            N,
            ctypes.cast(nibbles_u32.data_ptr(), ctypes.POINTER(ctypes.c_uint32)),
            nibbles_u32.numel(),
            ctypes.cast(scales_u16.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
            scales_u16.numel(),
        )
        if rc != 0:
            raise RuntimeError(f"uha_kinesis_pack_tmac_w1_panels failed with status {rc}")
        return nibbles_u32, scales_u16

    def convert_w1_to_gpu_interleaved(
        self, bits: torch.Tensor, scales_u16: torch.Tensor, K: int, N: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Losslessly converts CPU GotoBLAS W1 panels into GPU [N, K/32] Interleaved uint32 layout."""
        if K <= 0 or N <= 0 or K % 128 != 0:
            # GPU layout [N, K/128] has no tail group (docs/12 4.5).
            raise ValueError(f"convert_w1_to_gpu_interleaved requires K % 128 == 0 and N > 0 (got K={K}, N={N})")
        bits_c = bits.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        scales_c = scales_u16.detach().to(device="cpu", dtype=torch.int16).contiguous()
        gpu_u32 = torch.empty((N, K // 32), dtype=torch.int32, device="cpu")
        gpu_scales_bf16 = torch.empty((N, K // 128), dtype=torch.bfloat16, device="cpu")
        rc = self._lib.uha_kinesis_convert_w1_to_gpu_interleaved_u32(
            ctypes.cast(bits_c.data_ptr(), ctypes.POINTER(ctypes.c_uint8)),
            bits_c.numel(),
            ctypes.cast(scales_c.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
            scales_c.numel(),
            K,
            N,
            ctypes.cast(gpu_u32.data_ptr(), ctypes.POINTER(ctypes.c_uint32)),
            gpu_u32.numel(),
            ctypes.cast(gpu_scales_bf16.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
            gpu_scales_bf16.numel(),
        )
        if rc != 0:
            raise RuntimeError(f"uha_kinesis_convert_w1_to_gpu_interleaved_u32 failed with status {rc}")
        return gpu_u32, gpu_scales_bf16

    def gemm_fp32_4x8(self, a_mk: torch.Tensor, b_kn: torch.Tensor) -> torch.Tensor:
        a_c = a_mk.detach().to(device="cpu", dtype=torch.float32).contiguous()
        b_c = b_kn.detach().to(device="cpu", dtype=torch.float32).contiguous()
        M, K = a_c.shape
        K2, N = b_c.shape
        assert K == K2, (K, K2)
        c_mn = torch.empty((M, N), dtype=torch.float32, device="cpu")
        rc = self._lib.uha_kinesis_gemm_fp32_4x8(
            ctypes.cast(a_c.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(b_c.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(c_mn.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            M,
            N,
            K,
        )
        if rc != 0:
            raise RuntimeError(f"uha_kinesis_gemm_fp32_4x8 failed with status {rc}")
        return c_mn

    def smart_w1_gemm(
        self,
        a_mk: torch.Tensor,
        bits: torch.Tensor,
        scales_u16: torch.Tensor,
        N: int,
        error_budget_eps: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        a_c = a_mk.detach().to(device="cpu", dtype=torch.float32).contiguous()
        bits_c = bits.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        scales_c = scales_u16.detach().to(device="cpu", dtype=torch.int16).contiguous()
        M, K = a_c.shape
        c_mn = torch.empty((M, N), dtype=torch.float32, device="cpu")
        tot = ctypes.c_size_t(0)
        pruned = ctypes.c_size_t(0)
        rc = self._lib.uha_kinesis_smart_w1_gemm(
            ctypes.cast(a_c.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(bits_c.data_ptr(), ctypes.POINTER(ctypes.c_uint8)),
            bits_c.numel(),
            ctypes.cast(scales_c.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
            scales_c.numel(),
            ctypes.cast(c_mn.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            M,
            N,
            K,
            ctypes.c_float(error_budget_eps),
            ctypes.byref(tot),
            ctypes.byref(pruned),
        )
        if rc != 0:
            raise RuntimeError(f"uha_kinesis_smart_w1_gemm failed with status {rc}")
        total_req = int(tot.value)
        pruned_req = int(pruned.value)
        return c_mn, {
            "total_groups": total_req,
            "pruned_groups": pruned_req,
            "pruned_pct": (100.0 * pruned_req / total_req) if total_req > 0 else 0.0,
        }

    def smart_w2_gemm(
        self,
        a_mk: torch.Tensor,
        trits_u16: torch.Tensor,
        scales_u16: torch.Tensor,
        N: int,
        error_budget_eps: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        a_c = a_mk.detach().to(device="cpu", dtype=torch.float32).contiguous()
        trits_c = trits_u16.detach().to(device="cpu", dtype=torch.int16).contiguous()
        scales_c = scales_u16.detach().to(device="cpu", dtype=torch.int16).contiguous()
        M, K = a_c.shape
        c_mn = torch.empty((M, N), dtype=torch.float32, device="cpu")
        tot = ctypes.c_size_t(0)
        pruned = ctypes.c_size_t(0)
        rc = self._lib.uha_kinesis_smart_w2_gemm(
            ctypes.cast(a_c.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(trits_c.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
            trits_c.numel(),
            ctypes.cast(scales_c.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
            scales_c.numel(),
            ctypes.cast(c_mn.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            M,
            N,
            K,
            ctypes.c_float(error_budget_eps),
            ctypes.byref(tot),
            ctypes.byref(pruned),
        )
        if rc != 0:
            raise RuntimeError(f"uha_kinesis_smart_w2_gemm failed with status {rc}")
        total_req = int(tot.value)
        pruned_req = int(pruned.value)
        return c_mn, {
            "total_groups": total_req,
            "pruned_groups": pruned_req,
            "pruned_pct": (100.0 * pruned_req / total_req) if total_req > 0 else 0.0,
        }

    def tmac_lut_w1_gemm(
        self,
        a_mk: torch.Tensor,
        nibbles_u32: torch.Tensor,
        scales_u16: torch.Tensor,
        N: int,
        energy_sieve_tau: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        a_c = a_mk.detach().to(device="cpu", dtype=torch.float32).contiguous()
        nibbles_c = nibbles_u32.detach().to(device="cpu", dtype=torch.int32).contiguous()
        scales_c = scales_u16.detach().to(device="cpu", dtype=torch.int16).contiguous()
        M, K = a_c.shape
        c_mn = torch.empty((M, N), dtype=torch.float32, device="cpu")
        tot = ctypes.c_size_t(0)
        pruned = ctypes.c_size_t(0)
        rc = self._lib.uha_kinesis_tmac_lut_w1_gemm(
            ctypes.cast(a_c.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(nibbles_c.data_ptr(), ctypes.POINTER(ctypes.c_uint32)),
            nibbles_c.numel(),
            ctypes.cast(scales_c.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
            scales_c.numel(),
            ctypes.cast(c_mn.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            M,
            N,
            K,
            ctypes.c_float(energy_sieve_tau),
            ctypes.byref(tot),
            ctypes.byref(pruned),
        )
        if rc != 0:
            raise RuntimeError(f"uha_kinesis_tmac_lut_w1_gemm failed with status {rc}")
        total_req = int(tot.value)
        pruned_req = int(pruned.value)
        return c_mn, {
            "total_groups": total_req,
            "pruned_groups": pruned_req,
            "pruned_pct": (100.0 * pruned_req / total_req) if total_req > 0 else 0.0,
        }


# =============================================================================
# Spec-Compliant SafeTensors Binary Serializer & HF Checkpoint Converter
# =============================================================================

_TORCH_TO_SAFETENSORS_DTYPE = {
    torch.float32: ("F32", 4),
    torch.bfloat16: ("BF16", 2),
    torch.float16: ("F16", 2),
    torch.int32: ("I32", 4),
    torch.int16: ("I16", 2),
    torch.uint8: ("U8", 1),
    torch.int8: ("I8", 1),
}
if hasattr(torch, "uint32"):
    _TORCH_TO_SAFETENSORS_DTYPE[torch.uint32] = ("U32", 4)
if hasattr(torch, "uint16"):
    _TORCH_TO_SAFETENSORS_DTYPE[torch.uint16] = ("U16", 2)

_SAFETENSORS_TO_TORCH_DTYPE = {
    "F32": (torch.float32, 4),
    "BF16": (torch.bfloat16, 2),
    "F16": (torch.float16, 2),
    "I32": (torch.int32, 4),
    "U32": (getattr(torch, "uint32", torch.int32), 4),
    "I16": (torch.int16, 2),
    "U16": (getattr(torch, "uint16", torch.int16), 2),
    "U8": (torch.uint8, 1),
    "I8": (torch.int8, 1),
}


def save_safetensors_bytes(
    tensors: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]] = None
) -> bytes:
    """
    Serializes a dictionary of PyTorch tensors into the standard Hugging Face SafeTensors binary format:
      [8-byte uint64 little-endian header size N][N-byte UTF-8 JSON header][raw tensor byte buffers]
    """
    header_obj: Dict[str, object] = {}
    if metadata:
        header_obj["__metadata__"] = dict(metadata)

    raw_chunks = []
    offset = 0
    for name in sorted(tensors.keys()):
        t = tensors[name].detach().to(device="cpu").contiguous()
        if t.dtype not in _TORCH_TO_SAFETENSORS_DTYPE:
            raise ValueError(f"Unsupported tensor dtype {t.dtype} for tensor {name}")
        st_dtype, elem_bytes = _TORCH_TO_SAFETENSORS_DTYPE[t.dtype]
        nbytes = t.numel() * elem_bytes
        buf = ctypes.string_at(t.data_ptr(), nbytes) if nbytes > 0 else b""
        header_obj[name] = {
            "dtype": st_dtype,
            "shape": list(t.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        raw_chunks.append(buf)
        offset += nbytes

    header_json = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
    pad_len = (8 - (len(header_json) % 8)) % 8
    header_padded = header_json + (b" " * pad_len)
    return struct.pack("<Q", len(header_padded)) + header_padded + b"".join(raw_chunks)


def load_safetensors_bytes(blob: bytes) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """
    Deserializes a standard Hugging Face SafeTensors binary payload back into PyTorch CPU tensors.
    """
    if len(blob) < 8:
        raise ValueError("Invalid SafeTensors payload: fewer than 8 bytes")
    (header_len,) = struct.unpack("<Q", blob[:8])
    data_start = 8 + header_len
    if data_start > len(blob):
        raise ValueError("Invalid SafeTensors payload: header length exceeds buffer size")

    header_obj = json.loads(blob[8:data_start].decode("utf-8"))
    metadata = header_obj.pop("__metadata__", {})
    out_tensors: Dict[str, torch.Tensor] = {}

    for name, info in header_obj.items():
        st_dtype = info["dtype"]
        shape = tuple(info["shape"])
        start_off, end_off = info["data_offsets"]
        torch_dtype, elem_bytes = _SAFETENSORS_TO_TORCH_DTYPE[st_dtype]
        raw_slice = blob[data_start + start_off : data_start + end_off]
        numel = 1
        for d in shape:
            numel *= d
        assert len(raw_slice) == numel * elem_bytes, (name, len(raw_slice), numel * elem_bytes)
        if numel == 0:
            out_tensors[name] = torch.empty(shape, dtype=torch_dtype)
        else:
            u8 = torch.frombuffer(bytearray(raw_slice), dtype=torch.uint8)
            out_tensors[name] = u8.view(torch_dtype).reshape(shape).clone()

    return out_tensors, metadata


def convert_hf_transformer_layer_to_sub2bit(
    hf_state_dict: Dict[str, torch.Tensor],
    calibration_activations: torch.Tensor,
    bridge: Optional[KinesisNativeBridge] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Converts a Hugging Face Transformer layer state dict:
      - 'self_attn.o_proj.weight'  [D_model, D_model]
      - 'mlp.gate_proj.weight'     [N_ffn, D_model]    -> W1 Binary g128 (CPU GotoBLAS + T-MAC LUT + GPU u32)
      - 'mlp.up_proj.weight'       [N_ffn, D_model]    -> W2 Ternary b1.58 (CPU GotoBLAS)
    Applies Kronecker-Stiefel optimization R_kron = R_P (x) H_16 on post-o_proj activations,
    absorbs (R_P^T (x) I_16) offline into self_attn.o_proj.weight, and packs the rotated MLP weights.
    """
    kb = bridge if bridge is not None else KinesisNativeBridge()
    w_o = hf_state_dict["self_attn.o_proj.weight"].detach().to(torch.float32)
    w_gate = hf_state_dict["mlp.gate_proj.weight"].detach().to(torch.float32)
    w_up = hf_state_dict["mlp.up_proj.weight"].detach().to(torch.float32)

    h_pre = calibration_activations.detach().to(torch.float32) @ w_o.T
    r_kron, r_p, k_init, k_h16, k_kron = optimize_kronecker_stiefel_rotation(
        h_pre, q_hadamard=16, steps=15, lr=0.25
    )
    r_prev_absorb = torch.kron(r_p, torch.eye(16, dtype=torch.float32))
    w_o_absorbed = (r_prev_absorb.T @ w_o).to(torch.bfloat16)

    # Rotate MLP weights offline by R_kron: W_rot [N, K] = W @ R_kron; transpose to [K, N] for GotoBLAS panels
    w_gate_rot_kn = (w_gate @ r_kron).T.contiguous()
    w_up_rot_kn = (w_up @ r_kron).T.contiguous()

    w1_bits, w1_scales_u16 = kb.pack_w1_panels(w_gate_rot_kn)
    tmac_nibbles_u32, tmac_scales_u16 = kb.pack_tmac_w1_panels(w_gate_rot_kn)
    gpu_w1_u32, gpu_w1_scales_bf16 = kb.convert_w1_to_gpu_interleaved(
        w1_bits, w1_scales_u16, K=w_gate.shape[1], N=w_gate.shape[0]
    )
    w2_trits_u16, w2_scales_u16 = kb.pack_w2_panels(w_up_rot_kn)

    orig_bytes = (w_gate.numel() + w_up.numel()) * 4
    compressed_bytes = (
        w1_bits.numel() * 1
        + w1_scales_u16.numel() * 2
        + w2_trits_u16.numel() * 2
        + w2_scales_u16.numel() * 2
    )

    packed_dict = {
        "self_attn.o_proj.weight_absorbed_bf16": w_o_absorbed,
        "mlp.gate_proj.w1_gotoblas_bits_u8": w1_bits,
        "mlp.gate_proj.w1_gotoblas_scales_i16": w1_scales_u16,
        "mlp.gate_proj.w1_tmac_nibbles_i32": tmac_nibbles_u32,
        "mlp.gate_proj.w1_tmac_scales_i16": tmac_scales_u16,
        "mlp.gate_proj.w1_gpu_interleaved_i32": gpu_w1_u32,
        "mlp.gate_proj.w1_gpu_scales_bf16": gpu_w1_scales_bf16,
        "mlp.up_proj.w2_gotoblas_trits_i16": w2_trits_u16,
        "mlp.up_proj.w2_gotoblas_scales_i16": w2_scales_u16,
    }
    stats = {
        "kurtosis_init": k_init,
        "kurtosis_h16": k_h16,
        "kurtosis_kron": k_kron,
        "mlp_compression_vs_fp32": orig_bytes / max(compressed_bytes, 1),
    }
    return packed_dict, stats


def run_kinesis_bridge_verification() -> None:
    print("================================================================================")
    print("  SUITE F2: NATIVE C++20 CTYPES BRIDGE & SAFETENSORS SUB-2-BIT CONVERTER")
    print("================================================================================\n")

    torch.manual_seed(42)
    bridge = KinesisNativeBridge()

    # [1] Verify native C++20 4x8 register-blocked FP32 GEMM against PyTorch
    M, K, N = 19, 256, 67
    a_f32 = torch.randn((M, K), dtype=torch.float32)
    b_f32 = torch.randn((K, N), dtype=torch.float32)
    c_cpp = bridge.gemm_fp32_4x8(a_f32, b_f32)
    c_ref = (a_f32.double() @ b_f32.double()).float()
    fp32_max_err = float((c_cpp - c_ref).abs().max().item())
    assert fp32_max_err < 1e-4, f"FP32 4x8 C++ GEMM mismatch: {fp32_max_err}"
    print(f"[1] Native C++20 gemm_fp32_4x8 (19x67x256)      : MaxAbsErr = {fp32_max_err:.3e} | PASS")

    # [2] Verify native C++20 W1 GotoBLAS packer, CPU->GPU Interleaved u32 bridge, and AVX2 W1 GEMM
    M_w1, K_w1, N_w1 = 16, 256, 64
    a_w1 = torch.randn((M_w1, K_w1), dtype=torch.float32)
    w_nk = torch.randn((N_w1, K_w1), dtype=torch.float32) * 0.08
    w_kn = w_nk.T.contiguous()

    bits_u8, scales_i16 = bridge.pack_w1_panels(w_kn)
    gpu_u32_cpp, gpu_scales_cpp = bridge.convert_w1_to_gpu_interleaved(bits_u8, scales_i16, K_w1, N_w1)
    gpu_u32_py, gpu_scales_py = quantize_and_pack_w1_g128_interleaved(w_nk)

    u32_mismatches = int((gpu_u32_cpp.view(torch.int32) != gpu_u32_py.view(torch.int32)).sum().item())
    scale_mismatches = int(
        (gpu_scales_cpp.view(torch.int16) != gpu_scales_py.view(torch.int16)).sum().item()
    )
    assert u32_mismatches == 0 and scale_mismatches == 0, (u32_mismatches, scale_mismatches)

    w1_dequant_f32 = unpack_w1_g128_lop3_reference(gpu_u32_py, gpu_scales_py).to(torch.float32)
    c_w1_cpp, w1_stats = bridge.smart_w1_gemm(a_w1, bits_u8, scales_i16, N=N_w1, error_budget_eps=0.0)
    c_w1_ref = a_w1 @ w1_dequant_f32.T
    w1_max_err = float((c_w1_cpp - c_w1_ref).abs().max().item())
    assert w1_max_err < 1e-5, f"Native C++20 W1 GEMM mismatch vs lop3 reference: {w1_max_err}"

    # Verify guaranteed Holder error-budget sieve across the C ABI boundary
    a_w1_sparse = a_w1.clone()
    a_w1_sparse[:, 128:256] *= 1e-4
    c_w1_sieved, sieve_stats = bridge.smart_w1_gemm(
        a_w1_sparse, bits_u8, scales_i16, N=N_w1, error_budget_eps=0.05
    )
    c_w1_sparse_exact, _ = bridge.smart_w1_gemm(
        a_w1_sparse, bits_u8, scales_i16, N=N_w1, error_budget_eps=0.0
    )
    sieve_bound_err = float((c_w1_sieved - c_w1_sparse_exact).abs().max().item())
    assert sieve_stats["pruned_pct"] == 50.0 and sieve_bound_err <= 0.05 + 1e-5
    print(
        f"[2] Native C++20 W1 AVX2 + GPU u32 Bridge       : u32/Scale Err = 0 | "
        f"GEMM Err = {w1_max_err:.3e} | Holder Pruned = {sieve_stats['pruned_pct']:.1f}% (MaxErr={sieve_bound_err:.3e} <= 0.05) | PASS"
    )

    # [3] Verify native C++20 W2 Ternary AVX2 GEMM and T-MAC 256-bit 32-lane LUT GEMM
    trits_i16, w2_scales_i16 = bridge.pack_w2_panels(w_kn)
    c_w2_cpp, _ = bridge.smart_w2_gemm(a_w1, trits_i16, w2_scales_i16, N=N_w1, error_budget_eps=0.0)
    assert torch.isfinite(c_w2_cpp).all()

    nibbles_i32, tmac_scales_i16 = bridge.pack_tmac_w1_panels(w_kn)
    c_tmac_cpp, _ = bridge.tmac_lut_w1_gemm(a_w1, nibbles_i32, tmac_scales_i16, N=N_w1)
    tmac_cos = float(
        torch.nn.functional.cosine_similarity(c_tmac_cpp.flatten(), c_w1_cpp.flatten(), dim=0).item()
    )
    assert tmac_cos > 0.9999, f"T-MAC 256-bit LUT cosine similarity vs exact W1 too low: {tmac_cos}"
    print(
        f"[3] Native C++20 W2 Ternary & T-MAC 256b LUT    : T-MAC vs Exact W1 CosSim = {tmac_cos:.6f} | PASS"
    )

    # [4] Verify Hugging Face Transformer layer conversion + SafeTensors binary round-trip
    D_model, N_ffn = 256, 128
    hf_layer = {
        "self_attn.o_proj.weight": torch.randn((D_model, D_model), dtype=torch.float32) * 0.05,
        "mlp.gate_proj.weight": torch.randn((N_ffn, D_model), dtype=torch.float32) * 0.05,
        "mlp.up_proj.weight": torch.randn((N_ffn, D_model), dtype=torch.float32) * 0.05,
    }
    hf_layer["self_attn.o_proj.weight"][2, :] *= 14.0
    cal_x = torch.randn((32, D_model), dtype=torch.float32)

    packed_ckpt, ckpt_stats = convert_hf_transformer_layer_to_sub2bit(hf_layer, cal_x, bridge=bridge)
    blob = save_safetensors_bytes(packed_ckpt, metadata={"format": "uha_sub2bit_v1"})
    reloaded_ckpt, reloaded_meta = load_safetensors_bytes(blob)

    assert reloaded_meta.get("format") == "uha_sub2bit_v1"
    for k, v in packed_ckpt.items():
        assert k in reloaded_ckpt, f"Missing tensor {k} after SafeTensors round-trip"
        assert torch.equal(
            v.view(torch.uint8), reloaded_ckpt[k].view(torch.uint8)
        ), f"Byte mismatch on {k} after SafeTensors round-trip"

    print(
        f"[4] SafeTensors Sub-2-Bit HF Layer Converter    : Kurtosis {ckpt_stats['kurtosis_init']:.2f} -> "
        f"{ckpt_stats['kurtosis_kron']:.2f} | Compression = {ckpt_stats['mlp_compression_vs_fp32']:.2f}x | "
        f"SafeTensors Bytes = {len(blob)} (0 bit errors) | PASS"
    )

    # [5] docs/12 4.5 regression: the ABI size contract is explicit and enforced.
    #   (a) K % 128 != 0 is rejected before any buffer is allocated or written (used to overflow by
    #       N*16 bytes for K=160).
    #   (b) A caller buffer one element short gets status -3 and is left untouched (sentinel check),
    #       for both an output (convert) and an input (smart GEMM reconstruction) buffer.
    K_bad, N_bad = 160, 8
    bits_bad, scales_bad = bridge.pack_w1_panels(torch.randn((K_bad, N_bad)))
    try:
        bridge.convert_w1_to_gpu_interleaved(bits_bad, scales_bad, K_bad, N_bad)
        raise AssertionError("K=160 must be rejected by convert_w1_to_gpu_interleaved")
    except ValueError:
        pass
    c_u8, c_u16, c_u32 = (ctypes.POINTER(t) for t in (ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint32))
    rc_k160 = bridge._lib.uha_kinesis_convert_w1_to_gpu_interleaved_u32(
        ctypes.cast(bits_bad.data_ptr(), c_u8), bits_bad.numel(),
        ctypes.cast(scales_bad.data_ptr(), c_u16), scales_bad.numel(), K_bad, N_bad,
        ctypes.cast(torch.empty(N_bad * K_bad // 32, dtype=torch.int32).data_ptr(), c_u32), N_bad * K_bad // 32,
        ctypes.cast(torch.empty(N_bad, dtype=torch.int16).data_ptr(), c_u16), N_bad,
    )
    assert rc_k160 == -1, rc_k160
    sentinel = 0x5A5A
    short_scales = torch.full((N_w1 * (K_w1 // 128) - 1 + 4,), sentinel, dtype=torch.int16)
    rc_short_out = bridge._lib.uha_kinesis_convert_w1_to_gpu_interleaved_u32(
        ctypes.cast(bits_u8.data_ptr(), c_u8), bits_u8.numel(),
        ctypes.cast(scales_i16.data_ptr(), c_u16), scales_i16.numel(), K_w1, N_w1,
        ctypes.cast(torch.empty(N_w1 * K_w1 // 32, dtype=torch.int32).data_ptr(), c_u32), N_w1 * K_w1 // 32,
        ctypes.cast(short_scales.data_ptr(), c_u16), N_w1 * (K_w1 // 128) - 1,
    )
    assert rc_short_out == -3 and bool((short_scales == sentinel).all()), rc_short_out
    rc_short_in = bridge._lib.uha_kinesis_smart_w1_gemm(
        ctypes.cast(a_w1.contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(bits_u8.data_ptr(), c_u8), bits_u8.numel() - 1,
        ctypes.cast(scales_i16.data_ptr(), c_u16), scales_i16.numel(),
        ctypes.cast(torch.empty((M_w1, N_w1)).data_ptr(), ctypes.POINTER(ctypes.c_float)),
        M_w1, N_w1, K_w1, ctypes.c_float(0.0), None, None,
    )
    assert rc_short_in == -3, rc_short_in
    # docs/12 4.23: a packed input whose length does not match (K, N) is rejected (-4), not decoded with the
    # wrong panel stride. K = 128 activations against K = 256 panels used to return garbage with status 0.
    rc_k_mismatch = bridge._lib.uha_kinesis_smart_w1_gemm(
        ctypes.cast(a_w1[:, : K_w1 // 2].contiguous().data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(bits_u8.data_ptr(), c_u8), bits_u8.numel(),
        ctypes.cast(scales_i16.data_ptr(), c_u16), scales_i16.numel(),
        ctypes.cast(torch.empty((M_w1, N_w1)).data_ptr(), ctypes.POINTER(ctypes.c_float)),
        M_w1, N_w1, K_w1 // 2, ctypes.c_float(0.0), None, None,
    )
    assert rc_k_mismatch == -4, rc_k_mismatch
    try:
        bridge.smart_w1_gemm(a_w1[:, : K_w1 // 2], bits_u8, scales_i16, N=N_w1)
        raise AssertionError("K mismatch must raise")
    except RuntimeError:
        pass
    print("[5] C ABI Size Contract (docs/12 4.5)           : K=160 rejected (-1), short out/in buffers -> -3, "
          "sentinel intact, K mismatch -> -4 (4.23) | PASS")

    # [6] docs/12 4.14 regression: the CPU W2 GotoBLAS packer and the GPU (spinquant) W2 packer must produce the
    # same dequantized ternary weights for the same input, so a W2 tile decodes identically on either tier.
    torch.manual_seed(9)
    K_c, N_c = 512, 40
    w_c_nk = torch.randn((N_c, K_c)) * 0.03
    w_c_nk[3, :128] = 0.0                       # an all-zero group (exercises the 1e-6 clamp)
    trits_cpu, sc_cpu = bridge.pack_w2_panels(w_c_nk.T.contiguous())
    gc = K_c // 128
    tr = trits_cpu.view(torch.int16).to(torch.int32) & 0xFFFF
    cpu_deq = torch.empty((N_c, K_c))
    sc_f32 = sc_cpu.view(torch.bfloat16).to(torch.float32)
    for n in range(N_c):
        p, c = divmod(n, 8)
        codes = ((tr[p * K_c:(p + 1) * K_c] >> (2 * c)) & 3) - 1
        scales = sc_f32[(p * gc + torch.arange(gc)) * 8 + c].repeat_interleave(128)
        cpu_deq[n] = codes.to(torch.float32) * scales
    gpu_deq = unpack_w2_ternary_lop3_reference(*quantize_and_pack_w2_ternary_interleaved(w_c_nk)).to(torch.float32)
    w2_cross = int((cpu_deq != gpu_deq).sum().item())
    assert w2_cross == 0, w2_cross
    print(f"[6] W2 CPU GotoBLAS vs GPU spinquant (docs/12 4.14): dequantized mismatches = {w2_cross}/{N_c * K_c} | PASS")

    # [7] docs/12 4.19: the g128 scale rule (FP64 mean, one rounding to BF16) must agree between the C++ W1 packer
    # and the Python packer on many groups. Control: an FP32 torch mean (the old Python rule) disagrees.
    torch.manual_seed(13)
    K_s, N_s = 4096, 512                         # 32 groups x 512 columns = 16,384 scales
    w_s_nk = torch.randn((N_s, K_s)) * torch.rand((N_s, 1)) * 0.05
    _, sc_w1_cpu = bridge.pack_w1_panels(w_s_nk.T.contiguous())
    gs = K_s // 128
    sc_cpu_f = sc_w1_cpu.view(torch.bfloat16).to(torch.float32)
    idx = ((torch.arange(N_s) // 8)[:, None] * gs + torch.arange(gs)[None, :]) * 8 + (torch.arange(N_s) % 8)[:, None]
    cpu_nk = sc_cpu_f[idx]                                        # [N, groups]
    _, sc_py = quantize_and_pack_w1_g128_interleaved(w_s_nk)
    scale_rule_diff = int((cpu_nk != sc_py.to(torch.float32)).sum())
    # The disagreement this fixes is rare (one group in 1,024 in G1, on R_kron-rotated weights); G1's QAT-vs-packer
    # cross-check is the case that exposed it and stays as its regression.
    assert scale_rule_diff == 0, scale_rule_diff
    print(f"[7] g128 scale rule C++ vs Python (docs/12 4.19): {scale_rule_diff}/{N_s * gs} differ | PASS")

    # [8] docs/12 4.29 seam regression: mixed (+Inf, -Inf, NaN, finite) in the SAME g128 W2 group must produce
    # identical ternary codes [2, 0, 1, 2, 0, 1] across Python spinquant_packer and C++ pack_w2_panels, and
    # save_safetensors_bytes must round-trip torch.uint32 outputs from quantize_and_pack_w1_g128_interleaved.
    w_mix = torch.zeros((8, 128), dtype=torch.float32)
    w_mix[0, :6] = torch.tensor([float("inf"), float("-inf"), float("nan"), 2.5, -2.5, 0.0])
    w2_py_u32, _ = quantize_and_pack_w2_ternary_interleaved(w_mix)
    u0 = int(w2_py_u32[0, 0].item()) & 0xFFFFFFFF
    py_mix_codes = [((u0 >> ((2 * (k // 2)) if (k % 2 == 0) else (2 * (k // 2) + 16))) & 3) for k in range(6)]
    trits_mix, _ = bridge.pack_w2_panels(w_mix.T.contiguous())
    cpp_mix_codes = [int(trits_mix[k].item()) & 3 for k in range(6)]
    assert py_mix_codes == [2, 0, 1, 2, 0, 1] and cpp_mix_codes == py_mix_codes, (py_mix_codes, cpp_mix_codes)
    w1_u32_rt, w1_sc_rt = quantize_and_pack_w1_g128_interleaved(torch.randn((8, 128)))
    st_u32_blob = save_safetensors_bytes({"w1_u32": w1_u32_rt, "w1_sc": w1_sc_rt})
    st_u32_loaded, _ = load_safetensors_bytes(st_u32_blob)
    assert torch.equal(st_u32_loaded["w1_u32"].view(torch.int32), w1_u32_rt.view(torch.int32))
    print(f"[8] W2 Mixed NaN/Inf Group & U32 SafeTensors (4.29): Py={py_mix_codes} == C++={cpp_mix_codes} | U32 round-trip PASS")
    print("\n[+] ALL SUITE F2 NATIVE C++20 BRIDGE & SAFETENSORS CHECKS PASSED.")


if __name__ == "__main__":
    run_kinesis_bridge_verification()
