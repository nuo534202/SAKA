"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class KeypointAllocator:
	"""
	BETA-K keypoint allocation for reliability maps.

	This allocator replaces a plain global Top-K with block-aware bounded
	apportionment. Statistics are computed on a smoothed map, while final point
	selection is done on the original score map.
	"""

	def __init__(
		self,
		block_size: Tuple[int, int] = (8, 8),
		alpha: float = 0.5,
		beta: float = 0.5,
		eta: float = 3.0,
		gamma: float = 0.12,
		eps: float = 1e-6,
		clip_quantile: float = 0.99,
		top_m: int = 8,
		temperature_base: float = 1.0,
		temperature_min: float = 0.8,
		temperature_max: float = 2.0,
		gini_scale: float = 1.0,
		adaptive_temperature: bool = True,
		auto_sigmoid: bool = True,
		relax_mask_if_needed: bool = True,
		n_min: int = 2,
		score_tau_quantile: float = 0.3,
		quality_floor: bool = True,
		cand_quality_delta: float = 0.5,
		use_cand_quality: bool = True,
		entropy_norm_mode: str = 'area',
		n_ref: int = 3,
		rho: float = 0.5,
		use_count_factor: bool = True,
		cap_aware_refill: bool = True,
		shifted_grid: bool = False,
		debug: bool = False,
	):
		if block_size[0] <= 0 or block_size[1] <= 0:
			raise ValueError("block_size must be positive")

		self.block_size = (int(block_size[0]), int(block_size[1]))
		self.alpha = float(alpha)
		self.beta = float(beta)
		self.eta = float(eta)
		self.gamma = float(gamma)
		self.eps = float(eps)
		self.clip_quantile = float(clip_quantile)
		self.top_m = int(top_m)
		self.temperature_base = float(temperature_base)
		self.temperature_min = float(temperature_min)
		self.temperature_max = float(temperature_max)
		self.gini_scale = float(gini_scale)
		self.adaptive_temperature = bool(adaptive_temperature)
		self.auto_sigmoid = bool(auto_sigmoid)
		self.relax_mask_if_needed = bool(relax_mask_if_needed)

		self.n_min = int(n_min)
		self.score_tau_quantile = float(score_tau_quantile)
		self.quality_floor = bool(quality_floor)
		self.cand_quality_delta = float(cand_quality_delta)
		self.use_cand_quality = bool(use_cand_quality)
		self.entropy_norm_mode = str(entropy_norm_mode)
		self.n_ref = int(n_ref)
		self.rho = float(rho)
		self.use_count_factor = bool(use_count_factor)
		self.cap_aware_refill = bool(cap_aware_refill)
		self.shifted_grid = bool(shifted_grid)
		self.debug = bool(debug)

		self._last_debug_stats: Optional[Dict[str, Any]] = None

	@torch.inference_mode()
	def select_topk(self, score_map: torch.Tensor, top_k: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		scores = self.normalize_scores(score_map)
		if scores.dim() == 2:
			scores = scores.unsqueeze(0)
		elif scores.dim() == 4:
			if scores.shape[1] != 1:
				raise ValueError("score_map with 4 dims must have shape (B, 1, H, W)")
			scores = scores[:, 0]
		elif scores.dim() != 3:
			raise ValueError("score_map must be (H, W), (B, H, W) or (B, 1, H, W)")

		B, H, W = scores.shape
		K = min(max(int(top_k), 0), H * W)
		if K == 0:
			return torch.empty((B, 0), device=scores.device, dtype=torch.long)

		mask_bhw = self.prepare_mask(mask, B, H, W, scores.device)

		selected = []
		for b in range(B):
			sample_mask = None if mask_bhw is None else mask_bhw[b]
			selected.append(self.select_single(scores[b], K, sample_mask))

		return torch.stack(selected, dim=0)

	@torch.inference_mode()
	def select_xy(self, score_map: torch.Tensor, top_k: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		indices = self.select_topk(score_map, top_k=top_k, mask=mask)

		scores = self.normalize_scores(score_map)
		if scores.dim() == 2:
			scores = scores.unsqueeze(0)
		elif scores.dim() == 4:
			scores = scores[:, 0]

		_, H, W = scores.shape
		y = torch.div(indices, W, rounding_mode='floor')
		x = indices % W
		return torch.stack([x, y], dim=-1)

	@torch.inference_mode()
	def select_sparse_topk(
		self,
		reliability_map: torch.Tensor,
		mkpts: torch.Tensor,
		scores: torch.Tensor,
		top_k: int,
	) -> torch.Tensor:
		rel = self.normalize_scores(reliability_map)
		if rel.dim() == 4:
			if rel.shape[1] != 1:
				raise ValueError("reliability_map with 4 dims must have shape (B, 1, H_r, W_r)")
			rel = rel[:, 0]
		elif rel.dim() == 2:
			rel = rel.unsqueeze(0)
		elif rel.dim() != 3:
			raise ValueError("reliability_map must be (H_r, W_r), (B, H_r, W_r) or (B, 1, H_r, W_r)")

		B = rel.shape[0]
		K = min(max(int(top_k), 0), scores.shape[1])
		if K == 0:
			return torch.empty((B, 0), device=scores.device, dtype=torch.long)

		selected = []
		for b in range(B):
			selected.append(self.select_sparse_single(rel[b], mkpts[b], scores[b], K))

		max_len = max(s.shape[0] for s in selected)
		if max_len == 0:
			return torch.empty((B, 0), device=scores.device, dtype=torch.long)

		padded = torch.zeros((B, max_len), device=scores.device, dtype=torch.long)
		for b in range(B):
			n = selected[b].shape[0]
			if n > 0:
				padded[b, :n] = selected[b]

		return padded

	@torch.inference_mode()
	def select_sparse_single(
		self,
		reliability_map: torch.Tensor,
		mkpts: torch.Tensor,
		scores: torch.Tensor,
		K: int,
	) -> torch.Tensor:
		device = reliability_map.device
		H_r, W_r = reliability_map.shape
		N = mkpts.shape[0]

		valid_mask = scores > 0
		valid_indices = torch.where(valid_mask)[0]
		n_valid = valid_indices.numel()

		if n_valid == 0:
			return torch.empty((0,), device=device, dtype=torch.long)

		if n_valid <= K:
			order = torch.argsort(scores[valid_indices], descending=True)
			return valid_indices[order]

		valid_scores = scores[valid_indices]
		valid_mkpts = mkpts[valid_indices]

		bh, bw = self.block_size
		pad_h = (bh - (H_r % bh)) % bh
		pad_w = (bw - (W_r % bw)) % bw
		Hp = H_r + pad_h
		Wp = W_r + pad_w
		Gh = Hp // bh
		Gw = Wp // bw
		area = bh * bw

		rel_x = (valid_mkpts[:, 0] / 8).long().clamp(0, W_r - 1)
		rel_y = (valid_mkpts[:, 1] / 8).long().clamp(0, H_r - 1)
		block_col = rel_x // bw
		block_row = rel_y // bh
		block_idx = block_row * Gw + block_col

		n_i = torch.bincount(block_idx, minlength=Gh * Gw).long()
		if n_i.sum() == 0:
			topk_out = torch.topk(valid_scores, k=K, dim=-1)
			return valid_indices[topk_out.indices]

		score_tau = torch.quantile(valid_scores, self.score_tau_quantile) if self.quality_floor else 0.0
		block_score_top1 = torch.zeros(Gh * Gw, device=device, dtype=valid_scores.dtype)
		block_score_topm_mean = torch.zeros(Gh * Gw, device=device, dtype=valid_scores.dtype)
		for i in range(Gh * Gw):
			if n_i[i] == 0:
				continue
			in_block = (block_idx == i)
			bs = valid_scores[in_block]
			block_score_top1[i] = bs.max()
			m = min(self.top_m, bs.numel())
			block_score_topm_mean[i] = bs.topk(k=m).values.mean()

		s_bar = F.avg_pool2d(reliability_map[None, None], kernel_size=3, stride=1, padding=1)[0, 0]

		valid_rel = s_bar[s_bar > 0]
		if valid_rel.numel() == 0:
			topk_out = torch.topk(valid_scores, k=K, dim=-1)
			return valid_indices[topk_out.indices]

		q_hi = torch.quantile(valid_rel, self.clip_quantile)
		s_clip = torch.minimum(s_bar, q_hi)

		s_raw_pad = F.pad(reliability_map, (0, pad_w, 0, pad_h), value=0.0)
		s_clip_pad = F.pad(s_clip, (0, pad_w, 0, pad_h), value=0.0)

		raw_blocks = self.to_blocks(s_raw_pad, Gh, Gw, bh, bw)
		clip_blocks = self.to_blocks(s_clip_pad, Gh, Gw, bh, bw)

		A_i = clip_blocks.sum(dim=-1)
		mu_i = A_i / area

		m_i = clip_blocks.max(dim=-1).values
		m_i = torch.where(n_i > 0, m_i, torch.zeros_like(m_i))

		den = A_i[:, None] + area * self.eps
		p_ij = (clip_blocks + self.eps) / den
		entropy = -(p_ij * torch.log(p_ij.clamp_min(self.eps))).sum(dim=-1)

		H_i = torch.ones_like(entropy)
		if self.entropy_norm_mode == 'area':
			log_norm = math.log(area)
			H_i = (entropy / log_norm).clamp(0.0, 1.0)
		else:
			norm_mask = n_i > 1
			if torch.any(norm_mask):
				H_i[norm_mask] = entropy[norm_mask] / torch.log(n_i[norm_mask].float())
			H_i = H_i.clamp(0.0, 1.0)

		top_m = min(max(self.top_m, 1), area)
		top_vals = torch.topk(clip_blocks, k=top_m, dim=-1).values
		top_is_valid = top_vals > 0
		Q_i = top_vals.masked_fill(~top_is_valid, 0.0).sum(dim=-1) / top_is_valid.sum(dim=-1).clamp(min=1)

		w_i = (mu_i + self.eps).pow(self.alpha)
		w_i = w_i * (Q_i + self.eps).pow(1.0 - self.alpha)
		w_i = w_i * (H_i + self.eps).pow(self.beta)

		if self.use_cand_quality:
			cand_q_i = block_score_topm_mean
			cand_q_max = cand_q_i.max()
			if cand_q_max > 0:
				cand_q_i = cand_q_i / cand_q_max
			w_i = w_i * (cand_q_i + self.eps).pow(self.cand_quality_delta)

		if self.use_count_factor:
			count_factor = (n_i.float() / self.n_ref).clamp(0.0, 1.0)
			w_i = w_i * count_factor.pow(self.rho)

		w_i = torch.where(n_i > 0, w_i, torch.zeros_like(w_i))

		tau_m = torch.median(valid_rel)
		tau_A = tau_m * area
		active = ((m_i > tau_m) | (A_i > tau_A)) & (n_i > 0)

		if self.quality_floor:
			active = active & (n_i >= self.n_min) & (block_score_top1 > score_tau)

		n_active = int(active.sum().item())
		if n_active == 0:
			topk_out = torch.topk(valid_scores, k=K, dim=-1)
			return valid_indices[topk_out.indices]

		w_active = w_i[active]
		if torch.all(w_active <= 0):
			w_active = torch.ones_like(w_active)

		temperature = self.compute_temperature(w_active)
		P = torch.zeros_like(w_i)
		P[active] = torch.softmax(torch.log(w_active.clamp_min(self.eps)) / temperature, dim=0)

		L = torch.zeros_like(n_i)
		if K >= n_active:
			L[active] = 1

		cap_avg = max(1, int(math.ceil(self.eta * K / max(n_active, 1))))
		cap_global = max(1, int(math.ceil(self.gamma * K)))

		C = torch.zeros_like(n_i)
		C_active = torch.minimum(
			n_i,
			torch.full_like(n_i, cap_avg),
		)
		C_active = torch.minimum(C_active, torch.full_like(n_i, cap_global))
		C[active] = C_active[active]

		if C.sum() == 0:
			topk_out = torch.topk(valid_scores, k=K, dim=-1)
			return valid_indices[topk_out.indices]

		remaining = max(K - int(L.sum().item()), 0)
		k_hat = L.float() + remaining * P
		k_hat = torch.maximum(k_hat, L.float())
		k_hat = torch.minimum(k_hat, C.float())

		k_i = torch.floor(k_hat).long()
		frac = k_hat - k_i.float()

		residual = int(K - k_i.sum().item())
		if residual > 0:
			order = torch.argsort(frac, descending=True)
			for idx in order:
				if residual <= 0:
					break
				room = int((C[idx] - k_i[idx]).item())
				if room <= 0:
					continue
				add = min(room, residual)
				k_i[idx] += add
				residual -= add
		elif residual < 0:
			order = torch.argsort(frac, descending=False)
			for idx in order:
				if residual >= 0:
					break
				room = int((k_i[idx] - L[idx]).item())
				if room <= 0:
					continue
				drop = min(room, -residual)
				k_i[idx] -= drop
				residual += drop

		selected_in_valid = []
		selected_score_vals = []

		for i in torch.nonzero(k_i > 0, as_tuple=False).flatten():
			k_block = int(k_i[i].item())
			if k_block <= 0:
				continue

			in_block = (block_idx == i)
			block_candidate_indices = torch.where(in_block)[0]

			if block_candidate_indices.numel() == 0:
				continue

			block_candidate_scores = valid_scores[block_candidate_indices]
			k_take = min(k_block, block_candidate_indices.numel())
			topk_local = torch.topk(block_candidate_scores, k=k_take, dim=-1)

			selected_in_valid.append(block_candidate_indices[topk_local.indices])
			selected_score_vals.append(topk_local.values)

		if selected_in_valid:
			selected_in_valid = torch.cat(selected_in_valid, dim=0)
			selected_score_vals = torch.cat(selected_score_vals, dim=0)
		else:
			selected_in_valid = torch.empty((0,), device=device, dtype=torch.long)
			selected_score_vals = torch.empty((0,), device=device, dtype=valid_scores.dtype)

		if selected_in_valid.numel() < K:
			missing = K - selected_in_valid.numel()
			selected_set = torch.zeros(n_valid, dtype=torch.bool, device=device)
			if selected_in_valid.numel() > 0:
				selected_set[selected_in_valid] = True

			if self.cap_aware_refill:
				refill_idx = self._cap_aware_refill_sparse(
					k_i, C, block_idx, valid_scores, selected_set,
					missing, n_valid, active, w_i, device
				)
			else:
				refill_idx = self._global_refill_sparse(
					valid_scores, selected_set, missing, n_valid, device
				)

			if refill_idx is not None and refill_idx.numel() > 0:
				selected_in_valid = torch.cat([selected_in_valid, refill_idx.long()], dim=0)
				selected_score_vals = torch.cat([selected_score_vals, valid_scores[refill_idx]], dim=0)

		if selected_in_valid.numel() > K:
			top = torch.topk(selected_score_vals, k=K, dim=-1).indices
			selected_in_valid = selected_in_valid[top]
			selected_score_vals = selected_score_vals[top]

		if selected_in_valid.numel() > 1:
			order = torch.argsort(selected_score_vals, descending=True)
			selected_in_valid = selected_in_valid[order]

		if selected_in_valid.numel() != K:
			topk_out = torch.topk(valid_scores, k=min(K, n_valid), dim=-1)
			return valid_indices[topk_out.indices]

		if self.debug:
			self._last_debug_stats = {
				'n_active': n_active,
				'n_total_blocks': Gh * Gw,
				'K': K,
				'quality_floor_active': self.quality_floor,
				'score_tau': float(score_tau) if self.quality_floor else 0.0,
				'n_blocks_below_floor': int(((n_i < self.n_min) | (block_score_top1 <= score_tau)).sum().item()) if self.quality_floor else 0,
				'cand_quality_delta': self.cand_quality_delta if self.use_cand_quality else 0.0,
				'entropy_norm_mode': self.entropy_norm_mode,
				'cap_aware_refill': self.cap_aware_refill,
				'mean_w_active': float(w_active.mean().item()) if n_active > 0 else 0.0,
				'temperature': float(temperature.item()),
			}

		return valid_indices[selected_in_valid].long()

	def _cap_aware_refill_sparse(
		self,
		k_i: torch.Tensor,
		C: torch.Tensor,
		block_idx: torch.Tensor,
		valid_scores: torch.Tensor,
		selected_set: torch.Tensor,
		missing: int,
		n_valid: int,
		active: torch.Tensor,
		w_i: torch.Tensor,
		device: torch.device,
	) -> Optional[torch.Tensor]:
		refill_from_active = []
		refill_from_active_scores = []

		active_under_cap = torch.where(active & (k_i < C))[0]
		if active_under_cap.numel() > 0:
			sorted_by_utility = active_under_cap[torch.argsort(w_i[active_under_cap], descending=True)]
			for block_id in sorted_by_utility:
				if missing <= 0:
					break
				room = int((C[block_id] - k_i[block_id]).item())
				if room <= 0:
					continue
				in_block = (block_idx == block_id)
				block_cand_idx = torch.where(in_block & ~selected_set[:n_valid] if n_valid <= selected_set.shape[0] else in_block)[0]
				if n_valid <= selected_set.shape[0]:
					block_cand_idx = torch.where(in_block & ~selected_set[:n_valid])[0]
				else:
					block_cand_idx = torch.where(in_block)[0]
					block_cand_idx = block_cand_idx[~selected_set[block_cand_idx]]

				if block_cand_idx.numel() == 0:
					continue
				take = min(room, missing, block_cand_idx.numel())
				bs = valid_scores[block_cand_idx]
				topk_local = torch.topk(bs, k=take, dim=-1)
				refill_from_active.append(block_cand_idx[topk_local.indices])
				refill_from_active_scores.append(topk_local.values)
				missing -= take

		if missing > 0:
			inactive_with_cands = torch.where(~active)[0]
			for block_id in inactive_with_cands:
				if missing <= 0:
					break
				in_block = (block_idx == block_id)
				if n_valid <= selected_set.shape[0]:
					block_cand_idx = torch.where(in_block & ~selected_set[:n_valid])[0]
				else:
					block_cand_idx = torch.where(in_block)[0]
					block_cand_idx = block_cand_idx[~selected_set[block_cand_idx]]

				if block_cand_idx.numel() == 0:
					continue
				bs = valid_scores[block_cand_idx]
				if bs.max() <= 0:
					continue
				take = min(missing, block_cand_idx.numel())
				topk_local = torch.topk(bs, k=take, dim=-1)
				refill_from_active.append(block_cand_idx[topk_local.indices])
				refill_from_active_scores.append(topk_local.values)
				missing -= take

		if missing > 0:
			candidate_mask = ~selected_set[:n_valid] if n_valid <= selected_set.shape[0] else torch.ones(n_valid, dtype=torch.bool, device=device)
			candidate_idx = torch.where(candidate_mask)[0]
			if candidate_idx.numel() > 0:
				refill_k = min(missing, candidate_idx.numel())
				cand_scores = valid_scores[candidate_idx]
				top_refill = torch.topk(cand_scores, k=refill_k, dim=-1).indices
				refill_from_active.append(candidate_idx[top_refill])
				refill_from_active_scores.append(cand_scores[top_refill])

		if refill_from_active:
			return torch.cat(refill_from_active, dim=0)
		return None

	def _global_refill_sparse(
		self,
		valid_scores: torch.Tensor,
		selected_set: torch.Tensor,
		missing: int,
		n_valid: int,
		device: torch.device,
	) -> Optional[torch.Tensor]:
		candidate_mask = ~selected_set
		candidate_idx = torch.where(candidate_mask)[0]
		if candidate_idx.numel() > 0:
			refill_k = min(missing, candidate_idx.numel())
			cand_scores = valid_scores[candidate_idx]
			top_refill = torch.topk(cand_scores, k=refill_k, dim=-1).indices
			return candidate_idx[top_refill]
		return None

	def select_single(self, scores: torch.Tensor, K: int, mask: Optional[torch.Tensor]) -> torch.Tensor:
		device = scores.device
		H, W = scores.shape

		if self.shifted_grid:
			return self._select_single_shifted(scores, K, mask)

		if mask is None:
			valid = torch.ones((H, W), dtype=torch.bool, device=device)
		else:
			valid = mask.bool()

		if not torch.any(valid):
			return self.global_topk(scores, K, valid_mask=None)

		return self._select_single_core(scores, K, valid, H, W, device)

	def _select_single_shifted(self, scores: torch.Tensor, K: int, mask: Optional[torch.Tensor]) -> torch.Tensor:
		device = scores.device
		H, W = scores.shape

		if mask is None:
			valid = torch.ones((H, W), dtype=torch.bool, device=device)
		else:
			valid = mask.bool()

		if not torch.any(valid):
			return self.global_topk(scores, K, valid_mask=None)

		result_0 = self._select_single_core(scores, K, valid, H, W, device)

		bh, bw = self.block_size
		shift_h, shift_w = bh // 2, bw // 2

		pad_top = shift_h
		pad_bottom = (bh - (H % bh)) % bh
		pad_left = shift_w
		pad_right = (bw - (W % bw)) % bw

		scores_shifted = F.pad(scores, (pad_left, pad_right, pad_top, pad_bottom), value=float('-inf'))
		valid_shifted = F.pad(valid.float(), (pad_left, pad_right, pad_top, pad_bottom), value=0.0).bool()

		Hs, Ws = scores_shifted.shape
		result_1_shifted = self._select_single_core(scores_shifted, K, valid_shifted, Hs, Ws, device)

		result_1 = torch.zeros_like(result_1_shifted)
		shifted_y = result_1_shifted // Ws
		shifted_x = result_1_shifted % Ws
		orig_y = shifted_y - shift_h
		orig_x = shifted_x - shift_w
		in_bounds = (orig_y >= 0) & (orig_y < H) & (orig_x >= 0) & (orig_x < W)
		valid_results = in_bounds
		if valid_results.any():
			result_1[valid_results] = orig_y[valid_results] * W + orig_x[valid_results]
		else:
			return result_0

		flat_scores = scores.reshape(-1)
		scores_0 = flat_scores[result_0].mean()
		scores_1 = flat_scores[result_1].mean()

		if scores_0 >= scores_1:
			return result_0
		else:
			return result_1

	def _select_single_core(self, scores: torch.Tensor, K: int, valid: torch.Tensor, H: int, W: int, device: torch.device) -> torch.Tensor:
		valid_f = valid.float()
		s_weighted = scores * valid_f
		s_num = F.avg_pool2d(s_weighted[None, None], kernel_size=3, stride=1, padding=1)
		s_den = F.avg_pool2d(valid_f[None, None], kernel_size=3, stride=1, padding=1)
		s_bar = torch.where(s_den > 0, s_num / s_den.clamp_min(self.eps), torch.zeros_like(s_num))[0, 0]
		s_bar = torch.where(valid, s_bar, torch.zeros_like(s_bar))

		valid_scores = s_bar[valid]
		if valid_scores.numel() == 0:
			return self.global_topk(scores, K, valid_mask=valid)

		q_hi = torch.quantile(valid_scores, self.clip_quantile)
		s_clip = torch.minimum(s_bar, q_hi)

		bh, bw = self.block_size
		pad_h = (bh - (H % bh)) % bh
		pad_w = (bw - (W % bw)) % bw
		Hp = H + pad_h
		Wp = W + pad_w

		s_raw_pad = F.pad(scores, (0, pad_w, 0, pad_h), value=float('-inf'))
		s_clip_pad = F.pad(s_clip, (0, pad_w, 0, pad_h), value=0.0)
		valid_pad = F.pad(valid.float(), (0, pad_w, 0, pad_h), value=0.0).bool()

		Gh = Hp // bh
		Gw = Wp // bw
		area = bh * bw

		raw_blocks = self.to_blocks(s_raw_pad, Gh, Gw, bh, bw)
		clip_blocks = self.to_blocks(s_clip_pad, Gh, Gw, bh, bw)
		valid_blocks = self.to_blocks(valid_pad.float(), Gh, Gw, bh, bw).bool()

		n_i = valid_blocks.sum(dim=-1).long()
		if n_i.sum() == 0:
			return self.global_topk(scores, K, valid_mask=valid)

		clip_valid = clip_blocks * valid_blocks.float()
		A_i = clip_valid.sum(dim=-1)
		mu_i = A_i / n_i.clamp(min=1).float()

		max_blocks = raw_blocks.masked_fill(~valid_blocks, float('-inf'))
		max_blocks_clip = clip_blocks.masked_fill(~valid_blocks, float('-inf'))
		m_i = max_blocks_clip.max(dim=-1).values
		m_i = torch.where(n_i > 0, m_i, torch.zeros_like(m_i))

		den = A_i[:, None] + n_i.float()[:, None] * self.eps
		p_ij = (clip_valid + self.eps) / den
		p_ij = torch.where(valid_blocks, p_ij, torch.zeros_like(p_ij))
		entropy = -(p_ij * torch.log(p_ij.clamp_min(self.eps))).sum(dim=-1)

		H_i = torch.ones_like(entropy)
		if self.entropy_norm_mode == 'area':
			H_i = (entropy / math.log(area)).clamp(0.0, 1.0)
		else:
			norm_mask = n_i > 1
			if torch.any(norm_mask):
				H_i[norm_mask] = entropy[norm_mask] / torch.log(n_i[norm_mask].float())
			H_i = H_i.clamp(0.0, 1.0)

		top_m = min(max(self.top_m, 1), area)
		top_vals = torch.topk(max_blocks_clip, k=top_m, dim=-1).values
		top_is_valid = torch.isfinite(top_vals)
		Q_i = top_vals.masked_fill(~top_is_valid, 0.0).sum(dim=-1) / top_is_valid.sum(dim=-1).clamp(min=1)

		w_i = (mu_i + self.eps).pow(self.alpha)
		w_i = w_i * (Q_i + self.eps).pow(1.0 - self.alpha)
		w_i = w_i * (H_i + self.eps).pow(self.beta)
		w_i = torch.where(n_i > 0, w_i, torch.zeros_like(w_i))

		tau_m = torch.median(valid_scores)
		tau_A = tau_m * n_i.float()
		active = ((m_i > tau_m) | (A_i > tau_A)) & (n_i > 0)

		n_active = int(active.sum().item())
		if n_active == 0:
			return self.global_topk(scores, K, valid_mask=valid)

		w_active = w_i[active]
		if torch.all(w_active <= 0):
			w_active = torch.ones_like(w_active)

		temperature = self.compute_temperature(w_active)
		P = torch.zeros_like(w_i)
		P[active] = torch.softmax(torch.log(w_active.clamp_min(self.eps)) / temperature, dim=0)

		L = torch.zeros_like(n_i)
		if K >= n_active:
			L[active] = 1

		cap_avg = max(1, int(math.ceil(self.eta * K / max(n_active, 1))))
		cap_global = max(1, int(math.ceil(self.gamma * K)))

		C = torch.zeros_like(n_i)
		C_active = torch.minimum(
			n_i,
			torch.full_like(n_i, cap_avg),
		)
		C_active = torch.minimum(C_active, torch.full_like(n_i, cap_global))
		C[active] = C_active[active]

		if C.sum() == 0:
			return self.global_topk(scores, K, valid_mask=valid)

		remaining = max(K - int(L.sum().item()), 0)
		k_hat = L.float() + remaining * P
		k_hat = torch.maximum(k_hat, L.float())
		k_hat = torch.minimum(k_hat, C.float())

		k_i = torch.floor(k_hat).long()
		frac = k_hat - k_i.float()

		residual = int(K - k_i.sum().item())
		if residual > 0:
			order = torch.argsort(frac, descending=True)
			for idx in order:
				if residual <= 0:
					break
				room = int((C[idx] - k_i[idx]).item())
				if room <= 0:
					continue
				add = min(room, residual)
				k_i[idx] += add
				residual -= add
		elif residual < 0:
			order = torch.argsort(frac, descending=False)
			for idx in order:
				if residual >= 0:
					break
				room = int((k_i[idx] - L[idx]).item())
				if room <= 0:
					continue
				drop = min(room, -residual)
				k_i[idx] -= drop
				residual += drop

		yy, xx = torch.meshgrid(
			torch.arange(Hp, device=device),
			torch.arange(Wp, device=device),
			indexing='ij',
		)
		xx_blocks = self.to_blocks(xx, Gh, Gw, bh, bw)
		yy_blocks = self.to_blocks(yy, Gh, Gw, bh, bw)

		selected_idx = []
		selected_scores = []
		for i in torch.nonzero(k_i > 0, as_tuple=False).flatten():
			k_block = int(k_i[i].item())
			if k_block <= 0:
				continue

			valid_count = int(n_i[i].item())
			if valid_count <= 0:
				continue

			k_take = min(k_block, valid_count)
			vals, loc = torch.topk(max_blocks[i], k=k_take, dim=-1)
			x_sel = xx_blocks[i, loc]
			y_sel = yy_blocks[i, loc]
			flat_idx = y_sel * W + x_sel

			selected_idx.append(flat_idx.long())
			selected_scores.append(vals)

		if selected_idx:
			selected_idx = torch.cat(selected_idx, dim=0)
			selected_scores = torch.cat(selected_scores, dim=0)
		else:
			selected_idx = torch.empty((0,), device=device, dtype=torch.long)
			selected_scores = torch.empty((0,), device=device, dtype=scores.dtype)

		if selected_idx.numel() < K:
			missing = K - selected_idx.numel()
			flat_scores = scores.reshape(-1)
			flat_valid = valid.reshape(-1)
			selected_mask = torch.zeros_like(flat_valid)
			if selected_idx.numel() > 0:
				selected_mask[selected_idx] = True

			if self.cap_aware_refill:
				refill_result = self._cap_aware_refill_dense(
					k_i, C, xx_blocks, yy_blocks, max_blocks, W,
					selected_mask, flat_scores, flat_valid, missing, device
				)
				if refill_result is not None:
					r_idx, r_scores = refill_result
					selected_idx = torch.cat([selected_idx, r_idx], dim=0)
					selected_scores = torch.cat([selected_scores, r_scores], dim=0)
					missing = K - selected_idx.numel()

			if selected_idx.numel() < K:
				missing = K - selected_idx.numel()
				selected_mask2 = torch.zeros_like(flat_valid)
				if selected_idx.numel() > 0:
					selected_mask2[selected_idx] = True

				candidate_mask = (~selected_mask2) & flat_valid
				if candidate_mask.sum() < missing and self.relax_mask_if_needed:
					candidate_mask = ~selected_mask2

				candidate_idx = torch.nonzero(candidate_mask, as_tuple=False).flatten()
				if candidate_idx.numel() > 0:
					refill_k = min(missing, candidate_idx.numel())
					cand_scores = flat_scores[candidate_idx]
					top_refill = torch.topk(cand_scores, k=refill_k, dim=-1).indices
					refill_idx = candidate_idx[top_refill]
					selected_idx = torch.cat([selected_idx, refill_idx.long()], dim=0)
					selected_scores = torch.cat([selected_scores, flat_scores[refill_idx]], dim=0)

		if selected_idx.numel() > K:
			top = torch.topk(selected_scores, k=K, dim=-1).indices
			selected_idx = selected_idx[top]
			selected_scores = selected_scores[top]

		if selected_idx.numel() == K and K > 1:
			order = torch.argsort(selected_scores, descending=True)
			selected_idx = selected_idx[order]

		if selected_idx.numel() != K:
			return self.global_topk(scores, K, valid_mask=valid)

		return selected_idx.long()

	def _cap_aware_refill_dense(
		self,
		k_i: torch.Tensor,
		C: torch.Tensor,
		xx_blocks: torch.Tensor,
		yy_blocks: torch.Tensor,
		max_blocks: torch.Tensor,
		W: int,
		selected_mask: torch.Tensor,
		flat_scores: torch.Tensor,
		flat_valid: torch.Tensor,
		missing: int,
		device: torch.device,
	):
		refill_idx = []
		refill_scores = []

		active_under_cap = torch.where(k_i < C)[0]
		if active_under_cap.numel() > 0:
			for block_id in active_under_cap:
				if missing <= 0:
					break
				room = int((C[block_id] - k_i[block_id]).item())
				if room <= 0:
					continue

				x_sel = xx_blocks[block_id]
				y_sel = yy_blocks[block_id]
				flat_idx = y_sel * W + x_sel

				valid_in_block = flat_valid[flat_idx]
				not_selected = ~selected_mask[flat_idx]
				available = valid_in_block & not_selected
				available_flat = flat_idx[available]
				available_scores = max_blocks[block_id][available]

				if available_flat.numel() == 0:
					continue

				take = min(room, missing, available_flat.numel())
				if available_scores.numel() > take:
					topk_local = torch.topk(available_scores, k=take, dim=-1)
					refill_idx.append(available_flat[topk_local.indices])
					refill_scores.append(topk_local.values)
				else:
					finite_mask = torch.isfinite(available_scores)
					refill_idx.append(available_flat[finite_mask][:take])
					refill_scores.append(available_scores[finite_mask][:take])
				missing -= take

		if refill_idx:
			return torch.cat(refill_idx, dim=0).long(), torch.cat(refill_scores, dim=0)
		return None

	def global_topk(self, scores: torch.Tensor, K: int, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
		flat = scores.reshape(-1)
		if valid_mask is not None and torch.any(valid_mask):
			mask_flat = valid_mask.reshape(-1)
			if int(mask_flat.sum().item()) >= K:
				masked = flat.masked_fill(~mask_flat, float('-inf'))
				return torch.topk(masked, k=K, dim=-1).indices.long()

		return torch.topk(flat, k=K, dim=-1).indices.long()

	def normalize_scores(self, score_map: torch.Tensor) -> torch.Tensor:
		scores = torch.nan_to_num(score_map, nan=0.0, posinf=0.0, neginf=0.0).float()

		if self.auto_sigmoid:
			if scores.dim() == 2:
				s_min = scores.min()
				s_max = scores.max()
				if (s_min < 0) or (s_max > 1):
					scores = torch.sigmoid(scores)
			elif scores.dim() in (3, 4):
				dims = (-2, -1)
				s_min = scores.amin(dim=dims, keepdim=True)
				s_max = scores.amax(dim=dims, keepdim=True)
				needs_sigmoid = (s_min < 0) | (s_max > 1)
				if torch.any(needs_sigmoid):
					scores = torch.where(needs_sigmoid, torch.sigmoid(scores), scores)

		return scores

	def prepare_mask(
		self,
		mask: Optional[torch.Tensor],
		B: int,
		H: int,
		W: int,
		device: torch.device,
	) -> Optional[torch.Tensor]:
		if mask is None:
			return None

		m = mask.to(device)
		if m.dim() == 2:
			m = m.unsqueeze(0)
		elif m.dim() == 4:
			if m.shape[1] != 1:
				raise ValueError("mask with 4 dims must have shape (B, 1, H, W)")
			m = m[:, 0]
		elif m.dim() != 3:
			raise ValueError("mask must be (H, W), (B, H, W) or (B, 1, H, W)")

		if m.shape[0] == 1 and B > 1:
			m = m.expand(B, -1, -1)
		elif m.shape[0] != B:
			raise ValueError("mask batch dimension is incompatible with score_map")

		if m.shape[-2:] != (H, W):
			m = F.interpolate(m[:, None].float(), size=(H, W), mode='nearest')[:, 0]

		return m > 0.5

	def compute_temperature(self, w_active: torch.Tensor) -> torch.Tensor:
		if self.adaptive_temperature:
			g = self.gini(w_active)
			t = self.temperature_base + self.gini_scale * g
		else:
			t = w_active.new_tensor(self.temperature_base)

		return t.clamp(min=self.temperature_min, max=self.temperature_max)

	def gini(self, values: torch.Tensor) -> torch.Tensor:
		x = values.clamp_min(0)
		x = x[x > 0]
		if x.numel() <= 1:
			return values.new_tensor(0.0)

		x_sorted, _ = torch.sort(x)
		n = x_sorted.numel()
		idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
		x_sum = x_sorted.sum().clamp_min(self.eps)
		g = (2.0 * (idx * x_sorted).sum() / (n * x_sum)) - ((n + 1.0) / n)
		return g.clamp(0.0, 1.0)

	@staticmethod
	def to_blocks(t: torch.Tensor, gh: int, gw: int, bh: int, bw: int) -> torch.Tensor:
		return t.view(gh, bh, gw, bw).permute(0, 2, 1, 3).reshape(gh * gw, bh * bw)

	def get_debug_stats(self) -> Optional[Dict[str, Any]]:
		return self._last_debug_stats


@dataclass
class SceneStatistics:
	reliability_gini: float
	top10_mass_ratio: float
	active_block_ratio: float
	candidate_block_coverage: float
	reliability_mean: float
	reliability_std: float
	candidate_concentration: float
	block_score_top1_mean: float
	block_score_topm_mean: float
	block_candidate_count_mean: float
	coverage_need: float
	quality_concentration: float
	lambda_val: float


class SceneStatisticsComputer:

	@staticmethod
	@torch.inference_mode()
	def compute(
		reliability_map: torch.Tensor,
		mkpts: torch.Tensor,
		scores: torch.Tensor,
		block_size: Tuple[int, int] = (8, 8),
	) -> SceneStatistics:
		H_r, W_r = reliability_map.shape
		bh, bw = block_size

		rel_flat = reliability_map.reshape(-1)
		rel_positive = rel_flat[rel_flat > 0]

		reliability_gini = SceneStatisticsComputer._gini(rel_positive)
		top10_mass_ratio = SceneStatisticsComputer._top10_mass_ratio(rel_positive)

		pad_h = (bh - (H_r % bh)) % bh
		pad_w = (bw - (W_r % bw)) % bw
		Hp = H_r + pad_h
		Wp = W_r + pad_w
		Gh = Hp // bh
		Gw = Wp // bw
		total_blocks = Gh * Gw

		rel_pad = F.pad(reliability_map, (0, pad_w, 0, pad_h), value=0.0)
		rel_blocks = rel_pad.view(Gh, bh, Gw, bw).permute(0, 2, 1, 3).reshape(Gh * Gw, bh * bw)
		block_means = rel_blocks.mean(dim=-1)

		active_blocks = (block_means > 0).sum().item()
		active_block_ratio = active_blocks / total_blocks if total_blocks > 0 else 0.0

		N = mkpts.shape[0]
		block_idx = None
		cand_blocks_with_kp = 0
		block_score_top1 = torch.zeros(total_blocks, device=reliability_map.device, dtype=torch.float32)
		block_score_topm_mean = torch.zeros(total_blocks, device=reliability_map.device, dtype=torch.float32)
		block_candidate_count = torch.zeros(total_blocks, device=reliability_map.device, dtype=torch.long)

		if N > 0:
			rel_x = (mkpts[:, 0] / 8).long().clamp(0, W_r - 1)
			rel_y = (mkpts[:, 1] / 8).long().clamp(0, H_r - 1)
			block_col = rel_x // bw
			block_row = rel_y // bh
			block_idx = block_row * Gw + block_col

			cand_blocks_with_kp = torch.unique(block_idx).numel()

			valid_mask = scores > 0
			n_i = torch.bincount(block_idx, minlength=total_blocks).long()
			block_candidate_count = n_i

			for i in range(total_blocks):
				if n_i[i] == 0:
					continue
				in_block = (block_idx == i) & valid_mask
				if in_block.sum() == 0:
					continue
				bs = scores[in_block]
				block_score_top1[i] = bs.max()
				m = min(8, bs.numel())
				block_score_topm_mean[i] = bs.topk(k=m).values.mean()

		candidate_block_coverage = cand_blocks_with_kp / total_blocks if total_blocks > 0 else 0.0

		reliability_mean = rel_positive.mean().item() if rel_positive.numel() > 0 else 0.0
		reliability_std = rel_positive.std().item() if rel_positive.numel() > 1 else 0.0

		candidate_concentration = SceneStatisticsComputer._candidate_concentration(
			mkpts, scores, Gh, Gw, bw, bh, W_r, H_r
		)

		active_with_cands = block_candidate_count > 0
		block_score_top1_mean = block_score_top1[active_with_cands].mean().item() if active_with_cands.any() else 0.0
		block_score_topm_mean_val = block_score_topm_mean[active_with_cands].mean().item() if active_with_cands.any() else 0.0
		block_candidate_count_mean = block_candidate_count[active_with_cands].float().mean().item() if active_with_cands.any() else 0.0

		coverage_need = SceneStatisticsComputer._compute_coverage_need(
			active_block_ratio, candidate_block_coverage,
			reliability_gini, top10_mass_ratio,
		)

		quality_concentration = SceneStatisticsComputer._compute_quality_concentration(
			reliability_gini, top10_mass_ratio,
			candidate_concentration, active_block_ratio,
		)

		lambda_val = SceneStatisticsComputer._compute_lambda(coverage_need, quality_concentration)

		return SceneStatistics(
			reliability_gini=reliability_gini,
			top10_mass_ratio=top10_mass_ratio,
			active_block_ratio=active_block_ratio,
			candidate_block_coverage=candidate_block_coverage,
			reliability_mean=reliability_mean,
			reliability_std=reliability_std,
			candidate_concentration=candidate_concentration,
			block_score_top1_mean=block_score_top1_mean,
			block_score_topm_mean=block_score_topm_mean_val,
			block_candidate_count_mean=block_candidate_count_mean,
			coverage_need=coverage_need,
			quality_concentration=quality_concentration,
			lambda_val=lambda_val,
		)

	@staticmethod
	def _compute_coverage_need(
		active_block_ratio: float,
		candidate_block_coverage: float,
		reliability_gini: float,
		top10_mass_ratio: float,
	) -> float:
		coverage_score = (
			0.35 * active_block_ratio
			+ 0.35 * candidate_block_coverage
			+ 0.15 * (1.0 - reliability_gini)
			+ 0.15 * (1.0 - top10_mass_ratio)
		)
		return float(max(0.0, min(1.0, coverage_score)))

	@staticmethod
	def _compute_quality_concentration(
		reliability_gini: float,
		top10_mass_ratio: float,
		candidate_concentration: float,
		active_block_ratio: float,
	) -> float:
		concentration_score = (
			0.35 * reliability_gini
			+ 0.25 * top10_mass_ratio
			+ 0.25 * candidate_concentration
			+ 0.15 * (1.0 - active_block_ratio)
		)
		return float(max(0.0, min(1.0, concentration_score)))

	@staticmethod
	def _compute_lambda(coverage_need: float, quality_concentration: float) -> float:
		raw = coverage_need - quality_concentration
		lambda_val = 1.0 / (1.0 + math.exp(-6.0 * raw))
		return float(max(0.0, min(1.0, lambda_val)))

	@staticmethod
	def _gini(values: torch.Tensor) -> float:
		if values.numel() <= 1:
			return 0.0
		x, _ = torch.sort(values.clamp_min(0))
		n = x.numel()
		idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
		x_sum = x.sum().clamp_min(1e-6)
		g = (2.0 * (idx * x).sum() / (n * x_sum)) - ((n + 1.0) / n)
		return float(g.clamp(0.0, 1.0).item())

	@staticmethod
	def _top10_mass_ratio(values: torch.Tensor) -> float:
		if values.numel() == 0:
			return 0.0
		total = values.sum().item()
		if total <= 0:
			return 0.0
		k = max(1, int(values.numel() * 0.1))
		top_vals, _ = torch.topk(values, k=min(k, values.numel()))
		return float(top_vals.sum().item() / total)

	@staticmethod
	def _candidate_concentration(
		mkpts: torch.Tensor,
		scores: torch.Tensor,
		Gh: int, Gw: int,
		bw: int, bh: int,
		W_r: int, H_r: int,
	) -> float:
		N = mkpts.shape[0]
		if N <= 1:
			return 0.0
		rel_x = (mkpts[:, 0] / 8).long().clamp(0, W_r - 1)
		rel_y = (mkpts[:, 1] / 8).long().clamp(0, H_r - 1)
		block_col = rel_x // bw
		block_row = rel_y // bh
		block_idx = block_row * Gw + block_col

		n_i = torch.bincount(block_idx, minlength=Gh * Gw).float()
		total_blocks = Gh * Gw
		if total_blocks == 0:
			return 0.0
		probs = n_i / n_i.sum().clamp_min(1e-6)
		probs = probs[probs > 0]
		entropy = -(probs * torch.log(probs)).sum().item()
		max_entropy = math.log(total_blocks)
		if max_entropy <= 0:
			return 0.0
		concentration = 1.0 - (entropy / max_entropy)
		return float(max(0.0, min(1.0, concentration)))


class TextureAwareParamGenerator:

	GAMMA_RANGE = (0.10, 0.18)
	BETA_RANGE = (0.25, 0.60)
	TEMPERATURE_BASE_RANGE = (0.85, 1.30)
	SCORE_TAU_QUANTILE_RANGE = (0.30, 0.50)
	GLOBAL_RESERVE_RATIO_RANGE = (0.20, 0.35)

	@staticmethod
	def generate(lambda_val: float) -> Dict[str, Any]:
		gamma = TextureAwareParamGenerator._lerp(
			lambda_val, TextureAwareParamGenerator.GAMMA_RANGE, reverse=True
		)
		beta = TextureAwareParamGenerator._lerp(
			lambda_val, TextureAwareParamGenerator.BETA_RANGE, reverse=False
		)
		temperature_base = TextureAwareParamGenerator._lerp(
			lambda_val, TextureAwareParamGenerator.TEMPERATURE_BASE_RANGE, reverse=False
		)
		score_tau_quantile = TextureAwareParamGenerator._lerp(
			lambda_val, TextureAwareParamGenerator.SCORE_TAU_QUANTILE_RANGE, reverse=True
		)
		global_reserve_ratio = TextureAwareParamGenerator._lerp(
			lambda_val, TextureAwareParamGenerator.GLOBAL_RESERVE_RATIO_RANGE, reverse=True
		)

		temperature_min = max(0.5, temperature_base - 0.3)
		temperature_max = temperature_base + 1.0

		return {
			"block_size": (8, 8),
			"gamma": gamma,
			"beta": beta,
			"temperature_base": temperature_base,
			"temperature_min": temperature_min,
			"temperature_max": temperature_max,
			"quality_floor": True,
			"n_min": 2,
			"score_tau_quantile": score_tau_quantile,
			"global_reserve_ratio": global_reserve_ratio,
		}

	@staticmethod
	def _lerp(t: float, value_range: Tuple[float, float], reverse: bool = False) -> float:
		lo, hi = value_range
		if reverse:
			return lo + (1.0 - t) * (hi - lo)
		return lo + t * (hi - lo)


class AdaptiveKeypointAllocator:

	def __init__(
		self,
		debug: bool = False,
	):
		self.debug = debug
		self._default_allocator = KeypointAllocator()
		self._last_scene_stats: Optional[SceneStatistics] = None
		self._last_allocator_params: Optional[Dict[str, Any]] = None
		self._last_global_reserve_ratio: float = 0.25

	@torch.inference_mode()
	def select_sparse_single(
		self,
		reliability_map: torch.Tensor,
		mkpts: torch.Tensor,
		scores: torch.Tensor,
		K: int,
	) -> torch.Tensor:
		block_size = self._resolve_block_size(reliability_map)
		stats = SceneStatisticsComputer.compute(
			reliability_map, mkpts, scores,
			block_size=block_size,
		)
		self._last_scene_stats = stats

		params = TextureAwareParamGenerator.generate(stats.lambda_val)
		self._last_allocator_params = params

		global_reserve_ratio = params.pop("global_reserve_ratio", 0.25)
		self._last_global_reserve_ratio = global_reserve_ratio

		if self.debug:
			logger.info(
				"lambda=%.4f | cov_need=%.4f | qual_conc=%.4f | "
				"gini=%.4f | top10_mass=%.4f | active_ratio=%.4f | "
				"cand_coverage=%.4f | cand_conc=%.4f | "
				"blk_top1_mean=%.4f | blk_topm_mean=%.4f | blk_cand_mean=%.4f | "
				"gamma=%.4f | beta=%.4f | temp_base=%.4f | "
				"score_tau_q=%.4f | reserve=%.4f",
				stats.lambda_val,
				stats.coverage_need,
				stats.quality_concentration,
				stats.reliability_gini,
				stats.top10_mass_ratio,
				stats.active_block_ratio,
				stats.candidate_block_coverage,
				stats.candidate_concentration,
				stats.block_score_top1_mean,
				stats.block_score_topm_mean,
				stats.block_candidate_count_mean,
				params.get("gamma", 0),
				params.get("beta", 0),
				params.get("temperature_base", 0),
				params.get("score_tau_quantile", 0),
				global_reserve_ratio,
			)

		allocator = KeypointAllocator(**params)

		device = reliability_map.device
		N = mkpts.shape[0]
		if N == 0:
			return torch.empty((0,), device=device, dtype=torch.long)

		valid_mask = scores > 0
		valid_indices = torch.where(valid_mask)[0]
		n_valid = valid_indices.numel()

		if n_valid == 0:
			return torch.empty((0,), device=device, dtype=torch.long)

		if n_valid <= K:
			order = torch.argsort(scores[valid_indices], descending=True)
			return valid_indices[order]

		reserve_k = max(0, min(int(round(global_reserve_ratio * K)), n_valid))
		spatial_k = K - reserve_k

		if spatial_k <= 0:
			topk_out = torch.topk(scores[valid_indices], k=K, dim=-1)
			return valid_indices[topk_out.indices]

		if reserve_k > 0:
			topk_reserve = torch.topk(scores[valid_indices], k=reserve_k, dim=-1)
			reserved_valid_idx = valid_indices[topk_reserve.indices]
			reserved_set = torch.zeros(N, dtype=torch.bool, device=device)
			reserved_set[reserved_valid_idx] = True
		else:
			reserved_set = torch.zeros(N, dtype=torch.bool, device=device)

		spatial_scores = scores.clone()
		spatial_scores[reserved_set] = -1

		spatial_valid = spatial_scores > 0
		spatial_valid_indices = torch.where(spatial_valid)[0]
		n_spatial_valid = spatial_valid_indices.numel()

		if n_spatial_valid == 0:
			topk_out = torch.topk(scores[valid_indices], k=K, dim=-1)
			return valid_indices[topk_out.indices]

		if n_spatial_valid <= spatial_k:
			order = torch.argsort(spatial_scores[spatial_valid_indices], descending=True)
			spatial_result = spatial_valid_indices[order]
		else:
			spatial_result = allocator.select_sparse_single(
				reliability_map, mkpts, spatial_scores, spatial_k
			)

		if reserve_k > 0:
			combined = torch.cat([reserved_valid_idx, spatial_result], dim=0)
		else:
			combined = spatial_result

		if combined.numel() > K:
			combined_scores = scores[combined]
			top = torch.topk(combined_scores, k=K, dim=-1).indices
			combined = combined[top]
		elif combined.numel() < K:
			missing = K - combined.numel()
			selected_set = torch.zeros(N, dtype=torch.bool, device=device)
			selected_set[combined] = True
			candidate_mask = (~selected_set) & (scores > 0)
			candidate_idx = torch.where(candidate_mask)[0]
			if candidate_idx.numel() > 0:
				refill_k = min(missing, candidate_idx.numel())
				cand_scores = scores[candidate_idx]
				top_refill = torch.topk(cand_scores, k=refill_k, dim=-1).indices
				combined = torch.cat([combined, candidate_idx[top_refill]], dim=0)

		if combined.numel() > 1:
			order = torch.argsort(scores[combined], descending=True)
			combined = combined[order]

		return combined.long()

	@torch.inference_mode()
	def select_topk(
		self,
		score_map: torch.Tensor,
		top_k: int,
		mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		return self._default_allocator.select_topk(score_map, top_k, mask)

	def normalize_scores(self, score_map: torch.Tensor) -> torch.Tensor:
		return self._default_allocator.normalize_scores(score_map)

	def _resolve_block_size(self, reliability_map: torch.Tensor) -> Tuple[int, int]:
		H_r, W_r = reliability_map.shape
		if H_r >= 64 and W_r >= 64:
			return (8, 8)
		return (4, 4)

	def get_last_scene_stats(self) -> Optional[SceneStatistics]:
		return self._last_scene_stats

	def get_last_allocator_params(self) -> Optional[Dict[str, Any]]:
		return self._last_allocator_params

	def get_last_global_reserve_ratio(self) -> float:
		return self._last_global_reserve_ratio
