"""
Step 5 — .TIR Tire Comparator & FSAE Benchmark (TirAnalyzerTab / TirAnalyzerDialog).

Carbon Motorsport Telemetry Edition:
- Structured Benchmark KPI table (ttk.Treeview with zebra rows and Delta vs Base)
- 4 Color-coded tire comparison slots
- 7 Comparison plots (Fy, Fx, Mz, G-G Friction Circle, Load Sensitivity, CFa, KPI Radar)
"""

from __future__ import annotations

import csv
import os
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from mf51_model import MF51Model
from tir_parser import load_tir_to_model
from ui_scroll import ScrollableFrame
from ui_theme import (
    FONT_BOLD,
    FONT_MONO_BOLD,
    FONT_SMALL,
    FONT_SMALL_BOLD,
    PALETTE,
    ModernCard,
    StepHeader,
    apply_ttk_theme,
    load_brand_logo,
    style_axes_2d,
    style_legend,
    style_treeview_zebra,
)


TIRE_PALETTE = [
    PALETTE["cyan"],     # Slot 1 (Active / Base)
    PALETTE["accent"],   # Slot 2
    PALETTE["warn"],     # Slot 3
    PALETTE["purple"],   # Slot 4
]

VIEWS = [
    "Fy(SA) — Lateral Force Comparison",
    "Fx(SL) — Longitudinal Force Comparison",
    "Mz(SA) — Aligning Torque Comparison",
    "G-G Friction Circle — Fx vs Fy Envelope",
    "Load Sensitivity — Peak Fy / Fx vs Fz",
    "Cornering Stiffness — CFa(Fz)",
    "KPI Radar — Normalized Benchmark",
]


def _model_desc(m: MF51Model) -> str:
    w = m.get("WIDTH", 0.1524)
    r0 = m.get("UNLOADED_RADIUS", 0.2286)
    if abs(w - 0.180) < 0.015 or abs(r0 - 0.205) < 0.012:
        return f"Hoosier 16x7.5-10 (W={w*1000:.0f}mm)"
    return f"MF 5.1 (R0={r0*1000:.0f}mm)"


@dataclass
class TirSlot:
    index: int
    name: str
    color: str
    model: Optional[MF51Model] = None
    path: str = ""
    enabled: bool = False
    is_active_slot: bool = False

    def compute_kpis(self, fz_n: float, ia_deg: float) -> Dict[str, float]:
        if self.model is None:
            return {}
        m = self.model
        fz_s = max(float(fz_n), 1.0)
        ia_r = float(np.radians(ia_deg))
        sa_v = np.linspace(-14.0, 14.0, 281)
        sa_r = np.radians(sa_v)
        sl_v = np.linspace(-0.25, 0.25, 201)

        fy, _ = m.evaluate_fy(sa_r, fz_s, ia_r)
        fx, _ = m.evaluate_fx(sl_v, fz_s, ia_r)
        mz, _, _, _ = m.evaluate_mz(sa_r, fz_s, ia_r)

        sa_sm = np.radians(np.linspace(-0.5, 0.5, 21))
        fy_sm, _ = m.evaluate_fy(sa_sm, fz_s, ia_r)
        cfa = abs(float(np.polyfit(np.degrees(sa_sm), fy_sm, 1)[0]))

        peak_fy = float(np.max(np.abs(fy)))
        peak_fx = float(np.max(np.abs(fx)))
        peak_mz = float(np.max(np.abs(mz)))
        max_mu = max(peak_fy, peak_fx) / fz_s
        r0 = float(m.get("UNLOADED_RADIUS", 0.2286))
        return {
            "peak_fy": peak_fy,
            "peak_fx": peak_fx,
            "cfa": cfa,
            "peak_mz": peak_mz,
            "max_mu": max_mu,
            "r0": r0,
        }


class TirAnalyzerTab(ttk.Frame):
    """Step 5 — Compare up to 4 .TIR / MF 5.1 tire models with Benchmark KPI table."""

    def __init__(
        self,
        parent,
        *,
        colors=None,
        fonts=None,
        set_status: Optional[Callable[[str], None]] = None,
        get_active_model: Optional[Callable[[], MF51Model]] = None,
        on_next_step: Optional[Callable[[], None]] = None,
        show_step_header: bool = True,
    ):
        super().__init__(parent)
        self.c = colors or PALETTE
        self.fonts = fonts or {}
        self._set_status_cb = set_status or (lambda _msg: None)
        self._get_active_model_cb = get_active_model
        self._on_next_step_cb = on_next_step
        self._show_step_header = show_step_header

        self._draw_job: Optional[str] = None

        self.var_view = tk.StringVar(value=VIEWS[0])
        self.var_fz = tk.DoubleVar(value=900.0)
        self.var_ia = tk.DoubleVar(value=0.0)

        active_m = get_active_model() if get_active_model is not None else MF51Model()
        self.slots: List[TirSlot] = [
            TirSlot(
                index=0,
                name=f"Base: {_model_desc(active_m)}",
                color=TIRE_PALETTE[0],
                model=active_m.copy(),
                path="(Active model from Step 3)",
                enabled=True,
                is_active_slot=True,
            ),
            TirSlot(index=1, name="Slot 2 (empty)", color=TIRE_PALETTE[1], model=None, enabled=False),
            TirSlot(index=2, name="Slot 3 (empty)", color=TIRE_PALETTE[2], model=None, enabled=False),
            TirSlot(index=3, name="Slot 4 (empty)", color=TIRE_PALETTE[3], model=None, enabled=False),
        ]
        self._slot_widgets: List[dict] = []

        self._build_ui()
        self._refresh_slot_cards()
        self.draw()

    def _status(self, msg: str):
        self._set_status_cb(msg)

    # ------------------------------------------------------------------
    # UI CONSTRUCTION
    # ------------------------------------------------------------------
    def _build_ui(self):
        c = self.c
        self.configure(style="TFrame")

        if self._show_step_header:
            StepHeader(
                self,
                "STEP 5 / 5",
                ".TIR Tire Comparator & FSAE Benchmark",
                "Benchmark up to 4 tire compounds or sizes across Delta Fy, Delta Fx, CFa stiffness, G-G friction cone, and KPI Radar.",
                badge_kind="purple",
                next_text="Next: Cornering Stiffness ->" if self._on_next_step_cb else None,
                next_command=self._on_next_step_cb,
                colors=c,
            ).pack(fill="x", pady=(0, 8))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        left_wrap = tk.Frame(body, bg=c["bg"], width=420)
        left_wrap.pack(side="left", fill="y", padx=(0, 10))
        left_wrap.pack_propagate(False)

        self._left_scroll = ScrollableFrame(left_wrap, bg=c["bg"], padx=2, pady=2)
        self._left_scroll.pack(fill="both", expand=True)
        ctrl = self._left_scroll.inner

        card_cond = ModernCard(
            ctrl,
            "1. Test Operating Conditions & View",
            accent_color=c["purple"],
            colors=c,
            pad_x=10,
            pad_y=8,
        )
        card_cond.pack(fill="x", pady=(0, 8))

        tk.Label(card_cond.body, text="Comparison plot:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD, anchor="w").pack(fill="x")
        cb_view = ttk.Combobox(card_cond.body, textvariable=self.var_view, values=VIEWS, state="readonly")
        cb_view.pack(fill="x", pady=(3, 8))
        cb_view.bind("<<ComboboxSelected>>", lambda _e: self._schedule_draw(10))

        fz_hdr = tk.Frame(card_cond.body, bg=c["panel"])
        fz_hdr.pack(fill="x")
        tk.Label(fz_hdr, text="Vertical Load Fz [N]:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD).pack(side="left")
        self.lbl_fz = tk.Label(fz_hdr, text="900 N", bg=c["panel"], fg=c["accent_hover"], font=FONT_MONO_BOLD)
        self.lbl_fz.pack(side="right")

        ttk.Scale(card_cond.body, from_=150.0, to=2000.0, variable=self.var_fz, command=self._on_fz_change).pack(fill="x", pady=(2, 4))

        fz_pills = tk.Frame(card_cond.body, bg=c["panel"])
        fz_pills.pack(fill="x", pady=(0, 8))
        for label, fz_v, ia_v in (
            ("Inside (450 N)", 450.0, 0.0),
            ("Corner (900 N)", 900.0, 2.0),
            ("Braking (1350 N)", 1350.0, 0.0),
        ):
            ttk.Button(
                fz_pills,
                text=label,
                style="Pill.TButton",
                command=lambda f=fz_v, i=ia_v: self._apply_scenario(f, i),
            ).pack(side="left", padx=(0, 4))

        ia_hdr = tk.Frame(card_cond.body, bg=c["panel"])
        ia_hdr.pack(fill="x")
        tk.Label(ia_hdr, text="Camber IA [deg]:", bg=c["panel"], fg=c["text_secondary"], font=FONT_SMALL_BOLD).pack(side="left")
        self.lbl_ia = tk.Label(ia_hdr, text="0.0°", bg=c["panel"], fg=c["cyan"], font=FONT_MONO_BOLD)
        self.lbl_ia.pack(side="right")

        ttk.Scale(card_cond.body, from_=-5.0, to=5.0, variable=self.var_ia, command=self._on_ia_change).pack(fill="x", pady=(2, 4))

        card_slots = ModernCard(
            ctrl,
            "2. Tire Slots for Comparison (1–4)",
            badge="4 SLOTS",
            badge_color="cyan",
            accent_color=c["cyan"],
            colors=c,
            pad_x=10,
            pad_y=8,
        )
        card_slots.pack(fill="x", pady=(0, 8))

        for slot in self.slots:
            row_frame = tk.Frame(
                card_slots.body,
                bg=c["panel_alt"],
                highlightthickness=1,
                highlightbackground=c["border"],
            )
            row_frame.pack(fill="x", pady=(0, 6))

            stripe = tk.Frame(row_frame, bg=slot.color, width=4)
            stripe.pack(side="left", fill="y")

            inner = tk.Frame(row_frame, bg=c["panel_alt"], padx=8, pady=6)
            inner.pack(side="left", fill="both", expand=True)

            top_line = tk.Frame(inner, bg=c["panel_alt"])
            top_line.pack(fill="x")

            var_en = tk.BooleanVar(value=slot.enabled)
            ttk.Checkbutton(
                top_line,
                text=f"Slot {slot.index + 1}",
                variable=var_en,
                command=lambda idx=slot.index, v=var_en: self._toggle_slot(idx, v.get()),
            ).pack(side="left")

            var_name = tk.StringVar(value=slot.name)
            tk.Label(
                top_line,
                textvariable=var_name,
                bg=c["panel_alt"],
                fg=c["text"],
                font=FONT_BOLD,
                anchor="w",
            ).pack(side="left", fill="x", expand=True, padx=(6, 0))

            lbl_path = tk.Label(inner, text="", bg=c["panel_alt"], fg=c["text_muted"], font=FONT_SMALL, anchor="w")
            lbl_path.pack(fill="x", pady=(2, 4))

            btn_line = tk.Frame(inner, bg=c["panel_alt"])
            btn_line.pack(fill="x")

            ttk.Button(
                btn_line,
                text="Browse .TIR",
                style="Pill.TButton",
                command=lambda idx=slot.index: self._load_tir_for_slot(idx),
            ).pack(side="left", padx=(0, 4))

            if slot.index == 0:
                ttk.Button(btn_line, text="Sync Step 3", style="Pill.TButton", command=self.sync_active_model).pack(side="left", padx=(0, 4))
            else:
                ttk.Button(btn_line, text="R25B Preset", style="Pill.TButton", command=lambda idx=slot.index: self._load_preset_for_slot(idx)).pack(side="left", padx=(0, 4))
                ttk.Button(btn_line, text="Clear", style="Pill.TButton", command=lambda idx=slot.index: self._clear_slot(idx)).pack(side="left")

            self._slot_widgets.append({"var_en": var_en, "var_name": var_name, "lbl_path": lbl_path})

        right = ttk.Frame(body)
        right.pack(side="right", fill="both", expand=True)

        plot_card = ModernCard(right, "Comparative Tire Overlay", colors=c, pad_x=8, pad_y=6)
        plot_card.pack(fill="both", expand=True, pady=(0, 8))

        self.fig = Figure(figsize=(7.6, 4.1), dpi=100, facecolor=c["panel"])
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_card.body)
        self.canvas.get_tk_widget().configure(bg=c["panel"], highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        kpi_card = ModernCard(right, "Benchmark KPI Table (Delta vs Slot 1)", badge="LIVE", badge_color="cyan", colors=c, pad_x=8, pad_y=6)
        kpi_card.pack(fill="x")

        cols = ("slot", "name", "fy", "dfy", "fx", "dfx", "cfa", "mz", "mu", "r0")
        self.kpi_tree = ttk.Treeview(kpi_card.body, columns=cols, show="headings", height=4)
        for key, title, width, anchor in [
            ("slot", "Slot", 52, "center"),
            ("name", "Tire / Source", 210, "w"),
            ("fy", "Peak Fy [N]", 88, "e"),
            ("dfy", "Delta Fy", 78, "e"),
            ("fx", "Peak Fx [N]", 88, "e"),
            ("dfx", "Delta Fx", 78, "e"),
            ("cfa", "CFa [N/deg]", 88, "e"),
            ("mz", "Peak |Mz|", 82, "e"),
            ("mu", "Max mu", 70, "e"),
            ("r0", "R0 [m]", 70, "e"),
        ]:
            self.kpi_tree.heading(key, text=title)
            self.kpi_tree.column(key, width=width, anchor=anchor, stretch=(key == "name"))

        self.kpi_tree.pack(fill="x", expand=False)
        style_treeview_zebra(self.kpi_tree, c)

    # ------------------------------------------------------------------
    # SLOT & CONDITION MANAGEMENT
    # ------------------------------------------------------------------
    def sync_active_model(self, redraw: bool = True):
        if self._get_active_model_cb is None:
            return
        m = self._get_active_model_cb()
        s0 = self.slots[0]
        s0.model = m.copy()
        s0.name = f"Active: {_model_desc(m)}"
        s0.path = f"Active model (LMUX={m.get('LMUX', 1.0):.2f}, LMUY={m.get('LMUY', 1.0):.2f})"
        s0.enabled = True
        self._refresh_slot_cards()
        if redraw:
            self._schedule_draw(10)

    def refresh_active_model(self, redraw: bool = True):
        self.sync_active_model(redraw=redraw)

    def _refresh_slot_cards(self):
        for slot, w in zip(self.slots, self._slot_widgets):
            w["var_en"].set(slot.enabled)
            w["var_name"].set(slot.name)
            w["lbl_path"].config(
                text=slot.path if slot.path else ("Empty — load a .TIR via Browse or Header Import" if slot.model is None else slot.name)
            )

    def _on_fz_change(self, val):
        self.lbl_fz.config(text=f"{float(val):.0f} N")
        self._update_kpi_table()
        self._schedule_draw(30)

    def _on_ia_change(self, val):
        self.lbl_ia.config(text=f"{float(val):+.1f}°")
        self._update_kpi_table()
        self._schedule_draw(30)

    def _apply_scenario(self, fz_val: float, ia_val: float):
        self.var_fz.set(fz_val)
        self.var_ia.set(ia_val)
        self.lbl_fz.config(text=f"{fz_val:.0f} N")
        self.lbl_ia.config(text=f"{ia_val:+.1f}°")
        self._update_kpi_table()
        self._schedule_draw(10)

    def _toggle_slot(self, idx: int, enabled: bool):
        slot = self.slots[idx]
        if enabled and slot.model is None:
            self._load_tir_for_slot(idx)
            return
        slot.enabled = enabled
        self._schedule_draw(10)

    def load_tir_path_to_slot(self, idx: int, path: str) -> str:
        """Load a .tir file into the given slot index (0-3) and redraw."""
        model, meta = load_tir_to_model(path)
        slot = self.slots[idx]
        slot.model = model
        slot.path = os.path.basename(path)
        tire_label = meta.get("TIRE_NAME") or os.path.splitext(os.path.basename(path))[0]
        slot.name = tire_label
        slot.enabled = True
        self._refresh_slot_cards()
        self._status(f"Slot {idx + 1}: loaded {slot.name}")
        self._schedule_draw(10)
        return slot.name

    def _load_tir_for_slot(self, idx: int):
        path = filedialog.askopenfilename(
            title=f"Select .TIR File for Slot {idx + 1}",
            filetypes=[("Tire Property Files (*.tir)", "*.tir"), ("All Files", "*.*")],
        )
        if not path:
            self._slot_widgets[idx]["var_en"].set(self.slots[idx].enabled)
            return
        try:
            self.load_tir_path_to_slot(idx, path)
        except Exception as exc:
            messagebox.showerror(".TIR Read Error", f"Failed to read .tir file:\n{exc}")
            self._slot_widgets[idx]["var_en"].set(self.slots[idx].enabled)

    def _load_preset_for_slot(self, idx: int):
        slot = self.slots[idx]
        m = MF51Model()
        if idx == 1:
            name = "Hoosier 18x6 R25B (Asphalt x0.65)"
            m.set("LMUX", 0.65)
            m.set("LMUY", 0.65)
        elif idx == 2:
            name = "Hoosier 16x7.5 R20 (Asphalt x0.68)"
            m.set("LMUX", 0.68)
            m.set("LMUY", 0.68)
            m.set("PKY1", m.get("PKY1") * 1.12)
            m.set("WIDTH", 0.180)
            m.set("UNLOADED_RADIUS", 0.205)
        else:
            name = f"R25B Reference Variant {idx + 1}"
        slot.model = m
        slot.name = name
        slot.path = "(Built-in Preset)"
        slot.enabled = True
        self._refresh_slot_cards()
        self._schedule_draw(10)

    def _clear_slot(self, idx: int):
        if idx == 0:
            return
        slot = self.slots[idx]
        slot.model = None
        slot.path = ""
        slot.name = f"Slot {idx + 1} (empty)"
        slot.enabled = False
        self._refresh_slot_cards()
        self._schedule_draw(10)

    # ------------------------------------------------------------------
    # KPI TABLE & EXPORT HELPERS
    # ------------------------------------------------------------------
    def _update_kpi_table(self):
        for item in self.kpi_tree.get_children():
            self.kpi_tree.delete(item)

        fz_n = float(self.var_fz.get())
        ia_deg = float(self.var_ia.get())

        active_slots = [s for s in self.slots if s.enabled and s.model is not None]
        if not active_slots:
            return

        base_kpi = active_slots[0].compute_kpis(fz_n, ia_deg)

        for row_idx, slot in enumerate(active_slots):
            k = slot.compute_kpis(fz_n, ia_deg)
            if row_idx == 0:
                dfy_str = "Base"
                dfx_str = "Base"
            else:
                dfy_pct = 100.0 * (k["peak_fy"] - base_kpi["peak_fy"]) / max(base_kpi["peak_fy"], 1.0)
                dfx_pct = 100.0 * (k["peak_fx"] - base_kpi["peak_fx"]) / max(base_kpi["peak_fx"], 1.0)
                dfy_str = f"{dfy_pct:+.1f}%"
                dfx_str = f"{dfx_pct:+.1f}%"

            tag = "even" if row_idx % 2 == 0 else "odd"
            self.kpi_tree.insert(
                "",
                "end",
                values=(
                    f"Slot {slot.index + 1}",
                    slot.name,
                    f"{k['peak_fy']:.0f}",
                    dfy_str,
                    f"{k['peak_fx']:.0f}",
                    dfx_str,
                    f"{k['cfa']:.0f}",
                    f"{k['peak_mz']:.1f}",
                    f"{k['max_mu']:.3f}",
                    f"{k['r0']:.4f}",
                ),
                tags=(tag,),
            )

    def get_kpi_dataframe(self) -> pd.DataFrame:
        active_slots = [s for s in self.slots if s.enabled and s.model is not None]
        if not active_slots:
            return pd.DataFrame()
        fz_n = float(self.var_fz.get())
        ia_deg = float(self.var_ia.get())
        base_kpi = active_slots[0].compute_kpis(fz_n, ia_deg)
        rows = []
        for idx, slot in enumerate(active_slots):
            k = slot.compute_kpis(fz_n, ia_deg)
            dfy = 0.0 if idx == 0 else 100.0 * (k["peak_fy"] - base_kpi["peak_fy"]) / max(base_kpi["peak_fy"], 1.0)
            dfx = 0.0 if idx == 0 else 100.0 * (k["peak_fx"] - base_kpi["peak_fx"]) / max(base_kpi["peak_fx"], 1.0)
            rows.append(
                {
                    "Slot": f"Slot {slot.index + 1}",
                    "Tire": slot.name,
                    "Source": slot.path,
                    "Fz_N": round(fz_n, 1),
                    "IA_deg": round(ia_deg, 2),
                    "Peak_Fy_N": round(k["peak_fy"], 2),
                    "Delta_Fy_pct": round(dfy, 2),
                    "Peak_Fx_N": round(k["peak_fx"], 2),
                    "Delta_Fx_pct": round(dfx, 2),
                    "CFa_N_per_deg": round(k["cfa"], 2),
                    "Peak_Mz_Nm": round(k["peak_mz"], 2),
                    "Max_mu": round(k["max_mu"], 4),
                    "R0_m": round(k["r0"], 4),
                }
            )
        return pd.DataFrame(rows)

    def _export_kpi_csv(self, path: Optional[str] = None, *, sep: str = ";"):
        df = self.get_kpi_dataframe()
        if df.empty:
            messagebox.showwarning("No Active Models", "Enable at least one model in comparison slots.")
            return
        if not path:
            path = filedialog.asksaveasfilename(
                title="Export KPI Benchmark Report (.CSV)",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("All Files", "*.*")],
                initialfile="PUTM_TIR_Benchmark_KPI.csv",
            )
        if not path:
            return
        df.to_csv(path, sep=sep, index=False, encoding="utf-8")
        self._status(f"Saved KPI benchmark report: {path}")

    def save_plot(self, path: Optional[str] = None, dpi: int = 220):
        if not path:
            path = filedialog.asksaveasfilename(
                title="Save Comparison Plot",
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
                initialfile="PUTM_TIR_Comparator.png",
            )
        if not path:
            return
        self.fig.savefig(path, dpi=dpi, facecolor=self.fig.get_facecolor())
        self._status(f"Saved comparison plot: {path}")

    # ------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------
    def _schedule_draw(self, delay_ms: int = 25):
        if self._draw_job is not None:
            try:
                self.after_cancel(self._draw_job)
            except tk.TclError:
                pass
            self._draw_job = None
        self._draw_job = self.after(delay_ms, self.draw)

    def draw(self):
        self._draw_job = None
        self._update_kpi_table()

        c = self.c
        self.fig.clear()
        self.fig.patch.set_facecolor(c["panel"])

        active_slots = [s for s in self.slots if s.enabled and s.model is not None]
        if not active_slots:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Enable or load a .TIR model on the left", ha="center", va="center", color=c["text_muted"])
            style_axes_2d(ax, zero_lines=False, c=c)
            self.canvas.draw_idle()
            return

        view = self.var_view.get()
        fz_n = max(float(self.var_fz.get()), 1.0)
        ia_deg = float(self.var_ia.get())
        ia_r = float(np.radians(ia_deg))

        if view.startswith("KPI Radar"):
            self._draw_radar(active_slots, fz_n, ia_deg)
            self.fig.tight_layout(pad=1.6)
            self.canvas.draw_idle()
            return

        ax = self.fig.add_subplot(111)

        if view.startswith("Fy("):
            sa = np.linspace(-14.0, 14.0, 260)
            sa_r = np.radians(sa)
            for slot in active_slots:
                fy, _ = slot.model.evaluate_fy(sa_r, fz_n, ia_r)
                ax.plot(sa, fy, color=slot.color, linewidth=2.3, label=slot.name)
            style_axes_2d(ax, title=f"Lateral Force Fy(SA) @ Fz={fz_n:.0f} N, IA={ia_deg:+.1f}°", xlabel="SA [deg]", ylabel="Fy [N]", c=c)

        elif view.startswith("Fx("):
            sl = np.linspace(-0.25, 0.25, 240)
            for slot in active_slots:
                fx, _ = slot.model.evaluate_fx(sl, fz_n, ia_r)
                ax.plot(sl, fx, color=slot.color, linewidth=2.3, label=slot.name)
            style_axes_2d(ax, title=f"Longitudinal Force Fx(SL) @ Fz={fz_n:.0f} N, IA={ia_deg:+.1f}°", xlabel="Slip Ratio SL [-]", ylabel="Fx [N]", c=c)

        elif view.startswith("Mz("):
            sa = np.linspace(-14.0, 14.0, 260)
            sa_r = np.radians(sa)
            for slot in active_slots:
                mz, _, _, _ = slot.model.evaluate_mz(sa_r, fz_n, ia_r)
                ax.plot(sa, mz, color=slot.color, linewidth=2.3, label=slot.name)
            style_axes_2d(ax, title=f"Aligning Torque Mz(SA) @ Fz={fz_n:.0f} N, IA={ia_deg:+.1f}°", xlabel="SA [deg]", ylabel="Mz [Nm]", c=c)

        elif view.startswith("G-G"):
            theta = np.linspace(0.0, 2.0 * np.pi, 200)
            sa_r = np.radians(13.0 * np.sin(theta))
            sl_arr = 0.18 * np.cos(theta)
            for slot in active_slots:
                fx = slot.model.evaluate_combined_fx(sl_arr, sa_r, fz_n, ia_r)
                fy = slot.model.evaluate_combined_fy(sa_r, sl_arr, fz_n, ia_r)
                ax.plot(fy, fx, color=slot.color, linewidth=2.3, label=slot.name)
            ax.set_aspect("equal", adjustable="datalim")
            style_axes_2d(ax, title=f"G-G Envelope (Fx vs Fy) @ Fz={fz_n:.0f} N, IA={ia_deg:+.1f}°", xlabel="Fy [N]", ylabel="Fx [N]", c=c)

        elif view.startswith("Load"):
            fz_grid = np.linspace(200.0, 1800.0, 22)
            sa_r = np.radians(np.linspace(-14.0, 14.0, 121))
            for slot in active_slots:
                peaks = [float(np.max(np.abs(slot.model.evaluate_fy(sa_r, float(f), ia_r)[0]))) for f in fz_grid]
                ax.plot(fz_grid, peaks, marker="o", markersize=4, color=slot.color, linewidth=2.2, label=slot.name)
            style_axes_2d(ax, title=f"Load Sensitivity: Peak Fy(Fz) @ IA={ia_deg:+.1f}°", xlabel="Vertical Load Fz [N]", ylabel="Peak |Fy| [N]", c=c)

        elif view.startswith("Cornering"):
            fz_grid = np.linspace(200.0, 1800.0, 22)
            sa_sm = np.radians(np.linspace(-0.5, 0.5, 21))
            for slot in active_slots:
                cfas = []
                for f in fz_grid:
                    fy_s, _ = slot.model.evaluate_fy(sa_sm, float(f), ia_r)
                    cfas.append(abs(float(np.polyfit(np.degrees(sa_sm), fy_s, 1)[0])))
                ax.plot(fz_grid, cfas, marker="s", markersize=4, color=slot.color, linewidth=2.2, label=slot.name)
            style_axes_2d(ax, title=f"Cornering Stiffness CFa(Fz) @ IA={ia_deg:+.1f}°", xlabel="Vertical Load Fz [N]", ylabel="CFa [N/deg]", c=c)

        style_legend(ax, c=c)
        self.fig.tight_layout(pad=1.6)
        self.canvas.draw_idle()

    def _draw_radar(self, active_slots: List[TirSlot], fz_n: float, ia_deg: float):
        c = self.c
        metrics = ["Peak Fy", "Peak Fx", "CFa Stiffness", "Peak |Mz|", "Max mu"]
        keys = ["peak_fy", "peak_fx", "cfa", "peak_mz", "max_mu"]
        kpis = [s.compute_kpis(fz_n, ia_deg) for s in active_slots]

        max_vals = {k: max(max(row[k] for row in kpis), 1e-6) for k in keys}
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        ax = self.fig.add_subplot(111, polar=True)
        ax.set_facecolor(c["panel_alt"])
        ax.spines["polar"].set_color(c["border"])
        ax.grid(color=c["border"], linewidth=0.7)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, color=c["text"], fontsize=9, fontweight="bold")
        ax.set_ylim(0.0, 1.05)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], color=c["text_muted"], fontsize=7.5)

        for slot, kpi in zip(active_slots, kpis):
            vals = [kpi[k] / max_vals[k] for k in keys]
            vals += vals[:1]
            ax.plot(angles, vals, color=slot.color, linewidth=2.2, label=slot.name)
            ax.fill(angles, vals, color=slot.color, alpha=0.14)

        ax.set_title(
            f"Normalized KPI Radar (@ Fz={fz_n:.0f} N, IA={ia_deg:+.1f}°)",
            color=c["text"],
            fontsize=10.5,
            fontweight="bold",
            pad=14,
        )
        style_legend(ax, loc="upper right", c=c)


class TirAnalyzerDialog(tk.Toplevel):
    """Standalone .TIR Tire Comparator window in Carbon Motorsport Telemetry style."""

    def __init__(self, parent, *, active_model: Optional[MF51Model] = None):
        super().__init__(parent)
        self.c = PALETTE
        self.title("PUTM Tire Lab — .TIR Tire Comparator")
        self.geometry("1260x760")
        self.minsize(980, 620)
        self.configure(bg=self.c["bg"])
        apply_ttk_theme(self, self.c)

        hdr = tk.Frame(self, bg=self.c["bg_header"], height=46, padx=14)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        self._logo = load_brand_logo(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "logo2.png"),
            max_size=(28, 28),
        )
        if self._logo is not None:
            tk.Label(hdr, image=self._logo, bg=self.c["bg_header"]).pack(side="left", padx=(0, 8))

        tk.Label(
            hdr,
            text=".TIR TIRE COMPARATOR (PAC2002 / MF 5.1)",
            bg=self.c["bg_header"],
            fg=self.c["text"],
            font=("Segoe UI Semibold", 11, "bold"),
        ).pack(side="left")

        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)

        self.tab = TirAnalyzerTab(
            body,
            colors=self.c,
            get_active_model=(lambda: active_model) if active_model is not None else None,
            show_step_header=False,
        )
        self.tab.pack(fill="both", expand=True)
