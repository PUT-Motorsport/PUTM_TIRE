"""
PUTM Tire - Tire Model Optimizer (Cornering + Drive/Brake & Friction Circle)
==========================================================================
Uruchomienie:
    python3 -m pip install -r requirements.txt
    python3 app.py
"""

import os
import sys
import json
import traceback
from dataclasses import asdict

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Importy z nowego backendu
from tire_model import (
    RunConfig, TireDataManager, 
    pacejka_fy, pacejka_fx, pacejka_mz, pacejka_fy_load, pacejka_comb_Gyk
)
from fitting import build_combined_slip_targets, fit_combined_slip, fit_load_sensitive, fit_pure_magic

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(APP_DIR, "img", "logo.png")

# --------------------------------------------------------------------------
# Paleta kolorów - motyw "PUTM Racing"
# --------------------------------------------------------------------------
COLORS = {
    "bg":          "#17181a",
    "bg_header":   "#0f0f10",
    "panel":       "#212226",
    "panel_alt":   "#2a2c30",
    "input":       "#2d2f34",
    "border":      "#3a3c42",
    "text":        "#eaeaea",
    "text_muted":  "#9a9ca3",
    "accent":      "#e0521c",
    "accent_hover":"#ff6a2e",
    "accent_dark": "#a83e14",
    "ok":          "#3ddc84",
    "warn":        "#f2c94c",
}

FONT_BASE = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI", 9, "bold")
FONT_HEAD = ("Segoe UI", 10, "bold")
FONT_MONO = ("Consolas", 9)


class PUTMTireApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PUTM Tire — Tire Model Optimizer")
        self.geometry("1340x840")
        self.minsize(1150, 740)

        self.runs = []           
        self.df_cornering = None 
        self.df_drive = None     
        self.fit_result = None   
        self.fit_results = {}

        self._build_style()
        self._build_header()
        self._build_body()
        self._build_statusbar()

    def _build_style(self):
        self.configure(bg=COLORS["bg"])

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=COLORS["bg"], foreground=COLORS["text"],
                        fieldbackground=COLORS["input"], font=FONT_BASE,
                        bordercolor=COLORS["border"], darkcolor=COLORS["panel"],
                        lightcolor=COLORS["panel"], troughcolor=COLORS["bg"])

        style.configure("TFrame", background=COLORS["bg"])
        style.configure("Card.TFrame", background=COLORS["panel"])

        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"])
        style.configure("Card.TLabel", background=COLORS["panel"], foreground=COLORS["text"])
        style.configure("Muted.TLabel", background=COLORS["panel"], foreground=COLORS["text_muted"])
        style.configure("Section.TLabel", background=COLORS["panel"], foreground=COLORS["accent"], font=FONT_HEAD)

        style.configure("TLabelframe", background=COLORS["panel"], foreground=COLORS["text"],
                        bordercolor=COLORS["border"], relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=COLORS["panel"], foreground=COLORS["accent"], font=FONT_HEAD)

        style.configure("TButton", background=COLORS["panel_alt"], foreground=COLORS["text"],
                        borderwidth=0, focusthickness=0, padding=(10, 7), font=FONT_BASE)
        style.map("TButton",
                  background=[("active", COLORS["border"]), ("disabled", COLORS["panel"])],
                  foreground=[("disabled", COLORS["text_muted"])])

        style.configure("Accent.TButton", background=COLORS["accent"], foreground="white",
                        borderwidth=0, focusthickness=0, padding=(12, 8), font=FONT_BOLD)
        style.map("Accent.TButton",
                  background=[("active", COLORS["accent_hover"]), ("disabled", COLORS["panel_alt"])])

        style.configure("Danger.TButton", background="#3a2020", foreground="#ff9a8a",
                        borderwidth=0, padding=(10, 7), font=FONT_BASE)
        style.map("Danger.TButton", background=[("active", "#552626")])

        style.configure("TEntry", fieldbackground=COLORS["input"], foreground=COLORS["text"],
                        bordercolor=COLORS["border"], insertcolor=COLORS["text"], padding=5)
        style.map("TEntry", bordercolor=[("focus", COLORS["accent"])])

        style.configure("TCombobox", fieldbackground=COLORS["input"], foreground=COLORS["text"],
                        background=COLORS["input"], arrowcolor=COLORS["text"], padding=5)
        style.map("TCombobox",
                  fieldbackground=[("readonly", COLORS["input"])],
                  foreground=[("readonly", COLORS["text"])])
        self.option_add("*TCombobox*Listbox.background", COLORS["input"])
        self.option_add("*TCombobox*Listbox.foreground", COLORS["text"])
        self.option_add("*TCombobox*Listbox.selectBackground", COLORS["accent"])

        style.configure("TNotebook", background=COLORS["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=COLORS["panel_alt"],
                        foreground=COLORS["text_muted"], padding=(16, 9), font=FONT_BOLD, borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", COLORS["accent"])],
                  foreground=[("selected", "white")],
                  expand=[("selected", [0, 0, 0, 0])])

        style.configure("Treeview", background=COLORS["panel"], fieldbackground=COLORS["panel"],
                        foreground=COLORS["text"], bordercolor=COLORS["border"],
                        rowheight=24, borderwidth=0, font=FONT_BASE)
        style.configure("Treeview.Heading", background=COLORS["panel_alt"],
                        foreground=COLORS["accent"], font=FONT_BOLD, relief="flat")
        style.map("Treeview.Heading", background=[("active", COLORS["border"])])
        style.map("Treeview",
                  background=[("selected", COLORS["accent"])],
                  foreground=[("selected", "white")])

        style.configure("Vertical.TScrollbar", background=COLORS["panel_alt"],
                        troughcolor=COLORS["bg"], bordercolor=COLORS["bg"], arrowcolor=COLORS["text_muted"])

    def _build_header(self):
        header = tk.Frame(self, bg=COLORS["bg_header"], height=78)
        header.pack(side="top", fill="x")
        header.pack_propagate(False)

        logo_holder = tk.Frame(header, bg=COLORS["bg_header"])
        logo_holder.pack(side="left", padx=18, pady=10)

        self._logo_img = None
        if PIL_AVAILABLE and os.path.exists(LOGO_PATH):
            try:
                img = Image.open(LOGO_PATH)
                img.thumbnail((240, 58))
                self._logo_img = ImageTk.PhotoImage(img)
                tk.Label(logo_holder, image=self._logo_img, bg=COLORS["bg_header"]).pack()
            except Exception:
                tk.Label(logo_holder, text="PUTM TIRE", fg="white", bg=COLORS["bg_header"], font=("Segoe UI", 19, "bold")).pack()
        else:
            tk.Label(logo_holder, text="PUTM TIRE", fg="white", bg=COLORS["bg_header"], font=("Segoe UI", 19, "bold")).pack()

        text_holder = tk.Frame(header, bg=COLORS["bg_header"])
        text_holder.pack(side="left", padx=(4, 6))
        tk.Label(text_holder, text="Tire Model Optimizer", fg="white", bg=COLORS["bg_header"], font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(14, 0))
        tk.Label(text_holder, text="Cornering  ·  Drive/Brake  ·  Tire Force Envelope", fg=COLORS["text_muted"], bg=COLORS["bg_header"], font=("Segoe UI", 9)).pack(anchor="w")

        tk.Frame(self, bg=COLORS["accent"], height=3).pack(side="top", fill="x")

    def _build_body(self):
        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(side="top", fill="both", expand=True)

        left = tk.Frame(body, width=490, bg=COLORS["bg"])
        left.pack(side="left", fill="y", padx=(10, 6), pady=10)
        left.pack_propagate(False)

        right = tk.Frame(body, bg=COLORS["bg"])
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)

        self._build_notebook_runs(left)
        self._build_fit_panel(left)
        self._build_plot_panel(right)

    def _build_notebook_runs(self, parent):
        self.nb_runs = ttk.Notebook(parent)
        self.nb_runs.pack(side="top", fill="x", pady=(0, 10))

        tab_corn = ttk.Frame(self.nb_runs, style="Card.TFrame")
        self.nb_runs.add(tab_corn, text="1A · CORNERING")
        self._build_cornering_inputs(tab_corn)

        tab_db = ttk.Frame(self.nb_runs, style="Card.TFrame")
        self.nb_runs.add(tab_db, text="1B · DRIVE / BRAKE")
        self._build_drivebrake_inputs(tab_db)

        ctrl_frame = ttk.Frame(parent, style="Card.TFrame")
        ctrl_frame.pack(side="top", fill="x", pady=(0, 10))

        header_row = ttk.Frame(ctrl_frame, style="Card.TFrame")
        header_row.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Label(header_row, text="Załadowane run'y", style="Section.TLabel").pack(side="left")

        btn_row = ttk.Frame(ctrl_frame, style="Card.TFrame")
        btn_row.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Button(btn_row, text="🗑  Usuń zaznaczone", command=self.remove_selected_runs, style="Danger.TButton").pack(side="left")
        ttk.Button(btn_row, text="⚙  Przetwórz wszystkie run'y", command=self.process_runs, style="Accent.TButton").pack(side="right")

        self.run_list = ttk.Treeview(
            ctrl_frame, columns=("type", "file", "fz", "param", "n"), show="headings", height=4, selectmode="extended"
        )
        for c, txt, w in [
            ("type", "Typ", 75), ("file", "Plik", 100),
            ("fz", "Fz±tol", 70), ("param", "Warunki", 175), ("n", "Pkt", 40)
        ]:
            self.run_list.heading(c, text=txt)
            self.run_list.column(c, width=w, anchor="w")
        self.run_list.pack(fill="x", padx=10, pady=(2, 10))

    def _form_row(self, parent, r, label, var):
        ttk.Label(parent, text=label, style="Card.TLabel").grid(row=r, column=0, sticky="w", pady=5, padx=(0, 8))
        ttk.Entry(parent, textvariable=var, width=10).grid(row=r, column=1, sticky="w")

    def _build_cornering_inputs(self, parent):
        form = ttk.Frame(parent, style="Card.TFrame")
        form.pack(fill="x", padx=14, pady=(14, 6))

        self.c_fz = tk.StringVar(value="1000")
        self.c_tol_fz = tk.StringVar(value="100")
        self.c_ia = tk.StringVar(value="0")
        self.c_tol_ia = tk.StringVar(value="0.25")

        self._form_row(form, 0, "Środek Fz [N]:", self.c_fz)
        self._form_row(form, 1, "Tolerancja Fz (± N):", self.c_tol_fz)
        self._form_row(form, 2, "Środek Camberu IA [deg]:", self.c_ia)
        self._form_row(form, 3, "Tolerancja IA (± deg):", self.c_tol_ia)

        ttk.Button(
            parent, text="＋  Dodaj pliki .mat (Cornering)",
            command=lambda: self.add_runs("Cornering"), style="Accent.TButton"
        ).pack(fill="x", padx=14, pady=(6, 14))

    def _build_drivebrake_inputs(self, parent):
        form = ttk.Frame(parent, style="Card.TFrame")
        form.pack(fill="x", padx=14, pady=(14, 6))

        self.d_fz = tk.StringVar(value="1000")
        self.d_tol_fz = tk.StringVar(value="100")
        self.d_sl_max = tk.StringVar(value="0.30")
        self.d_sa = tk.StringVar(value="0")
        self.d_tol_sa = tk.StringVar(value="6.5")
        self.d_ia = tk.StringVar(value="0")
        self.d_tol_ia = tk.StringVar(value="0.25")

        self._form_row(form, 0, "Środek Fz [N]:", self.d_fz)
        self._form_row(form, 1, "Tolerancja Fz (± N):", self.d_tol_fz)
        self._form_row(form, 2, "Maks. |Slip Ratio SL| [-]:", self.d_sl_max)
        self._form_row(form, 3, "Środek SA [deg]:", self.d_sa)
        self._form_row(form, 4, "Tolerancja SA (± deg):", self.d_tol_sa)
        self._form_row(form, 5, "Środek IA [deg]:", self.d_ia)
        self._form_row(form, 6, "Tolerancja IA (± deg):", self.d_tol_ia)

        ttk.Button(
            parent, text="＋  Dodaj pliki .mat (Drive/Brake)",
            command=lambda: self.add_runs("Drive/Brake"), style="Accent.TButton"
        ).pack(fill="x", padx=14, pady=(6, 14))

    def _build_fit_panel(self, parent):
        box = ttk.LabelFrame(parent, text="  2)  DOPASOWANIE I EKSPORT  ")
        box.pack(side="top", fill="both", expand=True, pady=(0, 0))

        self.fit_btn_text = tk.StringVar(value="📈  Dopasuj model Pacejki Fy0 (Pure Slip)")
        btn_fit = ttk.Button(box, textvariable=self.fit_btn_text, command=self.fit_model, style="Accent.TButton")
        btn_fit.pack(fill="x", padx=12, pady=(12, 8))

        self.txt_results = tk.Text(
            box, height=8, font=FONT_MONO, state="disabled",
            bg=COLORS["input"], fg=COLORS["text"], insertbackground=COLORS["text"],
            relief="flat", padx=10, pady=8, borderwidth=0, highlightthickness=1,
            highlightbackground=COLORS["border"], highlightcolor=COLORS["accent"]
        )
        self.txt_results.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        exp = ttk.Frame(box, style="Card.TFrame")
        exp.pack(fill="x", padx=12, pady=(0, 12))
        ttk.Button(exp, text="⬇  Eksport danych (CSV)", command=self.export_data).pack(side="left")
        ttk.Button(exp, text="⬇  Eksport współczynników", command=self.export_coeffs).pack(side="left", padx=8)

    def _build_plot_panel(self, parent):
        box = ttk.LabelFrame(parent, text="  PODGLĄD WYKRESÓW  ")
        box.pack(fill="both", expand=True)

        mode_frame = ttk.Frame(box, style="Card.TFrame")
        mode_frame.pack(fill="x", padx=10, pady=(10, 4))

        ttk.Label(mode_frame, text="Wybierz wykres:", style="Card.TLabel").pack(side="left", padx=(0, 8))
        self.plot_mode = tk.StringVar(value="Fy (Siła boczna)")

        modes = [
            "Fy (Siła boczna)", 
            "Fy (Load Sensitivity)",
            "Fx (Siła wzdłużna)", 
            "Mz (Moment stabilizujący)",
            "Tire Force Envelope",
            "Combined Slip (Gyk)"
        ]
        self.cb_mode = ttk.Combobox(mode_frame, textvariable=self.plot_mode, values=modes, state="readonly", width=25)
        self.cb_mode.pack(side="left", padx=4)
        self.cb_mode.bind("<<ComboboxSelected>>", lambda e: self.on_mode_change())

        ttk.Button(mode_frame, text="⟳  Odśwież", command=self.update_plot_view).pack(side="left", padx=6)
        ttk.Button(mode_frame, text="✕  Wyczyść", command=self._reset_plot).pack(side="right")

        self.fig = Figure(figsize=(6, 5), dpi=100, facecolor=COLORS["panel"])
        self.ax = self.fig.add_subplot(111, facecolor=COLORS["panel"])
        self._cbar = None
        self._reset_plot()

        canvas_holder = ttk.Frame(box, style="Card.TFrame")
        canvas_holder.pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_holder)
        self.canvas.get_tk_widget().configure(bg=COLORS["panel"], highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_statusbar(self):
        self.status = tk.StringVar(value="Gotowy. Załaduj pliki Cornering lub Drive/Brake.")
        bar = tk.Frame(self, bg=COLORS["bg_header"], height=28)
        bar.pack(side="bottom", fill="x")
        bar.pack_propagate(False)
        tk.Frame(bar, bg=COLORS["accent"], width=4).pack(side="left", fill="y")
        tk.Label(bar, textvariable=self.status, anchor="w", bg=COLORS["bg_header"], fg=COLORS["text_muted"], font=("Segoe UI", 9), padx=10).pack(side="left", fill="both", expand=True)

    def _set_status(self, text):
        self.status.set(text)
        self.update_idletasks()

    def _style_axes(self):
        self.ax.set_facecolor(COLORS["panel"])
        self.fig.patch.set_facecolor(COLORS["panel"])
        for spine in self.ax.spines.values():
            spine.set_color(COLORS["border"])
        self.ax.tick_params(colors=COLORS["text_muted"], labelsize=8)
        self.ax.title.set_color(COLORS["text"])
        self.ax.xaxis.label.set_color(COLORS["text_muted"])
        self.ax.yaxis.label.set_color(COLORS["text_muted"])
        self.ax.grid(True, color=COLORS["border"], alpha=0.5, linewidth=0.6)

    def _style_colorbar(self, cbar, label):
        cbar.set_label(label, color=COLORS["text_muted"])
        cbar.ax.yaxis.set_tick_params(color=COLORS["text_muted"], labelsize=8)
        cbar.outline.set_edgecolor(COLORS["border"])
        for t in cbar.ax.get_yticklabels():
            t.set_color(COLORS["text_muted"])

    def on_mode_change(self):
        mode = self.plot_mode.get()
        if mode == "Fy (Siła boczna)":
            self.fit_btn_text.set("📈  Dopasuj model Fy0 (Pure Slip)")
        elif mode == "Fy (Load Sensitivity)":
            self.fit_btn_text.set("📈  Dopasuj model Fy_load (Fz)")
        elif mode == "Fx (Siła wzdłużna)":
            self.fit_btn_text.set("📈  Dopasuj model Fx0 (Pure Slip)")
        elif mode == "Mz (Moment stabilizujący)":
            self.fit_btn_text.set("📈  Dopasuj model Mz0 (Aligning Torque)")
        elif mode == "Combined Slip (Gyk)":
            self.fit_btn_text.set("📈  Dopasuj Combined Slip (Gyk)")
        else:
            self.fit_btn_text.set("📈  Brak dopasowania dla obwiedni")
        self.update_plot_view()

    def _reset_plot(self):
        if getattr(self, "_cbar", None) is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        self.ax.clear()
        legend = self.ax.get_legend()
        if legend is not None:
            legend.remove()

        mode = self.plot_mode.get() if hasattr(self, "plot_mode") else "Fy (Siła boczna)"

        # ``Axes.clear()`` does not reliably restore an aspect changed by a
        # previous view.  The force envelope uses equal X/Y scaling, while all
        # slip plots combine quantities with very different numerical ranges.
        self.ax.set_aspect("auto", adjustable="box")

        if "Fy" in mode:
            self.ax.set_xlabel("Kąt znoszenia SA [deg]")
            self.ax.set_ylabel("Siła boczna Fy [N]")
            self.ax.set_title("Pacejka Fy – Siła boczna od kąta znoszenia")
        elif "Fx" in mode:
            self.ax.set_xlabel("Poślizg wzdłużny Slip Ratio (SL) [-]")
            self.ax.set_ylabel("Siła wzdłużna Fx [N]")
            self.ax.set_title("Charakterystyka wzdłużna (Drive/Brake Fx)")
        elif "Mz" in mode:
            self.ax.set_xlabel("Kąt znoszenia SA [deg]")
            self.ax.set_ylabel("Moment stabilizujący Mz [Nm]")
            self.ax.set_title("Aligning Torque (Mz0 vs SA)")
        elif mode == "Combined Slip (Gyk)":
            self.ax.set_xlabel("Poślizg wzdłużny SL [-]")
            self.ax.set_ylabel("Współczynnik redukcji Gyk [-]")
            self.ax.set_title("Combined Slip – zmierzona redukcja Fy")
        else:
            self.ax.set_xlabel("Siła wzdłużna Fx [N]")
            self.ax.set_ylabel("Siła boczna Fy [N]")
            self.ax.set_title("Tire Force Envelope (Fx vs Fy)")
            self.ax.set_aspect("equal", adjustable="box")

        self.ax.axhline(0, color=COLORS["text_muted"], linewidth=0.8, linestyle="--")
        self.ax.axvline(0, color=COLORS["text_muted"], linewidth=0.8, linestyle="--")
        self._style_axes()

        if hasattr(self, "canvas"):
            self.canvas.draw()
        if hasattr(self, "status"):
            self._set_status("Wyczyszczono wykres.")

    # ------------------------------------------------------------ akcje --
    def add_runs(self, test_type: str):
        paths = filedialog.askopenfilenames(
            title=f"Wybierz pliki .mat dla {test_type}",
            filetypes=[("MATLAB files", "*.mat"), ("Wszystkie pliki", "*.*")]
        )
        if not paths:
            return
        try:
            if test_type == "Cornering":
                fz = float(self.c_fz.get())
                tol_fz = float(self.c_tol_fz.get())
                param = float(self.c_ia.get())
                tol_param = float(self.c_tol_ia.get())
            else:
                fz = float(self.d_fz.get())
                tol_fz = float(self.d_tol_fz.get())
                param = 0.0
                tol_param = abs(float(self.d_sl_max.get()))
                target_sa = float(self.d_sa.get())
                tol_sa = abs(float(self.d_tol_sa.get()))
                target_ia = float(self.d_ia.get())
                tol_ia = abs(float(self.d_tol_ia.get()))
        except ValueError:
            messagebox.showerror("Błąd", "Wszystkie pola numeryczne muszą być poprawnymi liczbami.")
            return

        for path in paths:
            kwargs = {}
            if test_type == "Drive/Brake":
                kwargs = {
                    "target_sa": target_sa, "tol_sa": tol_sa,
                    "target_ia": target_ia, "tol_ia": tol_ia,
                }
            run = RunConfig(
                filepath=path, test_type=test_type, target_fz=abs(fz), tol_fz=abs(tol_fz),
                target_param=param, tol_param=abs(tol_param), **kwargs
            )
            self.runs.append(run)

        self._refresh_run_list()
        self._set_status(f"Dodano {len(paths)} run'ów [{test_type}].")

    def remove_selected_runs(self):
        sel = self.run_list.selection()
        if not sel:
            return
        indices = sorted([self.run_list.index(item) for item in sel], reverse=True)
        for idx in indices:
            del self.runs[idx]
        self._refresh_run_list()
        self._set_status(f"Usunięto {len(indices)} run(ów).")

    def _refresh_run_list(self):
        for item in self.run_list.get_children():
            self.run_list.delete(item)
        for r in self.runs:
            if r.test_type == "Cornering":
                condition = f"IA {r.target_param:g}±{r.tol_param:g}"
            else:
                condition = f"SA {r.target_sa:g}±{r.tol_sa:g}; IA {r.target_ia:g}±{r.tol_ia:g}; |SL|≤{r.tol_param:g}"
            self.run_list.insert("", "end", values=(
                r.test_type, os.path.basename(r.filepath),
                f"{r.target_fz:g}±{r.tol_fz:g}", condition, r.n_points
            ))

    def process_runs(self):
        if not self.runs:
            messagebox.showwarning("Brak runów", "Dodaj przynajmniej jeden plik .mat.")
            return

        corn_frames, drive_frames, errors = [], [], []
        for run in self.runs:
            try:
                self._set_status(f"Przetwarzanie {os.path.basename(run.filepath)}...")
                fdf = TireDataManager.load_and_filter(run)
                if fdf.empty:
                    raise ValueError("Brak punktów zgodnych z wybranymi warunkami.")
                run.n_points = len(fdf)
                if run.test_type == "Cornering":
                    corn_frames.append(fdf)
                else:
                    drive_frames.append(fdf)
            except Exception as e:
                run.n_points = 0
                errors.append(f"{os.path.basename(run.filepath)}: {e}")

        self._refresh_run_list()

        self.df_cornering = (
            pd.concat(corn_frames, ignore_index=True).drop_duplicates(["__source_path__", "__row__"], ignore_index=True)
            if corn_frames else None
        )
        self.df_drive = (
            pd.concat(drive_frames, ignore_index=True).drop_duplicates(["__source_path__", "__row__"], ignore_index=True)
            if drive_frames else None
        )
        self.fit_result = None
        self.fit_results.clear()
        self._show_plain_results("Dane zmieniły się. Poprzednie wyniki dopasowania zostały unieważnione.")

        total = (len(self.df_cornering) if self.df_cornering is not None else 0) + \
                (len(self.df_drive) if self.df_drive is not None else 0)

        if total == 0:
            self._set_status("Nie załadowano żadnych poprawnych punktów.")
            return messagebox.showerror("Błąd przetwarzania", "\n".join(errors) if errors else "Brak punktów zgodnych z filtrami.")

        self._set_status(f"Załadowano {total} punktów; błędne pliki: {len(errors)}.")
        self.update_plot_view()
        if errors:
            messagebox.showwarning("Przetworzono z ostrzeżeniami", f"Załadowano punktów: {total}\n\n" + "\n".join(errors))
        else:
            messagebox.showinfo("Gotowe", f"Przetworzono pomyślnie. Łącznie punktów: {total}")

    def update_plot_view(self):
        mode = self.plot_mode.get()
        self._reset_plot()

        if mode == "Fy (Siła boczna)":
            if self.df_cornering is None or len(self.df_cornering) == 0:
                self._set_status("Brak danych Cornering.")
                return
            alpha = self.df_cornering["SA"].values
            fy = -self.df_cornering["FY"].values
            ia = self.df_cornering["IA"].values

            sc = self.ax.scatter(alpha, fy, c=ia, cmap="coolwarm", s=10, alpha=0.75, edgecolors="none")
            self._cbar = self.fig.colorbar(sc, ax=self.ax, pad=0.02)
            self._style_colorbar(self._cbar, "Camber IA [deg]")

            self.ax.set_xlim(np.min(alpha) - 1, np.max(alpha) + 1)
            self.ax.set_ylim(np.min(fy) - 200, np.max(fy) + 200)

        elif mode == "Fy (Load Sensitivity)":
            if self.df_cornering is None or len(self.df_cornering) == 0:
                self._set_status("Brak danych Cornering.")
                return
            alpha = self.df_cornering["SA"].values
            fy = -self.df_cornering["FY"].values
            fz_abs = np.abs(self.df_cornering["FZ"].values)

            # Kolorowanie względem obciążenia żeby widzieć "warstwy" Fz
            sc = self.ax.scatter(alpha, fy, c=fz_abs, cmap="viridis", s=10, alpha=0.75, edgecolors="none")
            self._cbar = self.fig.colorbar(sc, ax=self.ax, pad=0.02)
            self._style_colorbar(self._cbar, "Obciążenie pionowe Fz [N]")
            self.ax.set_title("Load Sensitivity – rozkład i przekroje Fz")

            self.ax.set_xlim(np.min(alpha) - 1, np.max(alpha) + 1)
            self.ax.set_ylim(np.min(fy) - 200, np.max(fy) + 200)

        elif mode == "Mz (Moment stabilizujący)":
            if self.df_cornering is None or len(self.df_cornering) == 0:
                self._set_status("Brak danych Cornering.")
                return
            valid = np.isfinite(self.df_cornering["MZ"])
            if not valid.any():
                self._set_status("Załadowane dane Cornering nie zawierają kanału MZ.")
                return
            alpha = self.df_cornering.loc[valid, "SA"].values
            mz = self.df_cornering.loc[valid, "MZ"].values
            fz_abs = np.abs(self.df_cornering.loc[valid, "FZ"].values)

            sc = self.ax.scatter(alpha, mz, c=fz_abs, cmap="viridis", s=10, alpha=0.75, edgecolors="none")
            self._cbar = self.fig.colorbar(sc, ax=self.ax, pad=0.02)
            self._style_colorbar(self._cbar, "Obciążenie pionowe Fz [N]")

            self.ax.set_xlim(np.min(alpha) - 1, np.max(alpha) + 1)
            self.ax.set_ylim(np.min(mz) - 5, np.max(mz) + 5)

        elif mode == "Fx (Siła wzdłużna)":
            if self.df_drive is None or len(self.df_drive) == 0:
                self._set_status("Brak danych Drive/Brake.")
                return
            sl = self.df_drive["SL"].values
            fx = self.df_drive["FX"].values
            fz_vals = self.df_drive["FZ"].values

            sc = self.ax.scatter(sl, fx, c=fz_vals, cmap="viridis", s=10, alpha=0.75, edgecolors="none")
            self._cbar = self.fig.colorbar(sc, ax=self.ax, pad=0.02)
            self._style_colorbar(self._cbar, "Obciążenie Fz [N]")

            self.ax.set_xlim(np.min(sl) - 0.05, np.max(sl) + 0.05)
            self.ax.set_ylim(np.min(fx) - 200, np.max(fx) + 200)

        elif mode == "Combined Slip (Gyk)":
            if self.df_drive is None or len(self.df_drive) == 0:
                self._set_status("Do Combined Slip potrzebne są dane Drive/Brake z kanałami SA, IA i FY.")
                return
            try:
                targets = build_combined_slip_targets(self.df_drive)
            except ValueError as exc:
                self._set_status(str(exc))
                return

            sc = self.ax.scatter(
                targets["SL"], targets["GYK"], c=np.abs(targets["SA"]),
                cmap="plasma", s=10, alpha=0.65, edgecolors="none"
            )
            self._cbar = self.fig.colorbar(sc, ax=self.ax, pad=0.02)
            self._style_colorbar(self._cbar, "|SA| [deg]")
            upper_gyk = min(max(float(np.percentile(targets["GYK"], 99)) * 1.1, 1.2), 3.0)
            self.ax.set_ylim(0.0, upper_gyk)
            self.ax.set_title("Combined Slip – Gyk z bazą zmierzoną przy SL≈0")

        else:  # Tire Force Envelope
            has_corn = self.df_cornering is not None and len(self.df_cornering) > 0
            has_drive = self.df_drive is not None and len(self.df_drive) > 0

            if not has_corn and not has_drive:
                self._set_status("Brak danych do wykreślenia obwiedni sił opony.")
                return

            all_fx, all_fy = [], []

            if has_corn:
                fx_c = np.zeros_like(self.df_cornering["SA"].values)
                fy_c = -self.df_cornering["FY"].values
                self.ax.scatter(fx_c, fy_c, color=COLORS["accent"], s=8, alpha=0.65, edgecolors="none", label="Cornering")
                all_fx.append(fx_c)
                all_fy.append(fy_c)

            if has_drive:
                fx_d = self.df_drive["FX"].values
                fy_values = self.df_drive["FY"].values
                fy_d = np.where(np.isfinite(fy_values), -fy_values, 0.0)
                self.ax.scatter(fx_d, fy_d, color="#4fb0ff", s=8, alpha=0.65, edgecolors="none", label="Drive/Brake")
                all_fx.append(fx_d)
                all_fy.append(fy_d)

            fx_all = np.concatenate(all_fx)
            fy_all = np.concatenate(all_fy)
            r_max = float(np.max(np.sqrt(fx_all**2 + fy_all**2))) if len(fx_all) else 1000

            theta = np.linspace(0, 2 * np.pi, 200)
            for frac in (0.25, 0.5, 0.75, 1.0):
                r = r_max * frac
                self.ax.plot(r * np.cos(theta), r * np.sin(theta), color=COLORS["border"], linewidth=0.8, linestyle=":", zorder=0)

            limit = r_max * 1.15
            self.ax.set_xlim(-limit, limit)
            self.ax.set_ylim(-limit, limit)
            self.ax.set_aspect("equal", adjustable="box")
            self.ax.legend(loc="upper right", facecolor=COLORS["panel_alt"], edgecolor=COLORS["border"], labelcolor=COLORS["text"])

        self._draw_saved_fit(mode)
        self.canvas.draw()
        self._set_status(f"Wyświetlono widok: {mode}")

    def _draw_saved_fit(self, mode):
        model_by_mode = {
            "Fy (Siła boczna)": "Fy0",
            "Fy (Load Sensitivity)": "Fy_load",
            "Fx (Siła wzdłużna)": "Fx0",
            "Mz (Moment stabilizujący)": "Mz0",
            "Combined Slip (Gyk)": "CombinedSlip",
        }
        result = self.fit_results.get(model_by_mode.get(mode))
        if result is None:
            return

        if mode == "Fy (Siła boczna)":
            alpha_deg = np.linspace(float(self.df_cornering["SA"].min()), float(self.df_cornering["SA"].max()), 400)
            parameters = [result[name] for name in ("B", "C", "D", "E", "Sh", "Sv")]
            self.ax.plot(alpha_deg, pacejka_fy(np.deg2rad(alpha_deg), *parameters), color="#ffb703", linewidth=3.0, label="Robust fit Fy0")
        elif mode == "Mz (Moment stabilizujący)":
            alpha_deg = np.linspace(float(self.df_cornering["SA"].min()), float(self.df_cornering["SA"].max()), 400)
            parameters = [result[name] for name in ("Bz", "Cz", "Dz", "Ez", "Shz", "Svz")]
            self.ax.plot(alpha_deg, pacejka_mz(np.deg2rad(alpha_deg), *parameters), color="#ffb703", linewidth=3.0, label="Robust fit Mz0")
        elif mode == "Fx (Siła wzdłużna)":
            kappa = np.linspace(float(self.df_drive["SL"].min()), float(self.df_drive["SL"].max()), 400)
            parameters = [result[name] for name in ("Bx", "Cx", "Dx", "Ex", "Shx", "Svx")]
            self.ax.plot(kappa, pacejka_fx(kappa, *parameters), color="#ffb703", linewidth=3.0, label="Robust fit Fx0")
        elif mode == "Combined Slip (Gyk)":
            kappa = np.linspace(float(self.df_drive["SL"].min()), float(self.df_drive["SL"].max()), 400)
            self.ax.plot(kappa, pacejka_comb_Gyk(kappa, result["rBy1"], result["rCy1"]), color="#ffb703", linewidth=3.0, label="Robust fit Gyk")
        else:
            alpha_deg = np.linspace(float(self.df_cornering["SA"].min()), float(self.df_cornering["SA"].max()), 400)
            alpha = np.deg2rad(alpha_deg)
            parameters = [result[name] for name in ("pCy1", "pDy1", "pDy2", "pEy1", "pKy1", "pKy2", "pUy1", "pVy1")]
            fz_values = [float(self.df_cornering["FZ"].min()), float(self.df_cornering["FZ"].median()), float(self.df_cornering["FZ"].max())]
            for fz, color in zip(fz_values, ["#3ddc84", "#ffb703", "#e0521c"]):
                prediction = pacejka_fy_load((alpha, np.full_like(alpha, fz)), *parameters, fz0=result["fz0"])
                self.ax.plot(alpha_deg, prediction, color=color, linewidth=2.5, label=f"Fit Fz={fz:.0f} N")
        self.ax.legend(loc="best", facecolor=COLORS["panel_alt"], edgecolor=COLORS["border"], labelcolor=COLORS["text"])

    def _show_plain_results(self, text):
        self.txt_results.configure(state="normal")
        self.txt_results.delete("1.0", "end")
        self.txt_results.insert("end", text)
        self.txt_results.configure(state="disabled")

    def _record_fit(self, result):
        self.fit_result = result
        self.fit_results[result["model"]] = result

    def _diagnostic_lines(self, result):
        lines = [
            f"RMSE: {result['rmse']:.2f} | NRMSE: {100 * result['nrmse']:.2f}% | R²: {result['r2']:.5f}",
            f"Bias: {result['bias']:.2f} | Punkty: {result['n_points']} | Wywołania: {result['nfev']}",
            f"Condition number: {result['condition_number']:.3e}",
        ]
        if result.get("warnings"):
            lines.append("UWAGA: " + "; ".join(result["warnings"]))
        return lines

    def _fit_pure_model(self, mode):
        if mode in {"Fy (Siła boczna)", "Mz (Moment stabilizujący)"}:
            if self.df_cornering is None or len(self.df_cornering) == 0:
                return messagebox.showwarning("Błąd", "Brak danych Cornering.")
            frame = self.df_cornering.replace([np.inf, -np.inf], np.nan)
            x = np.deg2rad(frame["SA"].to_numpy(dtype=float))
            if mode == "Fy (Siła boczna)":
                y = -frame["FY"].to_numpy(dtype=float)
                model_name, title = "Fy0", "Fy0 – Pure Slip"
            else:
                y = frame["MZ"].to_numpy(dtype=float)
                model_name, title = "Mz0", "Mz0 – Aligning Moment"
            slip_unit = "rad"
        else:
            if self.df_drive is None or len(self.df_drive) == 0:
                return messagebox.showwarning("Błąd", "Brak danych Drive/Brake.")
            frame = self.df_drive.replace([np.inf, -np.inf], np.nan)
            if "SA" in frame and frame["SA"].notna().any():
                pure_mask = np.abs(frame["SA"]) <= 0.30
                frame = frame[pure_mask]
            if len(frame) < 12:
                return messagebox.showwarning("Błąd", "Za mało danych Pure Fx przy |SA| ≤ 0.30°.")
            x = frame["SL"].to_numpy(dtype=float)
            y = frame["FX"].to_numpy(dtype=float)
            model_name, title, slip_unit = "Fx0", "Fx0 – Pure Slip", "ratio"

        try:
            output = fit_pure_magic(x, y, model_name=model_name, slip_unit=slip_unit)
        except Exception as exc:
            return messagebox.showerror("Błąd optymalizacji", str(exc))

        generic = output.as_dict()
        if model_name == "Fy0":
            result = generic
            parameter_line = (
                f"B={result['B']:.5g}  C={result['C']:.4g} (fixed)  D={result['D']:.2f} N\n"
                f"E={result['E']:.5g}  Sh={np.rad2deg(result['Sh']):.4f}°  Sv={result['Sv']:.2f} N"
            )
        elif model_name == "Fx0":
            diagnostics = {key: value for key, value in generic.items() if key not in {"B", "C", "D", "E", "Sh", "Sv"}}
            result = {
                **diagnostics,
                "Bx": generic["B"], "Cx": generic["C"], "Dx": generic["D"],
                "Ex": generic["E"], "Shx": generic["Sh"], "Svx": generic["Sv"],
            }
            parameter_line = (
                f"Bx={result['Bx']:.5g}  Cx={result['Cx']:.4g} (fixed)  Dx={result['Dx']:.2f} N\n"
                f"Ex={result['Ex']:.5g}  Shx={result['Shx']:.6f}  Svx={result['Svx']:.2f} N"
            )
        else:
            diagnostics = {key: value for key, value in generic.items() if key not in {"B", "C", "D", "E", "Sh", "Sv"}}
            result = {
                **diagnostics,
                "Bz": generic["B"], "Cz": generic["C"], "Dz": generic["D"],
                "Ez": generic["E"], "Shz": generic["Sh"], "Svz": generic["Sv"],
            }
            parameter_line = (
                f"Bz={result['Bz']:.5g}  Cz={result['Cz']:.4g} (fixed)  Dz={result['Dz']:.3f} Nm\n"
                f"Ez={result['Ez']:.5g}  Shz={np.rad2deg(result['Shz']):.4f}°  Svz={result['Svz']:.3f} Nm"
            )

        result["schema_version"] = 2
        self._record_fit(result)
        self._show_plain_results("\n".join([title, parameter_line, *self._diagnostic_lines(result)]))
        self.update_plot_view()
        self._set_status(f"Dopasowano {model_name}; NRMSE={100 * result['nrmse']:.2f}%")

    def _fit_combined_model(self):
        if self.df_drive is None or len(self.df_drive) == 0:
            return messagebox.showwarning("Błąd", "Brak danych Drive/Brake.")
        try:
            output = fit_combined_slip(self.df_drive)
        except Exception as exc:
            return messagebox.showerror("Błąd optymalizacji Combined Slip", str(exc))
        result = {**output.as_dict(), "schema_version": 2}
        self._record_fit(result)
        parameters = f"rBy1={result['rBy1']:.5g}  rCy1={result['rCy1']:.5g}\nGrupy warunków: {result['n_condition_groups']}"
        self._show_plain_results("\n".join(["Combined Slip – Gyk", parameters, *self._diagnostic_lines(result)]))
        self.update_plot_view()
        self._set_status(f"Dopasowano Combined Slip; NRMSE={100 * result['nrmse']:.2f}%")

    def _fit_load_model(self):
        if self.df_cornering is None or len(self.df_cornering) == 0:
            return messagebox.showwarning("Błąd", "Brak danych Cornering.")

        df = self.df_cornering.replace([np.inf, -np.inf], np.nan).dropna(subset=["SA", "FY", "FZ"])
        configured_loads = df["__target_fz__"].nunique() if "__target_fz__" in df else 0
        if configured_loads < 3:
            return messagebox.showwarning(
                "Błąd",
                "Fit load-sensitive wymaga danych dodanych dla co najmniej trzech docelowych poziomów Fz.",
            )
        alpha = np.deg2rad(df["SA"].to_numpy(dtype=float))
        fy = -df["FY"].to_numpy(dtype=float)
        fz_abs = np.abs(df["FZ"].to_numpy(dtype=float))
        try:
            output = fit_load_sensitive(alpha, fz_abs, fy)
        except Exception as exc:
            return messagebox.showerror("Błąd optymalizacji", str(exc))
        result = {**output.as_dict(), "schema_version": 2}
        self._record_fit(result)
        parameter_lines = [
            "Fy Load Sensitivity",
            f"pCy1={result['pCy1']:.5g}  pDy1={result['pDy1']:.5g}  pDy2={result['pDy2']:.5g}",
            f"pEy1={result['pEy1']:.5g}  pKy1={result['pKy1']:.5g}  pKy2={result['pKy2']:.5g}",
            f"pUy1={result['pUy1']:.5g} rad  pVy1={result['pVy1']:.5g}  Fz0={result['fz0']:.1f} N",
            f"Poziomy Fz (100 N): {result['n_load_levels_100N']}",
        ]
        self._show_plain_results("\n".join([*parameter_lines, *self._diagnostic_lines(result)]))
        self.update_plot_view()
        self._set_status(f"Dopasowano Fy_load; NRMSE={100 * result['nrmse']:.2f}%")

    def fit_model(self):
        mode = self.plot_mode.get()
        if mode in {"Fy (Siła boczna)", "Fx (Siła wzdłużna)", "Mz (Moment stabilizujący)"}:
            return self._fit_pure_model(mode)
        if mode == "Fy (Load Sensitivity)":
            return self._fit_load_model()
        if mode == "Combined Slip (Gyk)":
            return self._fit_combined_model()
        messagebox.showinfo("Info", "Ten widok nie ma modelu do dopasowania.")

    def export_data(self):
        if self.df_cornering is None and self.df_drive is None:
            return messagebox.showwarning("Brak danych", "Brak danych do eksportu.")
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        combined = pd.concat([df for df in [self.df_cornering, self.df_drive] if df is not None], ignore_index=True)
        combined.to_csv(path, index=False)
        self._set_status(f"Zapisano dane do {path}")

    def export_coeffs(self):
        if not self.fit_results:
            return messagebox.showwarning("Brak", "Najpierw wykonaj dopasowanie.")
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json"), ("CSV", "*.csv")])
        if not path:
            return
        if path.lower().endswith(".csv"):
            rows = []
            for result in self.fit_results.values():
                rows.append({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in result.items()})
            pd.DataFrame(rows).to_csv(path, index=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                project = {
                    "schema_version": 2,
                    "conventions": {
                        "FZ": "positive compression [N]",
                        "SA_model": "radians",
                        "SA_UI": "degrees",
                        "SL": "dimensionless ratio",
                    },
                    "runs": [asdict(run) for run in self.runs],
                    "fits": self.fit_results,
                }
                json.dump(project, f, indent=2, ensure_ascii=False)
        self._set_status(f"Zapisano współczynniki do {path}")

def main():
    try:
        app = PUTMTireApp()
        app.mainloop()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
