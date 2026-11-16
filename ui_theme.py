"""ui_theme.py — Spójny system projektowy i komponenty UI dla PUTM Tire Studio
===========================================================================
Motyw: Carbon Motorsport Telemetry (Dark Obsidian + Racing Orange + Telemetry Cyan).
Dostarcza paletę kolorów, typografię, stylowanie ttk, bezpieczne ładowanie logo z
przezroczystością, stylizację Matplotlib oraz nowoczesne komponenty (ModernCard,
CollapsibleCard, StepHeader, KpiTile, SegmentedToggle, SidebarNav, StatusBanner).
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ==============================================================================
# PALETA KOLORÓW & TYPOGRAFIA
# ==============================================================================
PALETTE: Dict[str, str] = {
    # Główne powierzchnie (warstwy od najciemniejszej do najwyższej)
    "bg":             "#080B10",   # Głębokie tło aplikacji
    "bg_header":      "#0C1017",   # Pasek górny & dolny
    "bg_sidebar":     "#0B0F16",   # Lewy panel nawigacji krok po kroku
    "panel":          "#111722",   # Główna karta / kontener
    "panel_alt":      "#172030",   # Nagłówek karty / nagłówek tabeli / hover
    "panel_elevated": "#1E293B",   # Podniesiona kontrolka / aktywna pigułka
    "card":           "#141B27",   # Wewnętrzna sekcja fokusowa
    "input":          "#090D14",   # Pole tekstowe / konsola / well

    # Obramowania
    "border":         "#212C3D",   # Subtelna linia oddzielająca
    "border_bright":  "#334155",   # Wyraźniejsze obramowanie / focus

    # Tekst
    "text":           "#F1F5F9",   # Główny tekst o wysokim kontraście
    "text_secondary": "#CBD5E1",   # Tekst pomocniczy / opisy
    "text_muted":     "#7C8EA6",   # Etykiety / jednostki / opisy osi

    # Kolory akcentowe zespołu PUT Motorsport
    "accent":         "#FF5A1F",   # Oficjalny PUTM Racing Orange
    "accent_hover":   "#FF7542",   # Jasny hover
    "accent_dark":    "#D9430D",   # Wciśnięty pomarańcz
    "accent_soft":    "#2A1610",   # Delikatne tło pigułki akcentowej

    # Kolory telemetryczne
    "cyan":           "#06D6F2",   # Telemetry Cyan
    "cyan_hover":     "#3FE2FA",   # Jasny cyan hover
    "cyan_soft":      "#0A2530",   # Tło pigułki cyan

    # Statusy
    "ok":             "#10B981",   # Szmaragdowy zielony
    "ok_soft":        "#0B2920",
    "warn":           "#F59E0B",   # Bursztynowy
    "warn_soft":      "#2B1F0A",
    "danger":         "#F43F5E",   # Koralowa czerwień
    "danger_soft":    "#2E1018",
    "purple":         "#A855F7",   # Fiolet dla momentu Mz / 3. serii

    # Wykresy
    "axes_bg":        "#090D14",
}

CHART_PALETTE: List[str] = [
    "#FF5A1F",  # Racing Orange
    "#06D6F2",  # Telemetry Cyan
    "#10B981",  # Emerald
    "#F59E0B",  # Amber
    "#A855F7",  # Purple
    "#38BDF8",  # Sky
    "#F43F5E",  # Rose
]

FONT_DISPLAY    = ("Segoe UI", 14, "bold")
FONT_TITLE      = ("Segoe UI", 11, "bold")
FONT_HEAD       = ("Segoe UI", 10, "bold")
FONT_BASE       = ("Segoe UI", 9)
FONT_BOLD       = ("Segoe UI", 9, "bold")
FONT_SMALL      = ("Segoe UI", 8)
FONT_SMALL_BOLD = ("Segoe UI", 8, "bold")
FONT_MONO       = ("Consolas", 9)
FONT_MONO_BOLD  = ("Consolas", 9, "bold")
FONT_KPI_VAL    = ("Segoe UI", 14, "bold")
FONT_KPI_LG     = ("Segoe UI", 16, "bold")


# ==============================================================================
# ŁADOWANIE LOGO BEZ CZARNEJ RAMKI
# ==============================================================================
def load_brand_logo(
    path: str,
    max_size: Tuple[int, int] = (36, 36),
    tint_rgb: Optional[Tuple[int, int, int]] = None,
):
    """Ładuje logo PNG z czarnym tłem i zamienia czerń na czystą przezroczystość."""
    if not PIL_AVAILABLE or not path or not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGBA")
        # Wyliczamy kanał alfa z jasności (idealne dla białych/pomarańczowych znaków na czarnym tle)
        gray = img.convert("L")
        # Delikatne przycięcie szumu tła (< 18 -> 0) i wzmocnienie jasnych pikseli
        alpha = gray.point(lambda p: 0 if p < 18 else min(255, int((p - 18) * 255 / 220)))
        if tint_rgb is not None:
            colored = Image.new("RGBA", img.size, (*tint_rgb, 255))
            colored.putalpha(alpha)
            img = colored
        else:
            img.putalpha(alpha)

        resample = getattr(Image, "Resampling", Image).LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS
        img.thumbnail(max_size, resample)
        return ImageTk.PhotoImage(img)
    except Exception:
        return None


# ==============================================================================
# KONFIGURACJA MOTYWU TTK
# ==============================================================================
def apply_ttk_theme(root: tk.Misc, c: Optional[Dict[str, str]] = None) -> ttk.Style:
    """Konfiguruje spójny ciemny motyw ttk dla całej aplikacji i okien modalnych."""
    c = c or PALETTE
    try:
        root.configure(bg=c["bg"])
    except tk.TclError:
        pass

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(
        ".",
        background=c["bg"],
        foreground=c["text"],
        fieldbackground=c["input"],
        font=FONT_BASE,
        bordercolor=c["border"],
        darkcolor=c["panel"],
        lightcolor=c["panel"],
        troughcolor=c["bg"],
    )

    # Ramki i etykiety
    style.configure("TFrame", background=c["bg"])
    style.configure("Card.TFrame", background=c["panel"])
    style.configure("PanelAlt.TFrame", background=c["panel_alt"])
    style.configure("Sidebar.TFrame", background=c["bg_sidebar"])

    style.configure("TLabel", background=c["bg"], foreground=c["text"], font=FONT_BASE)
    style.configure("Card.TLabel", background=c["panel"], foreground=c["text"], font=FONT_BASE)
    style.configure("CardBold.TLabel", background=c["panel"], foreground=c["text"], font=FONT_BOLD)
    style.configure("CardSec.TLabel", background=c["panel"], foreground=c["text_secondary"], font=FONT_BASE)
    style.configure("Muted.TLabel", background=c["panel"], foreground=c["text_muted"], font=FONT_SMALL)
    style.configure("Section.TLabel", background=c["panel"], foreground=c["accent"], font=FONT_HEAD)
    style.configure("Cyan.TLabel", background=c["panel"], foreground=c["cyan"], font=FONT_SMALL_BOLD)

    # LabelFrame (dla kompatybilności i okien)
    style.configure(
        "TLabelframe",
        background=c["panel"],
        foreground=c["text"],
        bordercolor=c["border"],
        relief="solid",
        borderwidth=1,
    )
    style.configure(
        "TLabelframe.Label",
        background=c["panel"],
        foreground=c["cyan"],
        font=FONT_BOLD,
    )

    # Przyciski standardowe
    style.configure(
        "TButton",
        background=c["panel_alt"],
        foreground=c["text"],
        borderwidth=1,
        relief="solid",
        bordercolor=c["border_bright"],
        focusthickness=0,
        padding=(11, 6),
        font=FONT_BOLD,
    )
    style.map(
        "TButton",
        background=[("pressed", c["panel"]), ("active", c["panel_elevated"]), ("disabled", c["panel"])],
        bordercolor=[("active", c["cyan"]), ("focus", c["cyan"])],
        foreground=[("disabled", c["text_muted"])],
    )

    # Przycisk główny (Racing Orange)
    style.configure(
        "Accent.TButton",
        background=c["accent"],
        foreground="#FFFFFF",
        borderwidth=0,
        relief="flat",
        focusthickness=0,
        padding=(14, 7),
        font=FONT_BOLD,
    )
    style.map(
        "Accent.TButton",
        background=[("pressed", c["accent_dark"]), ("active", c["accent_hover"]), ("disabled", c["panel_alt"])],
        foreground=[("disabled", c["text_muted"])],
    )

    # Przycisk telemetryczny (Cyan)
    style.configure(
        "Cyan.TButton",
        background=c["cyan"],
        foreground="#071018",
        borderwidth=0,
        relief="flat",
        focusthickness=0,
        padding=(12, 6),
        font=FONT_BOLD,
    )
    style.map(
        "Cyan.TButton",
        background=[("pressed", "#04A9C2"), ("active", c["cyan_hover"]), ("disabled", c["panel_alt"])],
        foreground=[("disabled", c["text_muted"])],
    )

    # Przycisk destrukcyjny (Subtle Danger)
    style.configure(
        "Danger.TButton",
        background=c["danger_soft"],
        foreground="#FDA4AF",
        bordercolor="#6B1D2F",
        borderwidth=1,
        relief="solid",
        focusthickness=0,
        padding=(10, 6),
        font=FONT_BOLD,
    )
    style.map(
        "Danger.TButton",
        background=[("active", "#451522"), ("pressed", "#260B12")],
        foreground=[("active", "#FECDD3")],
    )

    # Małe przyciski pigułek (presety Fz / IA / SA)
    style.configure(
        "Pill.TButton",
        background=c["panel_alt"],
        foreground=c["text_secondary"],
        borderwidth=1,
        relief="solid",
        bordercolor=c["border"],
        focusthickness=0,
        padding=(6, 3),
        font=FONT_SMALL_BOLD,
    )
    style.map(
        "Pill.TButton",
        background=[("active", c["panel_elevated"]), ("pressed", c["accent_soft"])],
        bordercolor=[("active", c["accent"])],
        foreground=[("active", c["text"])],
    )

    # Pola wejściowe i Combobox
    style.configure(
        "TEntry",
        fieldbackground=c["input"],
        foreground=c["text"],
        bordercolor=c["border_bright"],
        insertcolor=c["text"],
        padding=(7, 5),
    )
    style.map(
        "TEntry",
        bordercolor=[("focus", c["accent"]), ("active", c["border_bright"])],
    )

    style.configure(
        "TCombobox",
        fieldbackground=c["input"],
        foreground=c["text"],
        background=c["panel_alt"],
        arrowcolor=c["cyan"],
        bordercolor=c["border_bright"],
        padding=(6, 4),
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", c["input"]), ("disabled", c["panel"])],
        foreground=[("readonly", c["text"]), ("disabled", c["text_muted"])],
        bordercolor=[("focus", c["accent"])],
    )
    root.option_add("*TCombobox*Listbox.background", c["input"])
    root.option_add("*TCombobox*Listbox.foreground", c["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", c["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")
    root.option_add("*TCombobox*Listbox.font", FONT_BASE)

    # Checkbox & Radio
    style.configure(
        "TCheckbutton",
        background=c["panel"],
        foreground=c["text"],
        indicatorbackground=c["input"],
        indicatorforeground=c["accent"],
        font=FONT_BASE,
    )
    style.map(
        "TCheckbutton",
        background=[("active", c["panel"])],
        foreground=[("active", c["text"])],
        indicatorbackground=[("selected", c["accent"])],
    )

    style.configure(
        "TRadiobutton",
        background=c["panel"],
        foreground=c["text"],
        indicatorbackground=c["input"],
        indicatorforeground=c["accent"],
        font=FONT_BASE,
    )
    style.map(
        "TRadiobutton",
        background=[("active", c["panel"])],
        indicatorbackground=[("selected", c["accent"])],
    )

    # Suwaki (TScale)
    style.configure(
        "Horizontal.TScale",
        background=c["panel"],
        troughcolor=c["panel_elevated"],
        bordercolor=c["border"],
        sliderthickness=16,
        sliderlength=22,
    )

    # Paski postępu
    style.configure(
        "Horizontal.TProgressbar",
        background=c["accent"],
        troughcolor=c["panel_alt"],
        bordercolor=c["border"],
        lightcolor=c["accent"],
        darkcolor=c["accent_dark"],
        thickness=8,
    )

    # Ciemne paski przewijania
    for orient in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
        style.configure(
            orient,
            background=c["panel_elevated"],
            troughcolor=c["bg_sidebar"],
            bordercolor=c["bg_sidebar"],
            arrowcolor=c["text_muted"],
            relief="flat",
            borderwidth=0,
            width=10,
        )
        style.map(
            orient,
            background=[("active", c["border_bright"]), ("pressed", c["accent"])],
            arrowcolor=[("active", c["text"])],
        )

    # Sub-notebooks
    style.configure("TNotebook", background=c["panel"], borderwidth=0)
    style.configure(
        "TNotebook.Tab",
        background=c["panel_alt"],
        foreground=c["text_muted"],
        padding=(12, 6),
        font=FONT_BOLD,
        borderwidth=0,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", c["panel"])],
        foreground=[("selected", c["cyan"])],
    )

    # Tabele Treeview
    style.configure(
        "Treeview",
        background=c["input"],
        fieldbackground=c["input"],
        foreground=c["text"],
        bordercolor=c["border"],
        rowheight=26,
        borderwidth=1,
        relief="solid",
        font=FONT_BASE,
    )
    style.configure(
        "Treeview.Heading",
        background=c["panel_alt"],
        foreground=c["cyan"],
        font=FONT_SMALL_BOLD,
        relief="flat",
        bordercolor=c["border"],
        padding=(8, 5),
    )
    style.map("Treeview.Heading", background=[("active", c["panel_elevated"])])
    style.map(
        "Treeview",
        background=[("selected", c["accent_soft"])],
        foreground=[("selected", c["text"])],
    )

    return style


def style_treeview_zebra(tree: ttk.Treeview, c: Optional[Dict[str, str]] = None) -> None:
    """Dodaje delikatną zebrę w tabeli Treeview."""
    c = c or PALETTE
    tree.tag_configure("odd", background=c["input"])
    tree.tag_configure("even", background="#0D131D")
    tree.tag_configure("highlight", background=c["accent_soft"], foreground=c["text"])


# ==============================================================================
# STYLIZACJA MATPLOTLIB
# ==============================================================================
def style_axes_2d(
    ax,
    fig=None,
    *,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    zero_lines: bool = True,
    c: Optional[Dict[str, str]] = None,
) -> None:
    """Nowoczesny, czysty styl 2D dla wykresów Matplotlib."""
    c = c or PALETTE
    ax.set_facecolor(c["axes_bg"])
    if fig is not None:
        fig.patch.set_facecolor(c["panel"])

    if zero_lines:
        ax.axhline(0, color=c["border_bright"], linewidth=0.8, linestyle="--", alpha=0.65, zorder=1)
        ax.axvline(0, color=c["border_bright"], linewidth=0.8, linestyle="--", alpha=0.65, zorder=1)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(c["border_bright"])
        ax.spines[side].set_linewidth(0.9)

    ax.tick_params(colors=c["text_muted"], labelsize=8.5)
    if title is not None:
        ax.set_title(title, color=c["text"], fontsize=10.5, fontweight="bold", pad=10)
    else:
        ax.title.set_color(c["text"])
        ax.title.set_fontweight("bold")

    if xlabel is not None:
        ax.set_xlabel(xlabel, color=c["text_secondary"], fontsize=9)
    else:
        ax.xaxis.label.set_color(c["text_secondary"])

    if ylabel is not None:
        ax.set_ylabel(ylabel, color=c["text_secondary"], fontsize=9)
    else:
        ax.yaxis.label.set_color(c["text_secondary"])

    ax.grid(True, color=c["border"], alpha=0.45, linewidth=0.7, linestyle=":")


def style_legend(ax, *, loc: str = "best", fontsize: float = 8.0, c: Optional[Dict[str, str]] = None):
    c = c or PALETTE
    handles, _ = ax.get_legend_handles_labels()
    if not handles:
        return None
    leg = ax.legend(
        loc=loc,
        facecolor=c["panel_alt"],
        edgecolor=c["border_bright"],
        framealpha=0.92,
        fontsize=fontsize,
    )
    if leg:
        for txt in leg.get_texts():
            txt.set_color(c["text"])
    return leg


def style_colorbar(cbar, label: str, c: Optional[Dict[str, str]] = None) -> None:
    c = c or PALETTE
    cbar.set_label(label, color=c["text_secondary"], fontsize=8.5)
    cbar.ax.yaxis.set_tick_params(color=c["text_muted"], labelsize=8)
    cbar.outline.set_edgecolor(c["border"])
    for t in cbar.ax.get_yticklabels():
        t.set_color(c["text_muted"])


# ==============================================================================
# KOMPONENTY UI
# ==============================================================================
class ModernCard(tk.Frame):
    """Nowoczesna karta sekcji z subtelną ramką, nagłówkiem i opcjonalnym badge'em."""

    def __init__(
        self,
        parent,
        title: str = "",
        *,
        badge: str = "",
        badge_color: str = "cyan",
        accent_color: Optional[str] = None,
        colors: Optional[Dict[str, str]] = None,
        pad_x: int = 12,
        pad_y: int = 10,
    ):
        c = colors or PALETTE
        self.colors = c
        super().__init__(
            parent,
            bg=c["panel"],
            highlightbackground=c["border"],
            highlightthickness=1,
        )

        if title:
            hdr = tk.Frame(self, bg=c["panel_alt"], padx=10, pady=7)
            hdr.pack(fill="x")
            self.header = hdr

            if accent_color:
                tk.Frame(hdr, bg=accent_color, width=3, height=14).pack(side="left", padx=(0, 8), fill="y")

            self.lbl_title = tk.Label(
                hdr, text=title, fg=c["text"], bg=c["panel_alt"], font=FONT_BOLD, anchor="w"
            )
            self.lbl_title.pack(side="left")

            if badge:
                fg_b = c.get(badge_color, c["cyan"])
                bg_b = c.get(f"{badge_color}_soft", c["cyan_soft"])
                self.lbl_badge = tk.Label(
                    hdr, text=badge, fg=fg_b, bg=bg_b, font=FONT_SMALL_BOLD, padx=7, pady=1
                )
                self.lbl_badge.pack(side="right")

            tk.Frame(self, bg=c["border"], height=1).pack(fill="x")

        self.body = tk.Frame(self, bg=c["panel"], padx=pad_x, pady=pad_y)
        self.body.pack(fill="both", expand=True)


class CollapsibleCard(tk.Frame):
    """Rozwijana karta akordeonowa z płynnym hoverem i wskaźnikiem stanu."""

    def __init__(
        self,
        parent,
        title: str,
        *,
        is_open: bool = True,
        badge: str = "",
        badge_color: str = "cyan",
        colors: Optional[Dict[str, str]] = None,
        pad_x: int = 10,
        pad_y: int = 8,
    ):
        c = colors or PALETTE
        self.colors = c
        super().__init__(
            parent,
            bg=c["panel"],
            highlightbackground=c["border"],
            highlightthickness=1,
        )

        self._is_open = is_open
        self._header = tk.Frame(self, bg=c["panel_alt"], cursor="hand2", padx=10, pady=7)
        self._header.pack(fill="x")

        self._arrow = tk.Label(
            self._header,
            text="▾" if is_open else "▸",
            fg=c["accent"] if is_open else c["text_muted"],
            bg=c["panel_alt"],
            font=("Segoe UI", 9, "bold"),
        )
        self._arrow.pack(side="left", padx=(0, 6))

        self._title = tk.Label(
            self._header, text=title, fg=c["text"], bg=c["panel_alt"], font=FONT_BOLD
        )
        self._title.pack(side="left")

        self._badge = None
        if badge:
            fg_b = c.get(badge_color, c["cyan"])
            bg_b = c.get(f"{badge_color}_soft", c["cyan_soft"])
            self._badge = tk.Label(
                self._header, text=badge, fg=fg_b, bg=bg_b, font=FONT_SMALL_BOLD, padx=7, pady=1
            )
            self._badge.pack(side="right")

        self._sep = tk.Frame(self, bg=c["border"], height=1)
        self.body = tk.Frame(self, bg=c["panel"], padx=pad_x, pady=pad_y)

        if is_open:
            self._sep.pack(fill="x")
            self.body.pack(fill="both", expand=True)

        hover_targets = [self._header, self._arrow, self._title]
        for w in hover_targets:
            w.bind("<Button-1>", self.toggle)
            w.bind("<Enter>", lambda _e: self._set_hover(True))
            w.bind("<Leave>", lambda _e: self._set_hover(False))

    def _set_hover(self, hovering: bool):
        bg = self.colors["panel_elevated"] if hovering else self.colors["panel_alt"]
        self._header.configure(bg=bg)
        self._arrow.configure(bg=bg)
        self._title.configure(bg=bg)

    def toggle(self, _event=None):
        self.set_open(not self._is_open)

    def set_open(self, want_open: bool):
        if self._is_open == want_open:
            return
        self._is_open = want_open
        if want_open:
            self._sep.pack(fill="x")
            self.body.pack(fill="both", expand=True)
            self._arrow.configure(text="▾", fg=self.colors["accent"])
        else:
            self.body.pack_forget()
            self._sep.pack_forget()
            self._arrow.configure(text="▸", fg=self.colors["text_muted"])


class StepHeader(tk.Frame):
    """Górny pasek kontekstu kroku z numerem etapu, opisem i przyciskiem Dalej."""

    def __init__(
        self,
        parent,
        step_badge: str,
        title: str,
        subtitle: str,
        *,
        badge_kind: str = "accent",
        next_text: Optional[str] = None,
        next_command: Optional[Callable[[], None]] = None,
        colors: Optional[Dict[str, str]] = None,
    ):
        c = colors or PALETTE
        super().__init__(
            parent,
            bg=c["panel"],
            highlightbackground=c["border"],
            highlightthickness=1,
            padx=14,
            pady=8,
        )

        left = tk.Frame(self, bg=c["panel"])
        left.pack(side="left", fill="x", expand=True)

        top_row = tk.Frame(left, bg=c["panel"])
        top_row.pack(anchor="w", fill="x")

        fg_b = c.get(badge_kind, c["accent"])
        bg_b = c.get(f"{badge_kind}_soft", c["accent_soft"])
        tk.Label(
            top_row, text=step_badge, fg=fg_b, bg=bg_b, font=FONT_SMALL_BOLD, padx=8, pady=2
        ).pack(side="left", padx=(0, 10))

        tk.Label(
            top_row, text=title, fg=c["text"], bg=c["panel"], font=FONT_TITLE
        ).pack(side="left")

        tk.Label(
            left, text=subtitle, fg=c["text_muted"], bg=c["panel"], font=FONT_SMALL
        ).pack(anchor="w", pady=(2, 0))

        self.right_box = tk.Frame(self, bg=c["panel"])
        self.right_box.pack(side="right", padx=(10, 0))

        if next_text and next_command:
            ttk.Button(
                self.right_box, text=next_text, command=next_command, style="Cyan.TButton"
            ).pack(side="right")


class KpiTile(tk.Frame):
    """Kafelek telemetryczny KPI z kolorowym paskiem akcentu."""

    def __init__(
        self,
        parent,
        title: str,
        value: str = "--",
        *,
        accent: Optional[str] = None,
        subtext: str = "",
        large: bool = False,
        colors: Optional[Dict[str, str]] = None,
    ):
        c = colors or PALETTE
        accent_hex = accent or c["accent"]
        super().__init__(
            parent,
            bg=c["input"],
            highlightbackground=c["border"],
            highlightthickness=1,
        )

        # Cienki lewy pasek akcentu
        tk.Frame(self, bg=accent_hex, width=3).pack(side="left", fill="y")

        content = tk.Frame(self, bg=c["input"], padx=9, pady=6)
        content.pack(side="left", fill="both", expand=True)

        self.lbl_title = tk.Label(
            content, text=title, fg=c["text_muted"], bg=c["input"], font=FONT_SMALL_BOLD, anchor="w"
        )
        self.lbl_title.pack(anchor="w")

        self.lbl_val = tk.Label(
            content,
            text=value,
            fg=accent_hex,
            bg=c["input"],
            font=FONT_KPI_LG if large else FONT_KPI_VAL,
            anchor="w",
        )
        self.lbl_val.pack(anchor="w", pady=(1, 0))

        self.lbl_sub = None
        if subtext:
            self.lbl_sub = tk.Label(
                content, text=subtext, fg=c["text_muted"], bg=c["input"], font=FONT_SMALL, anchor="w"
            )
            self.lbl_sub.pack(anchor="w")

    def set_value(self, text: str, color: Optional[str] = None) -> None:
        self.lbl_val.configure(text=text)
        if color:
            self.lbl_val.configure(fg=color)


class SegmentedToggle(tk.Frame):
    """Nowoczesny przełącznik segmentowy (pill toggle) powiązany ze StringVar."""

    def __init__(
        self,
        parent,
        variable: tk.StringVar,
        options: List[Tuple[str, str]],  # [(value, label), ...]
        *,
        command: Optional[Callable[[], None]] = None,
        accent_key: str = "accent",
        colors: Optional[Dict[str, str]] = None,
    ):
        c = colors or PALETTE
        self.colors = c
        self.variable = variable
        self.command = command
        self.accent_key = accent_key
        super().__init__(
            parent,
            bg=c["input"],
            highlightbackground=c["border"],
            highlightthickness=1,
            padx=3,
            pady=3,
        )

        self._buttons: Dict[str, tk.Label] = {}
        for val, label in options:
            btn = tk.Label(
                self,
                text=label,
                font=FONT_SMALL_BOLD,
                padx=10,
                pady=5,
                cursor="hand2",
                anchor="center",
            )
            btn.pack(side="left", fill="x", expand=True, padx=1)
            btn.bind("<Button-1>", lambda _e, v=val: self._select(v))
            self._buttons[val] = btn

        self.variable.trace_add("write", lambda *_: self.refresh())
        self.refresh()

    def _select(self, value: str):
        if self.variable.get() != value:
            self.variable.set(value)
            if self.command:
                self.command()

    def refresh(self):
        c = self.colors
        cur = self.variable.get()
        active_bg = c.get(self.accent_key, c["accent"])
        active_fg = "#071018" if self.accent_key == "cyan" else "#FFFFFF"
        for val, btn in self._buttons.items():
            if val == cur:
                btn.configure(bg=active_bg, fg=active_fg)
            else:
                btn.configure(bg=c["input"], fg=c["text_muted"])


class StatusBanner(tk.Frame):
    """Inline'owy baner informacyjny / ostrzegawczy z opcjonalnym przyciskiem akcji."""

    def __init__(
        self,
        parent,
        text: str,
        *,
        kind: str = "warn",
        button_text: Optional[str] = None,
        button_command: Optional[Callable[[], None]] = None,
        second_button_text: Optional[str] = None,
        second_button_command: Optional[Callable[[], None]] = None,
        colors: Optional[Dict[str, str]] = None,
    ):
        c = colors or PALETTE
        fg = c.get(kind, c["warn"])
        bg = c.get(f"{kind}_soft", c["warn_soft"])
        super().__init__(
            parent,
            bg=bg,
            highlightbackground=fg,
            highlightthickness=1,
            padx=12,
            pady=8,
        )

        self.lbl = tk.Label(
            self, text=text, fg=c["text"], bg=bg, font=FONT_SMALL_BOLD, justify="left", anchor="w"
        )
        self.lbl.pack(side="left", fill="x", expand=True)

        if second_button_text and second_button_command:
            ttk.Button(
                self, text=second_button_text, command=second_button_command, style="Accent.TButton"
            ).pack(side="right", padx=(6, 0))

        if button_text and button_command:
            ttk.Button(
                self, text=button_text, command=button_command, style="TButton"
            ).pack(side="right", padx=(6, 0))


class SidebarNav(tk.Frame):
    """Pionowa nawigacja krok po kroku z sekcjami, pigułkami stanu i zwijaniem."""

    def __init__(
        self,
        parent,
        on_select: Callable[[str], None],
        *,
        width: int = 236,
        compact_width: int = 62,
        colors: Optional[Dict[str, str]] = None,
    ):
        c = colors or PALETTE
        self.colors = c
        self._on_select = on_select
        self._full_width = width
        self._compact_width = compact_width
        self._collapsed = False
        self._active_key: Optional[str] = None

        super().__init__(
            parent,
            bg=c["bg_sidebar"],
            width=width,
            highlightbackground=c["border"],
            highlightthickness=0,
        )
        self.pack_propagate(False)

        # Top sidebar header with collapse toggle
        top_bar = tk.Frame(self, bg=c["bg_sidebar"], padx=10, pady=8)
        top_bar.pack(fill="x")
        self._lbl_nav_title = tk.Label(
            top_bar, text="WORKFLOW", fg=c["text_muted"], bg=c["bg_sidebar"], font=FONT_SMALL_BOLD
        )
        self._lbl_nav_title.pack(side="left")

        self._btn_collapse = tk.Label(
            top_bar,
            text="«",
            fg=c["text_muted"],
            bg=c["panel_alt"],
            font=("Segoe UI", 9, "bold"),
            padx=6,
            pady=1,
            cursor="hand2",
        )
        self._btn_collapse.pack(side="right")
        self._btn_collapse.bind("<Button-1>", lambda _e: self.toggle_collapse())

        self._items_container = tk.Frame(self, bg=c["bg_sidebar"])
        self._items_container.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self._items: Dict[str, dict] = {}
        self._section_labels: List[tk.Label] = []

        # Prawa krawędź separatora
        tk.Frame(self, bg=c["border"], width=1).place(relx=1.0, rely=0.0, relheight=1.0, anchor="ne")

    def add_section(self, title: str) -> None:
        c = self.colors
        lbl = tk.Label(
            self._items_container,
            text=title,
            fg=c["text_muted"],
            bg=c["bg_sidebar"],
            font=("Segoe UI", 7, "bold"),
            anchor="w",
            padx=8,
            pady=4,
        )
        lbl.pack(fill="x", pady=(8, 2))
        self._section_labels.append(lbl)

    def add_item(
        self,
        key: str,
        step_code: str,
        title: str,
        subtitle: str,
        *,
        status_text: str = "",
        is_workflow: bool = True,
    ) -> None:
        c = self.colors
        row = tk.Frame(self._items_container, bg=c["bg_sidebar"], cursor="hand2")
        row.pack(fill="x", pady=2)

        indicator = tk.Frame(row, bg=c["bg_sidebar"], width=3)
        indicator.pack(side="left", fill="y")

        inner = tk.Frame(row, bg=c["bg_sidebar"], padx=8, pady=6)
        inner.pack(side="left", fill="both", expand=True)

        top_line = tk.Frame(inner, bg=c["bg_sidebar"])
        top_line.pack(fill="x")

        badge_bg = c["panel_alt"]
        badge_fg = c["text_secondary"]
        lbl_step = tk.Label(
            top_line, text=step_code, fg=badge_fg, bg=badge_bg, font=FONT_SMALL_BOLD, padx=5, pady=1
        )
        lbl_step.pack(side="left", padx=(0, 7))

        lbl_title = tk.Label(
            top_line, text=title, fg=c["text_secondary"], bg=c["bg_sidebar"], font=FONT_BOLD, anchor="w"
        )
        lbl_title.pack(side="left", fill="x", expand=True)

        lbl_dot = tk.Label(
            top_line, text="●", fg=c["border_bright"], bg=c["bg_sidebar"], font=("Segoe UI", 8)
        )
        lbl_dot.pack(side="right")

        lbl_sub = tk.Label(
            inner, text=subtitle, fg=c["text_muted"], bg=c["bg_sidebar"], font=("Segoe UI", 7), anchor="w"
        )
        lbl_sub.pack(fill="x", pady=(2, 0))

        self._items[key] = {
            "row": row,
            "indicator": indicator,
            "inner": inner,
            "top_line": top_line,
            "lbl_step": lbl_step,
            "lbl_title": lbl_title,
            "lbl_dot": lbl_dot,
            "lbl_sub": lbl_sub,
            "is_workflow": is_workflow,
            "status_state": "idle",
        }

        clickable = [row, indicator, inner, top_line, lbl_step, lbl_title, lbl_dot, lbl_sub]
        for w in clickable:
            w.bind("<Button-1>", lambda _e, k=key: self._on_select(k))
            w.bind("<Enter>", lambda _e, k=key: self._on_hover(k, True))
            w.bind("<Leave>", lambda _e, k=key: self._on_hover(k, False))

    def set_active(self, key: str) -> None:
        self._active_key = key
        for k in self._items:
            self._paint_item(k, hovering=False)

    def set_item_status(self, key: str, state: str = "idle", tooltip_sub: Optional[str] = None) -> None:
        """state: 'idle' | 'warn' | 'ok' | 'active'"""
        if key not in self._items:
            return
        item = self._items[key]
        item["status_state"] = state
        if tooltip_sub is not None:
            item["lbl_sub"].configure(text=tooltip_sub)
        color_map = {
            "idle": self.colors["border_bright"],
            "warn": self.colors["warn"],
            "ok": self.colors["ok"],
            "active": self.colors["cyan"],
        }
        item["lbl_dot"].configure(fg=color_map.get(state, self.colors["border_bright"]))

    def toggle_collapse(self) -> None:
        self._collapsed = not self._collapsed
        if self._collapsed:
            self.configure(width=self._compact_width)
            self._lbl_nav_title.pack_forget()
            self._btn_collapse.configure(text="»")
            for lbl in self._section_labels:
                lbl.pack_forget()
            for item in self._items.values():
                item["lbl_title"].pack_forget()
                item["lbl_sub"].pack_forget()
                item["lbl_dot"].pack_forget()
                item["lbl_step"].pack_configure(padx=0)
                item["inner"].configure(padx=6, pady=7)
        else:
            self.configure(width=self._full_width)
            self._lbl_nav_title.pack(side="left")
            self._btn_collapse.configure(text="«")
            for lbl in self._section_labels:
                lbl.pack_forget()
            for item in self._items.values():
                item["row"].pack_forget()
            for item in self._items.values():
                item["inner"].configure(padx=8, pady=6)
                item["lbl_step"].pack_configure(padx=(0, 7))
                item["lbl_title"].pack(side="left", fill="x", expand=True)
                item["lbl_dot"].pack(side="right")
                item["lbl_sub"].pack(fill="x", pady=(2, 0))
            for child in list(self._items_container.winfo_children()):
                if child in self._section_labels:
                    child.pack(fill="x", pady=(8, 2))
                else:
                    child.pack(fill="x", pady=2)

    def _on_hover(self, key: str, hovering: bool) -> None:
        if key == self._active_key:
            return
        self._paint_item(key, hovering=hovering)

    def _paint_item(self, key: str, *, hovering: bool) -> None:
        c = self.colors
        item = self._items[key]
        is_active = (key == self._active_key)

        if is_active:
            bg = c["panel_elevated"] if item["is_workflow"] else c["panel_alt"]
            accent = c["accent"] if item["is_workflow"] else c["cyan"]
            badge_bg = c["accent"] if item["is_workflow"] else c["cyan"]
            badge_fg = "#FFFFFF" if item["is_workflow"] else "#071018"
            title_fg = c["text"]
            sub_fg = c["text_secondary"]
        elif hovering:
            bg = c["panel"]
            accent = c["border_bright"]
            badge_bg = c["panel_elevated"]
            badge_fg = c["text"]
            title_fg = c["text"]
            sub_fg = c["text_muted"]
        else:
            bg = c["bg_sidebar"]
            accent = c["bg_sidebar"]
            badge_bg = c["panel_alt"]
            badge_fg = c["text_secondary"]
            title_fg = c["text_secondary"]
            sub_fg = c["text_muted"]

        for w_name in ("row", "inner", "top_line", "lbl_title", "lbl_dot", "lbl_sub"):
            item[w_name].configure(bg=bg)
        item["indicator"].configure(bg=accent)
        item["lbl_step"].configure(bg=badge_bg, fg=badge_fg)
        item["lbl_title"].configure(fg=title_fg)
        item["lbl_sub"].configure(fg=sub_fg)
