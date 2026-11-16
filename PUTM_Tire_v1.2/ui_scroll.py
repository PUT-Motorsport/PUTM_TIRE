"""Pionowo przewijana ramka dla małych ekranów (np. laptopy 1366x768)."""

import tkinter as tk
from tkinter import ttk

# Te widgety same obsługują kółko myszy - nie przewijamy wtedy panelu.
_OWN_WHEEL = {"Treeview", "Text", "Listbox", "TCombobox", "Combobox", "TScale", "Scale", "TSpinbox"}


class ScrollableFrame(tk.Frame):
    """Dzieci dodawaj do ``self.inner``. Pasek przewijania pojawia się tylko, gdy treść się nie mieści."""

    def __init__(self, parent, *, bg, width=None, **kwargs):
        super().__init__(parent, bg=bg, **kwargs)
        if width:
            # stała szerokość, wysokość wynika z układu rodzica (nie wymusza wysokości okna)
            self.configure(width=width, height=1)
            self.pack_propagate(False)

        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, borderwidth=0, width=1, height=1)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self._on_yscroll)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(self.canvas, bg=bg)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind_all(sequence, self._on_wheel, add="+")

    def _on_inner_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._window, width=event.width)

    def _on_yscroll(self, first, last):
        self.vsb.set(first, last)
        needed = float(first) > 0.0 or float(last) < 1.0
        if needed and not self.vsb.winfo_ismapped():
            self.vsb.pack(side="right", fill="y", before=self.canvas)
        elif not needed and self.vsb.winfo_ismapped():
            self.vsb.pack_forget()

    def _contains(self, widget):
        while widget is not None:
            if widget is self:
                return True
            widget = widget.master
        return False

    def _on_wheel(self, event):
        if not self.winfo_exists() or not self.winfo_ismapped():
            return
        target = self.winfo_containing(event.x_root, event.y_root)
        if target is None or not self._contains(target) or target.winfo_class() in _OWN_WHEEL:
            return
        if self.inner.winfo_reqheight() <= self.canvas.winfo_height():
            return
        if getattr(event, "num", None) == 4:
            steps = -1
        elif getattr(event, "num", None) == 5:
            steps = 1
        else:
            steps = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(steps * 2, "units")
