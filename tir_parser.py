"""tir_parser.py — Parser plików parametrów opony Pacejka .tir (TYDEX / PAC2002)
================================================================================
Odczytuje standardowe pliki .tir używane w Adams/Car, IPG CarMaker, OptimumLap itp.
Ekstrahuje parametry modelu do słownika liczb zmiennoprzecinkowych oraz metadane opony.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Any, Tuple
from mf51_model import MF51Model, DEFAULT_MF51_PARAMS


def parse_tir_file(filepath: str) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Odczytuje plik .tir i zwraca (parametry_numeryczne, metadane).
    
    Parametry numeryczne to np. {'PCY1': 1.3, 'PDY1': 1.65, ...}.
    Metadane to sekcje, nazwa opony, promień, wersja, komentarze.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    return parse_tir_content(content, fallback_name=os.path.basename(filepath))


def parse_tir_content(content: str, fallback_name: str = "Tire_Model") -> Tuple[Dict[str, float], Dict[str, str]]:
    """Parsuje zawartość tekstową pliku .tir."""
    parameters: Dict[str, float] = dict(DEFAULT_MF51_PARAMS)
    metadata: Dict[str, str] = {
        "TIRE_NAME": fallback_name,
        "FILE_TYPE": "tir",
        "PROPERTY_FILE_FORMAT": "PAC2002",
        "UNITS": "meter / newton / radians",
    }

    current_section = ""
    lines = content.splitlines()

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Komentarze nagłówkowe z nazwą opony
        if "TIRE_VERSION" in line.upper() or "COMMENT" in line.upper():
            parts = line.split(":", 2)
            if len(parts) >= 3:
                val = parts[2].strip().strip("'\"")
                if val:
                    metadata["COMMENT"] = val
                    if metadata["TIRE_NAME"] == fallback_name:
                        metadata["TIRE_NAME"] = val

        # Komentarze zaczynające się od ! lub $
        if line_clean.startswith("!") or line_clean.startswith("$"):
            # Sprawdź czy w komentarzu nie ma nazwy opony
            if "PROPERTY FILE:" in line_clean.upper():
                name_part = line_clean.split(":", 1)[1].strip()
                if name_part:
                    metadata["TIRE_NAME"] = name_part
            continue

        # Sekcje np. [LATERAL_COEFFICIENTS]
        sec_match = re.match(r"^\[([A-Za-z0-9_]+)\]", line_clean)
        if sec_match:
            current_section = sec_match.group(1).upper()
            continue

        # Usunięcie komentarzy w linii (po znaku $ lub !)
        code_part = line_clean.split("$")[0].split("!")[0].strip()
        if "=" not in code_part:
            continue

        key, val_str = code_part.split("=", 1)
        key = key.strip().upper()
        val_str = val_str.strip().strip("'\"")

        # Zapisz metadane
        if key in ("FILE_TYPE", "FILE_VERSION", "FILE_FORMAT", "TYRESIDE", "PROPERTY_FILE_FORMAT"):
            metadata[key] = val_str
            continue
        if key == "COMMENT" or key == "TIRE_NAME":
            metadata["TIRE_NAME"] = val_str
            continue

        # Próba konwersji na float
        try:
            val_float = float(val_str)
            parameters[key] = val_float
        except ValueError:
            metadata[key] = val_str

    # Sprawdzenie nazwy opony
    if metadata.get("TIRE_NAME") == fallback_name and "COMMENT" in metadata:
        metadata["TIRE_NAME"] = metadata["COMMENT"]

    return parameters, metadata


def load_tir_to_model(filepath: str) -> Tuple[MF51Model, Dict[str, str]]:
    """Ładuje plik .tir bezpośrednio do instancji kalkulatora MF51Model."""
    params, meta = parse_tir_file(filepath)
    model = MF51Model(p=params)
    return model, meta
