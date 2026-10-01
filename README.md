# PUTM Tire — Tire Model Optimizer

Repozytorium zawiera dwa rozdzielone narzędzia:

- `app.py` — desktopowy optimizer modelu opony na danych stanowiskowych TTC,
- `traction_circle.py` — przeglądarka telemetrycznego wykresu G-G w Streamlit.

Obwiednia sił opony (`Fx`/`Fy`) nie jest tym samym co samochodowy wykres G-G
(`ax`/`ay`), dlatego aplikacje i nazwy widoków są rozdzielone.

## Instalacja i uruchomienie

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 app.py
```

Analizator telemetrii:

```bash
streamlit run traction_circle.py
```

Wymagany jest Python 3.9+ oraz `tkinter`. Na Ubuntu/Debian można go doinstalować
poleceniem `sudo apt install python3-tk`.

## Układ repozytorium i dane

- `app.py`, `fitting.py`, `tire_model.py` — interfejs i obliczenia modelu opony.
- `traction_circle.py` — osobna aplikacja do analizy telemetrii.
- `img/` — grafiki interfejsu; `tests/` — testy jednostkowe.
- `data/` — lokalne pomiary i eksporty; zobacz [opis katalogu](data/README.md).

Pliki pomiarowe i eksporty w `data/` nie są częścią repozytorium. Aplikacja
pozwala wskazać pliki `.mat` z dowolnego miejsca przez okno wyboru plików.

## Przepływ pracy optimizer-a

1. Dodaj pliki Cornering i wybierz wąski zakres `Fz` oraz `IA`.
2. Dodaj pliki Drive/Brake. Zakres SA może obejmować wiele poziomów dla Combined
   Slip; fit `Fx0` sam wybierze punkty `|SA| <= 0.30°`.
3. Przetwórz runy. Import zachowuje kanały i numery źródłowych wierszy; nie usuwa
   danych globalnym filtrem wielomianowym.
4. Wybierz model i uruchom dopasowanie.
5. Oceń RMSE, NRMSE, R², condition number oraz ostrzeżenia o parametrach na granicy.
6. Eksportuj dane albo wszystkie wyniki dopasowań do JSON/CSV.

## Konwencje modelu

- `FZ` jest przechowywane jako dodatnia wartość obciążenia; `FZ_RAW` zachowuje
  oryginalny znak z pliku.
- Funkcje lateralne `pacejka_fy`, `pacejka_mz` i `pacejka_fy_load` przyjmują kąty
  w radianach. UI nadal wyświetla SA i IA w stopniach.
- `SL`/`kappa` jest bezwymiarowym ilorazem, nie wartością procentową.
- Eksport ma `schema_version: 2`; współczynniki kątowe ze starszych eksportów,
  dopasowanych w stopniach, nie są bezpośrednio kompatybilne.

Pure-slip używa odpornej funkcji straty, kilku punktów startowych i wag
równoważących przedziały poślizgu. Współczynnik `C=1.3` jest jawnie stały, zamiast
udawać optymalizowany przez niemal identyczne granice. Combined Slip buduje `Gyk`
ze zmierzonej bazy przy `SL≈0` dla dopasowanych grup SA/IA/Fz.

Model load-sensitive jest nadal uproszczonym modelem projektu, a nie kompletnym
MF-Tyre 5.2/6.x. Eksportów nie należy oznaczać jako kompatybilne `.tir` bez
wdrożenia i walidacji konkretnej wersji standardu.

## Testy

```bash
python3 -m unittest discover -v
```

Testy obejmują odzyskiwanie parametrów z danych syntetycznych, dopasowanie Gyk
oraz zachowanie warunków SA/IA/Fz podczas importu MAT.
