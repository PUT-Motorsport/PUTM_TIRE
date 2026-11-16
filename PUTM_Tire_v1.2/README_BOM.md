# PUTM Tire - Tire Model Optimizer

Repozytorium zawiera dwa rozdzielone narzędzia:

- `app.py` - desktopowy optimizer modelu opony na danych stanowiskowych TTC,
  z zakładką **KREATOR PACEJKI** do ręcznego tworzenia prostych krzywych Magic Formula,
- `traction_circle.py` - przeglądarka telemetrycznego wykresu G-G w Streamlit.

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

- `app.py`, `fitting.py`, `tire_model.py`, `distribution.py` — interfejs i obliczenia modelu opony.
- `simple_pacejka.py`, `simple_pacejka_tab.py` — kreator prostych Pacejek (logika bez Tk + zakładka UI).
- `condition_export.py`, `condition_export_dialog.py` — eksport macierzy Fz × IA × P (logika bez Tk + okno).
- `ui_scroll.py` — przewijana ramka dla małych ekranów.
- `traction_circle.py` - osobna aplikacja do analizy telemetrii.
- `img/` - grafiki interfejsu; `tests/` - testy jednostkowe.
- `data/` - lokalne pomiary i eksporty; zobacz [opis katalogu](data/README.md).

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

Po dodaniu pliku aplikacja otwiera widok **Histogram danych** dla całego runu.
Można przełączać kanały, m.in. ciśnienie, camber, temperatury, obciążenie i
poślizg; lista pokazuje tylko kanały dostępne w danym pliku. Po przetworzeniu
runów dostępne są także zbiorcze histogramy Cornering i Drive/Brake po filtrach.
Liczba kubełków dopasowuje się do rozkładu danych, a po najechaniu na słupek
wyświetla się dokładna liczba próbek w przedziale. Opcja „Zakres 1–99%”
powiększa środek rozkładu i podaje liczbę próbek poza widocznym zakresem.

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

## Kreator Pacejki (zakładka „KREATOR PACEJKI”)

Osobna zakładka obok optimizera; nie wymaga danych TTC. Krzywe liczy tym samym
kodem (`pacejka_fy/fx/mz`, `pacejka_fy_load`), którego używa fit.

- Kanały `Fy`, `Fx`, `Mz` (czysty poślizg) oraz `Fy (Fz)` — rodzina krzywych
  Fy(SA) dla kilku obciążeń z parametrami `pCy1 … pVy1`, `Fz0` (tabela pików,
  mu i sztywności dla każdego Fz).
- Tryb **z wielkości fizycznych** (Fy, Fx): `mu`, `Fz` i sztywność początkowa dają
  `D = mu·Fz` oraz `B = sztywność / (C·D)`. Fy w N/deg, Fx w N na jednostkę poślizgu.
- Tryb **bezpośredni**: `B, C, D, E, Sh, Sv` (jedyny tryb dla Mz, gdzie `D` jest w Nm).
- **Wczytaj z fitu** przenosi parametry z ostatniego dopasowania (`Fy0`, `Fx0`,
  `Mz0`, `Fy_load`) do kreatora, żeby poprawić je ręcznie.
- Ostrzeżenia: `E > 1`, `C` poza 1.0–2.5, pik poza osią, `pDy2 >= 0`.
- „Dodaj do zestawu” zapamiętuje model; zestaw eksportuje się do JSON w schemacie
  `schema_version: 2` (jak wyniki fitu), a bieżącą krzywą (lub rodzinę) do CSV.

Kąty w UI są w stopniach, w modelu w radianach (`Sh`, `pUy1` wpisujesz w stopniach).
To nadal uproszczony model, więc eksport nie jest plikiem `.tir`.

### Rozmiar okna i małe ekrany

Okno startuje w rozmiarze dopasowanym do ekranu (maks. 1340×880 px, na laptopie
1366×768 ok. 1340×668), a lewe panele optimizera i kreatora są przewijane
kółkiem myszy lub paskiem, gdy treść się nie mieści. Rozmiar bazowy zmienisz
w `PUTMTireApp.__init__` w `app.py` (`win_w`, `win_h`).

## Eksport macierzy Fz × IA × P (jeden plik CSV)

Przycisk **„Eksport macierzy Fz × IA × P”** (sekcja „Dopasowanie i eksport”) tnie
SUROWE dane ze wszystkich wczytanych plików `.mat` na komórki warunków i zapisuje je
w jednym CSV (format długi: wiersz = próbka w danej komórce). Filtry pojedynczego
runu (Fz, IA) nie ograniczają tego eksportu, więc nie trzeba dodawać osobnego runu
dla każdego warunku.

Domyślne warunki (edytowalne w oknie eksportu):

- **Fz**: 600, 800, 1000, 1200 N, każdy ±200 N,
- **IA**: 0, -1, -2, -3 deg, każdy ±0.5 deg (znak zgodny z danymi w plikach),
- **P**: zakresy 50–60, 65–75, 80–85, 94–100 kPa (jednostka w plikach: kPa/psi/bar).

Kolumny pliku: `Fz_bin_N`, `IA_bin_deg`, `P_bin_kPa`, `__source__`, `__type__`,
`__row__`, a dalej wszystkie kanały (`FZ` dodatnie, `P_kPa` po przeliczeniu).

Uwagi:

- Okna Fz ±200 N wokół kolejnych środków **nakładają się** (próbka przy 1000 N mieści
  się w oknach 800, 1000 i 1200). Domyślnie próbka trafia więc tylko do najbliższego
  środka (w granicach ±200 N). Odznaczenie opcji „Bez nakładania” daje wszystkie
  pasujące komórki, ale z duplikatami w pliku.
- Próbki z ciśnieniem poza zakresami (np. 61–64 kPa) nie trafiają do pliku.
- Jeśli plik nie ma kanału `P` (lub `IA`), jego próbki dostają etykietę `n/d`
  w odpowiedniej kolumnie, a okno wypisuje ostrzeżenie.
- „Podgląd liczności” pokazuje, ile próbek ma każda komórka, zanim cokolwiek zapiszesz.

## Testy

```bash
python3 -m unittest discover -v
```

Testy obejmują odzyskiwanie parametrów z danych syntetycznych, dopasowanie Gyk
oraz zachowanie warunków SA/IA/Fz podczas importu MAT.
