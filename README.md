# PUTM Tire — Formula Student Tire Dynamics Suite

Profesjonalny pakiet narzędzi telemetrycznych i dynamiki pojazdu zespołu **PUT Motorsport**:

- `app.py` — kompleksowy pakiet desktopowy (Tkinter) łączący:
  1. **DANE & TELEMETRIA** (Data Hub z wbudowanym inspektorem kanałów i histogramem rozkładu),
  2. **DOPASOWANIE (OPTIMIZER)** (dopasowanie modeli Pacejki z kartą telemetrii i wskaźnikami jakości R², RMSE, NRMSE),
  3. **MODEL MF 5.1 / PAC2002** (oficjalny model z czułością Fz, camberem i eksportem .tir),
  4. **PORÓWNYWARKA OPON (.TIR)** (wbudowany symultaniczny benchmark 1–4 opon z suwakami obciążenia Fz i pochylenia IA w czasie rzeczywistym),
  5. **ANALIZA SZTYWNOŚCI** (wyznaczanie $C_{F\alpha}$ i strefy liniowej),
  6. **KREATOR 2D (SZKIC)** (ręczne modelowanie prostych krzywych Pacejki),
  7. **CENTRUM EKSPORTU** (jedyne, zunifikowane miejsce generowania arkuszy Excel, macierzy CSV, plików .tir i zapisu projektu).
- `traction_circle.py` — przeglądarka telemetrycznego wykresu G-G w Streamlit.

Obwiednia sił opony (`Fx`/`Fy`) nie jest tym samym co samochodowy wykres G-G (`ax`/`ay`), dlatego aplikacje i nazwy widoków są rozdzielone.

## Instalacja i uruchomienie

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 app.py
```

Analizator telemetrii G-G:

```bash
streamlit run traction_circle.py
```

Wymagany jest Python 3.9+ oraz `tkinter`. Na Ubuntu/Debian można go doinstalować
poleceniem `sudo apt install python3-tk`.

## Struktura aplikacji i przepływ inżynierski (Workflow)

Aplikacja zorganizowana jest w 7 logicznych, niepowtarzających się etapów:

1. **📁 1. DANE & TELEMETRIA (Data Hub)**:
   - Szybkie dodawanie plików pomiarowych `.mat` (Cornering lub Drive/Brake) ze zdefiniowanymi filtrami Fz, IA, SA i poślizgu SL.
   - Tabela załadowanych runów z licznikiem próbek i jednym przyciskiem przetwarzania *„Przetwórz dane (TTC)”*.
   - **Wbudowany Inspektor Kanałów i Histogram**: bezpośredni podgląd rozkładu dowolnego kanału (ciśnienie, camber, temperatury TSTI/C/O, siły, poślizgi) z dynamicznymi kubełkami i interaktywnym dymkiem hover. Wykres histogramu nie koliduje już z wykresem dopasowania modelu.

2. **📈 2. DOPASOWANIE (OPTIMIZER STUDIO)**:
   - Wybór modelu docelowego (Fy0, Fy(Fz) Load-Sensitive, Fx0, Mz0, Combined Slip Gyk, Obwiednia Fx-Fy, Współczynnik tarcia).
   - Przycisk *„URUCHOM DOPASOWANIE (FIT)”* oraz *„Autonomiczny Matrix Fit”* w jednym spójnym kontekście.
   - **Karta Jakości Telemetrii (Scorecard)**: duży wskaźnik $R^2$ (z kodowaniem kolorystycznym: zielony $\ge 0.98$, żółty $\ge 0.90$, czerwony $< 0.90$), NRMSE, RMSE, Condition number oraz sformatowana tabela parametrów z jednostkami inżynierskimi.
   - Przycisk bezpośredniego transferu: *„Prześlij parametry do Modelu MF 5.1 (.TIR)”*.

3. **🏎️ 3. MODEL MF 5.1 / PAC2002 (.TIR)**:
   - Pełna parametryzacja standardu Pacejka 2002 (camber thrust, czułość obciążenia, pneumatic trail).
   - Jednoklikowa synchronizacja: z wyników fita lub automatyczna ekstrakcja Multi-Fz z całego datasetu TTC.
   - Gotowe presety Formula Student (Hoosier R25B 16", Hoosier R20 18", Continental).
   - Symulacja wariacji obciążenia Fz i camberu IA.

4. **🔍 4. PORÓWNYWARKA OPON (.TIR BENCHMARK)**:
   - Wbudowane narzędzie do symultanicznego porównywania 1–4 opon (.tir lub aktywny model z Kreatora MF 5.1).
   - Suwaki w czasie rzeczywistym dla docisku bolidu Fz (200–2500 N) oraz pochylenia Camber IA (-5° do +5°).
   - 7 widoków dynamicznych ($F_y$, $F_x$, $M_z$, ramię śladu $t$, $\mu(F_z)$, $K_y(F_z)$, elipsa tarcia) oraz tabela KPI.

5. **🎯 5. ANALIZA SZTYWNOŚCI ($C_{F\alpha}$)**:
   - Bezpośrednie wyznaczanie sztywności znoszenia $C_{F\alpha}$ [N/deg, N/rad] z danych TTC.
   - Porównanie różnych naciągów, felg (np. 6" vs 7") lub poziomów ciśnień.

6. **📐 6. KREATOR 2D (SZKIC)**:
   - Pomocniczy kalkulator do manualnego kształtowania krzywej Pacejki z parametrów fizycznych ($\mu$, $F_z$, sztywność) lub współczynników $B, C, D, E$.

7. **📊 7. CENTRUM EKSPORTU**:
   - Wszystkie opcje eksportu w jednym, przejrzystym miejscu:
     - 📗 Zaawansowany arkusz Excel Fy(SA) z tabelami, KPI i wykresami,
     - 💾 Oficjalny plik opony `.tir` (PAC2002),
     - 📊 Cięcie macierzy warunków Fz × IA × P do CSV,
     - 📋 Zrzut przefiltrowanych danych pomiarowych (CSV),
     - 📦 Zapis projektu i współczynników (JSON/CSV).

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
