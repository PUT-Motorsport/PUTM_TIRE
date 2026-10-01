# Dane lokalne

Ten katalog służy do przechowywania plików pomiarowych `.mat` oraz eksportów
`.csv` i `.json`. Jego zawartość (poza tym opisem) jest ignorowana przez Git.

Aplikacja `app.py` wczytuje pliki `.mat` wybrane w oknie dialogowym; nie wymaga
konkretnych nazw ani położenia danych w tym katalogu. Eksporty można zapisać
tutaj lub w innym wybranym miejscu. Testy korzystają z danych syntetycznych i
nie wymagają lokalnych pomiarów.

Jeśli dane mają zostać świadomie dodane do repozytorium, sprawdź najpierw ich
rozmiar i możliwość udostępnienia, a następnie użyj osobnego rozwiązania do
przechowywania dużych plików.
