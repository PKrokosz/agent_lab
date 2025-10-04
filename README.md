# Agent Lab

## Opis projektu
Agent Lab to demonstracyjny lokalny agent konwersacyjny działający w trybie narzędziowym (ReAct). Łączy modele językowe uruchamiane na CPU z lekkim RAG‑iem TF‑IDF i pamięcią roboczą, aby realizować zadania użytkownika krok po kroku.

### Kluczowe funkcjonalności
- Planowanie działań w stylu AGENT (cel → plan → kroki → wynik).
- Zewnętrzne narzędzia: wyszukiwarka KB, odczyt plików, kalkulator, zapisywanie notatek oraz zapytania HTTP.
- Lokalna pamięć konwersacji (`memory/state.json`) oraz notatnik (`memory/notes.md`).
- Prosty RAG oparty o TF‑IDF na plikach `./kb/*.txt`.
- Konfigurowalny model językowy (np. TinyLlama, Gemma 2, Qwen 2.5) i parametry generacji przez zmienne środowiskowe.

### Diagram głównych komponentów
```mermaid
flowchart TD
    U[Użytkownik] -->|polecenie| A[agent.py]
    A -->|zapytania| LLM[Model LLM (Transformers)]
    A -->|wyszukiwanie| KB[(Repozytorium KB)]
    A -->|odczyt/zapis| MEM[(Pamięć stanu i notatek)]
    A -->|narzędzia| Tools[Akcje: calc, http_get, save_note]
    KB --> IDX[MiniTfidf]
    IDX --> A
    MEM --> A
```

## Wymagania i instalacja
1. Utwórz i aktywuj środowisko wirtualne:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Zaktualizuj `pip` i zainstaluj zależności:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Przygotuj katalogi na bazę wiedzy i pamięć:
   ```bash
   mkdir -p kb memory
   ```

## Uruchomienie
Podstawowy start agenta:
```bash
python agent.py
```
Pierwsze uruchomienie może pobrać i zbuforować wybrany model.

### Konfiguracja przez zmienne środowiskowe
Najważniejsze zmienne (wszystkie opcjonalne):
- `AGENT_MODEL` – identyfikator modelu w HuggingFace Hub (np. `google/gemma-2-2b-it`).
- `AGENT_MAX_NEW_TOKENS`, `AGENT_MAX_INPUT_TOKENS`, `AGENT_MAX_HISTORY_TOKENS` – limity tokenów.
- `AGENT_TEMPERATURE`, `AGENT_TOPP` – kontrola losowości generacji.
- `AGENT_HISTORY_TURNS`, `AGENT_MAX_STEPS` – długość historii i liczba kroków planu.
- `AGENT_SEED` – deterministyczność.
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS` – optymalizacja wydajności CPU.

Przykład konfiguracji dla alternatywnego modelu i wydajności CPU:
```bash
export AGENT_MODEL="google/gemma-2-2b-it"
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
```

## Workflow interakcji
1. Użytkownik przekazuje cel lub pytanie (opcjonalnie poprzedzone `cel:`).
2. Agent tworzy plan działania i w razie potrzeby wywołuje narzędzia (`search_kb`, `read_kb_file`, `calc`, `http_get`, `save_note`).
3. Wynik końcowy jest zwracany po maksymalnie `AGENT_MAX_STEPS` krokach.
4. Historia konwersacji i notatki są zapisywane w `memory/` i mogą być ponownie wykorzystane.

### Wskazówki dotyczące KB i pamięci
- Pliki w katalogu `kb/` (np. `kb/dokument.txt`) są automatycznie indeksowane przez `MiniTfidf`; aktualizacja katalogu wymaga ponownego uruchomienia agenta.
- Pamięć długoterminowa jest przechowywana w `memory/notes.md`; użyj polecenia `save_note`, aby dodać nowe notatki.
- `memory/state.json` gromadzi historię rozmów – usunięcie pliku resetuje kontekst.

### Przykładowe polecenia użytkownika
- `cel: przygotuj streszczenie najnowszych notatek z KB`
- `wyszukaj w KB informacje o module MiniTfidf`
- `oblicz 3 * (4 + 5) narzędziem calc`
- `zapisz notatkę o nowym pomyśle na funkcję`
- `pobierz stronę https://example.com narzędziem http_get`

## Testy
Aby uruchomić testy jednostkowe:
```bash
pytest
```
