# Agent Lab

## Opis projektu
Agent Lab to demonstracyjny lokalny agent konwersacyjny działający w trybie narzędziowym (ReAct). Łączy modele językowe uruchamiane na CPU z lekkim RAG‑iem TF‑IDF i pamięcią roboczą, aby realizować zadania użytkownika krok po kroku.

### Kluczowe funkcjonalności
- Planowanie działań w stylu AGENT (cel → plan → kroki → wynik).
- Zewnętrzne narzędzia: wyszukiwarka KB, odczyt plików, kalkulator, zapisywanie notatek, zapytania HTTP, wyszukiwarka webowa oraz bezpieczne uruchamianie kodu Python.
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
- `AGENT_SEARXNG_URL` – baza adresu instancji SearxNG z API JSON (np. `https://searxng.example.com`).
- `AGENT_SEARXNG_API_KEY` – opcjonalny token Bearer do uwierzytelniania żądań (jeśli wymaga tego instancja).
- `AGENT_SEARXNG_LANGUAGE`, `AGENT_SEARXNG_SAFESEARCH` – nadpisanie języka wyników i poziomu safe-search.
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS` – optymalizacja wydajności CPU.

Przykład konfiguracji dla alternatywnego modelu i wydajności CPU:
```bash
export AGENT_MODEL="google/gemma-2-2b-it"
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
```

## Workflow interakcji
1. Użytkownik przekazuje cel lub pytanie (opcjonalnie poprzedzone `cel:`).
2. Agent tworzy plan działania i w razie potrzeby wywołuje narzędzia (`search_kb`, `read_kb_file`, `calc`, `python_run`, `http_get`, `save_note`).
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
- `uruchom w sandboxie python_run kod liczący średnią z listy`

### Narzędzie `python_run`

`python_run` umożliwia uruchamianie krótkich skryptów Python w kontrolowanym środowisku:

- kod jest parsowany do AST i walidowany (blokada `import`, dostępów do atrybutów specjalnych oraz wywołań niebezpiecznych funkcji),
- wykonanie odbywa się z ograniczonym zbiorem wbudowanych funkcji oraz modułów numerycznych (`math`, `statistics`, `fractions`, `decimal`, `itertools`, `functools`),
- wynik zwraca przechwycone `stdout` oraz końcowe wartości zmiennych (jako `repr`).

Przykład wywołania narzędzia:

```json
{"tool_name": "python_run", "arguments": {"code": "import math\nprint(math.sqrt(9))"}}
```

Jeśli walidator wykryje niedozwolone konstrukcje, narzędzie zwróci komunikat błędu, a kod nie zostanie wykonany.
- `wyszukaj w sieci "nowości ML" narzędziem web_search`

### Integracja z SearxNG

Aby korzystać z narzędzia `web_search`, wskaż instancję SearxNG udostępniającą API JSON:

```bash
export AGENT_SEARXNG_URL="https://searxng.example.com"
# opcjonalnie, jeśli instancja wymaga tokenu
export AGENT_SEARXNG_API_KEY="sekretny_token"
```

Agent wywoła endpoint `GET /search?format=json` i zmapuje wyniki na krótkie streszczenia (tytuł + snippet + adres URL).

## Testy
Aby uruchomić testy jednostkowe:
```bash
pytest
```
