# diploma_thesis_python

Numerika k diplomové práci o hyperboloidální evoluci. Česky: jak nacpat nekonečno
do konečné mřížky a tvářit se, že je to úplně normální.

Hyperboloidální řezy jsou všude prostorupodobné, ale k nekonečnu se přimykají podél
světelných kuželů, takže budoucí nulové nekonečno (𝓘⁺) leží uvnitř výpočetní oblasti.
Žádná umělá vnější hranice, žádné „tady svět končí, odražte se prosím zpátky“.
Zatím vlnová rovnice ve sférické symetrii; Einsteinovy rovnice přijdou na řadu, až se
vlnová rovnice přestane vzpouzet.

## Co tu je

| Kde | Co |
|---|---|
| `src/` | derivace (centrální, 6. řád), RK4, Kreiss-Oligerova disipace, pravé strany, verifikace |
| `*.py` v kořeni | analýzy (řád schématu, chyby, dlouhé běhy) a skripty, které generují obrázky a tabulky do textu práce |
| `notebooks/` | notebooky; ty v `colab_ntbks/` se stahují z Colabu a nikdo je nepřeformátovává |
| `docs/` | `rovnice.md` – rovnice, ze kterých se vychází; `toread.md` – seznam četby (zatím jedna položka, optimismus vydržel přesně jeden řádek) |

## Spuštění

Python ≥ 3.12, vlastní venv, nic do systémového Pythonu.

```
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate         # Linux
pip install numpy matplotlib sympy
pip install --group dev           # ruff + Jupyter, vyžaduje pip >= 25.1
```

Repo se neinstaluje jako balíček, `src` se importuje jako `src.*`. Skripty se proto
spouštějí z kořene repa:

```
python we_1d_order_analysis.py
```

Skripty `*_outputs.py` zapisují do `../diploma_thesis_tex/figures`, tedy počítají
s textem práce v sousedním adresáři. Kdo ho tam nemá, předá jiný adresář jako argument.

## Upozornění

- Stabilita není zaručená, je naměřená. Na konkrétních parametrech, v konkrétní dny.
- SBP znamená summation by parts, ne „something breaks probably“. Většinou.
- Když to bouchá u počátku, odpověď je nejspíš větší sigma v Kreiss-Oligerově
  disipaci. Když to bouchá v 𝓘⁺, odpověď je „to je na diplomku, ne na README“.
- Kód je psaný pro diplomku, ne pro produkci. Kdo na něm staví něco důležitého,
  má jiné problémy.
