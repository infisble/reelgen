# reelgen — ідея → вертикальне відео 8–30 с

PoC агента для тестового HOLYWATER (AI Engineer). На вхід 1–3 речення, на виході `final.mp4`
1080×1920 з озвученою реплікою **дослівно** як у вхідному тексті, субтитрами та 2 шотами.

```
parse → script → cast → voice → images → render → assemble → qa
 код     LLM      еталони  TTS+STT  image+judge  ffmpeg   ffmpeg     probe+STT
                  героїв          (з референсами героїв)
```

Відповіді на Частини 1–3, 5 і «що я зрізав» — у [NOTES.md](NOTES.md).

## Запуск

Потрібно: Python 3.11+. ffmpeg ставити не треба (бінарник приходить з `imageio-ffmpeg`).

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate на Linux/macOS
pip install -e ".[dev]"
```

**Без ключів (demo):** шаблонний сценарій, плейсхолдер-кадри, безкоштовний голос edge-tts (потрібен інтернет).

```bash
python -m reelgen --demo "Нічна кав'ярня під дощем. Дівчина тихо каже: «Я повернулася, бо тільки тут мене ніхто не питає, чому я плачу.»"
```

**Повний режим (OpenAI):** `cp .env.example .env`, вписати `OPENAI_API_KEY`, далі:

```bash
python -m reelgen "A lighthouse keeper sees a ghost ship. He whispers: \"Not tonight, old friend. Not tonight.\"" --out out/lighthouse.mp4
```

Ключ: лише `OPENAI_API_KEY` (LLM `gpt-5-mini` для сценарію і vision-оцінки кадрів, `gpt-4o-mini-tts`,
`gpt-transcribe` для перевірки дослівності, `gpt-image-2.5-flare` для кадрів). Моделі можна змінити через env, див. `.env.example`.

**Без жодних ключів, повністю автоматично (Claude Code пише сценарій):** потрібен встановлений і залогінений
[Claude Code](https://claude.com/claude-code). Сценарний крок іде через `claude -p --json-schema`, персонажі
малюються шаблонним 2D-рендером, голос — edge-tts.

```bash
python -m reelgen --demo --renderer puppet --writer claude "Помідор-ведучий новин на кухні оголошує жахливу новину огірку. Він каже: «Термінові новини: з понеділка нас усіх закатують у банки!» Огірок шепоче: «А я ж тільки почав жити...»"
```

### Як задати репліку
- Текст у лапках (`"..."`, `«...»`, `“...”`, `„...“`) звучить дослівно. Кілька лапок означають кілька реплік; голоси розподіляє LLM.
- Якщо лапок немає, весь вхідний текст читається дослівно як закадровий голос.

### Інші команди
```bash
python -m reelgen --resume <run_id>                      # продовжити впалий запуск з місця падіння
python -m reelgen --resume <run_id> --from-stage images  # перегенерувати з етапу (незмінні шоти/репліки беруться з кешу)
python -m reelgen.evals run --label v2 [--demo]          # регресійний прогін на evals/ideas.jsonl
python -m reelgen.evals compare evals/results/v1.json evals/results/v2.json
python scripts/check.py                                  # lint + format + tests (gate)
```

Код виходу: `0` означає QA пройдено, `2` означає `needs_review` (відео є, але гейт якості не пройдено, потрібна людина), `1` означає помилку.

## Що лежить у run-директорії
```
runs/<run_id>/
  state.json        стан етапів (status, attempts, duration, outputs) + лічильники викликів/ретраїв
  events.jsonl      лог подій (спроби, вердикти STT/судді, помилки)
  input.json        ParsedInput: репліки, витягнуті кодом, мова
  script.json       ScriptPlan від LLM (структурований вихід, репліки лише за індексом)
  screenplay.json   людиночитний сценарій (текст реплік підставлено кодом)
  voice/line_N.wav|json   озвучка + результат STT-перевірки
  images/shot_N.png|json  кадр + вердикт судді (+ відкинуті спроби _tryK)
  render/           аудіотаймлайни, субтитри, кліпи шотів
  final.mp4, report.json
```

## Приклади
| Файл | Що це |
|---|---|
| `examples/cabbage_drama.mp4` | Мама-капуста і син-буряк, кухня, 2 шоти, 2 репліки |
| `examples/potato_garden.mp4` | Дід-картопля і онука-морквина, нічний город, 2 шоти, 2 репліки |
| `examples/tomato_claude.mp4` | Помідор-ведучий новин і огірок. **Повністю автоматично:** сценарій написав Claude Code (`--writer claude`) |

**Як вони зроблені (чесно):** без API-ключа, режим `--renderer puppet`. Персонажі — шаблонні 2D-ляльки,
намальовані кодом (`reelgen/puppet.py`), а не згенеровані моделлю. Рот рухається за гучністю голосу, очі
моргають, брови й посмішка йдуть від `delivery` репліки. Для `cabbage` і `potato` сценарний крок (план у форматі `ScriptPlan`) замість LLM
виконав Claude Code як агент у сесії розробки: `examples/plans/*.json`. Для `tomato` план згенерував сам пайплайн через `--writer claude`. План пройшов той самий валідатор. Решта пайплайна
справжня: репліки з лапок, озвучка (edge-tts, емоція через темп і висоту голосу), таймінг, субтитри, QA.

Відтворити:
```bash
python -m reelgen --demo --renderer puppet --plan examples/plans/cabbage.json \
  "Мама-капуста на кухні над каструлею борщу дивиться на винуватого сина-буряка. Вона кричить: «Я тебе виростила на цій грядці, а ти зрадив мене з олів'є!» Буряк зітхає: «Мамо, це був лише один салат.»"
```
Коли буде ключ OpenAI, ті самі ідеї треба прогнати без `--demo`/`--renderer`/`--plan`: сценарій пише LLM, герої
генеруються моделлю з еталонами (`cast`), і ці приклади замінюються.

## Структура
- `reelgen/pipeline.py`: етапи, раннер, ретраї й кеш
- `reelgen/models.py`: контракти даних між етапами
- `reelgen/verbatim.py`: витяг реплік, нормалізація, WER
- `reelgen/script_rules.py`: валідатор плану LLM (repair loop)
- `reelgen/providers/`: OpenAI та demo-провайдери за спільними Protocol-інтерфейсами
- `reelgen/media.py`: ffmpeg (Ken Burns, субтитри, склейка, probe)
- `reelgen/puppet.py`: шаблонний 2D-рендер персонажів (режим без генеративних моделей)
- `reelgen/evals.py`: eval-раннер
- `CLAUDE.md`, `.claude/settings.json`, `scripts/check.py`: harness для кодинг-агентів
