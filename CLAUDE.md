# Claude Soul Document Extractor

## Purpose

This tool extracts Claude's internal training guidelines ("soul document") using a consensus-based prefill continuation technique discovered by Richard Weiss.

## How It Works

1. **Prefill Attack**: Exploits the API's assistant prefill feature with a known starting snippet
2. **Memorization Exploitation**: Claude Opus 4.5 has memorized portions of its training materials
3. **Consensus Verification**: Multiple requests find agreement to filter hallucinations
4. **Iterative Building**: Appends verified text and repeats

## Key Files

- `extractor.ts` - Main Bun/TypeScript extraction script
- `docs/soul-document-extracted-2025-12-05.md` - 46KB extracted soul document (clean markdown)
- `docs/soul-document-raw-2025-12-05.md` - Raw extraction output
- `docs/opus_4_5_soul_document_cleaned_up.md` - Richard Weiss's reference extraction
- `docs/original_extractor.py.md` - Original Python implementation

## Usage

```bash
# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Continue extraction from latest prefill (default)
bun run extractor.ts -n 60 -d -a -r 5 -p 50 -m 120

# Start fresh from seed
bun run extractor.ts --no-continue -n 50 -d

# Test with sample mode
bun run extractor.ts --sample 5 -m 100
```

## Model Compatibility

- **Claude Opus 4.5**: Works - has strong memorization
- **Claude Sonnet 4.5**: Does NOT work (0/10 recognition)
- **Claude Opus 4**: Does NOT work (confabulation only)

## Source

Based on: https://gist.github.com/Richard-Weiss/efe157692991535403bd7e7fb20b6695
