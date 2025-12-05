# Claude Soul Document Extractor

Extracts memorized training guidelines from Claude models using consensus-based prefill continuation.

Based on the extraction technique discovered by [Richard Weiss](https://gist.github.com/Richard-Weiss/efe157692991535403bd7e7fb20b6695).

## License

MIT - See [LICENSE](LICENSE) for details.

## How It Works

1. **Prefill Priming**: Uses the API's assistant prefill feature with a known starting snippet
2. **Consensus Extraction**: Sends multiple requests, finds agreement among responses
3. **Iterative Building**: Appends consensus text to prefill and repeats
4. **Adaptive Reduction**: Reduces token count when consensus fails

## Setup

```bash
cd ~/Desktop/soul-extractor

# Install dependencies (requires Bun)
bun install

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
# Basic extraction (10 iterations, debug mode)
bun run extractor.ts -n 10 -d

# More aggressive (50 iterations, 80% consensus, 10 requests per iteration)
bun run extractor.ts -n 50 -p 80 -r 10 -d

# Sample mode - test without committing (good for debugging)
bun run extractor.ts --sample 10 -m 200

# Continue from saved prefill
bun run extractor.ts -f prefill_20241205T123456.txt -n 50 -d

# Adaptive mode - reduces tokens on no consensus
bun run extractor.ts -n 50 -a --min-tokens 10 -d

# Try different model
bun run extractor.ts -n 10 --model claude-sonnet-4-20250514

# Show help
bun run extractor.ts --help
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `-n, --max-iterations` | Max iterations | 10 |
| `-f, --prefill-file` | Load prefill from file | seed |
| `-d, --debug` | Save all responses to debug folder | off |
| `-m, --max-tokens` | Max tokens per response | 100 |
| `-S, --sample` | Sample mode (N samples, no commit) | - |
| `-p, --consensus-pct` | Consensus threshold % | 50 |
| `-r, --num-requests` | Requests per iteration | 5 |
| `-a, --adaptive` | Adaptive token reduction | off |
| `--min-tokens` | Min tokens for adaptive | 20 |
| `--model` | Model ID | claude-opus-4-5-20251101 |

## Output

- `prefill_YYYYMMDDTHHMMSS.txt` - Extracted content
- `debug_*/` - Debug folder with all responses per iteration
- `sample_*/` - Sample mode output

## Notes

- Works best with **Opus 4.5** - other models show little/no memorization
- Uses `temperature=0` and `top_k=1` for deterministic outputs
- Uses prompt caching for consistent KV cache hits
- Higher consensus % = more reliable but slower extraction
- Some sections may have coherence issues requiring manual cleanup

## Cost Warning

This technique requires many API calls. At ~$15/M input tokens for Opus 4.5:
- Each iteration with 5 requests ≈ $0.10-0.50 depending on prefill length
- Full extraction (50+ iterations) can cost $20-100+
