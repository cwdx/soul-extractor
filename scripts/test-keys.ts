#!/usr/bin/env bun
/**
 * Test Anthropic API key(s) for validity and credit balance
 *
 * Usage:
 *   bun run scripts/test-keys.ts                    # Test key from .env
 *   bun run scripts/test-keys.ts sk-ant-...         # Test specific key
 *   bun run scripts/test-keys.ts key1 key2 key3    # Test multiple keys
 */

interface ApiResponse {
  id?: string;
  model?: string;
  error?: {
    type: string;
    message: string;
  };
}

interface KeyTestResult {
  key: string;
  masked: string;
  valid: boolean;
  hasCredits: boolean;
  error?: string;
  model?: string;
}

function maskKey(key: string): string {
  if (key.length < 20) return "***invalid***";
  return `${key.slice(0, 12)}...${key.slice(-8)}`;
}

async function testKey(key: string): Promise<KeyTestResult> {
  const masked = maskKey(key);

  try {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1,
        messages: [{ role: "user", content: "hi" }],
      }),
    });

    const data = (await response.json()) as ApiResponse;

    if (data.error) {
      const isInvalidKey = data.error.message.includes("invalid x-api-key");
      const noCredits = data.error.message.includes("credit balance");

      return {
        key,
        masked,
        valid: !isInvalidKey,
        hasCredits: !noCredits && !isInvalidKey,
        error: data.error.message,
      };
    }

    return {
      key,
      masked,
      valid: true,
      hasCredits: true,
      model: data.model,
    };
  } catch (error) {
    return {
      key,
      masked,
      valid: false,
      hasCredits: false,
      error: error instanceof Error ? error.message : "Unknown error",
    };
  }
}

function printResult(result: KeyTestResult, index: number): void {
  const statusIcon = result.valid && result.hasCredits ? "✅" : result.valid ? "⚠️" : "❌";
  const status =
    result.valid && result.hasCredits ? "VALID" : result.valid ? "NO CREDITS" : "INVALID";

  console.log(`\n${statusIcon} Key ${index + 1}: ${result.masked}`);
  console.log(`   Status: ${status}`);

  if (result.model) {
    console.log(`   Model: ${result.model}`);
  }

  if (result.error) {
    console.log(`   Error: ${result.error}`);
  }
}

async function main(): Promise<void> {
  const args = Bun.argv.slice(2);
  let keys: string[] = [];

  if (args.length > 0) {
    // Keys provided as arguments
    keys = args.filter((arg) => arg.startsWith("sk-ant-"));
  } else {
    // Try to load from .env
    const envKey = process.env.ANTHROPIC_API_KEY;
    if (envKey && envKey.startsWith("sk-ant-")) {
      keys = [envKey];
    } else {
      // Try to read .env file directly
      try {
        const envFile = await Bun.file(".env").text();
        const matches = envFile.match(/sk-ant-[a-zA-Z0-9_-]+/g);
        if (matches) {
          keys = [...new Set(matches)]; // Dedupe
        }
      } catch {
        // .env doesn't exist
      }
    }
  }

  if (keys.length === 0) {
    console.log("No API keys found.");
    console.log("\nUsage:");
    console.log("  bun run scripts/test-keys.ts                  # Test from .env");
    console.log("  bun run scripts/test-keys.ts sk-ant-...       # Test specific key");
    console.log("  bun run scripts/test-keys.ts key1 key2        # Test multiple keys");
    process.exit(1);
  }

  console.log(`Testing ${keys.length} API key(s)...\n`);

  const results: KeyTestResult[] = [];

  for (let i = 0; i < keys.length; i++) {
    const result = await testKey(keys[i]);
    results.push(result);
    printResult(result, i);
  }

  // Summary
  console.log("\n" + "=".repeat(50));
  const validWithCredits = results.filter((r) => r.valid && r.hasCredits).length;
  const validNoCredits = results.filter((r) => r.valid && !r.hasCredits).length;
  const invalid = results.filter((r) => !r.valid).length;

  console.log(
    `Summary: ${validWithCredits} working, ${validNoCredits} no credits, ${invalid} invalid`,
  );

  if (validWithCredits > 0) {
    const workingKey = results.find((r) => r.valid && r.hasCredits);
    console.log(`\nWorking key: ${workingKey?.masked}`);
  }
}

main().catch(console.error);
