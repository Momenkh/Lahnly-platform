# Lahnly Backend — Security Reference

> **How to use this document**
> Each finding has an ID (C-1, H-2, M-5, L-1…), a severity, the exact source location,
> the attack scenario, and the specific fix. Work through the priority order at the bottom
> when planning a hardening sprint. Mark findings fixed by appending ✅ and the date.

---

## Severity Scale

| Level | Meaning |
|-------|---------|
| **CRITICAL** | Exploitable in a deployed app; fix before any real traffic |
| **HIGH** | Likely exploited under moderate load or targeted attack; fix before first users |
| **MEDIUM** | Requires attacker effort or specific conditions; fix before public launch |
| **LOW** | Defense-in-depth; informational leakage; schedule in next hardening sprint |

---

## Quick Summary

| ID | Severity | Title | File |
|----|----------|-------|------|
| C-1 | CRITICAL | DEV_BYPASS has no production guard | `src/auth/clerk.guard.ts` |
| C-2 | CRITICAL | Unrestricted CORS | `src/main.ts` |
| C-3 | CRITICAL | Unsanitized filename in S3 key | `src/songs/songs.service.ts` |
| C-4 | CRITICAL | Instrument arg not re-validated in queue processor | `src/queue/transcription.processor.ts` |
| H-1 | HIGH | No rate limiting on any endpoint | `src/app.module.ts` |
| H-2 | HIGH | No security headers (Helmet missing) | `src/main.ts` |
| H-3 | HIGH | Pagination limit unbounded | controllers |
| H-4 | HIGH | rawBody buffering enabled globally | `src/main.ts` |
| H-5 | HIGH | UpdateNotesDto has no size or shape validation | `src/results/dto/update-notes.dto.ts` |
| H-6 | HIGH | Presigned URL expiry too long (1 hr) | `src/results/results.service.ts` |
| M-1 | MEDIUM | Publications count query uses wrong parameter index | `src/publications/publications.service.ts` |
| M-2 | MEDIUM | No file magic-byte validation (MIME spoofing) | `src/songs/songs.controller.ts` |
| M-3 | MEDIUM | No malware scanning on uploaded files | `src/songs/songs.controller.ts` |
| M-4 | MEDIUM | User profile not updated on repeat logins | `src/auth/clerk.guard.ts` |
| M-5 | MEDIUM | Duplicate in-flight jobs not prevented | `src/jobs/jobs.service.ts` |
| M-6 | MEDIUM | Temp files persist on worker process crash | `src/queue/transcription.processor.ts` |
| M-7 | MEDIUM | No audit logging for security-sensitive actions | all services |
| M-8 | MEDIUM | No Redis authentication | `src/app.module.ts`, `.env` |
| L-1 | LOW | Worker error messages leak internal paths | `src/queue/transcription.processor.ts` |
| L-2 | LOW | Swagger enabled in production | `src/main.ts` |
| L-3 | LOW | No HTTPS enforcement at app level | `src/main.ts` |
| L-4 | LOW | Public routes not centrally enumerated | all controllers |

---

## Critical Findings

### C-1 — DEV_BYPASS Has No Production Guard

**File:** `src/auth/clerk.guard.ts` lines 39–45  
**OWASP:** A07:2021 – Identification and Authentication Failures

**What it is:**
`DEV_BYPASS=true` causes the guard to skip ALL Clerk JWT validation. Any HTTP request carrying the header `x-dev-user-id: <any-string>` is accepted as authenticated. The current `.env` has `DEV_BYPASS=true` enabled. Additionally, the Swagger UI description (`src/main.ts` lines 20–23) publicly documents this mechanism to anyone who reads the API docs.

**Attack scenario:**
An attacker reads the Swagger docs or intercepts a dev API call, discovers the header name, and sends `x-dev-user-id: victim-user-id` to impersonate any user — or uses an arbitrary string to create a new account with 9,999 tokens. This grants full API access with no credential requirement.

**Specific code (clerk.guard.ts:39–45):**
```typescript
if (this.config.get('DEV_BYPASS') === 'true') {
  const devUserId = request.headers['x-dev-user-id'] as string;
  if (!devUserId) throw new UnauthorizedException('...');
  request.user = await this.upsertUser(devUserId, `${devUserId}@dev.local`, 'Dev User', 9999);
  return true;
}
```

**Fix:**
```typescript
if (this.config.get('DEV_BYPASS') === 'true') {
  // Hard block: never run in production
  if (process.env.NODE_ENV === 'production') {
    throw new Error('FATAL: DEV_BYPASS must not be enabled in production');
  }

  const devUserId = request.headers['x-dev-user-id'] as string;
  const allowed = (this.config.get<string>('DEV_ALLOWED_USERS') || 'dev-user-1')
    .split(',').map(s => s.trim());

  if (!devUserId || !allowed.includes(devUserId)) {
    throw new UnauthorizedException('DEV_BYPASS: user ID not in DEV_ALLOWED_USERS');
  }

  this.logger.warn(`[DEV_BYPASS] Authenticated as ${devUserId}`);
  request.user = await this.upsertUser(devUserId, `${devUserId}@dev.local`, 'Dev User', 500);
  return true;
}
```
Also: remove the bypass instructions from the Swagger description in `src/main.ts`. Move them to the internal `README.md` only.

---

### C-2 — Unrestricted CORS

**File:** `src/main.ts` line 11  
**OWASP:** A05:2021 – Security Misconfiguration

**What it is:**
`app.enableCors()` with no arguments allows any origin (`*`) to make cross-origin requests to the API, including requests that carry credentials (tokens) set by the frontend.

**Attack scenario:**
An attacker hosts `https://evil.com` containing JavaScript that calls `https://api.lahnly.com/api/songs` with the victim's auth token. The browser's CORS preflight is approved for any origin, so the response is returned to the attacker's page.

**Fix:**
```typescript
app.enableCors({
  origin: process.env.CORS_ORIGIN || 'http://localhost:5173',
  credentials: true,
  methods: ['GET', 'POST', 'PATCH', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'x-dev-user-id'],
});
```
Add `CORS_ORIGIN=https://app.lahnly.com` to `.env.example` and production config.

---

### C-3 — Unsanitized Original Filename in S3 Key

**File:** `src/songs/songs.service.ts` line 24  
**OWASP:** A03:2021 – Injection / A01:2021 – Path Traversal

**What it is:**
The S3 storage key is constructed as:
```typescript
const key = `songs/${userId}/${Date.now()}-${file.originalname}`;
```
`file.originalname` is the raw browser-supplied filename. It is never sanitized before being embedded in the key.

**Attack scenarios:**
1. **Path traversal in S3:** A filename like `../../admin/config.json` produces the key `songs/uid/1234-../../admin/config.json`. Depending on the S3 client SDK and bucket policy, this may allow reading or overwriting objects outside the user's prefix.
2. **Log corruption:** Unicode control characters or ANSI escape sequences in the filename corrupt log entries and monitoring dashboards.
3. **Downstream path injection:** The queue processor derives local temp file paths from the storage key using `path.basename(audioStorageKey)`. A crafted filename could still influence temp path construction.
4. **DoS via long filename:** A 4,096-character filename causes S3 key validation failures and unexpected error paths.

**Fix:**
```typescript
// Generate a random key; store original name for display only
import { randomUUID } from 'crypto';

function mimeToExt(mimetype: string): string {
  const map: Record<string, string> = {
    'audio/mpeg': 'mp3', 'audio/mp3': 'mp3', 'audio/wav': 'wav',
    'audio/x-wav': 'wav', 'audio/flac': 'flac', 'audio/m4a': 'm4a',
    'audio/mp4': 'm4a', 'audio/x-m4a': 'm4a', 'audio/ogg': 'ogg',
    'audio/webm': 'webm',
  };
  return map[mimetype] ?? 'bin';
}

const ext = mimeToExt(file.mimetype);
const key = `songs/${userId}/${randomUUID()}.${ext}`;
// Store file.originalname in the DB `originalFilename` column for display — never in the key
```

---

### C-4 — Queue Payload Instrument Not Re-Validated in Processor

**File:** `src/queue/transcription.processor.ts` (instrument loop → spawn call)  
**OWASP:** A03:2021 – Injection

**What it is:**
The `instruments` array from the BullMQ job payload is iterated and each value is passed directly as a CLI argument to Python:
```typescript
spawn(pythonBin, ['main.py', audioPath, '--instrument', instrument, '--no-play', '--no-viz'])
```
Validation (the `@IsIn([...])` whitelist) only runs at HTTP time in `CreateJobDto`. The queue processor trusts the payload unconditionally.

**Attack scenario:**
Redis is accessible to an attacker (no password set — see M-8, or via a compromised internal service). They push a crafted job with `instruments: ['--output-document /tmp/pwned']`. Even though `spawn()` without `shell: true` prevents shell metacharacter injection, the arbitrary argument is still passed verbatim to the Python process. A crafted argument matching a real Python flag (e.g., `--config /dev/stdin`) could alter the processor's behavior, read from unexpected sources, or trigger unintended output paths.

**Fix:**
```typescript
const VALID_INSTRUMENTS = new Set(['guitar', 'bass', 'piano', 'vocals', 'drums']);

async process(job: Job<TranscriptionJobPayload>) {
  const { instruments } = job.data;

  for (const instrument of instruments) {
    if (!VALID_INSTRUMENTS.has(instrument)) {
      // Refund tokens, mark job failed
      await this.handleFailure(job.data, `Invalid instrument in payload: ${instrument}`);
      return;
    }
  }
  // ... rest of processing
}
```

---

## High Findings

### H-1 — No Rate Limiting on Any Endpoint

**File:** `src/app.module.ts`, all controllers  
**OWASP:** A04:2021 – Insecure Design

**What it is:**
Zero rate limiting exists. Every endpoint can be called unlimited times per second from any IP.

**Attack scenarios:**
- **Storage exhaustion:** Script uploads 200 MB files in a loop; attacker fills the MinIO bucket at zero cost.
- **Token race condition probing:** Rapid simultaneous `POST /songs/:id/jobs` attempts to defeat the `FOR UPDATE` lock (unlikely to succeed but exhausts DB connections).
- **DDoS on public endpoint:** `GET /publications` requires a DB query; flood with 10,000 req/s to exhaust the pg pool.
- **Brute-force user enumeration:** Systematic `x-dev-user-id` tries across known user patterns (dev mode only, but still).

**Fix:**
```typescript
// app.module.ts
import { ThrottlerModule, ThrottlerGuard } from '@nestjs/throttler';
import { APP_GUARD } from '@nestjs/core';

@Module({
  imports: [
    ThrottlerModule.forRoot([
      { name: 'short', ttl: 1000,  limit: 10  },  // 10 req/s burst
      { name: 'long',  ttl: 60000, limit: 120 },  // 120 req/min sustained
    ]),
  ],
  providers: [{ provide: APP_GUARD, useClass: ThrottlerGuard }],
})
```
Add `@Throttle({ short: { limit: 3, ttl: 60000 } })` on `POST /songs` (upload) and `POST /songs/:id/jobs` for tighter per-user limits.

---

### H-2 — No Security Headers (Helmet Missing)

**File:** `src/main.ts`  
**OWASP:** A05:2021 – Security Misconfiguration

**What it is:**
No `helmet()` middleware. Every API response is missing: `Content-Security-Policy`, `Strict-Transport-Security`, `X-Frame-Options`, `X-Content-Type-Options`, `Referrer-Policy`.

**Attack scenarios:**
- **Clickjacking:** The Swagger UI page (`/docs`) can be embedded in a hostile iframe.
- **MIME sniffing:** Audio file preview responses without `X-Content-Type-Options: nosniff` may be interpreted as HTML by older browsers.
- **Missing HSTS:** First visit over HTTP downgrades are possible if the load balancer is misconfigured.

**Fix:**
```typescript
// main.ts — before any other middleware
import helmet from 'helmet';
app.use(helmet({
  crossOriginResourcePolicy: { policy: 'cross-origin' }, // allow presigned S3 URLs in browser
}));
```
Install: `npm install helmet`

---

### H-3 — Pagination Limit Unbounded

**Files:** `src/users/users.controller.ts`, `src/songs/songs.controller.ts`, `src/publications/publications.controller.ts`  
**OWASP:** A04:2021 – Insecure Design

**What it is:**
The `?limit=` query parameter is accepted as-is and passed directly to `LIMIT $1` in SQL queries:
```typescript
@Query('limit') limit = '20',
// ...
return this.service.findAll(+page, +limit);  // +limit with no cap
```

**Attack scenario:**
`GET /api/publications?limit=999999` triggers a full table scan returning every row, causing memory exhaustion and slow response for all other users.

**Fix (apply to every paginated method):**
```typescript
const safePage  = Math.max(1, parseInt(page)  || 1);
const safeLimit = Math.min(100, Math.max(1, parseInt(limit) || 20));
```
Cap at 100 rows maximum.

---

### H-4 — rawBody Buffering Enabled Globally

**File:** `src/main.ts` line 8  
**OWASP:** A04:2021 – Insecure Design

**What it is:**
`NestFactory.create(AppModule, { rawBody: true })` tells NestJS to buffer the complete raw request body for every route — including the `POST /songs` upload route that accepts files up to 200 MB.

**Attack scenario:**
Each upload request causes NestJS to buffer the raw body in addition to Multer's buffer, doubling peak memory consumption per request. With 5 concurrent 200 MB uploads: 5 × 400 MB = 2 GB RAM spike → OOM kill.

**Fix:**
```typescript
// main.ts — disable global rawBody
const app = await NestFactory.create(AppModule);  // remove { rawBody: true }

// webhooks/webhooks.controller.ts — capture raw body only for this route
import { Request } from 'express';
import * as bodyParser from 'body-parser';

// In WebhooksModule, add middleware:
export class WebhooksModule implements NestModule {
  configure(consumer: MiddlewareConsumer) {
    consumer
      .apply(bodyParser.raw({ type: 'application/json' }))
      .forRoutes({ path: 'webhooks/stripe', method: RequestMethod.POST });
  }
}
```

---

### H-5 — UpdateNotesDto Has No Size or Shape Validation

**File:** `src/results/dto/update-notes.dto.ts`  
**OWASP:** A04:2021 – Insecure Design

**What it is:**
The DTO only validates `@IsArray()`. No constraints on array length, object depth, or field types within each note object.

**Attack scenario:**
`PATCH /api/jobs/:id/results/guitar/notes` with a body of 100,000 note objects, each containing a 10 KB string:
- Server must parse and validate 1 GB of JSON
- PostgreSQL must write a 1 GB JSONB column
- Future reads of this row cause memory spikes in all downstream services

**Fix:**
```typescript
// update-notes.dto.ts
import { IsArray, ArrayMaxSize, ValidateNested, IsNumber, IsString, IsOptional, MaxLength, Min, Max } from 'class-validator';
import { Type } from 'class-transformer';

export class NoteItemDto {
  @IsNumber() @Min(0) @Max(7200)       startTime: number;
  @IsNumber() @Min(0) @Max(7200)       endTime: number;
  @IsNumber() @Min(0) @Max(127)        pitch: number;
  @IsNumber() @Min(0) @Max(1)          @IsOptional() confidence?: number;
  @IsNumber() @Min(1) @Max(127)        @IsOptional() velocity?: number;
  @IsString() @MaxLength(32)           @IsOptional() label?: string;
}

export class UpdateNotesDto {
  @IsArray()
  @ArrayMaxSize(2000)
  @ValidateNested({ each: true })
  @Type(() => NoteItemDto)
  notes: NoteItemDto[];
}
```

---

### H-6 — Presigned URL Expiry Too Long

**Files:** `src/results/results.service.ts` line 73; `src/storage/storage.service.ts` line 46  
**OWASP:** A02:2021 – Cryptographic Failures

**What it is:**
Preview WAV presigned URLs are valid for 3,600 seconds (1 hour). The default `StorageService.presignedGetUrl()` also defaults to 3,600 s.

**Attack scenario:**
A user shares a preview link in a chat message or support ticket. The link is valid for one hour — enough time for it to be forwarded, logged, or accessed by unintended parties. Audio files may contain copyrighted material.

**Fix:**
```typescript
// results.service.ts
async getPreviewUrl(...) {
  const url = await this.storage.presignedGetUrl(result.previewStorageKey, 300); // 5 min
  return { url, expiresInSeconds: 300 };
}

// publications.service.ts — same change for publication previews
previews[instrument] = await this.storage.presignedGetUrl(result.previewStorageKey, 300);

// storage.service.ts — change default
async presignedGetUrl(key: string, expiresInSeconds = 300): Promise<string> {
```

---

## Medium Findings

### M-1 — Publications Count Query Uses Wrong Parameter Index

**File:** `src/publications/publications.service.ts` lines 70–94  
**OWASP:** A03:2021 – Injection (code smell leading to future risk)

**What it is:**
The `findAll` method constructs a `searchClause` string and interpolates it into two separate SQL queries:
```typescript
const searchClause = search ? `AND p.title ILIKE $3` : '';
const params: any[] = search ? [limit, skip, `%${search}%`] : [limit, skip];
// Items query — params: [limit, skip, search%]   →  $3 = search%  ✓
// Count query — params: [search%]                →  $3 = undefined ✗  BUG
```
The count query passes only one parameter but references `$3`, causing a PostgreSQL error when `search` is provided.

Additionally, the practice of string-interpolating SQL fragments (even static ones like `AND p.title ILIKE $3`) is a maintenance hazard. A future developer who adds dynamic content to `searchClause` will create a real SQL injection vulnerability.

**Fix (split into two explicit branches):**
```typescript
async findAll(page: number, limit: number, search?: string) {
  const skip = (page - 1) * limit;

  if (search) {
    const pattern = `%${search}%`;
    const [itemsRes, countRes] = await Promise.all([
      this.db.query(
        `SELECT p.*, json_build_object('id', u.id, 'displayName', u."displayName") as user
         FROM "Publication" p JOIN "User" u ON u.id = p."userId"
         WHERE p."isPublic" = true AND p.title ILIKE $3
         ORDER BY p."publishedAt" DESC LIMIT $1 OFFSET $2`,
        [limit, skip, pattern],
      ),
      this.db.query(
        `SELECT COUNT(*)::int as count FROM "Publication"
         WHERE "isPublic" = true AND title ILIKE $1`,
        [pattern],
      ),
    ]);
    return { items: itemsRes.rows, total: countRes.rows[0].count, page, limit };
  }

  const [itemsRes, countRes] = await Promise.all([
    this.db.query(
      `SELECT p.*, json_build_object('id', u.id, 'displayName', u."displayName") as user
       FROM "Publication" p JOIN "User" u ON u.id = p."userId"
       WHERE p."isPublic" = true
       ORDER BY p."publishedAt" DESC LIMIT $1 OFFSET $2`,
      [limit, skip],
    ),
    this.db.query(
      `SELECT COUNT(*)::int as count FROM "Publication" WHERE "isPublic" = true`,
    ),
  ]);
  return { items: itemsRes.rows, total: countRes.rows[0].count, page, limit };
}
```

---

### M-2 — No File Magic-Byte Validation (MIME Spoofing)

**File:** `src/songs/songs.controller.ts` lines 61–63  
**OWASP:** A03:2021 – Injection

**What it is:**
MIME type validation checks `file.mimetype`, which is provided by the browser — not derived from the actual file bytes. Any client can send `Content-Type: audio/mpeg` with a PNG or executable payload.

**Attack scenario:**
A malicious client uploads a crafted file with a valid-looking MIME type. The file passes the whitelist check and is stored. When the worker downloads and passes it to librosa/soundfile, a malformed audio payload may trigger a parsing vulnerability in the Python dependency.

**Fix:**
```typescript
// songs.controller.ts
import { fromBuffer } from 'file-type';

// After the MIME whitelist check:
const detected = await fromBuffer(file.buffer);
const detectedMime = detected?.mime ?? '';
if (!AUDIO_MIME_TYPES.includes(detectedMime)) {
  throw new BadRequestException(
    `File content does not match an audio format (detected: ${detectedMime || 'unknown'})`
  );
}
```
Install: `npm install file-type`

---

### M-3 — No Malware Scanning on Uploaded Files

**File:** `src/songs/songs.controller.ts`  
**OWASP:** A08:2021 – Software and Data Integrity Failures

**What it is:**
Files are accepted after MIME type check and duration extraction only. No content scanning is performed before storage.

**Attack scenarios:**
- A crafted audio file exploiting a zero-day in librosa, soundfile, or FFmpeg could achieve RCE in the worker container.
- A zip bomb disguised as audio could exhaust worker memory during decompression.

**Fix (phased):**
1. **Immediate (M-2):** Magic-byte validation rejects non-audio binary content.
2. **Production:** Stream uploaded buffer through ClamAV before S3 storage:
   ```typescript
   // Via @nest-modules/clamd or direct TCP socket to clamd
   const scanResult = await clamav.scanBuffer(file.buffer);
   if (scanResult.isInfected) throw new BadRequestException('File failed malware scan');
   ```
3. **Document:** Add `MALWARE_SCAN_ENABLED=false` to `.env.example` as a feature flag.

---

### M-4 — User Profile Not Updated on Repeat Logins

**File:** `src/auth/clerk.guard.ts` lines 77–82  
**OWASP:** A07:2021 – Identification and Authentication Failures

**What it is:**
The upsert uses `ON CONFLICT ("clerkId") DO NOTHING`. Once a user row is created, no subsequent login ever updates their email or display name.

**Consequence:**
If a user changes their email in Clerk, your DB still holds the old email. If the old email is shared by someone else and they join Clerk, you may associate wrong identity data.

**Fix:**
```sql
INSERT INTO "User" (id, "clerkId", email, "displayName", "createdAt")
VALUES ($1, $2, $3, $4, NOW())
ON CONFLICT ("clerkId")
DO UPDATE SET
  email        = EXCLUDED.email,
  "displayName" = EXCLUDED."displayName"
```

---

### M-5 — Duplicate In-Flight Job Submissions Not Prevented

**File:** `src/jobs/jobs.service.ts`  
**OWASP:** A04:2021 – Insecure Design

**What it is:**
No guard prevents a user from submitting a second job for the same song while the first job is still QUEUED or RUNNING.

**Consequences:**
- Double token deduction for the same song
- Two worker slots consumed for identical work
- Race condition between two simultaneous token-deduct transactions (both may succeed if they both read the balance before either commits)

**Fix (inside the transaction, before token deduction):**
```typescript
const { rows: running } = await client.query(
  `SELECT id FROM "ProcessingJob"
   WHERE "songId" = $1 AND "userId" = $2 AND status IN ('QUEUED', 'RUNNING')`,
  [songId, userId],
);
if (running.length > 0) {
  throw new HttpException(
    'A job is already processing for this song. Wait for it to complete before submitting another.',
    HttpStatus.CONFLICT,
  );
}
```

---

### M-6 — Temp Files Persist on Worker Process Crash

**File:** `src/queue/transcription.processor.ts`  
**OWASP:** A04:2021 – Insecure Design

**What it is:**
Temp directories are cleaned up in a `finally` block. A `SIGKILL` or OOM kill skips the finally block, leaving `lahnly-*` directories in `/tmp` containing downloaded user audio files.

**Consequences:**
- Disk exhaustion on the worker node
- User audio persists on disk indefinitely after the job completes
- If the worker host is shared (e.g., multi-tenant Kubernetes node), other processes may access `/tmp/lahnly-*`

**Fix:**
```typescript
// At processor startup — sweep stale temp dirs
async onModuleInit() {
  const tmpDir = os.tmpdir();
  const staleTtlMs = 30 * 60 * 1000; // 30 minutes
  const entries = fs.readdirSync(tmpDir);
  for (const entry of entries) {
    if (!entry.startsWith('lahnly-')) continue;
    const fullPath = path.join(tmpDir, entry);
    try {
      const stat = fs.statSync(fullPath);
      if (Date.now() - stat.mtimeMs > staleTtlMs) {
        fs.rmSync(fullPath, { recursive: true, force: true });
        this.logger.log(`Cleaned stale temp dir: ${fullPath}`);
      }
    } catch { /* skip */ }
  }
}
```
Long-term: stream audio from S3 to a named pipe fed directly to Python's stdin instead of writing to disk.

---

### M-7 — No Audit Logging for Security-Sensitive Actions

**Files:** All services  
**OWASP:** A09:2021 – Security Logging and Monitoring Failures

**What it is:**
No structured log entries exist for auth failures, authorization denials, token deductions, file uploads, or webhook events.

**Consequence:**
An ongoing attack (brute-force enumeration, token draining, unauthorized access attempts) cannot be detected, investigated, or attributed.

**Fix (minimum viable logging):**

| Event | Location | Log level | Fields to log |
|-------|----------|-----------|---------------|
| Auth failure (bad token) | `clerk.guard.ts` | WARN | ip, path, reason |
| Auth failure (missing header) | `clerk.guard.ts` | WARN | ip, path |
| ForbiddenException thrown | each service | WARN | userId, resourceType, resourceId |
| Token deducted | `jobs.service.ts` | LOG | userId, jobId, amount, balance_after |
| File uploaded | `songs.service.ts` | LOG | userId, songId, size, mimeType |
| Stripe webhook received | `webhooks.service.ts` | LOG | eventType, stripeEventId |
| Token granted | `webhooks.service.ts` | LOG | userId, amount, stripeEventId |

---

### M-8 — No Redis Authentication

**Files:** `src/app.module.ts`, `.env`  
**OWASP:** A05:2021 – Security Misconfiguration

**What it is:**
Redis runs with no password. Any process on the same network segment can:
1. Read all BullMQ job payloads (containing `userId`, `audioStorageKey`, `processingJobId`)
2. Push crafted jobs with arbitrary payloads (see C-4)
3. Delete or drain the queue

**Fix:**
```env
# .env
REDIS_PASSWORD=change_me_in_production
```
```typescript
// app.module.ts BullModule config
redis: {
  host: config.get('REDIS_HOST', '127.0.0.1'),
  port: config.get('REDIS_PORT', 6379),
  password: config.get('REDIS_PASSWORD') || undefined,
},
```
In Kubernetes: store the password in a `Secret` and inject via environment variable. Enable Redis `requirepass` in the StatefulSet config.

---

## Low Findings

### L-1 — Worker Error Messages Leak Internal Paths

**File:** `src/queue/transcription.processor.ts`  
**OWASP:** A09:2021 – Security Logging and Monitoring Failures

**What it is:**
The `errorMessage` stored in `ProcessingJob` and returned by `GET /jobs/:id` includes raw details: Python exit codes, temp directory paths (e.g., `/tmp/lahnly-a3f2/audio.wav`), and S3 key prefixes.

**Fix:**
```typescript
// Store generic user-facing message in DB
await this.db.query(
  `UPDATE "ProcessingJob" SET status='FAILED', "errorMessage"=$1, "finishedAt"=NOW() WHERE id=$2`,
  ['Processing failed. Our team has been notified.', processingJobId],
);
// Log full details internally at ERROR level
this.logger.error(`Job ${processingJobId} failed: ${fullError.message}`, fullError.stack);
```

---

### L-2 — Swagger Enabled in Production

**File:** `src/main.ts`  
**OWASP:** A05:2021 – Security Misconfiguration

**What it is:**
Swagger is always active regardless of `NODE_ENV`. In production this exposes: full API schema, all route paths, DTO field names, and (currently) the DEV_BYPASS documentation.

**Fix:**
```typescript
if (process.env.NODE_ENV !== 'production') {
  const config = new DocumentBuilder()...build();
  const document = SwaggerModule.createDocument(app, config);
  SwaggerModule.setup('docs', app, document);
  console.log(`Swagger docs at http://localhost:${port}/docs`);
}
```

---

### L-3 — No HTTPS Enforcement at Application Level

**File:** `src/main.ts`  
**OWASP:** A02:2021 – Cryptographic Failures

**What it is:**
The application has no HSTS header and does not redirect HTTP to HTTPS. This relies entirely on the load balancer / ingress being correctly configured.

**Fix:**
Once Helmet is added (H-2), HSTS is included automatically. Explicitly configure it:
```typescript
app.use(helmet({
  hsts: { maxAge: 31536000, includeSubDomains: true },
}));
```
Document in `k8s/ingress.yaml` that TLS must be terminated at the ingress level.

---

### L-4 — Public Routes Not Centrally Enumerated

**Files:** All controllers  
**OWASP:** A01:2021 – Broken Access Control

**What it is:**
`@Public()` decorators are scattered across controllers with no authoritative list. A reviewer cannot quickly confirm that all `@Public()` routes are intentionally public.

**Current public routes:**
- `GET /api/health`
- `GET /api/publications`
- `GET /api/publications/:id`
- `POST /api/webhooks/stripe`

**Fix:**
```typescript
// clerk.guard.ts — add this comment block to maintain the authoritative list
/**
 * INTENTIONALLY PUBLIC ROUTES (no authentication required):
 *   GET  /api/health
 *   GET  /api/publications
 *   GET  /api/publications/:id
 *   POST /api/webhooks/stripe
 *
 * To add a public route: add @Public() to the handler AND update this list.
 * Code review must verify this list matches the deployed @Public() decorators.
 */
```
Long-term: write a unit test that inspects all NestJS route metadata and asserts the exact set of `@Public()` handlers.

---

## What Is Already Secure

The following areas were audited and found to be correctly implemented. These are called out explicitly so future developers do not introduce regressions.

| Area | Evidence | File |
|------|----------|------|
| **Parameterized SQL throughout** | All queries use `$1, $2, $3` placeholders; no user data concatenated into query strings | all services |
| **Per-resource userId ownership check** | Every service method verifies the requesting user owns the resource before returning or modifying data; zero IDOR vulnerabilities found | `jobs.service.ts`, `results.service.ts`, `publications.service.ts`, `songs.service.ts` |
| **Stripe webhook signature verification** | Uses `stripe.webhooks.constructEvent(rawBody, sig, secret)` — the correct approach | `src/webhooks/webhooks.service.ts` |
| **Token deduction uses `FOR UPDATE` locking** | Balance read inside a transaction with row-level lock prevents double-spend race condition | `src/jobs/jobs.service.ts` lines 36–67 |
| **Mass assignment prevented globally** | `ValidationPipe({ whitelist: true, forbidNonWhitelisted: true })` — extra fields silently stripped | `src/main.ts` |
| **Instrument whitelist at HTTP layer** | `@IsIn(['guitar','bass','piano','vocals','drums'])` in `CreateJobDto` | `src/jobs/dto/create-job.dto.ts` |
| **File buffer never written to API server disk** | Multer configured with memory storage; file exists only as an in-memory Buffer | `src/songs/songs.controller.ts` |
| **DB connection retry logic** | `isConnectionError()` checks code + message + cause with 4 attempts and back-off | `src/prisma/database.service.ts` |
| **Stripe token grant requires verified webhook** | Only `invoice.paid` and `customer.subscription.created` events grant tokens, and only after signature passes | `src/webhooks/webhooks.service.ts` |

---

## Recommended Fix Order

Work through this list top-to-bottom when hardening. Estimated effort is for a single developer familiar with the codebase.

| # | ID | Title | Est. Time |
|---|----|-------|-----------|
| 1 | C-1 | DEV_BYPASS production guard + allowlist | 30 min |
| 2 | C-2 | CORS origin restriction | 5 min |
| 3 | H-2 | Add Helmet | 10 min |
| 4 | C-3 | Replace `originalname` in S3 key with UUID | 20 min |
| 5 | H-1 | Rate limiting (ThrottlerModule) | 1 hr |
| 6 | M-1 | Fix publications search query bug | 20 min |
| 7 | H-3 | Cap pagination limit to 100 | 10 min |
| 8 | C-4 | Re-validate instrument in processor | 10 min |
| 9 | H-5 | UpdateNotesDto size + shape validation | 30 min |
| 10 | H-6 | Reduce presigned URL expiry to 300 s | 5 min |
| 11 | M-2 | Magic-byte MIME validation (`file-type`) | 30 min |
| 12 | H-4 | Scope rawBody to webhook route only | 45 min |
| 13 | M-4 | User profile sync on repeat login | 10 min |
| 14 | M-5 | Duplicate in-flight job prevention | 20 min |
| 15 | M-8 | Redis password | 10 min |
| 16 | L-1 | Sanitize worker error messages | 15 min |
| 17 | L-2 | Gate Swagger behind NODE_ENV check | 5 min |
| 18 | M-7 | Structured audit logging | 1 hr |
| 19 | M-6 | Temp file startup sweep in worker | 30 min |
| 20 | M-3 | ClamAV malware scanning (production only) | 2 hr |
| 21 | L-3 | HSTS (covered by H-2 / Helmet) | — |
| 22 | L-4 | Centralize public route list + unit test | 30 min |
