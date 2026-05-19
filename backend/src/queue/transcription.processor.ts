import { Process, Processor } from '@nestjs/bull';
import { Logger } from '@nestjs/common';
import * as bull from 'bull';
import { randomUUID } from 'crypto';
import { ConfigService } from '@nestjs/config';
import { DatabaseService } from '../prisma/database.service';
import { StorageService } from '../storage/storage.service';
import { TranscriptionJobPayload, TRANSCRIPTION_QUEUE } from './transcription.types';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { spawn } from 'child_process';

@Processor(TRANSCRIPTION_QUEUE)
export class TranscriptionProcessor {
  private readonly logger = new Logger(TranscriptionProcessor.name);

  constructor(
    private db: DatabaseService,
    private storage: StorageService,
    private config: ConfigService,
  ) {}

  @Process()
  async handle(job: bull.Job<TranscriptionJobPayload>) {
    const { processingJobId, userId, audioStorageKey, instruments } = job.data;

    this.logger.log(`Processing job ${processingJobId} — instruments: ${instruments.join(', ')}`);

    await this.db.query(
      `UPDATE "ProcessingJob" SET status = 'RUNNING', "startedAt" = NOW() WHERE id = $1`,
      [processingJobId],
    );

    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'lahnly-'));
    const audioPath = path.join(tmpDir, path.basename(audioStorageKey));

    try {
      const presigned = await this.storage.presignedGetUrl(audioStorageKey, 600);
      const res = await fetch(presigned);
      if (!res.ok) throw new Error(`Failed to download audio: ${res.status}`);
      const audioBuffer = Buffer.from(await res.arrayBuffer());
      fs.writeFileSync(audioPath, audioBuffer);

      const processorPath = this.config.getOrThrow<string>('CORE_PROCESSOR_PATH');
      const pythonBin = this.config.get<string>('PYTHON_BIN', 'python');

      for (const instrument of instruments) {
        this.logger.log(`Running instrument: ${instrument}`);
        await this.runCoreProcessor(pythonBin, processorPath, audioPath, instrument);

        const songName = path.basename(audioPath, path.extname(audioPath));
        const outputDir = path.join(processorPath, 'outputs', songName, instrument);

        const notesKey = `results/${processingJobId}/${instrument}/notes.json`;
        const previewKey = `results/${processingJobId}/${instrument}/preview.wav`;

        await this.storage.upload(
          notesKey,
          fs.readFileSync(path.join(outputDir, '03_cleaned_notes.json')),
          'application/json',
        );
        await this.storage.upload(
          previewKey,
          fs.readFileSync(path.join(outputDir, '09_preview.wav')),
          'audio/wav',
        );

        const resultId = randomUUID();
        await this.db.query(
          `INSERT INTO "JobResult" (id, "jobId", instrument, "notesStorageKey", "previewStorageKey", "createdAt")
           VALUES ($1, $2, $3, $4, $5, NOW())`,
          [resultId, processingJobId, instrument, notesKey, previewKey],
        );

        this.logger.log(`Instrument ${instrument} done`);
      }

      await this.db.query(
        `UPDATE "ProcessingJob" SET status = 'DONE', "finishedAt" = NOW() WHERE id = $1`,
        [processingJobId],
      );
    } catch (err: any) {
      this.logger.error(`Job ${processingJobId} failed: ${err.message}`);

      await this.db.query(
        `UPDATE "ProcessingJob" SET status = 'FAILED', "errorMessage" = $1, "finishedAt" = NOW() WHERE id = $2`,
        [err.message, processingJobId],
      );

      const { rows } = await this.db.query(
        `SELECT "tokensDeducted", "userId" FROM "ProcessingJob" WHERE id = $1`,
        [processingJobId],
      );
      const processingJob = rows[0];
      if (processingJob && processingJob.tokensDeducted > 0) {
        const client = await this.db.connect();
        try {
          await client.query('BEGIN');
          await client.query(
            `UPDATE "TokenBalance" SET balance = balance + $1 WHERE "userId" = $2`,
            [processingJob.tokensDeducted, userId],
          );
          const txId = randomUUID();
          await client.query(
            `INSERT INTO "TokenTransaction" (id, "userId", delta, reason, "referenceId", "createdAt")
             VALUES ($1, $2, $3, 'JOB_REFUND', $4, NOW())`,
            [txId, userId, processingJob.tokensDeducted, processingJobId],
          );
          await client.query('COMMIT');
        } catch (refundErr) {
          await client.query('ROLLBACK');
          this.logger.error(`Refund failed for job ${processingJobId}: ${(refundErr as any).message}`);
        } finally {
          client.release();
        }
        this.logger.log(`Refunded ${processingJob.tokensDeducted} tokens to ${userId}`);
      }

      throw err;
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  }

  private runCoreProcessor(
    pythonBin: string,
    processorPath: string,
    audioPath: string,
    instrument: string,
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      const proc = spawn(
        pythonBin,
        ['main.py', audioPath, '--instrument', instrument, '--no-play', '--no-viz'],
        { cwd: processorPath },
      );

      proc.stdout.on('data', (d: Buffer) => {
        const line = d.toString().trim();
        if (line) this.logger.log(`[${instrument}] ${line}`);
      });
      proc.stderr.on('data', (d: Buffer) => {
        const line = d.toString().trim();
        if (line) this.logger.warn(`[${instrument}] STDERR: ${line}`);
      });

      proc.on('close', (code) => {
        if (code === 0) resolve();
        else reject(new Error(`core-processor exited with code ${code}`));
      });
      proc.on('error', reject);
    });
  }
}
