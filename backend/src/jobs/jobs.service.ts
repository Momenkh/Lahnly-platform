import {
  Injectable,
  NotFoundException,
  ForbiddenException,
  HttpException,
  HttpStatus,
} from '@nestjs/common';
import { randomUUID } from 'crypto';
import { InjectQueue } from '@nestjs/bull';
import * as bull from 'bull';
import { DatabaseService } from '../prisma/database.service';
import { SongsService } from '../songs/songs.service';
import { CreateJobDto } from './dto/create-job.dto';
import {
  TRANSCRIPTION_QUEUE,
  TranscriptionJobPayload,
} from '../queue/transcription.types';

@Injectable()
export class JobsService {
  constructor(
    private db: DatabaseService,
    private songsService: SongsService,
    @InjectQueue(TRANSCRIPTION_QUEUE) private queue: bull.Queue,
  ) {}

  async create(userId: string, songId: string, dto: CreateJobDto) {
    const song = await this.songsService.findOne(userId, songId);
    const cost = Math.ceil(song.durationSeconds / 60) * dto.instruments.length;

    const client = await this.db.connect();
    let jobId: string;
    try {
      await client.query('BEGIN');

      const balRes = await client.query(
        `SELECT balance FROM "TokenBalance" WHERE "userId" = $1 FOR UPDATE`,
        [userId],
      );
      const balance = balRes.rows[0]?.balance ?? 0;
      if (balance < cost) {
        throw new HttpException(
          `Insufficient tokens. Required: ${cost}, available: ${balance}`,
          HttpStatus.PAYMENT_REQUIRED,
        );
      }

      jobId = randomUUID();
      await client.query(
        `INSERT INTO "ProcessingJob" (id, "songId", "userId", instruments, status, "tokensDeducted", "createdAt")
         VALUES ($1, $2, $3, $4, 'QUEUED', $5, NOW())`,
        [jobId, songId, userId, dto.instruments, cost],
      );

      await client.query(
        `UPDATE "TokenBalance" SET balance = balance - $1 WHERE "userId" = $2`,
        [cost, userId],
      );

      const txId = randomUUID();
      await client.query(
        `INSERT INTO "TokenTransaction" (id, "userId", delta, reason, "referenceId", "createdAt")
         VALUES ($1, $2, $3, 'JOB_DEDUCT', $4, NOW())`,
        [txId, userId, -cost, jobId],
      );

      await client.query('COMMIT');
    } catch (err) {
      await client.query('ROLLBACK');
      throw err;
    } finally {
      client.release();
    }

    const payload: TranscriptionJobPayload = {
      processingJobId: jobId,
      songId,
      userId,
      audioStorageKey: song.audioStorageKey,
      instruments: dto.instruments,
      durationSeconds: song.durationSeconds,
    };

    const bullJob = await this.queue.add(payload, {
      attempts: 1,
      removeOnComplete: true,
      removeOnFail: false,
    });

    await this.db.query(
      `UPDATE "ProcessingJob" SET "queueJobId" = $1 WHERE id = $2`,
      [String(bullJob.id), jobId],
    );

    return this.findOne(userId, jobId);
  }

  async findAllForSong(userId: string, songId: string) {
    await this.songsService.findOne(userId, songId);
    const { rows } = await this.db.query(
      `SELECT j.*,
              COALESCE(
                json_agg(json_build_object('id', jr.id, 'instrument', jr.instrument, 'evalScore', jr."evalScore")
                         ORDER BY jr."createdAt")
                FILTER (WHERE jr.id IS NOT NULL),
                '[]'
              ) as results
       FROM "ProcessingJob" j
       LEFT JOIN "JobResult" jr ON jr."jobId" = j.id
       WHERE j."songId" = $1
       GROUP BY j.id
       ORDER BY j."createdAt" DESC`,
      [songId],
    );
    return rows;
  }

  async findOne(userId: string, jobId: string) {
    const { rows } = await this.db.query(
      `SELECT j.*,
              COALESCE(
                json_agg(json_build_object('id', jr.id, 'instrument', jr.instrument, 'evalScore', jr."evalScore", 'createdAt', jr."createdAt")
                         ORDER BY jr."createdAt")
                FILTER (WHERE jr.id IS NOT NULL),
                '[]'
              ) as results
       FROM "ProcessingJob" j
       LEFT JOIN "JobResult" jr ON jr."jobId" = j.id
       WHERE j.id = $1
       GROUP BY j.id`,
      [jobId],
    );
    if (!rows[0]) throw new NotFoundException('Job not found');
    if (rows[0].userId !== userId) throw new ForbiddenException();
    return rows[0];
  }
}
