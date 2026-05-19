import {
  Injectable,
  NotFoundException,
  ForbiddenException,
} from '@nestjs/common';
import { randomUUID } from 'crypto';
import { DatabaseService } from '../prisma/database.service';
import { StorageService } from '../storage/storage.service';
import { CreateSongDto } from './dto/create-song.dto';

@Injectable()
export class SongsService {
  constructor(
    private db: DatabaseService,
    private storage: StorageService,
  ) {}

  async create(
    userId: string,
    dto: CreateSongDto,
    file: Express.Multer.File,
    durationSeconds: number,
  ) {
    const key = `songs/${userId}/${Date.now()}-${file.originalname}`;
    await this.storage.upload(key, file.buffer, file.mimetype);

    const id = randomUUID();
    const { rows } = await this.db.query(
      `INSERT INTO "Song" (id, "userId", title, "originalFilename", "audioStorageKey", "durationSeconds", "createdAt")
       VALUES ($1, $2, $3, $4, $5, $6, NOW())
       RETURNING *`,
      [id, userId, dto.title, file.originalname, key, durationSeconds],
    );
    return rows[0];
  }

  async findAllForUser(userId: string, page: number, limit: number) {
    const skip = (page - 1) * limit;
    const [songsRes, countRes] = await Promise.all([
      this.db.query(
        `SELECT s.*,
                COALESCE(
                  json_agg(json_build_object('id', j.id, 'status', j.status, 'createdAt', j."createdAt")
                           ORDER BY j."createdAt")
                  FILTER (WHERE j.id IS NOT NULL),
                  '[]'
                ) as jobs
         FROM "Song" s
         LEFT JOIN "ProcessingJob" j ON j."songId" = s.id
         WHERE s."userId" = $1
         GROUP BY s.id
         ORDER BY s."createdAt" DESC
         LIMIT $2 OFFSET $3`,
        [userId, limit, skip],
      ),
      this.db.query(
        `SELECT COUNT(*)::int as count FROM "Song" WHERE "userId" = $1`,
        [userId],
      ),
    ]);
    return { items: songsRes.rows, total: countRes.rows[0].count, page, limit };
  }

  async findOne(userId: string, songId: string) {
    const { rows: songRows } = await this.db.query(
      `SELECT * FROM "Song" WHERE id = $1`,
      [songId],
    );
    if (!songRows[0]) throw new NotFoundException('Song not found');
    if (songRows[0].userId !== userId) throw new ForbiddenException();

    const [jobsRes, pubRes] = await Promise.all([
      this.db.query(
        `SELECT j.*, COALESCE(
           json_agg(json_build_object('id', jr.id, 'instrument', jr.instrument, 'evalScore', jr."evalScore", 'createdAt', jr."createdAt"))
           FILTER (WHERE jr.id IS NOT NULL), '[]'
         ) as results
         FROM "ProcessingJob" j
         LEFT JOIN "JobResult" jr ON jr."jobId" = j.id
         WHERE j."songId" = $1
         GROUP BY j.id
         ORDER BY j."createdAt" DESC`,
        [songId],
      ),
      this.db.query(`SELECT * FROM "Publication" WHERE "songId" = $1`, [songId]),
    ]);

    return {
      ...songRows[0],
      jobs: jobsRes.rows,
      publication: pubRes.rows[0] ?? null,
    };
  }

  async remove(userId: string, songId: string) {
    const song = await this.findOne(userId, songId);

    const resultsRes = await this.db.query(
      `SELECT jr.* FROM "JobResult" jr
       JOIN "ProcessingJob" j ON j.id = jr."jobId"
       WHERE j."songId" = $1`,
      [songId],
    );
    const results = resultsRes.rows;

    await Promise.all([
      this.storage.delete(song.audioStorageKey),
      ...results.flatMap((r: any) => [
        this.storage.delete(r.notesStorageKey),
        this.storage.delete(r.previewStorageKey),
      ]),
    ]);

    // Delete in FK-safe order
    const jobIds = song.jobs.map((j: any) => j.id);
    if (jobIds.length > 0) {
      const resultIds = results.map((r: any) => r.id);
      if (resultIds.length > 0) {
        await this.db.query(
          `DELETE FROM "NoteRevision" WHERE "resultId" = ANY($1::text[])`,
          [resultIds],
        );
        await this.db.query(
          `DELETE FROM "JobResult" WHERE "jobId" = ANY($1::text[])`,
          [jobIds],
        );
      }
      await this.db.query(
        `DELETE FROM "ProcessingJob" WHERE "songId" = $1`,
        [songId],
      );
    }
    await this.db.query(`DELETE FROM "Publication" WHERE "songId" = $1`, [songId]);
    await this.db.query(`DELETE FROM "Song" WHERE id = $1`, [songId]);

    return { deleted: true };
  }
}
