import {
  Injectable,
  NotFoundException,
  ForbiddenException,
} from '@nestjs/common';
import { randomUUID } from 'crypto';
import { DatabaseService } from '../prisma/database.service';
import { StorageService } from '../storage/storage.service';
import { UpdateNotesDto } from './dto/update-notes.dto';

@Injectable()
export class ResultsService {
  constructor(
    private db: DatabaseService,
    private storage: StorageService,
  ) {}

  private async getResult(userId: string, jobId: string, instrument: string) {
    const { rows: jobRows } = await this.db.query(
      `SELECT id, "userId" FROM "ProcessingJob" WHERE id = $1`,
      [jobId],
    );
    if (!jobRows[0]) throw new NotFoundException('Job not found');
    if (jobRows[0].userId !== userId) throw new ForbiddenException();

    const { rows: resultRows } = await this.db.query(
      `SELECT jr.*,
              COALESCE(
                json_agg(json_build_object('id', nr.id, 'resultId', nr."resultId", 'notes', nr.notes, 'createdAt', nr."createdAt")
                         ORDER BY nr."createdAt" DESC)
                FILTER (WHERE nr.id IS NOT NULL),
                '[]'
              ) as revisions
       FROM "JobResult" jr
       LEFT JOIN "NoteRevision" nr ON nr."resultId" = jr.id
       WHERE jr."jobId" = $1 AND jr.instrument = $2
       GROUP BY jr.id
       LIMIT 1`,
      [jobId, instrument],
    );
    if (!resultRows[0]) throw new NotFoundException('Result not found for this instrument');
    return resultRows[0];
  }

  async getNotes(userId: string, jobId: string, instrument: string) {
    const result = await this.getResult(userId, jobId, instrument);

    if (result.revisions.length > 0) {
      return { source: 'revision', notes: result.revisions[0].notes };
    }

    const url = await this.storage.presignedGetUrl(result.notesStorageKey, 300);
    const res = await fetch(url);
    const notes = await res.json();
    return { source: 'original', notes };
  }

  async saveNotes(userId: string, jobId: string, instrument: string, dto: UpdateNotesDto) {
    const result = await this.getResult(userId, jobId, instrument);

    const id = randomUUID();
    const { rows } = await this.db.query(
      `INSERT INTO "NoteRevision" (id, "resultId", notes, "createdAt")
       VALUES ($1, $2, $3, NOW())
       RETURNING *`,
      [id, result.id, JSON.stringify(dto.notes)],
    );
    return rows[0];
  }

  async getPreviewUrl(userId: string, jobId: string, instrument: string) {
    const result = await this.getResult(userId, jobId, instrument);
    const url = await this.storage.presignedGetUrl(result.previewStorageKey, 3600);
    return { url, expiresInSeconds: 3600 };
  }

  async listForJob(userId: string, jobId: string) {
    const { rows: jobRows } = await this.db.query(
      `SELECT id, "userId" FROM "ProcessingJob" WHERE id = $1`,
      [jobId],
    );
    if (!jobRows[0]) throw new NotFoundException('Job not found');
    if (jobRows[0].userId !== userId) throw new ForbiddenException();

    const { rows } = await this.db.query(
      `SELECT id, instrument, "evalScore", "createdAt" FROM "JobResult" WHERE "jobId" = $1 ORDER BY "createdAt"`,
      [jobId],
    );
    return rows;
  }
}
