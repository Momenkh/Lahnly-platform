import {
  Injectable,
  NotFoundException,
  ForbiddenException,
  BadRequestException,
} from '@nestjs/common';
import { randomUUID } from 'crypto';
import { DatabaseService } from '../prisma/database.service';
import { StorageService } from '../storage/storage.service';
import { CreatePublicationDto } from './dto/create-publication.dto';

@Injectable()
export class PublicationsService {
  constructor(
    private db: DatabaseService,
    private storage: StorageService,
  ) {}

  async publish(userId: string, dto: CreatePublicationDto) {
    const { rows: songRows } = await this.db.query(
      `SELECT * FROM "Song" WHERE id = $1`,
      [dto.songId],
    );
    if (!songRows[0]) throw new NotFoundException('Song not found');
    if (songRows[0].userId !== userId) throw new ForbiddenException();

    const { rows: pubRows } = await this.db.query(
      `SELECT id FROM "Publication" WHERE "songId" = $1`,
      [dto.songId],
    );
    if (pubRows[0]) throw new BadRequestException('Song is already published');

    // Verify requested instruments have completed results
    const { rows: resultRows } = await this.db.query(
      `SELECT DISTINCT jr.instrument
       FROM "JobResult" jr
       JOIN "ProcessingJob" j ON j.id = jr."jobId"
       WHERE j."songId" = $1 AND j.status = 'DONE'`,
      [dto.songId],
    );
    const availableInstruments = new Set(resultRows.map((r: any) => r.instrument));
    for (const inst of dto.instruments) {
      if (!availableInstruments.has(inst)) {
        throw new BadRequestException(`No completed result for instrument: ${inst}`);
      }
    }

    const id = randomUUID();
    const { rows } = await this.db.query(
      `INSERT INTO "Publication" (id, "songId", "userId", title, description, instruments, "isPublic", "publishedAt")
       VALUES ($1, $2, $3, $4, $5, $6, true, NOW())
       RETURNING *`,
      [id, dto.songId, userId, dto.title, dto.description ?? null, dto.instruments],
    );
    return rows[0];
  }

  async unpublish(userId: string, publicationId: string) {
    const { rows } = await this.db.query(
      `SELECT id, "userId" FROM "Publication" WHERE id = $1`,
      [publicationId],
    );
    if (!rows[0]) throw new NotFoundException('Publication not found');
    if (rows[0].userId !== userId) throw new ForbiddenException();

    await this.db.query(`DELETE FROM "Publication" WHERE id = $1`, [publicationId]);
    return { deleted: true };
  }

  async findAll(page: number, limit: number, search?: string) {
    const skip = (page - 1) * limit;
    const searchClause = search
      ? `AND p.title ILIKE $3`
      : '';
    const params: any[] = search ? [limit, skip, `%${search}%`] : [limit, skip];

    const [itemsRes, countRes] = await Promise.all([
      this.db.query(
        `SELECT p.*,
                json_build_object('id', u.id, 'displayName', u."displayName") as user
         FROM "Publication" p
         JOIN "User" u ON u.id = p."userId"
         WHERE p."isPublic" = true ${searchClause}
         ORDER BY p."publishedAt" DESC
         LIMIT $1 OFFSET $2`,
        params,
      ),
      this.db.query(
        `SELECT COUNT(*)::int as count FROM "Publication" WHERE "isPublic" = true ${searchClause}`,
        search ? [`%${search}%`] : [],
      ),
    ]);
    return { items: itemsRes.rows, total: countRes.rows[0].count, page, limit };
  }

  async findOne(publicationId: string) {
    const { rows: pubRows } = await this.db.query(
      `SELECT p.*,
              json_build_object('id', u.id, 'displayName', u."displayName") as user
       FROM "Publication" p
       JOIN "User" u ON u.id = p."userId"
       WHERE p.id = $1`,
      [publicationId],
    );
    if (!pubRows[0] || !pubRows[0].isPublic) throw new NotFoundException('Publication not found');
    const pub = pubRows[0];

    // Get completed results for this song
    const { rows: resultRows } = await this.db.query(
      `SELECT jr.instrument, jr."previewStorageKey"
       FROM "JobResult" jr
       JOIN "ProcessingJob" j ON j.id = jr."jobId"
       WHERE j."songId" = $1 AND j.status = 'DONE'`,
      [pub.songId],
    );

    const previews: Record<string, string> = {};
    for (const instrument of pub.instruments as string[]) {
      const result = resultRows.find((r: any) => r.instrument === instrument);
      if (result) {
        previews[instrument] = await this.storage.presignedGetUrl(
          result.previewStorageKey,
          3600,
        );
      }
    }

    return { ...pub, previews };
  }
}
