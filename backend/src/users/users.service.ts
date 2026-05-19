import { Injectable, NotFoundException } from '@nestjs/common';
import { DatabaseService } from '../prisma/database.service';

@Injectable()
export class UsersService {
  constructor(private db: DatabaseService) {}

  async getProfile(userId: string) {
    const { rows } = await this.db.query(
      `SELECT u.id, u."clerkId", u.email, u."displayName", u."createdAt",
              CASE WHEN tb."userId" IS NOT NULL
                THEN json_build_object('userId', tb."userId", 'balance', tb.balance)
                ELSE NULL
              END as "tokenBalance"
       FROM "User" u
       LEFT JOIN "TokenBalance" tb ON tb."userId" = u.id
       WHERE u.id = $1`,
      [userId],
    );
    if (!rows[0]) throw new NotFoundException('User not found');
    return rows[0];
  }

  async getTransactions(userId: string, page: number, limit: number) {
    const skip = (page - 1) * limit;
    const [itemsRes, countRes] = await Promise.all([
      this.db.query(
        `SELECT id, "userId", delta, reason, "referenceId", "createdAt"
         FROM "TokenTransaction"
         WHERE "userId" = $1
         ORDER BY "createdAt" DESC
         LIMIT $2 OFFSET $3`,
        [userId, limit, skip],
      ),
      this.db.query(
        `SELECT COUNT(*)::int as count FROM "TokenTransaction" WHERE "userId" = $1`,
        [userId],
      ),
    ]);
    return { items: itemsRes.rows, total: countRes.rows[0].count, page, limit };
  }
}
