import { randomUUID } from 'crypto';
import {
  CanActivate,
  ExecutionContext,
  Injectable,
  UnauthorizedException,
} from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { ConfigService } from '@nestjs/config';
import { createClerkClient } from '@clerk/backend';
import { IS_PUBLIC_KEY } from '../common/decorators/public.decorator';
import { DatabaseService } from '../prisma/database.service';
import type { AuthUser } from '../common/types/auth-user';

@Injectable()
export class ClerkGuard implements CanActivate {
  private readonly clerk;

  constructor(
    private reflector: Reflector,
    private config: ConfigService,
    private db: DatabaseService,
  ) {
    this.clerk = createClerkClient({
      secretKey: config.getOrThrow<string>('CLERK_SECRET_KEY'),
    });
  }

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const isPublic = this.reflector.getAllAndOverride<boolean>(IS_PUBLIC_KEY, [
      context.getHandler(),
      context.getClass(),
    ]);
    if (isPublic) return true;

    const request = context.switchToHttp().getRequest();

    // Dev bypass: set DEV_BYPASS=true in .env and pass x-dev-user-id header
    if (this.config.get('DEV_BYPASS') === 'true') {
      const devUserId = request.headers['x-dev-user-id'] as string;
      if (!devUserId) throw new UnauthorizedException('DEV_BYPASS: missing x-dev-user-id header');

      request.user = await this.upsertUser(devUserId, `${devUserId}@dev.local`, 'Dev User', 9999);
      return true;
    }

    const authHeader: string = request.headers['authorization'] ?? '';
    const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : null;
    if (!token) throw new UnauthorizedException('Missing bearer token');

    let payload: { sub: string; email?: string };
    try {
      const verified = await this.clerk.verifyToken(token);
      payload = { sub: verified.sub, email: (verified as any).email };
    } catch {
      throw new UnauthorizedException('Invalid or expired token');
    }

    const clerkUser = await this.clerk.users.getUser(payload.sub).catch(() => null);
    const email = clerkUser?.emailAddresses?.[0]?.emailAddress ?? payload.email ?? '';
    const displayName = clerkUser
      ? [clerkUser.firstName, clerkUser.lastName].filter(Boolean).join(' ') || null
      : null;

    request.user = await this.upsertUser(payload.sub, email, displayName, 0);
    return true;
  }

  private async upsertUser(
    clerkId: string,
    email: string,
    displayName: string | null,
    initialBalance: number,
  ): Promise<AuthUser> {
    const id = randomUUID();

    await this.db.query(
      `INSERT INTO "User" (id, "clerkId", email, "displayName", "createdAt")
       VALUES ($1, $2, $3, $4, NOW())
       ON CONFLICT ("clerkId") DO NOTHING`,
      [id, clerkId, email, displayName],
    );

    const { rows } = await this.db.query(
      `SELECT id, "clerkId", email, "displayName", "createdAt" FROM "User" WHERE "clerkId" = $1`,
      [clerkId],
    );
    const user = rows[0] as AuthUser;

    await this.db.query(
      `INSERT INTO "TokenBalance" ("userId", balance) VALUES ($1, $2) ON CONFLICT ("userId") DO NOTHING`,
      [user.id, initialBalance],
    );

    return user;
  }
}
