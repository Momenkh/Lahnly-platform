import { Injectable, OnModuleInit, OnModuleDestroy } from '@nestjs/common';
import { PrismaClient } from '@prisma/client';

@Injectable()
export class PrismaService extends PrismaClient implements OnModuleInit, OnModuleDestroy {
  constructor() {
    super({ datasourceUrl: process.env.DATABASE_URL });
  }

  async onModuleInit() {
    // Lazy connect — Prisma connects on first query automatically
    // Eager $connect() causes ECONNRESET on Docker Desktop at startup
  }

  async onModuleDestroy() {
    await this.$disconnect();
  }
}
