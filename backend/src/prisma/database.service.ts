import { Injectable, OnModuleDestroy, Logger } from '@nestjs/common';
import { Pool, PoolClient, QueryResult } from 'pg';

function isConnectionError(err: any): boolean {
  if (!err) return false;
  const code: string = err.code ?? '';
  const msg: string = err.message ?? '';
  return (
    code === 'ECONNRESET' ||
    code === 'ECONNREFUSED' ||
    code === 'EPIPE' ||
    code === 'ETIMEDOUT' ||
    code === 'ENOTFOUND' ||
    code === 'CONNECTION_TERMINATED' ||
    msg.includes('Connection terminated') ||
    msg.includes('connection timeout') ||
    msg.includes('Client was closed') ||
    // check cause (pg wraps errors)
    isConnectionError(err.cause)
  );
}

@Injectable()
export class DatabaseService implements OnModuleDestroy {
  readonly pool: Pool;
  private readonly logger = new Logger(DatabaseService.name);

  constructor() {
    this.pool = new Pool({
      connectionString: process.env.DATABASE_URL,
      max: 5,
      idleTimeoutMillis: 1000,
      connectionTimeoutMillis: 3000,
      keepAlive: true,
      keepAliveInitialDelayMillis: 0,
    });

    this.pool.on('error', (err) => {
      this.logger.warn(`pg pool idle client error: ${err.message}`);
    });
  }

  async onModuleDestroy() {
    await this.pool.end();
  }

  async query(text: string, params?: any[]): Promise<QueryResult> {
    for (let attempt = 1; attempt <= 4; attempt++) {
      let client: PoolClient | undefined;
      try {
        client = await this.pool.connect();
        const result = await client.query(text, params);
        client.release();
        return result;
      } catch (err: any) {
        if (client) {
          client.release(true); // destroy — don't recycle a potentially broken connection
          client = undefined;
        }
        if (isConnectionError(err) && attempt < 4) {
          const wait = attempt * 250;
          this.logger.warn(`DB connection error (attempt ${attempt}/4): ${err.message} — retrying in ${wait}ms`);
          await new Promise((r) => setTimeout(r, wait));
          continue;
        }
        throw err;
      }
    }
    throw new Error('unreachable');
  }

  async connect(): Promise<PoolClient> {
    for (let attempt = 1; attempt <= 4; attempt++) {
      let client: PoolClient | undefined;
      try {
        client = await this.pool.connect();
        return client;
      } catch (err: any) {
        if (client) {
          client.release(true);
          client = undefined;
        }
        if (isConnectionError(err) && attempt < 4) {
          const wait = attempt * 250;
          this.logger.warn(`DB connect error (attempt ${attempt}/4): ${err.message} — retrying in ${wait}ms`);
          await new Promise((r) => setTimeout(r, wait));
          continue;
        }
        throw err;
      }
    }
    throw new Error('unreachable');
  }
}
