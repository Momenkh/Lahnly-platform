import { Injectable, BadRequestException, Logger } from '@nestjs/common';
import { randomUUID } from 'crypto';
import { ConfigService } from '@nestjs/config';
import { DatabaseService } from '../prisma/database.service';
import Stripe from 'stripe';

@Injectable()
export class WebhooksService {
  private readonly stripe: InstanceType<typeof Stripe>;
  private readonly webhookSecret: string;
  private readonly logger = new Logger(WebhooksService.name);

  constructor(
    private config: ConfigService,
    private db: DatabaseService,
  ) {
    this.stripe = new Stripe(config.getOrThrow<string>('STRIPE_SECRET_KEY'));
    this.webhookSecret = config.getOrThrow<string>('STRIPE_WEBHOOK_SECRET');
  }

  async handleStripe(rawBody: Buffer, sig: string) {
    let event: any;
    try {
      event = this.stripe.webhooks.constructEvent(rawBody, sig, this.webhookSecret);
    } catch (err: any) {
      this.logger.warn(`Stripe webhook signature verification failed: ${err.message}`);
      throw new BadRequestException('Invalid Stripe signature');
    }

    switch (event.type as string) {
      case 'invoice.paid':
        await this.handleInvoicePaid(event.data.object);
        break;
      case 'customer.subscription.created':
        this.logger.log(`Subscription created: ${event.data.object?.id}`);
        break;
      default:
        this.logger.debug(`Unhandled Stripe event: ${event.type}`);
    }

    return { received: true };
  }

  private async handleInvoicePaid(invoice: any) {
    const clerkId: string | undefined = invoice?.metadata?.clerkId;
    if (!clerkId) {
      this.logger.warn(`No clerkId in invoice metadata for invoice ${invoice?.id}`);
      return;
    }

    const { rows: userRows } = await this.db.query(
      `SELECT id FROM "User" WHERE "clerkId" = $1`,
      [clerkId],
    );
    if (!userRows[0]) {
      this.logger.warn(`No user found for clerkId ${clerkId}`);
      return;
    }
    const userId = userRows[0].id;

    let tokensToGrant = 0;
    for (const line of (invoice?.lines?.data ?? []) as any[]) {
      const priceId: string | undefined = line?.price?.id;
      if (!priceId) continue;
      const envKey = `STRIPE_TOKENS_${priceId.toUpperCase().replace(/-/g, '_')}`;
      tokensToGrant += parseInt(this.config.get<string>(envKey, '0'), 10);
    }

    if (tokensToGrant <= 0) {
      this.logger.warn(`No token grant configured for invoice ${invoice?.id}`);
      return;
    }

    const client = await this.db.connect();
    try {
      await client.query('BEGIN');

      await client.query(
        `INSERT INTO "TokenBalance" ("userId", balance)
         VALUES ($1, $2)
         ON CONFLICT ("userId") DO UPDATE SET balance = "TokenBalance".balance + $2`,
        [userId, tokensToGrant],
      );

      const txId = randomUUID();
      await client.query(
        `INSERT INTO "TokenTransaction" (id, "userId", delta, reason, "referenceId", "createdAt")
         VALUES ($1, $2, $3, 'SUBSCRIPTION_GRANT', $4, NOW())`,
        [txId, userId, tokensToGrant, invoice?.id ?? null],
      );

      await client.query('COMMIT');
    } catch (err) {
      await client.query('ROLLBACK');
      throw err;
    } finally {
      client.release();
    }

    this.logger.log(`Granted ${tokensToGrant} tokens to user ${userId} (${clerkId})`);
  }
}
