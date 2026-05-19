import {
  Controller,
  Post,
  Req,
  Headers,
  BadRequestException,
} from '@nestjs/common';
import { ApiTags, ApiOperation } from '@nestjs/swagger';
import { Request } from 'express';
import { WebhooksService } from './webhooks.service';
import { Public } from '../common/decorators/public.decorator';

@ApiTags('Webhooks')
@Public()
@Controller('webhooks')
export class WebhooksController {
  constructor(private readonly webhooksService: WebhooksService) {}

  @ApiOperation({ summary: 'Stripe webhook endpoint (raw body, signature verified)' })
  @Post('stripe')
  async stripeWebhook(
    @Req() req: Request & { rawBody?: Buffer },
    @Headers('stripe-signature') sig: string,
  ) {
    if (!sig) throw new BadRequestException('Missing stripe-signature header');
    if (!req.rawBody) throw new BadRequestException('Missing raw body');
    return this.webhooksService.handleStripe(req.rawBody, sig);
  }
}
